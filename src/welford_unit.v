/*
 * welford_unit -- online activation statistics engine + multiply sequencer
 *
 * Maintains a running mean (Q8.8) and variance (Q16.8) of an 8-bit
 * activation stream using Welford's incremental update form with
 * power-of-2 gains (all divisions are arithmetic right shifts):
 *
 *   delta  = (x << 8) - mean
 *   mean' = mean + (delta >>> s)
 *   delta2 = (x << 8) - mean'
 *   var'  = var + (((delta * delta2) >>> 8) - var) >>> s
 *
 * The shift schedule s = min(bit_length(n), K) gives an expanding-window
 * average during warmup (s = 0,1,2,... so the first sample loads the mean
 * exactly) and converges to an exponential moving average with window 2^K
 * once n >= 2^K. K is tied to the warmup select so the averaging window
 * always equals the warmup length: sel = 0..3 -> N = 16/32/64/128, K = 4..7.
 *
 * v2 microarchitecture: this module also owns the design's single multiply
 * engine (serial_mac, radix-2 shift-add) and sequences ALL THREE products
 * each sample needs through it -- the two statistics products and the
 * decision's right-hand side. The FSM orders them so that predict-then-
 * update semantics are preserved exactly (every product that feeds the
 * decision reads PRE-update state) and so that the final product, dx^2,
 * is still sitting in the MAC's product register when the selector makes
 * its decision -- eliminating the wide LHS holding register v1 needed:
 *
 *   S_IDLE     accept sample
 *   S_DEC      latch delta and |delta|; selector snapshots dx sign +
 *              warmup flag; MAC starts alpha^2 * variance   (pre-update var)
 *   S_RHS      wait 17 MAC cycles
 *   S_MEAN     mean update; latch delta2, |delta2|; selector latched the
 *              RHS product this cycle (rhs_ld strobe)
 *   S_PROD_GO  MAC starts |delta| * |delta2|  (the Welford M2 increment)
 *   S_PROD     wait 17 MAC cycles
 *   S_VAR      variance update from the MAC product; MAC starts
 *              |delta| * |delta| (the decision LHS, dx^2)
 *   S_SQ       wait 17 MAC cycles
 *   S_OUT      selector decides straight from the MAC product register
 *
 * Fixed latency 57 clocks/sample: at the 25 MHz shuttle clock the unit
 * decides ~430k samples/s, comfortably ahead of the saturated 3 Mbaud
 * UART (~83 clocks/byte) that feeds it in the demo.
 *
 * Overflow: mean is provably confined to [0, 65280] and the variance
 * update term to [0, 16646400] < 2^24 (see ARCHITECTURE.txt for proofs),
 * so the 24-bit variance register cannot legitimately overflow. The clamp
 * below is defensive (SEU/X-prop hardening) and sets a sticky flag.
 * The Welford product delta*delta2 is provably non-negative (proof P2:
 * the mean update can never overshoot), which is what licenses feeding
 * the unsigned serial MAC with magnitudes.
 *
 * Copyright (c) 2026 Athithan Elamaran
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module welford_unit (
    input  wire        clk,
    input  wire        rst_n,          // synchronous, active low
    input  wire        start,          // 1-cycle pulse: process `x` (ignored unless idle)
    input  wire [7:0]  x,              // activation sample
    input  wire [1:0]  warmup_sel,     // N = 16<<sel warmup samples, K = 4+sel
    input  wire [3:0]  alpha,          // threshold gain (alpha_real = alpha/4)
    output wire [7:0]  x_r_out,        // latched sample, stable through the pass
    output reg  [15:0] mean,           // running mean, Q8.8 unsigned
    output reg  [23:0] variance,       // running variance, Q16.8 unsigned
    output wire        warmup_active,  // high while n < N (selector keeps all)
    output reg         clamp_sticky,   // defensive clamp engaged (never in normal op)
    output wire        snap,           // pulse: selector snapshots dx sign + warmup
    output wire        rhs_ld,         // pulse: MAC product = alpha^2 * var, latch it
    output wire        done,           // pulse: MAC product = dx^2, decide now
    output wire [40:0] mac_p           // the shared MAC's product register
);

  localparam [3:0] S_IDLE    = 4'd0,
                   S_DEC     = 4'd1,
                   S_RHS     = 4'd2,
                   S_MEAN    = 4'd3,
                   S_PROD_GO = 4'd4,
                   S_PROD    = 4'd5,
                   S_VAR     = 4'd6,
                   S_SQ      = 4'd7,
                   S_OUT     = 4'd8;

  reg [3:0]  state;
  reg [7:0]  x_r;
  reg [7:0]  n;                        // saturating sample counter
  reg signed [17:0] delta_r;           // Q9.8 signed deviation (pre-update mean)
  reg [16:0] da_r, dm2_r;              // |delta|, |delta2| magnitudes for the MAC

  assign x_r_out = x_r;
  assign snap    = (state == S_DEC);
  assign rhs_ld  = (state == S_MEAN);
  assign done    = (state == S_OUT);

  // ---- shift schedule: s = min(bit_length(n), K) --------------------------
  function [3:0] bitlen;
    input [7:0] v;
    casez (v)
      8'b1???????: bitlen = 4'd8;
      8'b01??????: bitlen = 4'd7;
      8'b001?????: bitlen = 4'd6;
      8'b0001????: bitlen = 4'd5;
      8'b00001???: bitlen = 4'd4;
      8'b000001??: bitlen = 4'd3;
      8'b0000001?: bitlen = 4'd2;
      8'b00000001: bitlen = 4'd1;
      default:     bitlen = 4'd0;
    endcase
  endfunction

  wire [3:0] bl = bitlen(n);
  wire [2:0] kk = 3'd4 + {1'b0, warmup_sel};            // 4..7
  wire [2:0] s  = (bl > {1'b0, kk}) ? kk : bl[2:0];

  // ---- warmup ------------------------------------------------------------
  wire [7:0] warmup_n = 8'd16 << warmup_sel;            // 16/32/64/128
  assign warmup_active = (n < warmup_n);

  // ---- mean datapath (adds and shifts only) -------------------------------
  wire signed [17:0] xq        = {2'b00, x_r, 8'b0};    // x in Q8.8, sign-safe
  wire signed [17:0] mean_s    = {2'b00, mean};
  wire signed [17:0] delta     = xq - mean_s;           // |delta| <= 65280
  wire signed [17:0] mean_next = mean_s + (delta_r >>> s); // provably in [0,65280]
  wire signed [17:0] delta2    = xq - mean_next;

  // magnitudes for the unsigned MAC (products are provably non-negative)
  wire [16:0] delta_abs  = delta[17]  ? -delta[16:0]  : delta[16:0];
  wire [16:0] delta2_abs = delta2[17] ? -delta2[16:0] : delta2[16:0];

  // ---- the shared serial multiply engine ----------------------------------
  wire        mac_busy, mac_done;
  wire [7:0]  alpha_sq = alpha * alpha;                 // 4x4, a handful of gates
  wire        mac_start = (state == S_DEC) | (state == S_PROD_GO)
                        | (state == S_VAR);
  // operand schedule (see header table)
  wire [23:0] mac_m = (state == S_DEC)     ? variance
                    : (state == S_PROD_GO) ? {7'b0, dm2_r}
                    :                        {7'b0, da_r};
  wire [16:0] mac_q = (state == S_DEC)     ? {9'b0, alpha_sq}
                    :                        da_r;

  serial_mac u_mac (
      .clk  (clk),
      .rst_n(rst_n),
      .start(mac_start),
      .m    (mac_m),
      .q    (mac_q),
      .busy (mac_busy),
      .done (mac_done),
      .p    (mac_p)
  );

  // ---- variance update (from the MAC's M2-increment product) --------------
  // prod <= 65280^2 < 2^33, so mac_p[40:33] = 0; term = prod >> 8 (Q16.8).
  // Every operand below is DECLARED signed: a single unsigned operand (e.g.
  // a bare concatenation) would flip the whole expression to an unsigned
  // context and silently turn `diff >>> s` into a logical shift.
  wire [25:0] term = mac_p[33:8];                       // <= 16,646,400 < 2^24
  wire signed [27:0] term_s  = {2'b00, term};
  wire signed [27:0] var_s   = {4'b0000, variance};
  wire signed [27:0] diff    = term_s - var_s;
  wire signed [27:0] var_upd = var_s + (diff >>> s);

  wire var_over  = (var_upd > $signed(28'sd16777215));  // 2^24 - 1
  wire var_under = (var_upd < 28'sd0);
  wire [23:0] var_next = var_over  ? 24'hFFFFFF :
                         var_under ? 24'h000000 : var_upd[23:0];

  // ---- FSM ---------------------------------------------------------------
  always @(posedge clk) begin
    if (!rst_n) begin
      state        <= S_IDLE;
      x_r          <= 8'd0;
      n            <= 8'd0;
      mean         <= 16'd0;
      variance     <= 24'd0;
      delta_r      <= 18'sd0;
      da_r         <= 17'd0;
      dm2_r        <= 17'd0;
      clamp_sticky <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          if (start) begin
            x_r   <= x;
            state <= S_DEC;
          end
        end
        S_DEC: begin                   // stats pre-update; selector snaps;
          delta_r <= delta;            // MAC starts alpha^2 * variance
          da_r    <= delta_abs;
          state   <= S_RHS;
        end
        S_RHS:
          if (mac_done) state <= S_MEAN;
        S_MEAN: begin                  // selector latches RHS product (rhs_ld)
          mean  <= mean_next[15:0];
          dm2_r <= delta2_abs;
          state <= S_PROD_GO;
        end
        S_PROD_GO:                     // MAC starts |delta| * |delta2|
          state <= S_PROD;
        S_PROD:
          if (mac_done) state <= S_VAR;
        S_VAR: begin                   // MAC starts |delta| * |delta| (dx^2)
          variance     <= var_next;
          clamp_sticky <= clamp_sticky | var_over | var_under;
          if (n != 8'hFF)
            n <= n + 8'd1;
          state <= S_SQ;
        end
        S_SQ:
          if (mac_done) state <= S_OUT;
        S_OUT:                         // selector decides from mac_p
          state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  // mac_busy is implied by the wait states; the variance path reads only
  // mac_p[33:8] (prod < 2^33; low 8 bits are the >>8 rescale truncation)
  wire _unused = &{mac_busy, mac_p[40:34], mac_p[7:0], 1'b0};

endmodule
