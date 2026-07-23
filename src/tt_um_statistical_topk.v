/*
 * tt_um_statistical_topk -- statistical top-k activation-sparsity primitive
 *
 * Open-silicon implementation of the statistical top-k operator from the
 * Spark Transformer paper (arXiv:2506.06644): an O(N), sort-free
 * approximate top-k for transformer FFN activation sparsity. The design
 * streams 8-bit quantized activations, learns their mean and variance
 * online (welford_unit), and gates each activation against the learned
 * Gaussian-tail threshold (topk_selector), emitting a 1-bit keep/skip
 * decision per sample.
 *
 * Pin map
 *   ui_in[7:0]  activation byte
 *   uio_in[0]   in_valid  -- rising edge: sample ui_in (>= 58 clk between edges)
 *   uio_in[1]   cfg_we    -- rising edge: latch the config fields below
 *   uio_in[5:2] cfg_alpha    (threshold gain, alpha = value/4; reset = 6 -> 1.5)
 *   uio_in[7:6] cfg_warmup   (warmup N = 16<<v samples & EMA window; reset = 2 -> 64)
 *   uo_out[0]   keep      -- 1 = keep this activation, 0 = skip
 *   uo_out[1]   out_valid -- 1-cycle pulse, 57 clks after in_valid edge
 *   uo_out[2]   warmup_active
 *   uo_out[3]   clamp_sticky (defensive overflow flag; 0 in normal operation)
 *   uo_out[7:4] keep_count[3:0] (low nibble of the running keep counter)
 *
 * Protocol: each rising edge on in_valid launches a fixed-latency
 * processing pass (57 clocks: three serial-MAC products plus the update
 * states -- see welford_unit.v); out_valid pulses when the decision is
 * ready, and edges arriving while a pass is in flight are ignored. At the
 * shuttle's 25 MHz clock the unit sustains ~430k decisions/s, ahead of a
 * saturated 3 Mbaud UART (~83 clocks/byte) feeding it in the demo.
 *
 * Copyright (c) 2026 Athithan Elamaran
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_statistical_topk (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  // ---- config registers (rising-edge cfg_we strobe) -----------------------
  reg [3:0] alpha;        // reset default 6: alpha = 1.5, ~6.7% Gaussian tail
  reg [1:0] warmup_sel;   // reset default 2: N = 64, EMA window 64
  reg       in_valid_q, cfg_we_q;

  wire in_valid_edge = uio_in[0] & ~in_valid_q;
  wire cfg_we_edge   = uio_in[1] & ~cfg_we_q;

  always @(posedge clk) begin
    if (!rst_n) begin
      alpha      <= 4'd6;
      warmup_sel <= 2'd2;
      in_valid_q <= 1'b0;
      cfg_we_q   <= 1'b0;
    end else begin
      in_valid_q <= uio_in[0];
      cfg_we_q   <= uio_in[1];
      if (cfg_we_edge) begin
        alpha      <= uio_in[5:2];
        warmup_sel <= uio_in[7:6];
      end
    end
  end

  // ---- the two submodules, wired with plain internal signals --------------
  wire [7:0]  x_r;
  wire [15:0] mean;
  wire [23:0] variance;
  wire [40:0] mac_p;
  wire        warmup_active, clamp_sticky, snap, rhs_ld, done;
  wire        keep, out_valid;
  wire [15:0] keep_count, index;

  welford_unit u_welford (
      .clk          (clk),
      .rst_n        (rst_n),
      .start        (in_valid_edge),
      .x            (ui_in),
      .warmup_sel   (warmup_sel),
      .alpha        (alpha),
      .x_r_out      (x_r),
      .mean         (mean),
      .variance     (variance),
      .warmup_active(warmup_active),
      .clamp_sticky (clamp_sticky),
      .snap         (snap),
      .rhs_ld       (rhs_ld),
      .done         (done),
      .mac_p        (mac_p)
  );

  topk_selector u_topk (
      .clk          (clk),
      .rst_n        (rst_n),
      .snap         (snap),
      .rhs_ld       (rhs_ld),
      .done         (done),
      .x_r          (x_r),
      .mean         (mean),
      .mac_p        (mac_p),
      .warmup_active(warmup_active),
      .keep         (keep),
      .out_valid    (out_valid),
      .keep_count   (keep_count),
      .index        (index)
  );

  // ---- outputs ------------------------------------------------------------
  assign uo_out  = {keep_count[3:0], clamp_sticky, warmup_active, out_valid, keep};
  assign uio_out = 8'b0;
  assign uio_oe  = 8'b0;

  // List all unused inputs to prevent warnings (variance stays exported by
  // welford_unit for FPGA builds/debug; the selector no longer consumes it --
  // its alpha^2*var product arrives pre-computed through the shared MAC)
  wire _unused = &{ena, index, keep_count[15:4], variance, 1'b0};

endmodule
