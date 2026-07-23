/*
 * topk_selector -- inference-time statistical top-k gate
 *
 * For each activation, emits a 1-bit keep/skip decision plus stream-position
 * and keep counters. The keep condition is the variance-space reformulation
 * of the Gaussian threshold x > mean + alpha * sigma -- squared to eliminate
 * the sqrt:
 *
 *   keep  <=>  dx > 0  AND  16 * dx^2  >  alpha^2 * (var << 8)
 *   where dx = (x << 8) - mean  (Q9.8) and alpha = alpha_reg / 4.
 *
 * This is mathematically exact, not an approximation: both sides are
 * integers and the scaling factors (2^16 from Q8.8 squaring, 2^16 from
 * var<<8, 16 from alpha^2 = alpha_reg^2/16) cancel precisely. The dx > 0
 * guard restores the sign information that squaring destroys (only the
 * upper tail is kept).
 *
 * v2: both products in the comparison come from welford_unit's serial MAC,
 * so this module contains NO multiplier at all:
 *   snap   (S_DEC) : latch dx sign and warmup flag -- PRE-update values.
 *                    dx > 0 reduces to the unsigned compare (x<<8) > mean.
 *   rhs_ld (S_MEAN): the MAC product register holds alpha^2 * variance
 *                    (computed from the pre-update variance); latch it.
 *   done   (S_OUT) : the MAC product register now holds dx^2 -- compare
 *                    directly against the latched RHS (with the exact
 *                    <<4 / <<8 scaling applied combinationally), register
 *                    the decision, pulse out_valid, bump counters.
 *
 * During warmup the decision is forced to keep-all: conservative, no
 * activation is ever dropped before the statistics are reliable.
 *
 * Copyright (c) 2026 Athithan Elamaran
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module topk_selector (
    input  wire        clk,
    input  wire        rst_n,          // synchronous, active low
    input  wire        snap,           // pulse: snapshot dx sign + warmup flag
    input  wire        rhs_ld,         // pulse: latch RHS product from the MAC
    input  wire        done,           // pulse: MAC holds dx^2, emit decision
    input  wire [7:0]  x_r,            // latched sample from welford_unit
    input  wire [15:0] mean,           // running mean, Q8.8 (pre-update at snap)
    input  wire [40:0] mac_p,          // shared MAC product register
    input  wire        warmup_active,
    output reg         keep,           // 1 = activation exceeds threshold
    output reg         out_valid,      // 1-cycle pulse qualifying `keep`
    output reg  [15:0] keep_count,     // running count of kept activations
    output reg  [15:0] index           // stream position (sparse index base)
);

  reg        dx_pos_r, warm_r;
  reg [31:0] rhs_r;                    // alpha^2 * variance < 2^32

  // dx > 0 reduces to an unsigned compare: (x << 8) > mean, both Q8.8
  wire dx_pos = ({x_r, 8'b0} > mean);

  // exact fixed-point predicate: 16 * dx^2 > alpha^2 * (var << 8).
  // dx^2 <= 65280^2 < 2^33, so mac_p[40:33] = 0 and the <<4 fits 37 bits.
  wire [40:0] lhs = {4'b0, mac_p[32:0], 4'b0};
  wire [40:0] rhs = {1'b0, rhs_r, 8'b0};
  wire keep_next  = warm_r | (dx_pos_r & (lhs > rhs));

  always @(posedge clk) begin
    if (!rst_n) begin
      dx_pos_r   <= 1'b0;
      warm_r     <= 1'b0;
      rhs_r      <= 32'd0;
      keep       <= 1'b0;
      out_valid  <= 1'b0;
      keep_count <= 16'd0;
      index      <= 16'd0;
    end else begin
      if (snap) begin
        dx_pos_r <= dx_pos;
        warm_r   <= warmup_active;
      end

      if (rhs_ld)
        rhs_r <= mac_p[31:0];          // alpha^2 * var < 225 * 2^24 < 2^32

      if (done) begin
        keep      <= keep_next;
        out_valid <= 1'b1;
        index     <= index + 16'd1;
        if (keep_next)
          keep_count <= keep_count + 16'd1;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end

  // mac_p[40:33] is provably zero when dx^2 is on the bus (dx^2 < 2^33)
  wire _unused = &{mac_p[40:33], 1'b0};

endmodule
