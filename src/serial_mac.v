/*
 * serial_mac -- radix-2 sequential multiplier (the design's ONE multiply engine)
 *
 * Every multiplication in the chip funnels through this unit:
 *
 *     op          multiplicand m       multiplier q        product
 *   ------------  -------------------  ------------------  --------------------
 *   RHS           variance (24b)       alpha^2 (8b, ext)   alpha^2 * var < 2^32
 *   M2 increment  |delta2| (17b, ext)  |delta| (17b)       delta*delta2 < 2^33
 *   decision LHS  |delta| (17b, ext)   |delta| (17b)       dx^2 < 2^33
 *
 * Implementation: the classic combined product/multiplier shift register.
 * One 25-bit adder, one 41-bit register, zero partial-product array:
 *
 *   r = {acc[23:0], q[16:0]}                       (41 bits)
 *   each cycle:  sum[24:0] = acc + (r[0] ? m : 0)  (25-bit add, carry kept)
 *                r         = {sum, r[16:1]}        (shift right absorbs carry)
 *   after 17 cycles: product = r
 *
 * This replaces v1's parallel 18x18 multiplier (a ~1,800-gate partial-product
 * array) and the selector's parallel 8x32 with ~1/10th the area, at 17 cycles
 * per product. Latency is fixed (always 17 iterations, even for the 8-bit
 * alpha^2 multiplier operand) so the sample pipeline is fully deterministic.
 *
 * The product register holds its value from `done` until the next `start`,
 * which the sequencer exploits: the final product of a sample pass (dx^2) is
 * consumed by the decision logic directly from `p` -- no extra wide register.
 *
 * Signedness: callers pass MAGNITUDES. The two Welford operands provably
 * share a sign (see ARCHITECTURE.txt proof P2), so |delta|*|delta2| equals
 * delta*delta2 exactly, and dx^2/alpha^2*var are non-negative by construction.
 * Keeping the engine unsigned removes all sign-extension logic from the
 * iteration loop.
 *
 * Copyright (c) 2026 Athithan Elamaran
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module serial_mac (
    input  wire        clk,
    input  wire        rst_n,      // synchronous, active low
    input  wire        start,      // 1-cycle pulse: latch m/q, begin iterating
    input  wire [23:0] m,          // multiplicand (magnitude)
    input  wire [16:0] q,          // multiplier (magnitude)
    output wire        busy,       // iterating (17 cycles after start)
    output wire        done,       // pulse: final iteration this cycle
    output wire [40:0] p           // product, valid after done until next start
);

  reg [40:0] r;                    // {acc[23:0], q[16:0]}
  reg [23:0] m_r;
  reg [4:0]  cnt;                  // 17 -> 0

  wire [24:0] sum = {1'b0, r[40:17]} + (r[0] ? {1'b0, m_r} : 25'd0);

  assign busy = (cnt != 5'd0);
  assign done = (cnt == 5'd1);
  assign p    = r;

  always @(posedge clk) begin
    if (!rst_n) begin
      r   <= 41'd0;
      m_r <= 24'd0;
      cnt <= 5'd0;
    end else if (start) begin
      r   <= {24'd0, q};
      m_r <= m;
      cnt <= 5'd17;
    end else if (busy) begin
      r   <= {sum, r[16:1]};
      cnt <= cnt - 5'd1;
    end
  end

endmodule
