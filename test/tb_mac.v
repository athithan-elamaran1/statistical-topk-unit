/*
 * tb_mac -- standalone directed + random testbench for serial_mac
 *
 * Not part of the cocotb regression (test/Makefile does not reference it);
 * run it directly when touching the multiply engine:
 *
 *   iverilog -g2005 -o tb_mac test/tb_mac.v src/serial_mac.v && vvp tb_mac
 *
 * Covers the directed corners (0, 1, max magnitudes 65280^2, the RHS
 * extreme (2^24-1) * 225, full-width q = 2^17-1) plus 500 random operand
 * pairs, each checked against the * operator. Fails with $fatal (nonzero
 * exit) on any mismatch.
 *
 * Copyright (c) 2026 Athithan Elamaran
 * SPDX-License-Identifier: Apache-2.0
 */

`timescale 1ns / 1ps
`default_nettype none

module tb_mac;

  reg clk = 0, rst_n = 0, start = 0;
  reg  [23:0] m;
  reg  [16:0] q;
  wire        busy, done;
  wire [40:0] p;
  integer     errors = 0;
  integer     i;
  reg  [23:0] rm;
  reg  [16:0] rq;

  serial_mac dut (
      .clk(clk), .rst_n(rst_n), .start(start),
      .m(m), .q(q), .busy(busy), .done(done), .p(p)
  );

  always #5 clk = ~clk;

  task mult(input [23:0] a, input [16:0] b);
    reg [40:0] expect_p;
    begin
      expect_p = a * b;
      @(negedge clk); m = a; q = b; start = 1;
      @(negedge clk); start = 0;
      wait (!busy);
      @(negedge clk);
      if (p !== expect_p) begin
        $display("FAIL: %0d * %0d = %0d (expected %0d)", a, b, p, expect_p);
        errors = errors + 1;
      end
    end
  endtask

  initial begin
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    // directed corners
    mult(24'd0,        17'd0);
    mult(24'd0,        17'd131071);
    mult(24'd16777215, 17'd0);
    mult(24'd1,        17'd1);
    mult(24'd1,        17'd131071);
    mult(24'd16777215, 17'd131071);   // full-width product
    mult(24'd65280,    17'd65280);    // |delta|^2 maximum (proof P2 bound)
    mult(24'd16777215, 17'd225);      // RHS maximum: var_max * alpha_sq_max
    mult(24'd3,        17'd3);

    // random sweep
    for (i = 0; i < 500; i = i + 1) begin
      rm = $random;
      rq = $random;
      mult(rm, rq);
    end

    if (errors != 0)
      $fatal(1, "tb_mac: %0d mismatches", errors);
    $display("tb_mac: all products exact (9 directed + 500 random)");
    $finish;
  end

endmodule
