# SPDX-FileCopyrightText: © 2026 Athithan Elamaran
# SPDX-License-Identifier: Apache-2.0
#
# cocotb verification of tt_um_statistical_topk against the bit-exact Python
# golden model (sim/topk_model.py). Every test drives the RTL and the model
# with identical stimulus and requires bit-for-bit agreement on every
# observable output: keep, warmup flag, clamp flag, and the keep-count
# nibble. Only TT-level pins are used, so the same tests pass in RTL and
# gate-level simulation.

import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sim"))
from topk_model import StatTopkModel  # noqa: E402

# uo_out bit positions
BIT_KEEP = 0
BIT_VALID = 1
BIT_WARMUP = 2
BIT_CLAMP = 3

# fixed decision latency of the serial-MAC pipeline: in_valid edge ->
# out_valid pulse. 3 serial products (18 cycles each incl. operand load)
# plus the decision/update states; see welford_unit.v's FSM timeline.
LATENCY = 57


async def setup(dut):
    clock = Clock(dut.clk, 40, unit="ns")  # 25 MHz, the shuttle clock
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def write_config(dut, model, alpha, warmup_sel):
    """Pulse cfg_we (uio[1]) with the config fields on uio[7:2]."""
    dut.uio_in.value = (warmup_sel << 6) | (alpha << 2) | 0b10
    await ClockCycles(dut.clk, 2)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)
    model.write_config(alpha, warmup_sel)


async def send_sample(dut, x):
    """Rising edge on in_valid (uio[0]) with x on ui_in; wait for the
    out_valid pulse and return the full uo_out byte at that cycle."""
    dut.ui_in.value = x
    dut.uio_in.value = 0b01
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    for _ in range(LATENCY + 8):
        await ClockCycles(dut.clk, 1)
        v = int(dut.uo_out.value)
        if v & (1 << BIT_VALID):
            return v
    raise AssertionError("out_valid never pulsed")


async def stream_and_check(dut, model, xs, ctx=""):
    """Drive samples through RTL and model; compare every output bit."""
    for i, x in enumerate(xs):
        got = await send_sample(dut, x)
        exp = model.step(x)
        assert (got >> BIT_KEEP) & 1 == exp["keep"], \
            f"{ctx} sample {i} (x={x}): keep={got & 1} expected {exp['keep']}"
        assert (got >> BIT_WARMUP) & 1 == exp["warmup"], \
            f"{ctx} sample {i}: warmup flag mismatch"
        assert (got >> BIT_CLAMP) & 1 == exp["clamp"], \
            f"{ctx} sample {i}: clamp flag mismatch"
        assert (got >> 4) & 0xF == exp["keep_count_nib"], \
            f"{ctx} sample {i}: keep_count nibble {(got >> 4) & 0xF} " \
            f"expected {exp['keep_count_nib']}"


def gauss_stream(n, mu, sigma, seed):
    rng = random.Random(seed)
    return [max(0, min(255, round(rng.gauss(mu, sigma)))) for _ in range(n)]


@cocotb.test()
async def test_reset_state(dut):
    """After reset: warmup active, no valid, no keep, clamp clear, count 0."""
    await setup(dut)
    v = int(dut.uo_out.value)
    assert v == (1 << BIT_WARMUP), f"uo_out after reset = {v:#04x}, expected 0x04"
    assert int(dut.uio_oe.value) == 0
    assert int(dut.uio_out.value) == 0


@cocotb.test()
async def test_out_valid_timing(dut):
    """out_valid pulses exactly once, LATENCY clocks after the in_valid edge
    (fixed-latency serial-MAC pipeline: 3 x 18-cycle products + FSM states)."""
    await setup(dut)
    dut.ui_in.value = 42
    dut.uio_in.value = 0b01
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    for cyc in range(1, LATENCY + 8):
        await ClockCycles(dut.clk, 1)
        valid = (int(dut.uo_out.value) >> BIT_VALID) & 1
        assert valid == (1 if cyc == LATENCY else 0), \
            f"cycle {cyc}: out_valid={valid} (must pulse only at cycle {LATENCY})"


@cocotb.test()
async def test_bitexact_gaussian_stream(dut):
    """2000 Gaussian samples, default config: bit-exact vs golden model,
    and the post-warmup keep rate lands near the Gaussian-tail theory."""
    await setup(dut)
    model = StatTopkModel()
    xs = gauss_stream(2000, 80, 25, seed=1)
    await stream_and_check(dut, model, xs, ctx="gauss")
    rate = model.keep_count / len(xs)
    assert 0.03 < rate < 0.15, f"keep rate {rate:.2%} implausible for alpha=1.5"


@cocotb.test()
async def test_warmup_boundary_and_constant_stream(dut):
    """Warmup flag drops after exactly 64 samples (default); a constant
    stream keeps everything during warmup and nothing after."""
    await setup(dut)
    model = StatTopkModel()
    for i in range(200):
        got = await send_sample(dut, 100)
        exp = model.step(100)
        warm = (got >> BIT_WARMUP) & 1
        keep = (got >> BIT_KEEP) & 1
        # keep-all covers the full 64 warmup samples; the flag is the live
        # post-update view, so it clears one sample earlier (see topk_model)
        assert warm == (1 if i < 63 else 0), f"warmup flag wrong at sample {i}"
        assert keep == exp["keep"] == (1 if i < 64 else 0), \
            f"constant stream: keep={keep} at sample {i}"


@cocotb.test()
async def test_config_reconfiguration(dut):
    """Live alpha/warmup reconfiguration mid-stream stays bit-exact."""
    await setup(dut)
    model = StatTopkModel()
    await stream_and_check(dut, model, gauss_stream(300, 80, 25, seed=2),
                           ctx="pre-cfg")
    # tighter threshold (alpha=2.0), shorter window (N=32, K=5)
    await write_config(dut, model, alpha=8, warmup_sel=1)
    await stream_and_check(dut, model, gauss_stream(500, 80, 25, seed=3),
                           ctx="post-cfg")
    # sparsest option and widest window
    await write_config(dut, model, alpha=15, warmup_sel=3)
    await stream_and_check(dut, model, gauss_stream(500, 80, 25, seed=4),
                           ctx="post-cfg2")


@cocotb.test()
async def test_extreme_inputs(dut):
    """0x00/0xFF spikes and random extremes: no overflow (clamp stays 0),
    still bit-exact."""
    await setup(dut)
    model = StatTopkModel()
    rng = random.Random(5)
    xs = [rng.choice([0, 255, rng.randrange(256)]) for _ in range(800)]
    await stream_and_check(dut, model, xs, ctx="extreme")
    assert model.var_clamp_sticky == 0
    assert (int(dut.uo_out.value) >> BIT_CLAMP) & 1 == 0


@cocotb.test()
async def test_distribution_shift(dut):
    """Mean jumps 60 -> 160 mid-stream: unit adapts without reset,
    bit-exact throughout, and resumes keeping upper-tail samples."""
    await setup(dut)
    model = StatTopkModel()
    xs = gauss_stream(600, 60, 20, seed=6) + gauss_stream(900, 160, 20, seed=7)
    await stream_and_check(dut, model, xs, ctx="shift")
    kept_late = model.keep_count
    xs2 = gauss_stream(400, 160, 20, seed=8)
    await stream_and_check(dut, model, xs2, ctx="post-shift")
    assert model.keep_count > kept_late, \
        "unit stopped keeping after the shift -- threshold failed to adapt"


@cocotb.test()
async def test_busy_edge_ignored(dut):
    """An in_valid edge during a processing pass is dropped, not queued."""
    await setup(dut)
    model = StatTopkModel()
    # prime with a few samples
    await stream_and_check(dut, model, [50, 60, 70], ctx="prime")
    # first sample launches; second edge arrives 2 clks later, mid-pass
    dut.ui_in.value = 80
    dut.uio_in.value = 0b01
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)
    dut.ui_in.value = 200
    dut.uio_in.value = 0b01   # mid-pass edge: must be ignored
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    saw_valid = 0
    for _ in range(LATENCY + 8):
        await ClockCycles(dut.clk, 1)
        saw_valid += (int(dut.uo_out.value) >> BIT_VALID) & 1
    assert saw_valid == 1, f"expected exactly 1 decision, saw {saw_valid}"
    model.step(80)  # only the first sample counts
    # stream continues bit-exact afterwards -- proves x=200 never entered stats
    await stream_and_check(dut, model, gauss_stream(200, 80, 25, seed=9),
                           ctx="post-busy")
