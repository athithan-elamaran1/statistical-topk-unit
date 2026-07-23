# statistical-topk-unit

![test](https://github.com/athithan-elamaran1/statistical-topk-unit/actions/workflows/test.yaml/badge.svg)
![gds](https://github.com/athithan-elamaran1/statistical-topk-unit/actions/workflows/gds.yaml/badge.svg)

Open-source silicon implementation of the **statistical top-k** primitive from the
Spark Transformer paper (Google DeepMind, NeurIPS 2025, [arXiv:2506.06644](https://arxiv.org/abs/2506.06644)) —
the first open-silicon implementation of online activation-statistics tracking for
transformer FFN sparsity, targeting the SkyWater 130nm open PDK via
[Tiny Tapeout](https://tinytapeout.com) (SKY26c shuttle, 1x2 tiles).

The chip accepts a stream of 8-bit quantized FFN activations, maintains a running mean
and variance using Welford-style online updates with power-of-2 gains (every division is
a bit shift), and gates each activation against the learned Gaussian-tail threshold —
emitting a 1-bit keep/skip decision per activation in real time. **No sorting, no
priority encoder, no square root, no divider, no offline calibration pass.** O(N) where
standard top-k is O(N log N) — the exact bottleneck the Spark paper measures as up to a
10x training slowdown on TPUs with conventional operators.

## Why

In trained transformers only ~8% of FFN neurons meaningfully activate per token (the
"lazy neuron" phenomenon). Exploiting that sparsity yields 2.5x FLOPs reduction and up
to 1.79x decode speedup — if the top ~8% can be identified fast. Statistical top-k fits
a Gaussian to the activation stream and reduces selection to one comparison per
activation:

```
keep  ⇔  x > mean + α·σ        (hardware form: dx > 0  AND  16·dx² > α²·(var << 8))
```

The squared ("variance-space") form on the right is mathematically **exact** in integer
arithmetic and eliminates the square root entirely — the most expensive block this
design would otherwise need.

## Architecture

One Verilog design, two submodules joined by plain wires — and **exactly one
multiplier in the entire chip**: a radix-2 serial MAC (the textbook combined
product/multiplier shift register — one 25-bit adder, no partial-product array)
through which *every* product the algorithm needs is sequenced. Fixed 57-clock
latency per sample = ~430k decisions/s at 25 MHz, still ~1.4x ahead of a saturated
3 Mbaud UART.

- **`welford_unit`** — running mean (Q8.8) and variance (Q16.8); expanding-window
  average during warmup morphing into an exponential moving average (window 16–128,
  reconfigurable) for live adaptation to distribution shift. Owns the serial MAC and
  sequences its three products per sample (α²·var, the Welford M2 increment
  delta·delta2, and the decision's dx²) so that predict-then-update semantics hold
  exactly and the *last* product is still sitting in the MAC's register when the
  decision fires — no wide holding register needed. The MAC runs unsigned on
  magnitudes, licensed by a proof that Welford's two deviations always share a sign.
- **`topk_selector`** — variance-space threshold compare (41-bit), warmup keep-all
  override, keep/index counters. Contains no arithmetic heavier than an increment.
  α is a live-reconfigurable 4-bit register (α = value/4: keep rates ~40% down to
  ~0.01%).

Bit-width proofs (mean confinement, product non-negativity, 24-bit variance
no-overflow) and the full module-by-module walkthrough are in
[ARCHITECTURE.txt](ARCHITECTURE.txt).

**v1 → v2:** the first hardened version used a parallel 18×18 multiplier shared
between two products — the GDS flow measured it at **138% of a 1x2-tile core**,
forcing 2x2 tiles. Profiling the layout showed the multiplier partial-product arrays
dominated combinational area, so v2 serializes all multiplication through the MAC.
Measured results of the swap (same LibreLane flow, same 25 MHz constraint):

| | v1 (parallel) | v2 (serial MAC) |
|---|---|---|
| tiles | 2x2 | **1x2** |
| standard cells | 5,772 | **2,492 (−57%)** |
| utilization | 67% of a 2x2 core | **64.5% of a 1x2 core** |
| worst-corner setup slack | +5.5 ns | **+17.1 ns (~3x)** |
| latency | 4 clks | 57 clks (UART byte period: 83) |

The counterintuitive part: serializing the multiplier made timing *better* — a
25-bit adder is a far shorter critical path than an 18×18 array, so worst-corner
slack tripled. (As-built under the relaxed 40 ns constraint the worst-corner path
is 22.9 ns, so 50 MHz would still need the optimizer to tighten it — plausible
under a 20 ns constraint, but untested and pointless: the UART is the bottleneck.)
The cost is latency the UART-bound system cannot observe. Both microarchitectures
were verified bit-for-bit against the same golden model; the v1 figures above are
from our earlier parallel-multiplier version, which motivated this redesign.

## Measured results

From the bit-exact fixed-point simulation ([sim/analysis_output.txt](sim/analysis_output.txt)):

| α register | α    | keep rate (Gaussian in) | 1-sided tail theory |
|-----------:|------|------------------------:|--------------------:|
| 5          | 1.25 | 10.97%                  | 10.56%              |
| 6 (default)| 1.50 | 7.07%                   | 6.68%               |
| 7          | 1.75 | 4.25%                   | 4.01%               |
| 8          | 2.00 | 2.43%                   | 2.28%               |

On ReLU-style FFN activation distributions, α = 1.75 lands at **8.1% keep — the Spark
paper's operating point**. Threshold error plateaus at ~64 warmup samples; after a
mid-stream distribution shift the threshold re-converges within one averaging window
with no reset. Fixed-point error vs. exact float Welford: 0.14 LSB (mean), 0.7%
(variance).

Verification: 8 cocotb tests, ~4,900 activations streamed through the RTL with
**bit-for-bit agreement** against the Python golden model on every output of every
sample (decisions, flags, counters) — including live reconfiguration, extremes, and
protocol abuse; the same suite passes on the routed gate-level netlist. Final
signoff (LibreLane/sky130A): **2,492 standard cells at 64.5% of the 1x2-tile core,
+17.1 ns worst-corner setup slack, zero DRC/LVS errors** (see the v1 → v2 table
above for how it got there).

## Repository layout

```
src/            Verilog RTL (top + welford_unit + topk_selector + serial_mac)
sim/            bit-exact Python golden model + design-space analysis
test/           cocotb testbench (runs identically against RTL and gate-level netlist)
demo/           HuggingFace → UART → chip demo (with zero-dependency --mock mode)
docs/info.md    Tiny Tapeout datasheet
ARCHITECTURE.txt  full design deep-dive: every decision, every bit width, every module
info.yaml       Tiny Tapeout submission config (1x2 tiles, 25 MHz)
```

## Reproduce everything

```bash
# golden model self-checks + design-space analysis (stdlib only)
python3 sim/topk_model.py && python3 sim/run_analysis.py

# RTL verification (needs iverilog; pip install -r test/requirements.txt)
cd test && make

# demo without hardware or ML deps
python3 demo/llama_stream_demo.py

# demo on a real model (pip install torch transformers)
python3 demo/llama_stream_demo.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The GDS build, precheck, and gate-level test run in CI via the Tiny Tapeout actions on
every push.

## Pinout

| pin | dir | function |
|-----|-----|----------|
| `ui[7:0]` | in | activation byte |
| `uio[0]` | in | IN_VALID — rising edge samples the activation (≥58 clks apart) |
| `uio[1]` | in | CFG_WE — rising edge latches config |
| `uio[5:2]` | in | CFG_ALPHA (α = value/4) |
| `uio[7:6]` | in | CFG_WARMUP (N = 16 << value) |
| `uo[0]` | out | KEEP decision |
| `uo[1]` | out | OUT_VALID (1-cycle pulse, 57 clks after IN_VALID) |
| `uo[2]` | out | WARMUP active |
| `uo[3]` | out | CLAMP sticky flag (0 in normal operation — provably) |
| `uo[7:4]` | out | keep counter low nibble |

## License

Apache-2.0
