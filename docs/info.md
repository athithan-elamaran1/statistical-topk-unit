<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This is a silicon implementation of the **statistical top-k** primitive from the Spark
Transformer paper (arXiv:2506.06644): a sort-free, O(N) approximate top-k operator for
exploiting transformer FFN activation sparsity. Instead of sorting activations to find the
top k (O(N log N), the step the paper shows can slow training by up to 10x on TPUs), the
unit fits a Gaussian to the activation stream online and keeps an activation iff it exceeds
`mean + alpha * sigma` — one comparison per activation.

Two submodules connected by plain wires:

**welford_unit** — streams 8-bit quantized activations and maintains a running mean (Q8.8)
and variance (Q16.8) using Welford's incremental update with power-of-2 gains, so every
division is a bit shift. The gain schedule `2^-min(bitlength(n), K)` gives an
expanding-window average during warmup and converges to an exponential moving average
(window `2^K`) afterwards, so the threshold keeps adapting when the activation distribution
shifts. No divider, no multiplier for the mean path.

**topk_selector** — gates each activation against the learned threshold in **variance
space**: instead of computing `sigma = sqrt(var)`, the keep condition
`x > mean + alpha * sigma` is squared into `dx > 0 AND 16 * dx^2 > alpha^2 * (var << 8)`
(with `dx = (x << 8) - mean`, `alpha = alpha_reg / 4`). This is mathematically exact in
integer arithmetic — the sqrt disappears entirely.

The chip contains **exactly one multiplier**: a radix-2 serial MAC (classic combined
product/multiplier shift register — one 25-bit adder, no partial-product array) through
which the sequencer funnels all three products each sample needs: `alpha^2 * var`, the
Welford M2 increment `delta * delta2`, and the decision's `dx^2`. The FSM orders them so
every decision input reads pre-update statistics, and the final product is consumed for
the keep decision straight out of the MAC's product register. The MAC runs unsigned on
magnitudes — valid because Welford's two deviations provably always share a sign. Fixed
57-clock latency per sample: ~430k decisions/s at 25 MHz, still ahead of a saturated
3 Mbaud UART. (v1 of this design used a parallel 18x18 multiplier and needed 4 tiles;
serializing the multiplies fits it in 2 at 57% fewer cells — and *tripled* the
worst-corner setup slack to +17.1 ns, since a 25-bit adder is a far shorter critical
path than a multiplier array.)

During the first N samples (warmup, default 64) the unit outputs keep-all — conservative,
nothing is dropped before the statistics are reliable. Alpha (sparsity knob, 0.25 steps,
default 1.5 ~ 6.7% Gaussian keep rate) and the warmup/averaging window (16/32/64/128) are
live-reconfigurable through the config pins. The RTL is verified bit-for-bit against a
fixed-point Python golden model (~5,000-sample cocotb regression, all decisions exact).

## How to test

1. Reset the design (`rst_n` low, then high). `WARMUP` (uo[2]) reads 1.
2. Put an activation byte on `ui[7:0]` and give `IN_VALID` (uio[0]) a rising edge.
   Leave at least 58 clock cycles between edges.
3. 57 clocks after the edge, `OUT_VALID` (uo[1]) pulses for one cycle and `KEEP` (uo[0])
   holds the decision. The low nibble of the running keep counter appears on uo[7:4].
4. The first 64 samples are all kept (warmup). After that, stream any roughly-Gaussian
   byte source and the keep rate settles near the one-sided tail of the configured alpha
   (default alpha = 1.5 -> ~7%). A constant stream keeps nothing (no activation exceeds
   its own mean); step the stream's level and the unit re-adapts within a few hundred
   samples.
5. To reconfigure: drive `CFG_ALPHA[3:0]` (uio[5:2], alpha = value/4) and
   `CFG_WARMUP[1:0]` (uio[7:6], N = 16 << value), then pulse `CFG_WE` (uio[1]).
   E.g. alpha = 8 (2.0) drops the keep rate to ~2%; alpha = 5 (1.25) raises it to ~11%.

The repository ships a host-side demo (`demo/llama_stream_demo.py`) that hooks a
HuggingFace transformer's FFN activations, quantizes them to 8 bits, streams them to the
chip over a UART bridge (RP2040/FPGA), and overlays the chip's keep/skip decisions on the
model's — reproducing the Spark Transformer "lazy neuron" statistics on live silicon.

## External hardware

None required. For the LLM demo: any 3.3V UART bridge (the Tiny Tapeout demo board's
RP2040 works) carrying activation bytes from the host Python script.
