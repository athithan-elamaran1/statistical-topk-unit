# Demo: transformer FFN activations through the statistical top-k unit

`llama_stream_demo.py` hooks a HuggingFace transformer's FFN intermediate
activations (`register_forward_hook` on each layer's activation function),
quantizes them to 8 bits, streams them through the statistical top-k unit, and
applies the chip's keep/skip mask back onto the live forward pass. It reports
the achieved keep rate (the Spark Transformer "lazy neuron" statistics) and the
effect of sparsification on the model's next-token prediction.

## Zero-dependency smoke run

Uses the bit-exact Python golden model as the chip and synthetic activation
streams — no hardware, no ML libraries:

```bash
python3 demo/llama_stream_demo.py
```

## Real model, simulated chip

```bash
pip install torch transformers
python3 demo/llama_stream_demo.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "The capital of France is"
```

(Any Llama-style `mlp.act_fn` or GPT2-style `mlp.act` architecture works;
`--hf sshleifer/tiny-gpt2` is a fast download for a quick test.)

## Real model, real silicon

The chip sits behind a UART bridge that translates bytes into pin strobes.
Byte protocol (host side is `SerialChip` in the demo script):

- host -> bridge: one activation byte (`0xFF` is reserved as an escape:
  `0xFF, cfg` writes the config register; data `0xFF` is clamped to `0xFE`)
- bridge -> host: one status byte per activation: bit0 = KEEP, bit2 = WARMUP

Bridge behavior per data byte (any 3.3V MCU works; the Tiny Tapeout demo
board's RP2040 is ideal):

1. drive `ui[7:0]` = activation byte
2. rising edge on `uio[0]` (IN_VALID), then drop it
3. wait for `uo[1]` (OUT_VALID) to pulse — 57 chip clocks
4. return `uo[7:0]` to the host

For a config frame (`0xFF, cfg`): drive `uio[7:2]` = cfg[7:2], pulse `uio[1]`
(CFG_WE). Config fields: bits [5:2] = alpha (threshold gain = value/4),
bits [7:6] = warmup select (N = 16 << value).

At 3 Mbaud the UART delivers ~300 kB/s (~83 chip clocks per byte at 25 MHz); the
chip decides in a fixed 57 clocks, so it always finishes before the next byte
arrives — the link, not the silicon, remains the bottleneck.

```bash
python3 demo/llama_stream_demo.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port /dev/ttyUSB0 --baud 3000000 --alpha 7
```

## What to expect

- Keep rates around 5–15% depending on alpha and layer (the paper's operating
  point of ~8% corresponds to alpha ≈ 7 on ReLU-like activation distributions;
  see `sim/analysis_output.txt`).
- Warmup → active transition: the first N=64 activations of each stream are
  all kept, then the threshold engages.
- Live adaptation: change prompts (or `--alpha` mid-run over the config
  channel) and watch the keep rate re-settle within a few hundred samples.
- Near-identical next-token predictions between the dense and sparse passes
  at moderate alpha — the lazy-neuron phenomenon in action.
