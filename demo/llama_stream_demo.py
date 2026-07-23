#!/usr/bin/env python3
"""Live demo: transformer FFN activations -> statistical top-k chip.

Hooks a HuggingFace transformer's FFN intermediate activations, quantizes
them to 8 bits, streams them to the statistical top-k unit, and applies the
chip's keep/skip mask back onto the model — measuring the achieved sparsity
and the effect on the model's output. This reproduces the Spark Transformer
"lazy neuron" statistics (arXiv:2506.06644) with the selection decision made
in (real or simulated) silicon.

Transports
  --mock          (default) bit-exact Python golden model of the chip
                  (sim/topk_model.py) -- runs anywhere, no hardware
  --port DEV      real chip behind a UART bridge (e.g. /dev/ttyUSB0);
                  1 byte out (activation), 1 byte back (bit0 = keep).
                  See demo/README.md for the demo-board bridge sketch.

Sources
  --synthetic     (default) synthetic Gaussian / ReLU-Gaussian streams --
                  no ML dependencies at all
  --hf MODEL      real model activations, e.g. --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0
                  (requires: pip install torch transformers)

Examples
  python3 demo/llama_stream_demo.py                       # zero-dep smoke demo
  python3 demo/llama_stream_demo.py --hf sshleifer/tiny-gpt2 --prompt "Hello"
  python3 demo/llama_stream_demo.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --port /dev/ttyUSB0 --alpha 7
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sim"))
from topk_model import StatTopkModel  # noqa: E402


# --------------------------------------------------------------------------
# chip transports
# --------------------------------------------------------------------------

class MockChip:
    """Bit-exact software stand-in for the silicon (same golden model the
    RTL is verified against). A fresh instance is a freshly-reset chip."""

    def __init__(self, alpha, warmup_sel):
        self.model = StatTopkModel(alpha=alpha, warmup_sel=warmup_sel)

    def send(self, x):
        return self.model.step(x)["keep"]


class SerialChip:
    """Real chip behind a UART bridge. Protocol: host sends the activation
    byte; bridge strobes IN_VALID, waits for OUT_VALID, and returns a status
    byte (bit0 = KEEP, bit2 = WARMUP). Statistics reset only with the chip's
    rst_n (a bridge/board action, not a byte-protocol command)."""

    def __init__(self, port, baud, alpha, warmup_sel):
        import serial  # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=2)
        # config frame: 0xFF escape, then cfg byte with alpha in bits [5:2]
        # and warmup select in bits [7:6] (bits [1:0] are ignored -- the
        # bridge drives CFG_WE itself); matches the demo/README.md sketch
        self.ser.write(bytes([0xFF, (warmup_sel << 6) | (alpha << 2)]))

    def send(self, x):
        x = 0xFE if x == 0xFF else x  # 0xFF reserved as escape
        self.ser.write(bytes([x]))
        r = self.ser.read(1)
        if not r:
            raise RuntimeError("chip timeout -- check bridge and wiring")
        return r[0] & 1


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------

def quantize_u8(values, lo=None, hi=None):
    """Asymmetric min-max quantization of a float sequence to uint8."""
    vals = list(values)
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi <= lo:
        return [0] * len(vals), lo, 1.0
    scale = 255.0 / (hi - lo)
    return [max(0, min(255, round((v - lo) * scale))) for v in vals], lo, scale


# --------------------------------------------------------------------------
# synthetic demo (no ML dependencies)
# --------------------------------------------------------------------------

def run_synthetic(make_chip, args):
    import math
    rng = random.Random(0)
    warm = 16 << args.warmup_sel          # length of the keep-all warmup
    theory = 0.5 * math.erfc(args.alpha / 4 / math.sqrt(2))  # 1 - Phi(alpha)
    print("== synthetic activation streams ==")

    # Every sample is fed to the chip (it needs the early ones to warm up);
    # only post-warmup / post-transient decisions are COUNTED.

    chip = make_chip()
    xs = [max(0, min(255, round(rng.gauss(80, 25)))) for _ in range(8000)]
    keeps = [chip.send(x) for x in xs]
    rate = sum(keeps[warm:]) / len(keeps[warm:])
    print(f"gaussian(80,25) : keep rate {rate:6.2%}  "
          f"(alpha={args.alpha / 4:.2f}, gaussian tail theory {theory:.2%})")

    chip = make_chip()
    xs = [max(0, min(255, round(rng.gauss(0, 60)))) for _ in range(8000)]
    keeps = [chip.send(x) for x in xs]
    rate = sum(keeps[warm:]) / len(keeps[warm:])
    print(f"ReLU(N(0,60))   : keep rate {rate:6.2%}  "
          f"(the FFN-like 'lazy neuron' case)")

    chip = make_chip()
    for x in (max(0, min(255, round(rng.gauss(60, 20)))) for _ in range(4000)):
        chip.send(x)
    tail = [max(0, min(255, round(rng.gauss(160, 20)))) for _ in range(4000)]
    keeps = [chip.send(x) for x in tail]
    rate = sum(keeps[1000:]) / len(keeps[1000:])  # skip re-adaptation transient
    print(f"shift 60 -> 160 : keep rate {rate:6.2%}  "
          f"after re-adaptation (no reset, no recalibration)")


# --------------------------------------------------------------------------
# HuggingFace demo
# --------------------------------------------------------------------------

def find_ffn_act_modules(model):
    """Locate per-layer FFN intermediate activation modules for common
    architectures (Llama-style mlp.act_fn, GPT2-style mlp.act)."""
    mods = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("act_fn", "act") and ".mlp." in f".{name}.":
            mods.append((name, mod))
    if not mods:
        raise RuntimeError("could not find FFN activation modules; "
                           "supported: Llama-style mlp.act_fn, GPT2-style mlp.act")
    return mods


def run_hf(make_chip, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"== loading {args.hf} ==")
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float32)
    model.eval()

    acts = find_ffn_act_modules(model)
    print(f"hooked {len(acts)} FFN activation modules")
    chips = {name: make_chip() for name, _ in acts}
    stats = {name: [0, 0] for name, _ in acts}  # kept, total
    mask_enabled = [False]

    def hook(name):
        def fn(module, inp, out):
            if not mask_enabled[0]:
                return None
            flat = out.detach().flatten().tolist()
            q, _, _ = quantize_u8(flat)
            chip = chips[name]
            keep = [chip.send(v) for v in q]
            stats[name][0] += sum(keep)
            stats[name][1] += len(keep)
            mask = torch.tensor(keep, dtype=out.dtype,
                                device=out.device).reshape(out.shape)
            return out * mask
        return fn

    handles = [mod.register_forward_hook(hook(name)) for name, mod in acts]
    ids = tok(args.prompt, return_tensors="pt")

    with torch.no_grad():
        mask_enabled[0] = False
        dense = model(**ids).logits[0, -1]
        mask_enabled[0] = True
        sparse = model(**ids).logits[0, -1]

    for h in handles:
        h.remove()

    kept = sum(s[0] for s in stats.values())
    total = sum(s[1] for s in stats.values())
    print(f"\nprompt: {args.prompt!r}")
    print(f"overall keep rate: {kept / total:.2%}  "
          f"({kept}/{total} activations across {len(acts)} layers)")
    per_layer = [s[0] / s[1] for s in stats.values() if s[1]]
    print(f"per-layer keep rate: min {min(per_layer):.2%}  "
          f"max {max(per_layer):.2%}")

    dp = torch.softmax(dense, -1)
    sp = torch.softmax(sparse, -1)
    d_top = int(dense.argmax())
    s_top = int(sparse.argmax())
    print(f"dense  next token: {tok.decode([d_top])!r} (p={dp[d_top]:.3f})")
    print(f"sparse next token: {tok.decode([s_top])!r} (p={sp[s_top]:.3f})")
    print(f"top-1 agreement: {'YES' if d_top == s_top else 'NO'}   "
          f"L1(prob) = {float((dp - sp).abs().sum()):.4f}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mock", action="store_true", default=True,
                    help="use the Python golden model as the chip (default)")
    ap.add_argument("--port", default=None,
                    help="serial port of the UART bridge to the real chip")
    ap.add_argument("--baud", type=int, default=3_000_000)
    ap.add_argument("--hf", default=None,
                    help="HuggingFace model id (else synthetic streams)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--alpha", type=int, default=6,
                    help="alpha register 0..15, threshold gain = alpha/4 "
                         "(6 -> 1.5 ~ 7%% Gaussian keep rate)")
    ap.add_argument("--warmup-sel", type=int, default=2,
                    help="warmup select 0..3, N = 16 << sel samples")
    args = ap.parse_args()

    if args.port:
        def make_chip():
            return SerialChip(args.port, args.baud, args.alpha, args.warmup_sel)
    else:
        def make_chip():
            return MockChip(args.alpha, args.warmup_sel)

    if args.hf:
        run_hf(make_chip, args)
    else:
        run_synthetic(make_chip, args)


if __name__ == "__main__":
    main()
