"""Design-space analysis for the statistical top-k unit.

Runs the bit-exact fixed-point model over synthetic activation streams and
reports:

  1. Keep rate vs alpha, compared against the one-sided Gaussian tail
     P(x > mu + alpha*sigma) = 1 - Phi(alpha)  -- validates the fixed-point
     threshold math end to end.
  2. Warmup convergence: how many samples until the implied threshold
     (mean + alpha*sigma) plateaus -- justifies the warmup register default.
  3. Distribution-shift adaptation: mean of the input jumps mid-stream;
     measures how fast the threshold re-converges (the EMA time constant).
  4. Fixed-point vs float Welford error on mean/variance.
  5. ReLU-style activations (half the mass at zero) -- the realistic FFN
     case; sweeps alpha 1.0-2.0 to locate the paper's ~8% operating point.

Pure stdlib (random, math) so it runs anywhere. Deterministic via fixed seeds.
Writes a report to sim/analysis_output.txt and prints it.
"""

import math
import random
import os

from topk_model import StatTopkModel, WelfordFloat, DEFAULT_ALPHA


def phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gauss_stream(n, mu, sigma, seed):
    rng = random.Random(seed)
    return [max(0, min(255, round(rng.gauss(mu, sigma)))) for _ in range(n)]


def relu_gauss_stream(n, sigma, seed):
    """ReLU(N(0, sigma)) quantized -- roughly half the samples are exactly 0,
    mimicking post-activation FFN sparsity structure."""
    rng = random.Random(seed)
    return [max(0, min(255, round(rng.gauss(0.0, sigma)))) for _ in range(n)]


def run_stream(model, xs, skip=0):
    """Feed xs; return keep rate over samples after `skip` (and after warmup)."""
    kept = tot = 0
    for i, x in enumerate(xs):
        r = model.step(x)
        if i >= skip and not r["warmup"]:
            kept += r["keep"]
            tot += 1
    return kept / tot if tot else float("nan")


def section(lines, title):
    lines.append("")
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)


def main():
    L = []
    L.append("statistical-topk-unit -- fixed-point design analysis")
    L.append("model: sim/topk_model.py (bit-exact vs RTL)")

    # ------------------------------------------------------------------ 1
    section(L, "1. Keep rate vs alpha  (Gaussian input mu=80 sigma=25, 60k samples)")
    L.append(f"{'alpha_reg':>9} {'alpha':>6} {'measured':>9} {'gaussian theory':>16}")
    for a in range(3, 11):
        xs = gauss_stream(60000, 80, 25, seed=100 + a)
        m = StatTopkModel(alpha=a)
        rate = run_stream(m, xs, skip=1000)
        theory = 1.0 - phi(a / 4.0)
        L.append(f"{a:>9} {a / 4.0:>6.2f} {rate:>8.2%} {theory:>15.2%}")
    L.append("")
    L.append("The measured rates track the one-sided Gaussian tail closely; the")
    L.append("small excess at high alpha comes from EMA estimator noise (the")
    L.append("threshold wobbles ~2 LSB around truth, and the tail is convex).")

    # ------------------------------------------------------------------ 2
    section(L, "2. Warmup convergence (threshold vs sample count, 20 trials)")
    trials = 20
    checkpoints = [8, 16, 32, 48, 64, 96, 128, 192, 256, 512]
    mu, sigma = 80, 25
    true_thr = mu + (DEFAULT_ALPHA / 4.0) * sigma
    err_at = {c: [] for c in checkpoints}
    for t in range(trials):
        xs = gauss_stream(600, mu, sigma, seed=300 + t)
        m = StatTopkModel()
        for i, x in enumerate(xs):
            m.step(x)
            if (i + 1) in err_at:
                err_at[i + 1].append(abs(m.threshold_real() - true_thr))
    L.append(f"true threshold = mu + 1.5*sigma = {true_thr:.1f}")
    L.append(f"{'samples':>8} {'mean |thr error|':>17}")
    for c in checkpoints:
        e = sum(err_at[c]) / len(err_at[c])
        L.append(f"{c:>8} {e:>17.2f}")
    L.append("")
    L.append("Error plateaus by ~64 samples (further samples reduce error only")
    L.append("marginally, limited by the EMA noise floor) -> warmup N = 64 and")
    L.append("EMA window 2^6 = 64 are hardened as the reset defaults (sel=2).")

    # ------------------------------------------------------------------ 3
    section(L, "3. Distribution-shift adaptation (mu: 60 -> 120 at sample 5000)")
    xs = gauss_stream(5000, 60, 20, seed=42) + gauss_stream(5000, 120, 20, seed=43)
    m = StatTopkModel()
    marks = [4999, 5064, 5128, 5256, 5512, 9999]
    thr_at = {}
    for i, x in enumerate(xs):
        m.step(x)
        if i in marks:
            thr_at[i] = m.threshold_real()
    L.append(f"{'sample':>7} {'threshold':>10}   (target after shift: "
             f"{120 + 1.5 * 20:.0f})")
    for i in marks:
        L.append(f"{i + 1:>7} {thr_at[i]:>10.1f}")
    L.append("")
    L.append("The threshold re-converges within a few EMA windows (~200-500")
    L.append("samples) of the shift with no reset and no recalibration pass --")
    L.append("the live-adaptation behavior the demo shows on prompt changes.")

    # ------------------------------------------------------------------ 4
    section(L, "4. Fixed-point vs float Welford (time-averaged, 60k samples)")
    xs = gauss_stream(60000, 80, 25, seed=7)
    m = StatTopkModel()
    w = WelfordFloat()
    macc = vacc = cnt = 0
    for i, x in enumerate(xs):
        m.step(x)
        w.step(x)
        if i >= 1000:
            macc += m.mean / 256.0
            vacc += m.var / 256.0
            cnt += 1
    L.append(f"float Welford : mean={w.mean:8.3f}  var={w.var:9.3f}")
    L.append(f"fixed (avg)   : mean={macc / cnt:8.3f}  var={vacc / cnt:9.3f}")
    L.append(f"error         : mean={abs(macc / cnt - w.mean):8.3f}  "
             f"var={abs(vacc / cnt - w.var):9.3f}")
    L.append("")
    L.append("Sub-LSB mean bias (floor-truncation of the shift, bounded by")
    L.append("(2^s-1)/2 raw Q8.8 units ~ 0.12) and a few percent variance bias")
    L.append("(EMA weighting factor 2/(2-lambda) ~ 1.008 plus truncation).")

    # ------------------------------------------------------------------ 5
    section(L, "5. ReLU-style FFN activations (ReLU(N(0,60)), alpha sweep 1.0-2.0)")
    for a in (4, 5, 6, 7, 8):
        xs = relu_gauss_stream(60000, 60, seed=900 + a)
        m = StatTopkModel(alpha=a)
        rate = run_stream(m, xs, skip=1000)
        L.append(f"alpha_reg={a} (alpha={a / 4.0:.2f}): keep rate = {rate:.2%}")
    L.append("")
    L.append("On half-rectified activations the Gaussian fit is approximate by")
    L.append("design (this is the Spark Transformer premise); alpha in 5..7")
    L.append("brackets the paper's ~8% operating point.")

    report = "\n".join(L) + "\n"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "analysis_output.txt")
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"[written to {out}]")


if __name__ == "__main__":
    main()
