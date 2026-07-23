"""Bit-exact fixed-point golden model of the statistical top-k unit.

This model is the single source of truth for the RTL. Every arithmetic
operation here maps 1:1 onto an operation in the Verilog, with identical
bit widths, identical truncation (floor) semantics, and identical update
ordering. The cocotb testbench drives the RTL and this model with the same
stimulus and requires bit-for-bit agreement on every output.

Number formats
--------------
  x       : 8-bit unsigned activation, integer 0..255
  mean    : 16-bit unsigned, Q8.8   (raw value = real value * 256)
  var     : 24-bit unsigned, Q16.8  (raw value = real value * 256)
  alpha   : 4-bit unsigned, alpha_real = alpha / 4   (0.25 steps, 0..3.75)
  n       : 8-bit saturating sample counter (saturates at 255)
  s       : shift amount = min(bit_length(n), K)  -- the power-of-2 gain
            schedule: gain = 2^-s.  During warmup s grows 0,1,2,... which
            makes the estimator an expanding-window average; after warmup
            s sticks at K, making it an exponential moving average with
            window 2^K.
  warmup  : 2-bit select, N = 16 << sel samples of forced keep-all,
            K = 4 + sel  (so the EMA window equals the warmup length)

Keep condition (variance space, sqrt-free, mathematically exact)
----------------------------------------------------------------
  real:   keep  <=>  x > mean  AND  (x - mean)^2 > alpha^2 * var
  fixed:  dx = (x << 8) - mean                    (Q9.8 signed)
          keep <=> dx > 0 AND 16*dx*dx > alpha_reg^2 * (var << 8)

  Derivation: dx^2 = (x-mean)^2 * 2^16;  (var << 8) = sigma^2 * 2^16;
  alpha^2 = alpha_reg^2 / 16.  Multiply both sides by 16 to clear the
  fraction; every quantity is an integer, so the comparison is exact.

Python's ">>" on negative ints is an arithmetic floor shift, which matches
Verilog ">>>" on signed values in two's complement. All shifts below rely
on this equivalence.
"""

MEAN_W = 16          # mean register width (Q8.8, unsigned)
VAR_W = 24           # variance register width (Q16.8, unsigned)
VAR_MAX = (1 << VAR_W) - 1
N_W = 8              # sample counter width (saturating)
N_MAX = (1 << N_W) - 1
CNT_W = 16           # keep / index counter widths

DEFAULT_ALPHA = 6    # alpha = 6/4 = 1.5  (one-sided Gaussian tail ~6.7%)
DEFAULT_WARMUP_SEL = 2  # N = 64 samples, K = 6 (EMA window 64)


class StatTopkModel:
    """Bit-exact model of tt_um_statistical_topk: the welford_unit statistics
    and the topk_selector decision. Models values, not cycles, so it is
    microarchitecture-agnostic (v1 parallel multiplier and v2 serial MAC
    both verify against it unchanged)."""

    def __init__(self, alpha=DEFAULT_ALPHA, warmup_sel=DEFAULT_WARMUP_SEL):
        self.reset(alpha=alpha, warmup_sel=warmup_sel)

    def reset(self, alpha=DEFAULT_ALPHA, warmup_sel=DEFAULT_WARMUP_SEL):
        # welford_unit state
        self.mean = 0          # Q8.8 unsigned
        self.var = 0           # Q16.8 unsigned
        self.n = 0             # samples seen so far (saturating)
        self.var_clamp_sticky = 0
        # topk_selector state
        self.keep_count = 0    # CNT_W-bit running count of kept activations
        self.index = 0         # CNT_W-bit stream position counter
        # config registers
        self.alpha = alpha & 0xF
        self.warmup_sel = warmup_sel & 0x3

    # -- config interface (mirrors cfg_we strobe in RTL) --------------------
    def write_config(self, alpha, warmup_sel):
        self.alpha = alpha & 0xF
        self.warmup_sel = warmup_sel & 0x3

    @property
    def warmup_n(self):
        return 16 << self.warmup_sel      # 16, 32, 64, 128

    @property
    def k(self):
        return 4 + self.warmup_sel        # 4, 5, 6, 7

    @property
    def warmup_active(self):
        return self.n < self.warmup_n

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _shift_schedule(n, k):
        """s = min(bit_length(n), k).  n=0 -> 0 (first sample loads mean=x)."""
        return min(n.bit_length(), k)

    def threshold_real(self):
        """Equivalent real-valued threshold mean + alpha*sigma (for analysis
        only -- the hardware never computes this; it exists to plot warmup
        convergence and adaptation)."""
        import math
        return self.mean / 256.0 + (self.alpha / 4.0) * math.sqrt(self.var / 256.0)

    # -- one sample ----------------------------------------------------------
    def step(self, x):
        """Process one 8-bit activation; returns dict of all observable outputs.

        Update order is exactly the RTL's value-update order:
          S_DEC : decision inputs snapshot PRE-update mean/var
                  (predict-then-update)
          S_MEAN: mean update
          S_VAR : variance update, counters
        The model is cycle-agnostic: the RTL interleaves additional states
        around these (v2 adds serial-multiplier wait states), but the order
        in which VALUES change is identical, so this model verifies any
        microarchitecture of the same algorithm bit-for-bit.
        """
        assert 0 <= x <= 255
        warmup = self.warmup_active

        # ---- S_DEC: decision against pre-update statistics ----
        dx = (x << 8) - self.mean                     # signed, |dx| <= 65280
        lhs = (dx * dx) << 4                          # 37-bit unsigned
        rhs = (self.alpha * self.alpha) * (self.var << 8)  # <= 225 * 2^32, 40-bit
        keep_raw = 1 if (dx > 0 and lhs > rhs) else 0
        keep = 1 if warmup else keep_raw

        # ---- S_MEAN: mean update (delta == dx, same subtraction) ----
        s = self._shift_schedule(self.n, self.k)
        mean_next = self.mean + (dx >> s)             # floor shift == >>> in RTL
        assert 0 <= mean_next < (1 << MEAN_W), "mean provably in range"
        # delta2 uses the POST-update mean (Welford M2 term)
        delta2 = (x << 8) - mean_next

        # ---- S_VAR: variance update ----
        prod = dx * delta2                            # signed 34-bit; provably >= 0
        term = prod >> 8                              # Q16.8; <= 16,646,400 < 2^24
        var_next = self.var + ((term - self.var) >> s)
        if var_next > VAR_MAX:                        # defensive clamp (unreachable
            var_next = VAR_MAX                        # in normal operation -- see
            self.var_clamp_sticky = 1                 # ARCHITECTURE.txt proof)
        if var_next < 0:
            var_next = 0
            self.var_clamp_sticky = 1

        self.mean = mean_next
        self.var = var_next
        self.n = min(self.n + 1, N_MAX)

        # ---- topk_selector counters ----
        self.index = (self.index + 1) & ((1 << CNT_W) - 1)
        if keep:
            self.keep_count = (self.keep_count + 1) & ((1 << CNT_W) - 1)

        # The warmup flag on uo_out[2] is the LIVE n < N comparison; by the
        # time out_valid pulses, n has already incremented for this sample.
        # So the reported flag is the post-update view ("still in warmup for
        # future samples"), while the keep decision above used the pre-update
        # view (keep-all covers the full N warmup samples).
        return {
            "keep": keep,
            "keep_raw": keep_raw,
            "warmup": 1 if self.warmup_active else 0,
            "warmup_used": 1 if warmup else 0,
            "keep_count": self.keep_count,
            "keep_count_nib": self.keep_count & 0xF,
            "index": self.index,
            "mean_q": self.mean,
            "var_q": self.var,
            "s": s,
            "clamp": self.var_clamp_sticky,
        }


class WelfordFloat:
    """Textbook exact Welford (float, arbitrary-n division) -- the reference
    the fixed-point unit approximates. Used to quantify fixed-point error."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def step(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self):
        return self.m2 / self.n if self.n > 0 else 0.0


if __name__ == "__main__":
    # quick self-checks
    import random
    random.seed(1)

    # 1) constant stream: var -> 0, mean -> x, nothing kept after warmup
    m = StatTopkModel()
    for _ in range(500):
        r = m.step(100)
    assert m.mean == 100 << 8, m.mean
    assert m.var == 0
    assert r["keep"] == 0, "constant stream must keep nothing after warmup"

    # 2) mean/var track a Gaussian. The EMA estimate fluctuates around the
    # truth (std ~ sigma*sqrt(lambda/2) ~ 2.2 for window 64), so compare the
    # TIME-AVERAGED estimates, where the fluctuation averages out.
    m = StatTopkModel()
    w = WelfordFloat()
    mean_acc = var_acc = cnt = 0
    for i in range(20000):
        x = max(0, min(255, round(random.gauss(80, 25))))
        m.step(x)
        w.step(x)
        if i >= 1000:
            mean_acc += m.mean / 256.0
            var_acc += m.var / 256.0
            cnt += 1
    avg_mean, avg_var = mean_acc / cnt, var_acc / cnt
    assert abs(avg_mean - w.mean) < 0.5, (avg_mean, w.mean)
    assert abs(avg_var - w.var) / w.var < 0.10, (avg_var, w.var)

    # 3) invariants under random stimulus incl. extremes
    m = StatTopkModel(alpha=15, warmup_sel=0)
    for i in range(5000):
        x = random.choice([0, 255, random.randrange(256)])
        m.step(x)
        assert 0 <= m.mean < (1 << MEAN_W)
        assert 0 <= m.var <= VAR_MAX
    assert m.var_clamp_sticky == 0, "clamp must be unreachable"

    print("topk_model self-checks passed")
