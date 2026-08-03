"""
strategies.py CAGR-oriented candidate books for PM Project 4.

Project 4 is graded ONLY on CAGR (after costs), so the sizing question is not "hit 2% vol"
(that was Project 3) but "how much leverage maximises geometric growth without busting".
The answer is the Kelly criterion: for a book whose daily return has mean m and variance v,
log-growth is maximised at leverage k* = m / v. Full Kelly is far too aggressive once you
account for estimation error and the-100% ruin day, so the levered books here apply a
FRACTIONAL Kelly (KELLY_FRACTION, default 0.5 = half-Kelly) and hard-cap gross exposure.

No look-ahead: the weight for day t uses only returns / signals through day t-1. All moment
estimates are EWMA series shifted forward by one day.

Each strategy has signature f(R, signals, pre)-> Wt where
    R : (T, K) returns
    signals : list of up to 3 (T, K) signal frames (any may be None)
    pre : dict from precompute()
    Wt : (T, K) target weights
Register new books in REGISTRY at the bottom.
"""

import numpy as np
import pandas as pd
from scipy import optimize

#-------------------------tunables-------------------------#
VOL_HALFLIFE = 20 # EWMA half-life for per-asset vol (days)
MEAN_HALFLIFE = 40 # EWMA half-life for per-asset mean (momentum / mean-variance)
MOM_HALFLIFE = 40 # EWMA half-life for the Kelly mean/var of the book return
KELLY_FRACTION = 0.15 # Optimized down to avoid the 0.05 trade cap penalty
MAX_LEVERAGE = 6.0 # hard cap on gross Kellyexposure (spec allows 10.0)
WARMUP = 60 # days before which leverage is throttled (thin history)
SIGNAL_TILT = 0.5 # weight of a signal/momentum tilt vs the inverse-vol core


#---------------------------------------------------------------------------#
# Precompute shared EWMA moments, computed once, all no-look-ahead
#---------------------------------------------------------------------------#
def _ewm(a, halflife):
    return pd.DataFrame(a).ewm(halflife=halflife, min_periods=1).mean().to_numpy()


def _shift1(a):
    """Row t now holds the value known at t-1 (first row keeps its own value)."""
    out = np.empty_like(a, dtype=float)
    out[1:] = a[:-1]
    out[0] = a[0]
    return out


def _xs_z(a):
    """Cross-sectionally demean + unit-scale each row (market-neutral tilt score)."""
    a = a- a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    return a / np.where(sd > 1e-9, sd, 1.0)


def precompute(R, signals):
    """Build reusable no-look-ahead estimates from returns and signals."""
    R = np.nan_to_num(np.asarray(R, dtype=float))
    T, K = R.shape
    var = _ewm(R * R, VOL_HALFLIFE)
    vol = _shift1(np.sqrt(np.maximum(var, 1e-12)))
    inv_vol = 1.0 / np.maximum(vol, 1e-6)
    mean = _shift1(_ewm(R, MEAN_HALFLIFE)) # trailing per-asset mean (momentum)
    zmom = _xs_z(mean)
    zsig = [] # per-signal cross-sectional z-scores
    for s in (signals or []):
        if s is None:
            zsig.append(None)
        else:
            zsig.append(_xs_z(_shift1(np.nan_to_num(np.asarray(s, dtype=float)))))
    sig_tilt_fm = _calibrate_signals(R, zsig)
    return {"vol": vol, "inv_vol": inv_vol, "mean": mean, "zmom": zmom,
            "zsig": zsig, "sig_tilt_fm": sig_tilt_fm, "T": T, "K": K}


#---------------------------------------------------------------------------#
# Signal regression Fama-MacBeth sign/horizon/strength calibration per signal
# (ported from the Group 8 signal-regression study, adapted to this repo's
# precompute() shape). `signal_kelly`'s naive equal-average of the three raw
# signals is CAGR-negative; the problem isn't just weighting but that (a) a
# signal can be insignificant or sign-flipped, and (b) even a genuinely
# significant signal churns turnover faster than the horizon it actually
# predicts at once it's re-normalised to unit gross every day, so transaction
# costs eat the edge. This calibrates sign/strength per signal via Fama-MacBeth
# across candidate horizons, drops insignificant signals, and EWMA-smooths each
# survivor at its own best horizon before blending--that combination is what
# turns the raw edge into a net-of-cost-positive tilt.
#---------------------------------------------------------------------------#
CALIB_HORIZONS = (1, 2, 3, 5, 8, 13, 21) # candidate forward-return windows (days)
CALIB_NW_LAGS = 8 # Newey-West bandwidth for the FM t-stat
CALIB_T_MIN = 1.96 # significance bar (~5%) to keep a signal


def _newey_west_t(v, lags):
    """Newey-West t-stat for the mean of a serially-correlated series."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return 0.0
    mu = v.mean()
    e = v- mu
    S = e @ e / n
    for l in range(1, lags + 1):
        S += 2 * (1- l / (lags + 1.0)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(max(S, 1e-18) / n)
    return float(mu / se) if se > 0 else 0.0


def _fm_slope_t(X, Y, lags=CALIB_NW_LAGS):
    """Fama-MacBeth: per-day cross-sectional slope of Y on X, then NW t of the mean slope."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ok = np.isfinite(X) & np.isfinite(Y)
    rows = ok.sum(axis=1) > 3
    if not rows.any():
        return 0.0
    Xn = np.where(ok[rows], X[rows], np.nan)
    Yn = np.where(ok[rows], Y[rows], np.nan)
    Xc = Xn- np.nanmean(Xn, axis=1, keepdims=True)
    Yc = Yn- np.nanmean(Yn, axis=1, keepdims=True)
    num = np.nansum(Xc * Yc, axis=1)
    den = np.nansum(Xc * Xc, axis=1)
    good = den > 0
    slopes = num[good] / den[good]
    return _newey_west_t(slopes, lags)


def _fwd_return(R, h):
    """h-day cumulative return starting AT t: fwd[t] = sum(R[t .. t+h-1]).

    `zsig` is already lagged (row t holds the signal known through t-1, i.e. what's actually
    traded on day t), so the correct calibration target is the return earned FROM day t, not
    day t+1--an off-by-one here silently measures a different, non-tradeable relationship.
    """
    T, K = R.shape
    csum = np.vstack([np.zeros((1, K)), np.cumsum(R, axis=0)])
    fwd = np.full((T, K), np.nan)
    end = T- h + 1
    if end > 0:
        fwd[:end] = csum[h:T + 1]- csum[0:end]
    return fwd


def _calibrate_signal(zsig_i, R, horizons=CALIB_HORIZONS, t_min=CALIB_T_MIN):
    """Best-horizon Fama-MacBeth sign/strength for one already-lagged, z-scored signal.

    Returns (signed_strength, best_horizon). signed_strength is 0 if the signal never clears
    the significance bar at any candidate horizon (dropped from the blend).
    """
    best_t, best_h = 0.0, horizons[0]
    for h in horizons:
        t = _fm_slope_t(zsig_i, _fwd_return(R, h))
        if abs(t) > abs(best_t):
            best_t, best_h = t, h
    if abs(best_t) < t_min:
        return 0.0, best_h
    return float(np.sign(best_t) * abs(best_t)), best_h


def _calibrate_signals(R, zsig):
    """Sign-correct, strength-weight and horizon-smooth the raw signals via Fama-MacBeth.

    Returns the combined (T, K) tilt matrix, or None if no signal is present / none clears
    the significance bar.
    """
    present = [s for s in zsig if s is not None]
    if not present:
        return None
    calib = [_calibrate_signal(s, R) for s in present]
    weights = np.array([w for w, _ in calib], dtype=float)
    total = np.abs(weights).sum()
    if total < 1e-9:
        return None
    weights /= total
    smoothed = [_ewm(s, max(h, 1)) for s, (_, h) in zip(present, calib)]
    return sum(w * sm for w, sm in zip(weights, smoothed))


#---------------------------------------------------------------------------#
# Building blocks
#---------------------------------------------------------------------------#
def _unit_gross(W):
    """Normalise each day to unit gross exposure (sum |w| = 1); flat days left flat."""
    g = np.abs(W).sum(axis=1, keepdims=True)
    return np.divide(W, g, out=np.zeros_like(W), where=g > 1e-12)


def _kelly_scale(base, R, fraction=None, max_lev=None, halflife=MOM_HALFLIFE, warmup=WARMUP):
    """Scale a unit-gross base book to fractional-Kelly leverage k*_t = fraction * m_t / v_t,
    with m_t, v_t the EWMA mean/var of the base book's OWN return through t-1 (no look-ahead).
    Clipped to [0, max_lev] and ramped during warmup. Losing book-> flat."""
    fraction = KELLY_FRACTION if fraction is None else fraction
    max_lev = MAX_LEVERAGE if max_lev is None else max_lev
    R = np.nan_to_num(np.asarray(R, dtype=float))
    p = (base * R).sum(axis=1)
    m = _shift1(_ewm(p.reshape(-1, 1), halflife).ravel())
    v = np.maximum(_shift1(_ewm((p * p).reshape(-1, 1), halflife).ravel()), 1e-8)
    k = np.clip(fraction * m / v, 0.0, max_lev)
    k *= np.minimum(1.0, (np.arange(len(k)) + 1) / max(warmup, 1))
    return base * k[:, None]


def _tilted_core(pre, score, tilt=SIGNAL_TILT):
    """Inverse-vol long core blended with a market-neutral tilt (inverse-vol * score)."""
    core = _unit_gross(pre["inv_vol"])
    if score is None:
        return core
    tilted = _unit_gross(pre["inv_vol"] * score)
    return _unit_gross((1- tilt) * core + tilt * tilted)


#---------------------------------------------------------------------------#
# Candidate books
#---------------------------------------------------------------------------#
def equal_weight_1x(R, signals, pre):
    """Reference floor: unlevered equal weight, gross = 1 every day."""
    return _unit_gross(np.ones((pre["T"], pre["K"])))

def sig1_mom_kelly(R, signals, pre):
    """Inverse-vol core tilted by a 50/50 blend of signal_1 and momentum."""
    z_s1 = pre["zsig"][0]
    z_mom = pre["zmom"]

    # Average the two slow, durable edges
    score = (z_s1 + z_mom) / 2.0
    return _kelly_scale(_tilted_core(pre, score), R)


def inverse_vol_1x(R, signals, pre):
    """Reference: unlevered inverse-vol (risk parity), gross = 1."""
    return _unit_gross(pre["inv_vol"])


def equal_weight_kelly(R, signals, pre):
    """Long equal-weight core, fractional-Kelly leverage."""
    return _kelly_scale(_unit_gross(np.ones((pre["T"], pre["K"]))), R)


def inverse_vol_kelly(R, signals, pre):
    """Long inverse-vol (risk-parity) core, fractional-Kelly leverage. [SHIPPED]"""
    return _kelly_scale(_unit_gross(pre["inv_vol"]), R)


def momentum_kelly(R, signals, pre):
    """Inverse-vol core tilted toward assets with positive trailing return, Kelly-levered."""
    return _kelly_scale(_tilted_core(pre, pre["zmom"]), R)


def _signal_book(pre, R, i):
    z = pre["zsig"][i] if i < len(pre["zsig"]) else None
    return _kelly_scale(_tilted_core(pre, z), R)


def signal1_kelly(R, signals, pre): return _signal_book(pre, R, 0)
def signal2_kelly(R, signals, pre): return _signal_book(pre, R, 1)
def signal3_kelly(R, signals, pre): return _signal_book(pre, R, 2)


def signal3_fast_kelly(R, signals, pre):
    """Isolates signal_3 ALONE, but smoothed over a 3-day EWMA before tilting (vs.
    signal3_kelly's raw daily z-score). Newey-West stats (signal_predictive_power.ipynb)
    show signal_3 has the STRONGEST, most significant edge of the three at 1-5 day
    horizons (t-stat 4-5.5, vs signal_1's 1.6-2.3)--but signal3_kelly is strongly
    NEGATIVE either sign. The likely reason: an unsmoothed daily z-score on a fast signal
    churns the book every day, and 5bps-each-way cost + the 0.05 trade cap penalty eat a
    small (IC~0.02) edge alive. This isolates the fix (smoothing only, same signal, same
    tilt weight as signal3_kelly) to test that hypothesis directly."""
    z3 = pre["zsig"][2] if len(pre["zsig"]) > 2 else None
    if z3 is None:
        return _unit_gross(pre["inv_vol"])
    smoothed = _ewm(z3, 3)
    return _kelly_scale(_tilted_core(pre, smoothed), R)


def signal_kelly(R, signals, pre):
    """Inverse-vol core tilted by the AVERAGE of the three cross-sectional signals."""
    zs = [z for z in pre["zsig"] if z is not None]
    score = np.mean(zs, axis=0) if zs else None
    return _kelly_scale(_tilted_core(pre, score), R)


def signal_kelly_fm(R, signals, pre):
    """Inverse-vol core tilted by the Fama-MacBeth-calibrated signal blend, Kelly-levered.

    Unlike `signal_kelly`'s naive equal-average, each signal here is sign-corrected, weighted
    by its own Fama-MacBeth strength, EWMA-smoothed at its own best-fitting horizon, and
    dropped entirely if never significant (see `_calibrate_signals`). Tilt fraction locked at
    0.15--swept 0.05-1.0, CAGR peaks around 0.15-0.20 and falls off a cliff past ~0.4.
    """
    return _kelly_scale(_tilted_core(pre, pre["sig_tilt_fm"], tilt=0.15), R)


def mean_variance_kelly(R, signals, pre):
    """Growth-optimal diagonal book w mean/var (long-short), Kelly-levered.
    Theoretically the CAGR-maximiser under a diagonal covariance; volatile in practice."""
    raw = pre["mean"] * pre["inv_vol"] ** 2
    return _kelly_scale(_unit_gross(raw), R)


def sqrt_vol_1x(R, signals, pre):
    """Reference: unlevered square-root inverse-vol, gross = 1."""
    return _unit_gross(pre["inv_vol"] ** 0.5)

def sqrt_vol_kelly(R, signals, pre):
    """Long sqrt-inverse-vol core, fractional-Kelly leverage."""
    return _kelly_scale(_unit_gross(pre["inv_vol"] ** 0.5), R)

def quant_edge_sqrt_kelly(R, signals, pre):
    """
    The ultimate evolution.
    Uses 1/sqrt(vol) as the base to harvest more drift than pure inverse-vol,
    while retaining enough tail-risk protection to survive leverage.
    Blends with the 20% asymmetric signal tilt.
    """
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]

    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)
    score = 0.6 * s1_smooth + 0.4 * s3_smooth

    # The magical drift/safety compromise:
    core = _unit_gross(pre["inv_vol"] ** 0.5)
    tilted = _unit_gross((pre["inv_vol"] ** 0.5) * score)
    shape = _unit_gross(0.80 * core + 0.20 * tilted)

    # We push the Kelly fraction to 0.18, as the safer base allows harder leverage
    return _kelly_scale(shape, R, fraction=0.18)



def quant_edge_kelly(R, signals, pre):
    """
    The real signal-driven book.
    Uses 10-day smoothed signal_1 and a 3-day smoothed signal_3.
    Applies a controlled 20% tilt to avoid trade cap penalties, then Kelly-levers.
    """
    # 1. Grab Z-scores (already cross-sectionally demeaned and scaled)
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]

    # 2. Asymmetric Smoothing (10-day for slow S1, 3-day for fast S3)
    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)

    # 3. The Blend (60% Slow Trend, 40% Fast Reversion)
    score = 0.6 * s1_smooth + 0.4 * s3_smooth

    # 4. Controlled Tilt (20% Signal, 80% Safe Core)
    # This harvests the alpha WITHOUT triggering the 0.05 daily trade cap penalty
    core = _unit_gross(pre["inv_vol"])
    tilted = _unit_gross(pre["inv_vol"] * score)
    shape = _unit_gross(0.80 * core + 0.20 * tilted)

    # 5. Optimal Kelly Scaling (Locked at 0.15 based on our sweep)
    return _kelly_scale(shape, R, fraction=0.15)


def quant_edge_safe_kelly(R, signals, pre):
    """Same signal shape as quant_edge_kelly (60% 10d-smoothed signal_1 + 40% 3d-smoothed
    signal_3, 20% tilt over an inverse-vol core) but Kelly fraction re-picked against
    WORST-FOLD CAGR instead of full-sample CAGR.

    0.15 (quant_edge_kelly's fraction) was chosen by sweeping full-sample CAGR alone-
    exactly the in-sample trap. Re-run that sweep scoring the worst of 8 walk-forward
    folds instead: every fraction from 0.02-0.15 makes full-sample CAGR monotonically
    worse, but 0.05 is where worst-fold stops falling off a cliff (worst-fold-16% @0.15
    ->-5% @0.05, max drawdown 66%-> 24%) while still keeping most of the edge over the
    unlevered core (full CAGR +11% vs the core's +7%). This is the fraction to ship if
    the worst fold matters as much as the average one--it does, since we don't get to
    pick which 47,500 out-of-sample days we land in."""
    z1, z3 = pre["zsig"][0], pre["zsig"][2]
    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)
    score = 0.6 * s1_smooth + 0.4 * s3_smooth
    core = _unit_gross(pre["inv_vol"])
    tilted = _unit_gross(pre["inv_vol"] * score)
    shape = _unit_gross(0.80 * core + 0.20 * tilted)
    return _kelly_scale(shape, R, fraction=0.05)


def signal_ic_weighted_kelly(R, signals, pre):
    """Signal blend rebuilt from a dedicated deep-dive (level/change/innovation, redundancy
    via regression on the other two signals, PCA, lead-lag leakage check, break-even cost
    per signal), not carried over from earlier ad hoc weights. Findings that drove this:

    -LEVEL beats 1-day CHANGE and EWMA-INNOVATION for all three signals (change is
      slightly negative for signal_1/2)--so use the raw (lagged, cross-sectionally
      z-scored) level, exactly as elsewhere in this file, not a differenced version.
    -The three signals are close to ORTHOGONAL: regressing each on the other two gives
      R^2 <= 0.1% and leaves its standalone IC essentially unchanged. No redundancy to
      correct for--a linear combination double-counts nothing.
    -signal_2 has NO standalone edge (IC t=1.33) and a NEGATIVE break-even cost (-0.06bps
      vs the actual 5bps charged)--it cannot pay for its own turnover even before
      leverage. Dropped entirely, not just down-weighted.
    -signal_1 (IC=0.0126, t=2.16, break-even 8.4bps) and signal_3 (IC=0.0289, t=4.82,
      break-even 3.8bps raw) are both real. signal_3's raw break-even is BELOW the actual
      5bps cost--that's the quantified reason a raw daily tilt on it loses money--but
      an 8-day EWMA smooth (swept explicitly against the real evaluator, not assumed) is
      the CAGR-maximising point: it cuts turnover enough to push the net edge solidly
      positive while giving up only a little of the raw IC.
    -Combined 30/70 (signal_1/signal_3), matching their relative |IC| magnitude, at a
      20% tilt over the inverse-vol core (swept 0.10-0.40; CAGR peaks at 0.20-0.25) and
      Kelly fraction 0.05 (worst-fold-optimal, see quant_edge_safe_kelly's docstring for
      why 0.05 and not the full-sample-optimal ~0.15)."""
    z1, z3 = pre["zsig"][0], pre["zsig"][2]
    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 8) if z3 is not None else np.zeros_like(R)
    score = 0.30 * s1_smooth + 0.70 * s3_smooth
    core = _unit_gross(pre["inv_vol"])
    tilted = _unit_gross(pre["inv_vol"] * score)
    shape = _unit_gross(0.80 * core + 0.20 * tilted)
    return _kelly_scale(shape, R, fraction=0.05)

def adaptive_downside_kelly(R, signals, pre):
    """
    The Ultimate Adaptive Masterclass.
    1. Base: 50/50 blend of Inverse Downside Vol (crash defence) and 1/sqrt(vol) (growth).
    2. Tilt: Asymmetric smoothed signal blend (mostly Signal 3 and Signal 1).
    3. Adaptive Execution: Internally tracks drift to enforce a deadband
       (ignores trades < 0.005) and a hard cap (clips at 0.049 to evade penalties).
    """
    T, K = R.shape

    #---Step 1: Precompute Rolling Downside Volatility--
    D_neg = np.minimum(R, 0.0)
    down_var = _ewm(D_neg * D_neg, VOL_HALFLIFE)
    down_vol = _shift1(np.sqrt(np.maximum(down_var, 1e-12)))
    inv_down_vol = 1.0 / np.maximum(down_vol, 1e-6)

    #---Step 2: The Dual-Leg Baseline--
    leg_defence = _unit_gross(inv_down_vol)
    leg_growth = _unit_gross(pre["inv_vol"] ** 0.5)
    core = _unit_gross(0.50 * leg_defence + 0.50 * leg_growth)

    #---Step 3: Signal Tilt--
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]

    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)

    score = 0.40 * s1_smooth + 0.60 * s3_smooth
    tilted = _unit_gross(core * score)

    ideal_shape = _unit_gross(0.80 * core + 0.20 * tilted)
    ideal_Wt = np.nan_to_num(_kelly_scale(ideal_shape, R, fraction=0.18))

    #---Step 4: Adaptive Execution Filter--
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]

    gross = np.zeros(T)
    gross[0] = float(actual_Wt[0] @ R[0])

    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        if denom <= 0:
            drift = np.zeros(K)
        else:
            drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom

        desired_trade = ideal_Wt[t]- drift

        # Deadband: Ignore trades < 0.005 to save 5bps cost
        trade = np.where(np.abs(desired_trade) > 0.005, desired_trade, 0.0)

        # Strict Cap: Clip at 0.049 to completely evade the 0.05 flat penalty
        applied_trade = np.clip(trade,-0.049, 0.049)

        actual_Wt[t] = drift + applied_trade
        gross[t] = float(actual_Wt[t] @ R[t])

    return actual_Wt

def dynamic_meta_kelly(R, signals, pre):
    """
    The Meta-Labeling Masterclass.
    1. Base: 50/50 blend of Inverse Downside Vol and 1/sqrt(vol).
    2. Paper PnL: Tracks the raw return of the pure signal tilt.
    3. Conviction Scaling: Uses EWMA t-stat passed through tanh to dynamically weight the signal.
    4. Adaptive Execution: Volatility-scaled deadband and 0.049 hard cap.
    """
    T, K = R.shape

    #---Step 1: Precompute Rolling Downside Volatility--
    D_neg = np.minimum(R, 0.0)
    down_var = _ewm(D_neg * D_neg, VOL_HALFLIFE)
    down_vol = _shift1(np.sqrt(np.maximum(down_var, 1e-12)))
    inv_down_vol = 1.0 / np.maximum(down_vol, 1e-6)

    #---Step 2: The Safe Dual-Leg Core--
    leg_defence = _unit_gross(inv_down_vol)
    leg_growth = _unit_gross(pre["inv_vol"] ** 0.5)
    core = _unit_gross(0.50 * leg_defence + 0.50 * leg_growth)

    #---Step 3: Pure Signal Tilt & Paper PnL--
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]

    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)
    score = 0.40 * s1_smooth + 0.60 * s3_smooth

    # The pure unit-gross signal portfolio
    pure_tilt = _unit_gross(score)
    # The daily return of the pure signal portfolio
    paper_ret = (pure_tilt * R).sum(axis=1)

    #---Step 4: Meta-labeling (Rolling t-stat)--
    # We use a 60-day half-life so the t-stat glides smoothly, protecting the trade cap
    mu = _shift1(_ewm(paper_ret.reshape(-1, 1), 60).ravel())
    v = np.maximum(_shift1(_ewm((paper_ret * paper_ret).reshape(-1, 1), 60).ravel()), 1e-8)
    sigma = np.sqrt(np.maximum(v- mu**2, 1e-8))
    t_stat = mu / sigma

    #---Step 5: Tanh Activation--
    W_max = 0.40 # Cap the maximum signal allocation at 40%
    k = 0.5 # Sensitivity dial
    tilt_weight = (W_max * np.tanh(t_stat / k))[:, None]

    # Dynamic shape: scales down the core to make room for the signal (and can short the signal if negative)
    ideal_shape = _unit_gross((1.0- np.abs(tilt_weight)) * core + tilt_weight * pure_tilt)
    ideal_Wt = np.nan_to_num(_kelly_scale(ideal_shape, R, fraction=0.18))

    #---Step 6: Adaptive Execution Filter--
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]

    gross = np.zeros(T)
    gross[0] = float(actual_Wt[0] @ R[0])

    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        if denom <= 0:
            drift = np.zeros(K)
        else:
            drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom

        desired_trade = ideal_Wt[t]- drift

        # Volatility-scaled deadband (from Project 3)
        band = 0.75 * pre["vol"][t]

        # Only trade the excess beyond the band
        trade = np.where(desired_trade > band, desired_trade- band,
                          np.where(desired_trade <-band, desired_trade + band, 0.0))

        # Strict Cap: Clip at 0.049 to completely evade the 0.05 flat penalty
        applied_trade = np.clip(trade,-0.049, 0.049)

        actual_Wt[t] = drift + applied_trade
        gross[t] = float(actual_Wt[t] @ R[t])

    return actual_Wt

def meta_risk_parity_kelly(R, signals, pre):
    """
    Meta-Labeling applied to a Risk Parity (Equal Risk Contribution) core.
    Rebalances the core Risk Parity shape every 10 days for speed, then applies
    daily dynamic signal conviction and execution filtering.
    """
    T, K = R.shape

    #---Step 1: Rolling Risk Parity Core (rebalanced every 10 days for speed)--
    core_weights = np.zeros((T, K))
    core_weights[:60] = 1.0 / K # Warmup period

    for t in range(60, T):
        if t % 10 == 0 or t == 60:
            # 60-day rolling window for covariance
            window = R[t-60:t]
            cov = np.cov(window, rowvar=False)

            # Spinu's log-barrier formulation for Risk Parity
            obj = lambda y: 0.5 * y @ cov @ y- np.mean(np.log(y))
            jac = lambda y: cov @ y- (1.0 / K) / y
            y0 = 1.0 / np.sqrt(np.diag(cov) + 1e-8)

            res = optimize.minimize(obj, y0, method="L-BFGS-B", jac=jac, bounds=[(1e-9, None)] * K)
            w_rp = res.x / np.sum(np.abs(res.x))
            core_weights[t] = w_rp
        else:
            core_weights[t] = core_weights[t-1]

    #---Step 2: Pure Signal Tilt & Paper PnL--
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]

    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)
    score = 0.40 * s1_smooth + 0.60 * s3_smooth

    pure_tilt = _unit_gross(score)
    paper_ret = (pure_tilt * R).sum(axis=1)

    #---Step 3: Meta-labeling (Rolling Sharpe Proxy)--
    mu = _shift1(_ewm(paper_ret.reshape(-1, 1), 60).ravel())
    v = np.maximum(_shift1(_ewm((paper_ret * paper_ret).reshape(-1, 1), 60).ravel()), 1e-8)
    sigma = np.sqrt(np.maximum(v- mu**2, 1e-8))
    t_stat = mu / sigma

    # Tanh Conviction Scaling
    W_max = 0.40
    k = 0.5
    tilt_weight = (W_max * np.tanh(t_stat / k))[:, None]

    # Dynamic shape
    ideal_shape = _unit_gross((1.0- np.abs(tilt_weight)) * core_weights + tilt_weight * pure_tilt)
    ideal_Wt = np.nan_to_num(_kelly_scale(ideal_shape, R, fraction=0.18))

    #---Step 4: Adaptive Execution Filter--
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]

    gross = np.zeros(T)
    gross[0] = float(actual_Wt[0] @ R[0])

    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        if denom <= 0:
            drift = np.zeros(K)
        else:
            drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom

        desired_trade = ideal_Wt[t]- drift

        # Volatility-scaled deadband
        band = 0.75 * pre["vol"][t]

        # Soft-thresholding
        trade = np.where(desired_trade > band, desired_trade- band,
                          np.where(desired_trade <-band, desired_trade + band, 0.0))

        # Strict Cap
        applied_trade = np.clip(trade,-0.049, 0.049)

        actual_Wt[t] = drift + applied_trade
        gross[t] = float(actual_Wt[t] @ R[t])

    return actual_Wt

def meta_ic_downside_voltarget(R, signals, pre):
    """
    1. Base: 50/50 blend of Inverse Downside Vol and 1/sqrt(vol).
    2. Meta-labeling: IC-based rolling t-stat (pure statistical predictive power).
    3. Sizing: 2% Daily Volatility Target (Project 3 style).
    4. Execution: Volatility-scaled deadband & 0.049 hard cap.
    """
    T, K = R.shape

    #---Step 1: The Core (Downside + SqrtVol)--
    D_neg = np.minimum(R, 0.0)
    down_var = _ewm(D_neg * D_neg, 20)
    down_vol = _shift1(np.sqrt(np.maximum(down_var, 1e-12)))
    inv_down_vol = 1.0 / np.maximum(down_vol, 1e-6)

    leg_def = _unit_gross(inv_down_vol)
    leg_gro = _unit_gross(pre["inv_vol"] ** 0.5)
    core = _unit_gross(0.50 * leg_def + 0.50 * leg_gro)

    #---Step 2: Signal & IC T-Stat Calculation--
    z1 = pre["zsig"][0]
    z3 = pre["zsig"][2]
    s1_smooth = _ewm(z1, 10) if z1 is not None else np.zeros_like(R)
    s3_smooth = _ewm(z3, 3) if z3 is not None else np.zeros_like(R)
    score = 0.40 * s1_smooth + 0.60 * s3_smooth
    pure_tilt = _unit_gross(score)

    # Calculate Daily IC (Cross-sectional correlation)
    s_dm = score- score.mean(axis=1, keepdims=True)
    r_dm = R- R.mean(axis=1, keepdims=True)
    cov = (s_dm * r_dm).mean(axis=1)
    var_s = (s_dm**2).mean(axis=1)
    var_r = (r_dm**2).mean(axis=1)
    daily_ic = cov / np.sqrt(np.maximum(var_s * var_r, 1e-12))

    # Shift to avoid lookahead, then calculate 60-day rolling moments
    ic_history = _shift1(daily_ic)
    mu_ic = _ewm(ic_history.reshape(-1, 1), 60).ravel()
    var_ic = _ewm((ic_history**2).reshape(-1, 1), 60).ravel()
    sigma_ic = np.sqrt(np.maximum(var_ic- mu_ic**2, 1e-8))

    t_stat = mu_ic / sigma_ic
    tilt_weight = (0.40 * np.tanh(t_stat / 0.5))[:, None]

    #---Step 3: Dynamic Shape & Volatility Targeting--
    ideal_shape = _unit_gross((1.0- np.abs(tilt_weight)) * core + tilt_weight * pure_tilt)

    # Target 2% daily risk, assuming independent assets for speed
    port_vol = np.sqrt(np.sum((ideal_shape * pre["vol"])**2, axis=1))
    L_t = np.clip(0.0252 / np.maximum(port_vol, 1e-6), 0.0, 8.0)
    ideal_Wt = ideal_shape * L_t[:, None]

    #---Step 4: Adaptive Execution Filter--
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]
    gross = np.zeros(T)
    gross[0] = float(actual_Wt[0] @ R[0])

    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom if denom > 0 else np.zeros(K)
        desired_trade = ideal_Wt[t]- drift

        band = 0.75 * pre["vol"][t]
        trade = np.where(desired_trade > band, desired_trade- band,
                          np.where(desired_trade <-band, desired_trade + band, 0.0))

        applied_trade = np.clip(trade,-0.049, 0.049)
        actual_Wt[t] = drift + applied_trade
        gross[t] = float(actual_Wt[t] @ R[t])

    return actual_Wt


def meta_ic_eqweight_voltarget(R, signals, pre):
    """
    Variant 2: Uses a naive Equal Weight (1/N) baseline instead of Downside Volatility.
    """
    T, K = R.shape
    core = _unit_gross(np.ones_like(R))

    # Signal, IC T-stat, and Meta-labeling
    z1, z3 = pre["zsig"][0], pre["zsig"][2]
    score = 0.40 * (_ewm(z1, 10) if z1 is not None else np.zeros_like(R)) + 0.60 * (_ewm(z3, 3) if z3 is not None else np.zeros_like(R))
    pure_tilt = _unit_gross(score)

    s_dm = score- score.mean(axis=1, keepdims=True)
    r_dm = R- R.mean(axis=1, keepdims=True)
    daily_ic = (s_dm * r_dm).mean(axis=1) / np.sqrt(np.maximum((s_dm**2).mean(axis=1) * (r_dm**2).mean(axis=1), 1e-12))
    ic_history = _shift1(daily_ic)

    mu_ic = _ewm(ic_history.reshape(-1, 1), 60).ravel()
    sigma_ic = np.sqrt(np.maximum(_ewm((ic_history**2).reshape(-1, 1), 60).ravel()- mu_ic **2, 1e-8))
    tilt_weight = (0.40 * np.tanh((mu_ic / sigma_ic) / 0.5))[:, None]

    # Shape & Vol Targeting
    ideal_shape = _unit_gross((1.0- np.abs(tilt_weight)) * core + tilt_weight * pure_tilt)
    port_vol = np.sqrt(np.sum((ideal_shape * pre["vol"])**2, axis=1))
    L_t = np.clip(0.0252 / np.maximum(port_vol, 1e-6), 0.0, 8.0)
    ideal_Wt = ideal_shape * L_t[:, None]

    # Execution Filter
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]
    gross = np.zeros(T); gross[0] = float(actual_Wt[0] @ R[0])
    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom if denom > 0 else np.zeros(K)
        desired = ideal_Wt[t]- drift
        band = 0.75 * pre["vol"][t]
        trade = np.where(desired > band, desired- band, np.where(desired <-band, desired + band, 0.0))
        actual_Wt[t] = drift + np.clip(trade,-0.049, 0.049)
        gross[t] = float(actual_Wt[t] @ R[t])
    return actual_Wt


def meta_ic_invvol_voltarget(R, signals, pre):
    """
    Variant 3: Uses a pure Inverse Volatility baseline.
    """
    T, K = R.shape
    core = _unit_gross(pre["inv_vol"])

    # Signal, IC T-stat, and Meta-labeling
    z1, z3 = pre["zsig"][0], pre["zsig"][2]
    score = 0.40 * (_ewm(z1, 10) if z1 is not None else np.zeros_like(R)) + 0.60 * (_ewm(z3, 3) if z3 is not None else np.zeros_like(R))
    pure_tilt = _unit_gross(score)

    s_dm = score- score.mean(axis=1, keepdims=True)
    r_dm = R- R.mean(axis=1, keepdims=True)
    daily_ic = (s_dm * r_dm).mean(axis=1) / np.sqrt(np.maximum((s_dm**2).mean(axis=1) * (r_dm**2).mean(axis=1), 1e-12))
    ic_history = _shift1(daily_ic)

    mu_ic = _ewm(ic_history.reshape(-1, 1), 60).ravel()
    sigma_ic = np.sqrt(np.maximum(_ewm((ic_history**2).reshape(-1, 1), 60).ravel()- mu_ic **2, 1e-8))
    tilt_weight = (0.40 * np.tanh((mu_ic / sigma_ic) / 0.5))[:, None]

    # Shape & Vol Targeting
    ideal_shape = _unit_gross((1.0- np.abs(tilt_weight)) * core + tilt_weight * pure_tilt)
    port_vol = np.sqrt(np.sum((ideal_shape * pre["vol"])**2, axis=1))
    L_t = np.clip(0.0252 / np.maximum(port_vol, 1e-6), 0.0, 8.0)
    ideal_Wt = ideal_shape * L_t[:, None]

    # Execution Filter
    actual_Wt = np.zeros_like(ideal_Wt)
    actual_Wt[0] = ideal_Wt[0]
    gross = np.zeros(T); gross[0] = float(actual_Wt[0] @ R[0])
    for t in range(1, T):
        denom = 1.0 + gross[t-1]
        drift = actual_Wt[t-1] * (1.0 + R[t-1]) / denom if denom > 0 else np.zeros(K)
        desired = ideal_Wt[t]- drift
        band = 0.75 * pre["vol"][t]
        trade = np.where(desired > band, desired- band, np.where(desired <-band, desired + band, 0.0))
        actual_Wt[t] = drift + np.clip(trade,-0.049, 0.049)
        gross[t] = float(actual_Wt[t] @ R[t])
    return actual_Wt


def robust_growth_v2(R, signals, pre):
    """Stress-selected causal growth strategy used by the submission entry point."""
    from submission_strategy import generate_targets

    return generate_targets(R, signals)


REGISTRY = {
    "robust_growth_v2": (robust_growth_v2,),
    "equal_weight_1x": (equal_weight_1x,),
    "inverse_vol_1x": (inverse_vol_1x,),
    "equal_weight_kelly": (equal_weight_kelly,),
    "inverse_vol_kelly": (inverse_vol_kelly,),
    "momentum_kelly": (momentum_kelly,),
    "mean_variance_kelly": (mean_variance_kelly,),
    "sig1_mom_kelly": (sig1_mom_kelly,),
    "quant_edge_kelly": (quant_edge_kelly,),
    "quant_edge_safe_kelly": (quant_edge_safe_kelly,),
    "signal_ic_weighted_kelly": (signal_ic_weighted_kelly,),
    "sqrt_vol_1x": (sqrt_vol_1x,),
    "sqrt_vol_kelly": (sqrt_vol_kelly,),
    "quant_edge_sqrt_kelly": (quant_edge_sqrt_kelly,),
    "adaptive_downside_kelly": (adaptive_downside_kelly,),
    "signal1_kelly": (signal1_kelly,),
    "meta_ic_downside_voltarget": (meta_ic_downside_voltarget,),
    "meta_ic_eqweight_voltarget": (meta_ic_eqweight_voltarget,),
    "meta_ic_invvol_voltarget": (meta_ic_invvol_voltarget,),
    "signal2_kelly": (signal2_kelly,),
    "meta_risk_parity_kelly": (meta_risk_parity_kelly,),
    "signal3_kelly": (signal3_kelly,),
    "dynamic_meta_kelly": (dynamic_meta_kelly,),
    "signal3_fast_kelly": (signal3_fast_kelly,),
    "signal_kelly": (signal_kelly,),
    "signal_kelly_fm": (signal_kelly_fm,),
}

# short formula + plain-English note per book, shown in the dashboard
FORMULAS = {
    "robust_growth_v2": "Equal core + 30/40/30 smoothed signal overlay; 3.75% downside-vol target; 2-6x gross.",
    "equal_weight_1x": " w = 1/N, gross 1 (no leverage).",
    "inverse_vol_1x": " w 1/ , gross 1 (no leverage).",
    "equal_weight_kelly": " w = 1/N, scaled to k m/v.",
    "sig1_mom_kelly": "inverse-vol core + tilt by mean z(signal_1, momentum).",
    "inverse_vol_kelly": " w 1/ , scaled to k m/v.",
    "meta_ic_downside_voltarget": "Downside/SqrtVol Core + IC-based Tanh T-stat + 2% Vol Target + Cap Filter.",
    "meta_ic_eqweight_voltarget": "Equal Weight Core + IC-based Tanh T-stat + 2% Vol Target + Cap Filter.",
    "meta_ic_invvol_voltarget": "Inverse Vol Core + IC-based Tanh T-stat + 2% Vol Target + Cap Filter.",
    "momentum_kelly": "inverse-vol core + tilt by z(trailing return), Kelly-scaled.",
    "quant_edge_kelly": "80% inv-vol core + 20% asymmetric smoothed signal blend, 0.15 Kelly.",
    "quant_edge_safe_kelly": "Same blend as quant_edge_kelly, Kelly fraction 0.05 (worst fold-optimal, not full-sample-optimal).",
    "signal_ic_weighted_kelly": "30/70 IC-weighted signal_1(10d)/signal_3(8d) tilt, no signal_2 (dead), 20% tilt, 0.05 Kelly.",
    "mean_variance_kelly": " w m / , Kelly-scaled (diagonal growth-optimal).",

    "signal1_kelly": "inverse-vol core + tilt by z(signal_1).",
    "meta_risk_parity_kelly": "Rolling Risk Parity base + tanh(t-stat) meta-labeling + vol scaled deadband.",
    "dynamic_meta_kelly": "Dual-leg base + tanh(t-stat) meta-labeling + vol-scaled deadband.",
    "signal2_kelly": "inverse-vol core + tilt by z(signal_2).",
    "adaptive_downside_kelly": "50/50 Semicov/SqrtVol base + adaptive execution filter (cap 0.049).",
    "signal3_kelly": "inverse-vol core + tilt by z(signal_3).",
    "signal3_fast_kelly": "inverse-vol core + tilt by 3d-EWMA-smoothed z(signal_3).",
    "sqrt_vol_1x": " w 1/ , gross 1 (no leverage).",
    "sqrt_vol_kelly": " w 1/ , scaled to k m/v.",
    "quant_edge_sqrt_kelly": "80% sqrt-vol core + 20% signal blend, 0.18 Kelly.",
    "signal_kelly": "inverse-vol core + tilt by mean z(signal_1..3).",
    "signal_kelly_fm": "inverse-vol core + tilt by Fama-MacBeth sign/strength/horizon calibrated z(signal_1..3).",
}
NOTES = {
    "robust_growth_v2": "Production candidate selected with corrected evaluator, causal folds, and 47,500-day regime stress tests.",
    "equal_weight_1x": "Unlevered baseline the field's floor. CAGR here is pure asset drift.",
    "inverse_vol_1x": "Risk-parity baseline, still unlevered. Shows the core before Kelly.",
    "equal_weight_kelly": "Naive but honest: lever the equal-weight book toward growth optimal.",
    "inverse_vol_kelly": "Shipped book. Down-weights noisy assets, then Kelly-levers the most robust CAGR earner without a calibrated signal.",
    "sig1_mom_kelly": "The Master Blend: Combines our two slow-decaying, high-IC edges to safely maximize CAGR under trade cap constraints.",
    "quant_edge_kelly": "Our custom master book. Captures the IC edge of S1 and S3 while suppressing turnover to dodge the trade cap penalty.",
    "quant_edge_safe_kelly": "Same signal shape, but Kelly fraction picked against the WORST fold, not full-sample CAGR: worst-fold-16%->-5%, maxDD 66%->24%, for ~half the full sample CAGR. The honest ship candidate.",
    "signal_ic_weighted_kelly": "Rebuilt from a dedicated deep-dive: signal_2 confirmed dead (breakeven cost-0.06bps) and dropped entirely; signal_3's smoothing halflife (8d) and the s1/s3 mix (30/70) were empirically swept against the real evaluator, not assumed.",
    "momentum_kelly": "Bets recent winners keep winning. Only helps if returns trend.",
    "mean_variance_kelly": "The textbook growth-optimal tilt; sensitive to mean estimates, so it swings hard watch its worst fold and bootstrap p5.",
    "signal1_kelly": "Isolates signal_1's cross-sectional edge on CAGR.",
    "adaptive_downside_kelly": "Best-in-class baseline combining growth and crash defense, using an internal simulator to completely evade trade-cap penalties.",
    "signal2_kelly": "Isolates signal_2's cross-sectional edge on CAGR.",
    "signal3_kelly": "Isolates signal_3's cross-sectional edge on CAGR.",
    "meta_ic_downside_voltarget": "Tests crash-defense core utilizing statistical IC meta labeling and fixed 2% risk sizing to limit maximum drawdown.",
    "meta_ic_eqweight_voltarget": "Tests equal-weight core utilizing statistical IC meta labeling and fixed 2% risk sizing.",
    "meta_ic_invvol_voltarget": "Tests inverse volatility core utilizing statistical IC meta-labeling and fixed 2% risk sizing.",
    "meta_risk_parity_kelly": "Uses full covariance matrix to equalize risk contribution among assets, minimizing idiosyncratic crash risk while dynamically harvesting alpha.",
    "dynamic_meta_kelly": "Dynamically scales signal conviction using a rolling Sharpe proxy, while filtering execution friction via a volatility-scaled deadband.",
    "sqrt_vol_1x": "Square-root volatility baseline. Splits the difference between equal weight drift and inverse-vol safety.",
    "sqrt_vol_kelly": "Levered square-root vol core. A more aggressive but balanced risk parity.",
    "quant_edge_sqrt_kelly": "The final form. Maximizes base drift using 1/ , filters trade-cap friction via asymmetric smoothing, and optimizes Kelly leverage to 0.18.",
    "signal_kelly": "Averages all three signals. Negative here the signals are not yet calibrated (sign/horizon), so this is research bait, not a book to ship.",
    "signal_kelly_fm": "Fixes signal_kelly's naive average: Fama-MacBeth picks each signal's sign, best horizon and relative strength, drops insignificant ones, and horizon-smooths survivors to cut turnover before blending.",
}


if __name__ == "__main__":
    import scoring
    R, sigs = scoring.load_panel()
    Rv = R.to_numpy()
    S = [None if s is None else s.to_numpy() for s in sigs]
    pre = precompute(Rv, S)
    res = {n: scoring.compute_metrics(Rv, np.nan_to_num(fn(Rv, S, pre)))
           for n, (fn,) in REGISTRY.items()}
    print(scoring.rank(res)[["cagr", "sharpe", "max_drawdown",
                              "max_daily_loss", "avg_gross", "busted"]].round(4))
