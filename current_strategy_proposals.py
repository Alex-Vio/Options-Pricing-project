"""Standalone PM Project 4 strategy handoff for code or dashboard integration.

Public contract
---------------
``generate_proposal_targets(returns, signals, name)`` accepts a ``(T, N)``
return panel and three matching signal panels ordered signal_1, signal_2,
signal_3.  It returns a finite ``(T, N)`` target-weight panel.  Day-t targets
use information through day t-1 only.  ``PROPOSALS`` contains every candidate
and parameter.  The metric constants below are reproducible snapshots for
dashboard integration.

The current selection is ``linear_fast_shrunk_core``.  The other entries are
genuine ablation or safety alternatives, not aliases.  NumPy and pandas are
the only dependencies.  No project-local imports or hidden fitted objects are
needed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


RECOMMENDED_PROPOSAL = "linear_fast_shrunk_core"
EVALUATION_CONVENTIONS = {
    "forward_returns": True,
    "information_lag_days": 1,
    "transaction_cost_each_way": 0.0005,
    "trade_cap_per_security_per_day": 0.05,
    "trade_cap_penalty": 0.0001,
    "first_evaluated_day_is_free": True,
    "drift_denominator": "1 + previous gross return - previous cost",
    "annualization_days": 250,
}
STRESS_SCENARIOS = {
    "empirical": "60-day joint block bootstrap",
    "long_blocks": "250-day joint block bootstrap",
    "high_vol": "60-day joint blocks with demeaned returns scaled 1.5x",
    "adverse": "half drift and 1.5x demeaned volatility",
    "no_signal_alpha": "returns and signals resampled independently",
    "reversed_signal": "joint path with all signals sign-reversed",
    "zero_drift": "60-day joint blocks with return drift removed",
    "zero_drift_high_vol": "zero drift and 1.5x demeaned volatility",
}


@dataclass(frozen=True)
class Proposal:
    name: str
    signal_tilt: float
    downside_vol_target: float
    leverage_cap: float
    timing_amplitude: float
    position_cap: float
    signal_mix: tuple[float, float, float] = (0.30, 0.40, 0.30)
    signal_halflives: tuple[int, int, int] = (5, 15, 5)
    fixed_leverage: bool = False
    core_mode: str = "equal"
    core_strength: float = 0.0
    core_halflife: int = 1000
    core_warmup: int = 500
    tilt_mode: str = "fixed"
    tstat_halflife: int = 120
    tstat_slope: float = 2.0
    tstat_shift: float = 1.0
    rationale: str = ""


PROPOSALS = {
    item.name: item
    for item in (
        Proposal(
            "linear_fast_shrunk_core",
            0.475,
            0.0375,
            6.0,
            0.5,
            1.5,
            signal_halflives=(5, 8, 3),
            core_mode="ewm_mean_variance",
            core_strength=0.30,
            rationale=(
                "Expected-growth selection: faster signal smoothing plus a mild, "
                "slowly estimated long-only mean/variance core tilt."
            ),
        ),
        Proposal(
            "linear_fast_shrunk_core_survival",
            0.475,
            0.0300,
            6.0,
            0.5,
            1.5,
            signal_halflives=(5, 8, 3),
            core_mode="ewm_mean_variance",
            core_strength=0.30,
            rationale=(
                "Survival-first alternative: the selected rule with a 3.0% "
                "downside-volatility target to reduce extreme-tail exposure."
            ),
        ),
        Proposal(
            "linear_fast_shrunk_core_tstat",
            0.475,
            0.0375,
            6.0,
            0.5,
            1.5,
            signal_halflives=(5, 8, 3),
            core_mode="ewm_mean_variance",
            core_strength=0.30,
            tilt_mode="shifted_tanh",
            rationale=(
                "Regime-reversal insurance: the same book, with signal exposure "
                "controlled by a lagged EWMA IC t-stat."
            ),
        ),
        Proposal(
            "linear_prod",
            0.475,
            0.0375,
            6.0,
            0.5,
            1.5,
            rationale="Current leader: linear signal mix plus downside-volatility sizing.",
        ),
        Proposal(
            "linear_lower_s2_fast",
            0.475,
            0.0375,
            6.0,
            0.5,
            1.5,
            signal_mix=(0.30, 0.30, 0.40),
            signal_halflives=(5, 8, 3),
            rationale=(
                "Leading challenger: shrinks weak signal 2, increases signal 3, "
                "and uses faster causal smoothing; treat as a stationary-DGP candidate."
            ),
        ),
        Proposal(
            "linear_no_timing",
            0.475,
            0.0375,
            6.0,
            0.0,
            1.5,
            rationale="Removes the weaker market-level signal-2 timing component.",
        ),
        Proposal(
            "linear_lower_tilt",
            0.400,
            0.0375,
            6.0,
            0.5,
            1.5,
            rationale="Less signal dependence; stronger if future signal alpha decays.",
        ),
        Proposal(
            "linear_conservative",
            0.475,
            0.0350,
            5.5,
            0.25,
            1.25,
            rationale="Lower volatility, leverage, timing, and single-name concentration.",
        ),
        Proposal(
            "linear_fixed_5_5",
            0.475,
            0.0,
            5.5,
            0.0,
            1.5,
            fixed_leverage=True,
            rationale="Simple fixed-leverage signal benchmark.",
        ),
        Proposal(
            "equal_core_5_5",
            0.0,
            0.0,
            5.5,
            0.0,
            1.5,
            fixed_leverage=True,
            rationale="Signal-free control and maximum model simplicity.",
        ),
    )
}

# Dashboard-ready snapshot from five 47,500-day paths per scenario.  Values are
# median after-cost CAGR; all six proposals had zero busts in this focused run.
FOCUSED_STRESS_MEDIAN_CAGR = {
    "linear_prod": {
        "empirical": 0.653394,
        "long_blocks": 0.689241,
        "high_vol": 0.464713,
        "adverse": 0.214033,
        "no_signal_alpha": 0.366491,
        "reversed_signal": -0.002009,
        "zero_drift": -0.007832,
        "zero_drift_high_vol": 0.008417,
    },
    "linear_no_timing": {
        "empirical": 0.642654,
        "long_blocks": 0.676263,
        "high_vol": 0.414012,
        "adverse": 0.179273,
        "no_signal_alpha": 0.375952,
        "reversed_signal": 0.005663,
        "zero_drift": -0.025333,
        "zero_drift_high_vol": -0.020139,
    },
    "linear_lower_tilt": {
        "empirical": 0.647765,
        "long_blocks": 0.666981,
        "high_vol": 0.430611,
        "adverse": 0.172224,
        "no_signal_alpha": 0.393624,
        "reversed_signal": 0.075565,
        "zero_drift": -0.045765,
        "zero_drift_high_vol": -0.030387,
    },
    "linear_conservative": {
        "empirical": 0.611925,
        "long_blocks": 0.636296,
        "high_vol": 0.427514,
        "adverse": 0.201752,
        "no_signal_alpha": 0.350594,
        "reversed_signal": 0.007051,
        "zero_drift": -0.001119,
        "zero_drift_high_vol": 0.009306,
    },
    "linear_fixed_5_5": {
        "empirical": 0.619506,
        "long_blocks": 0.634550,
        "high_vol": 0.418664,
        "adverse": 0.122572,
        "no_signal_alpha": 0.362498,
        "reversed_signal": 0.008987,
        "zero_drift": -0.023361,
        "zero_drift_high_vol": -0.107757,
    },
    "equal_core_5_5": {
        "empirical": 0.411678,
        "long_blocks": 0.366377,
        "high_vol": 0.142470,
        "adverse": -0.156223,
        "no_signal_alpha": 0.409977,
        "reversed_signal": 0.424926,
        "zero_drift": -0.175499,
        "zero_drift_high_vol": -0.335236,
    },
}

LINEAR_PROD_TRAINING_METRICS = {
    "cagr": 0.751603,
    "sharpe": 1.291125,
    "annual_vol": 0.551712,
    "max_drawdown": 0.625289,
    "max_daily_loss": 0.181350,
    "average_gross": 5.605954,
    "maximum_gross": 6.315678,
    "cost_bps_per_day": 1.257618,
    "busted": False,
}

RECOMMENDED_TRAINING_METRICS = {
    "cagr": 0.775727,
    "sharpe": 1.321101,
    "annual_vol": 0.548165,
    "max_drawdown": 0.577259,
    "max_daily_loss": 0.179366,
    "average_gross": 5.585819,
    "maximum_gross": 6.406938,
    "turnover": 0.326514,
    "cost_bps_per_day": 1.632570,
    "busted": False,
}


SIGNAL_MIX = (0.30, 0.40, 0.30)
SIGNAL_HALFLIVES = (5, 15, 5)
RISK_HALFLIFE = 60
TRADE_DEADBAND_HALFLIFE = 20
PLANNED_TRADE_CAP = 0.049
TRANSACTION_COST = 0.0005
CAP_PENALTY = 0.0001
GROSS_LIMIT = 9.5


def _ewm(values, halflife):
    return pd.DataFrame(values).ewm(halflife=halflife, min_periods=1).mean().to_numpy()


def _lag(values):
    result = np.empty_like(values, dtype=float)
    result[0] = 0.0
    result[1:] = values[:-1]
    return result


def _xs_z(values):
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _unit_gross(values):
    gross = np.abs(values).sum(axis=1, keepdims=True)
    return np.divide(values, gross, out=np.zeros_like(values), where=gross > 1e-12)


def _timing_feature(signal_2):
    values = signal_2.mean(axis=1)
    mean = _lag(_ewm(values[:, None], 500).ravel())
    second = _lag(_ewm((values * values)[:, None], 500).ravel())
    z = (_lag(values) - mean) / np.sqrt(np.maximum(second - mean * mean, 1e-12))
    z[:20] = 0.0
    return _ewm(np.clip(z, -3.0, 3.0)[:, None], 5).ravel()


def _core_weights(returns, proposal):
    dates, assets = returns.shape
    if proposal.core_mode == "equal":
        return np.full((dates, assets), 1.0 / assets)
    if proposal.core_mode != "ewm_mean_variance":
        raise KeyError(f"Unknown core mode {proposal.core_mode!r}")
    mean = _lag(_ewm(returns, proposal.core_halflife))
    second = _lag(_ewm(returns * returns, proposal.core_halflife))
    variance = np.maximum(second - mean * mean, 1e-10)
    quality = _xs_z(mean / variance)
    raw = np.maximum(1.0 + proposal.core_strength * quality, 0.25)
    raw[: proposal.core_warmup] = 1.0
    return raw / raw.sum(axis=1, keepdims=True)


def _daily_ic(score, returns):
    score = score - score.mean(axis=1, keepdims=True)
    returns = returns - returns.mean(axis=1, keepdims=True)
    covariance = np.mean(score * returns, axis=1)
    variance = np.mean(score * score, axis=1) * np.mean(returns * returns, axis=1)
    return np.divide(
        covariance,
        np.sqrt(np.maximum(variance, 1e-18)),
        out=np.zeros_like(covariance),
        where=variance > 1e-18,
    )


def _causal_t_stat(score, returns, halflife):
    ic = _daily_ic(score, returns)
    mean = _lag(_ewm(ic[:, None], halflife).ravel())
    second = _lag(_ewm((ic * ic)[:, None], halflife).ravel())
    standard_deviation = np.sqrt(np.maximum(second - mean * mean, 1e-8))
    decay = 2.0 ** (-1.0 / halflife)
    observations = np.arange(1, len(ic) + 1)
    sum_weights = (1.0 - decay**observations) / (1.0 - decay)
    sum_squared = (1.0 - decay ** (2 * observations)) / (1.0 - decay**2)
    effective_n = sum_weights**2 / sum_squared
    result = mean / standard_deviation * np.sqrt(effective_n)
    result[:20] = 0.0
    return result


def _ideal_weights(returns, signals, proposal):
    dates, assets = returns.shape
    core = _core_weights(returns, proposal)
    if proposal.signal_tilt == 0.0:
        shape = core
    else:
        lagged = [_lag(_xs_z(signal)) for signal in signals[:3]]
        smooth = [
            _ewm(signal, half)
            for signal, half in zip(lagged, proposal.signal_halflives)
        ]
        score = _xs_z(
            sum(
                weight * signal
                for weight, signal in zip(proposal.signal_mix, smooth)
            )
        )
        signal_book = _unit_gross(score)
        if proposal.tilt_mode == "fixed":
            tilt = np.full(dates, proposal.signal_tilt)
        elif proposal.tilt_mode == "shifted_tanh":
            t_stat = _causal_t_stat(score, returns, proposal.tstat_halflife)
            tilt = proposal.signal_tilt * np.tanh(
                proposal.tstat_slope * (t_stat + proposal.tstat_shift)
            )
            tilt[0] = 0.0
        else:
            raise KeyError(f"Unknown tilt mode {proposal.tilt_mode!r}")
        shape = _unit_gross(
            (1.0 - np.abs(tilt[:, None])) * core
            + tilt[:, None] * signal_book
        )

    if proposal.fixed_leverage:
        leverage = np.full(dates, proposal.leverage_cap)
    else:
        paper_return = np.sum(shape * returns, axis=1)
        downside = 2.0 * np.minimum(paper_return, 0.0) ** 2
        risk = np.sqrt(np.maximum(_lag(_ewm(downside[:, None], RISK_HALFLIFE).ravel()), 1e-10))
        leverage = np.clip(proposal.downside_vol_target / risk, 2.0, proposal.leverage_cap)
        leverage[:20] = min(5.0, proposal.leverage_cap)
        if proposal.timing_amplitude and len(signals) >= 2:
            leverage = np.clip(
                leverage + proposal.timing_amplitude * _timing_feature(signals[1]),
                2.0,
                proposal.leverage_cap,
            )
    return np.clip(
        shape * leverage[:, None],
        -proposal.position_cap,
        proposal.position_cap,
    )


def _make_executable(returns, ideal):
    dates = len(returns)
    targets = np.zeros_like(ideal)
    actual = np.zeros_like(ideal)
    net = np.zeros(dates)
    cost = np.zeros(dates)
    asset_vol = _lag(np.sqrt(np.maximum(_ewm(returns * returns, TRADE_DEADBAND_HALFLIFE), 1e-12)))

    targets[0] = ideal[0]
    actual[0] = targets[0]
    net[0] = actual[0] @ returns[0]
    for t in range(1, dates):
        growth = 1.0 + net[t - 1]
        if growth <= 0.0:
            continue
        drift = actual[t - 1] * (1.0 + returns[t - 1]) / growth
        desired = ideal[t] - drift
        trade = np.sign(desired) * np.maximum(np.abs(desired) - asset_vol[t], 0.0)
        target = drift + np.clip(trade, -PLANNED_TRADE_CAP, PLANNED_TRADE_CAP)
        gross = np.abs(target).sum()
        if gross > GROSS_LIMIT:
            target *= GROSS_LIMIT / gross
        targets[t] = target
        applied = np.clip(target - drift, -0.05, 0.05)
        hit = np.abs(target - drift) >= 0.05 - 1e-12
        actual[t] = drift + applied
        cost[t] = np.abs(applied).sum() * TRANSACTION_COST + hit.sum() * CAP_PENALTY
        net[t] = actual[t] @ returns[t] - cost[t]
    return targets


def generate_proposal_targets(returns, signals, name=RECOMMENDED_PROPOSAL):
    """Generate weights for one named proposal from wide NumPy-like panels."""
    if name not in PROPOSALS:
        raise KeyError(f"Unknown proposal {name!r}; choose from {sorted(PROPOSALS)}")
    returns = np.nan_to_num(np.asarray(returns, dtype=float))
    signals = [np.nan_to_num(np.asarray(signal, dtype=float)) for signal in signals]
    ideal = _ideal_weights(returns, signals, PROPOSALS[name])
    return _make_executable(returns, ideal)


if __name__ == "__main__":
    for proposal in PROPOSALS.values():
        print(proposal)
