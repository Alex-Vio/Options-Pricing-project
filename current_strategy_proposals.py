"""Standalone PM Project 4 strategy handoff for code or dashboard integration.

Public contract
---------------
``generate_proposal_targets(returns, signals, name)`` accepts a ``(T, N)``
return panel and three matching signal panels ordered signal_1, signal_2,
signal_3.  It returns a finite ``(T, N)`` target-weight panel.  Day-t targets
use information through day t-1 only.  ``PROPOSALS`` contains every candidate
and parameter. Empirical results are intentionally excluded from this handoff.

The current selection is ``linear_prod``.  The other five entries are genuine
ablation or safety alternatives, not aliases.  NumPy and pandas are the only
dependencies.  No project-local imports or hidden fitted objects are needed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


RECOMMENDED_PROPOSAL = "linear_prod"
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
    fixed_leverage: bool = False
    rationale: str = ""


PROPOSALS = {
    item.name: item
    for item in (
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


def _ideal_weights(returns, signals, proposal):
    dates, assets = returns.shape
    core = _unit_gross(np.ones((dates, assets)))
    if proposal.signal_tilt == 0.0:
        shape = core
    else:
        lagged = [_lag(_xs_z(signal)) for signal in signals[:3]]
        smooth = [_ewm(signal, half) for signal, half in zip(lagged, SIGNAL_HALFLIVES)]
        score = _xs_z(sum(weight * signal for weight, signal in zip(SIGNAL_MIX, smooth)))
        signal_book = _unit_gross(core * score)
        shape = _unit_gross(
            (1.0 - proposal.signal_tilt) * core
            + proposal.signal_tilt * signal_book
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


def generate_proposal_targets(returns, signals, name="linear_prod"):
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
