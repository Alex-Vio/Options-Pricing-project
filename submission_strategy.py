"""Causal production strategy for PM Project 4.

The public entry point is ``generate_targets``.  It depends only on NumPy and
Pandas and uses no information from day t to set day-t weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_MIX = (0.30, 0.40, 0.30)
SIGNAL_HALFLIVES = (5, 15, 5)
SIGNAL_TILT = 0.475
DOWNSIDE_VOL_TARGET = 0.0375
DOWNSIDE_VOL_HALFLIFE = 60
LEVERAGE_FLOOR = 2.0
LEVERAGE_CAP = 6.0
MARKET_TIMING_HALFLIFE = 5
MARKET_TIMING_SCALE_HALFLIFE = 500
MARKET_TIMING_AMPLITUDE = 0.5
POSITION_CAP = 1.5
ASSET_VOL_HALFLIFE = 20
DEADBAND_MULTIPLE = 1.0
PLANNED_TRADE_CAP = 0.049
EVALUATOR_TRADE_CAP = 0.05
TRANSACTION_COST = 0.0005
CAP_PENALTY = 0.0001
SUBMITTED_GROSS_LIMIT = 9.5


def _ewm(values, halflife):
    return (
        pd.DataFrame(np.asarray(values, dtype=float))
        .ewm(halflife=halflife, min_periods=1)
        .mean()
        .to_numpy()
    )


def _lag1(values, first=0.0):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    result[0] = first
    result[1:] = values[:-1]
    return result


def _xs_z(values):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 1e-12,
    )


def _unit_gross(values):
    values = np.asarray(values, dtype=float)
    gross = np.abs(values).sum(axis=1, keepdims=True)
    return np.divide(
        values,
        gross,
        out=np.zeros_like(values),
        where=gross > 1e-12,
    )


def _causal_standardize(values, halflife=MARKET_TIMING_SCALE_HALFLIFE):
    values = np.asarray(values, dtype=float)
    mean = _lag1(_ewm(values[:, None], halflife).ravel())
    second = _lag1(_ewm((values * values)[:, None], halflife).ravel())
    scale = np.sqrt(np.maximum(second - mean * mean, 1e-12))
    result = (_lag1(values) - mean) / scale
    result[:20] = 0.0
    return np.clip(result, -3.0, 3.0)


def _signal_shape(returns, signals):
    n_dates, n_assets = returns.shape
    if len(signals) < 3 or any(signal is None for signal in signals[:3]):
        return _unit_gross(np.ones((n_dates, n_assets)))

    lagged = [_lag1(_xs_z(signal)) for signal in signals[:3]]
    smooth = [
        _ewm(signal, halflife)
        for signal, halflife in zip(lagged, SIGNAL_HALFLIVES)
    ]
    score = _xs_z(
        sum(weight * signal for weight, signal in zip(SIGNAL_MIX, smooth))
    )
    core = _unit_gross(np.ones_like(returns))
    signal_book = _unit_gross(core * score)
    return _unit_gross(
        (1.0 - SIGNAL_TILT) * core + SIGNAL_TILT * signal_book
    )


def _leverage_series(returns, signals, shape):
    paper_return = np.sum(shape * returns, axis=1)
    downside_observation = 2.0 * np.minimum(paper_return, 0.0) ** 2
    downside_risk = np.sqrt(
        np.maximum(
            _lag1(
                _ewm(
                    downside_observation[:, None],
                    DOWNSIDE_VOL_HALFLIFE,
                ).ravel()
            ),
            1e-10,
        )
    )
    leverage = np.clip(
        DOWNSIDE_VOL_TARGET / downside_risk,
        LEVERAGE_FLOOR,
        LEVERAGE_CAP,
    )
    leverage[:20] = min(5.0, LEVERAGE_CAP)

    if len(signals) >= 2 and signals[1] is not None:
        common_signal_2 = np.asarray(signals[1], dtype=float).mean(axis=1)
        timing = _ewm(
            _causal_standardize(common_signal_2)[:, None],
            MARKET_TIMING_HALFLIFE,
        ).ravel()
        leverage = np.clip(
            leverage + MARKET_TIMING_AMPLITUDE * timing,
            LEVERAGE_FLOOR,
            LEVERAGE_CAP,
        )
    return leverage


def _executable_targets(returns, ideal):
    n_dates = len(returns)
    targets = np.zeros_like(ideal)
    actual = np.zeros_like(ideal)
    gross_return = np.zeros(n_dates)
    net_return = np.zeros(n_dates)
    cost = np.zeros(n_dates)

    asset_variance = _ewm(returns * returns, ASSET_VOL_HALFLIFE)
    asset_volatility = _lag1(np.sqrt(np.maximum(asset_variance, 1e-12)))

    targets[0] = ideal[0]
    opening_gross = np.abs(targets[0]).sum()
    if opening_gross > SUBMITTED_GROSS_LIMIT:
        targets[0] *= SUBMITTED_GROSS_LIMIT / opening_gross
    actual[0] = targets[0]
    gross_return[0] = float(actual[0] @ returns[0])
    net_return[0] = gross_return[0]

    for t in range(1, n_dates):
        nav_growth = 1.0 + net_return[t - 1]
        if nav_growth <= 0.0:
            continue
        drift = actual[t - 1] * (1.0 + returns[t - 1]) / nav_growth
        desired = ideal[t] - drift
        band = DEADBAND_MULTIPLE * asset_volatility[t]
        trade = np.sign(desired) * np.maximum(np.abs(desired) - band, 0.0)
        planned = drift + np.clip(
            trade,
            -PLANNED_TRADE_CAP,
            PLANNED_TRADE_CAP,
        )
        planned_gross = np.abs(planned).sum()
        if planned_gross > SUBMITTED_GROSS_LIMIT:
            planned *= SUBMITTED_GROSS_LIMIT / planned_gross
        targets[t] = planned

        evaluator_trade = targets[t] - drift
        hit = np.abs(evaluator_trade) >= EVALUATOR_TRADE_CAP - 1e-12
        applied = np.clip(
            evaluator_trade,
            -EVALUATOR_TRADE_CAP,
            EVALUATOR_TRADE_CAP,
        )
        actual[t] = drift + applied
        cost[t] = (
            np.abs(applied).sum() * TRANSACTION_COST
            + hit.sum() * CAP_PENALTY
        )
        gross_return[t] = float(actual[t] @ returns[t])
        net_return[t] = gross_return[t] - cost[t]
    return targets


def generate_targets(returns, signals):
    """Return wide target weights for every input date."""
    returns = np.nan_to_num(np.asarray(returns, dtype=float))
    clean_signals = [
        None if signal is None else np.nan_to_num(np.asarray(signal, dtype=float))
        for signal in (signals or [])
    ]
    shape = _signal_shape(returns, clean_signals)
    leverage = _leverage_series(returns, clean_signals, shape)
    ideal = np.clip(
        shape * leverage[:, None],
        -POSITION_CAP,
        POSITION_CAP,
    )
    return _executable_targets(returns, ideal)
