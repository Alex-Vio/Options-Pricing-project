# Independent Strategy Review Guide

## Objective

Audit whether `robust_growth_v2` is a defensible CAGR-maximizing submission
without assuming that its design hypotheses are true.

## Hypotheses to test independently

1. Signal 3 has useful short-horizon cross-sectional information that decays
   faster than signals 1 and 2.
2. Signals 1 and 2 are persistent enough that smoothing reduces turnover
   without discarding most predictive content.
3. A linear blend is more stable than powers, squares, logs, or products.
4. A diversified equal-weight core compounds more reliably than estimated
   covariance or inverse-volatility cores.
5. Downside-risk leverage improves survival relative to high fixed leverage.
6. The common component of signal 2 may contain weak market-timing information,
   but the term must be ablated because it is the least certain component.

## Required checks

- Verify strict day-t causality through truncation tests.
- Reproduce the evaluator's free opening, post-cost drift, trade cap, cap
  penalty, and transaction cost exactly.
- Compare `linear_prod` with `linear_no_timing`, `linear_lower_tilt`, and
  `linear_conservative` under identical paths.
- Test persistent high volatility, reduced drift, destroyed signal alpha,
  reversed signals, and near-ruin deleveraging.
- Inspect actual rather than submitted gross after drift.
- Prefer counterexamples and reproducible metrics over stylistic objections.

## Main risks

- Signal decay or sign reversal.
- Regime dependence hidden by one train/test split.
- Underestimated tail dependence in resampling.
- Leverage interacting badly with post-loss drift and the daily trade cap.
- False confidence from selecting parameters on a short sample.

The exact code is authoritative. The PDF explains the reasoning, not proof.
