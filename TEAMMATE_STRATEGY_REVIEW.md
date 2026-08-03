# PM Project strategy review

## Recommendation

Keep the current strategy as the reference implementation. Add this as the leading challenger:

- Core: equal weight.
- Signal weights: 30% signal 1, 30% signal 2, 40% signal 3.
- Signal half-lives: 5, 8, 3 days.
- Alpha tilt: fixed 0.475.
- Risk control: 3.75% daily downside-risk target, 60-day risk half-life, leverage clipped to 2x-6x.
- Keep the common component of signal 2 as the market-timing input.

This challenger reduces the disputed signal-2 allocation, increases the weight on the statistically stronger signal 3, improves the training CAGR and drawdown, and wins 14 of 15 matched paths in the five stress regimes that preserve the estimated signal relationship. It loses to the incumbent when all alpha is deliberately removed or reversed. That is the central model-risk trade-off.

## Test design

All features and rolling estimates are causal. PC1 uses returns strictly before the weight date. Portfolio tests use the evaluator's free opening, weight drift, 0.05 per-security trade cap, 5 bp transaction cost, cap penalty, and bust rule. The review used full-sample metrics, ten contiguous 250-day folds, paired first/second-half comparisons, HAC tests, moving-block bootstrap, and 47,500-day block-bootstrap stress paths. The stress Monte Carlo has only three paths per scenario, so small ranking differences are not statistically decisive.

## 1. PC1 versus equal weight

Reject pure PC1. Covariance PC1 maximises explained return variance, not expected log growth. It concentrates the core in the dominant high-volatility common factor. The causal 250-day covariance PC1 reduced training CAGR from 75.2% to 72.5% and increased maximum drawdown from 62.5% to 74.1%. Across all stress regimes its median CAGR was 35.9%, versus 42.1% for equal weight. A 50/50 equal-PC1 blend was less damaging but still inferior overall. Full-sample PCA would also leak future covariance information and is invalid.

## 2. Signal 2

The teammate's objection is valid. Signal 2 does not have strong standalone cross-sectional evidence:

- Smoothed one-day IC: HAC t-stat about 1.3.
- Incremental signal-2 coefficient after controlling for signals 1 and 3: about 1.1.
- Paired implemented-return advantage of 40% signal 2 over timing-only signal 2: about 0.6.
- The paired gain is positive in the first half and negative in the second.
- Moving-block bootstrap intervals include zero.

There is suggestive long-horizon IC with the same positive sign in both halves, but the horizon was searched and the evidence is not enough to call 40% an estimated optimum. Signal 2 should therefore be treated as a weak diversifier with shrinkage, not proven alpha.

Deleting it entirely is also too strong. Timing-only signal 2 lost every one of the 15 matched plausible-regime stress paths and gave up substantial median growth. Conditional on the simulated out-of-sample data preserving the training relationship, 30% signal 2 is the best compromise tested. Keep its separate common-component timing role, whose market-return interaction is positive in both the aggregate and second half.

## 3. Downside volatility and 0.0375

The downside observation is `2 * min(r, 0)^2`. The factor 2 puts it on the same scale as ordinary variance for a symmetric return distribution while reacting specifically to harmful volatility. It is preferable for a CAGR objective because positive volatility does not create ruin in the same way as negative volatility.

The value is 0.0375, meaning 3.75% daily risk, not 0.375. It came from a causal grid over 3.0%-4.25%, not a theoretical identity. The ten-fold score peaked locally at 3.75%. Symmetric targeting was close in CAGR but produced worse stressed tail loss and drawdown. Hence 3.75% is a robust grid choice, not a precisely known constant.

## 4. Dynamic parameter tuning

Leverage is already dynamic: estimated downside risk changes the leverage every day. Dynamically moving the target using signal confidence is a second adaptation layer. An 80% fixed prior plus 20% confidence adjustment reduced both training and stressed growth. The confidence estimate is too noisy relative to the 2,500-day sample.

Dynamic tuning remains possible, but only as a one-sided safety overlay with a strong prior and a pre-specified trigger. It should not continuously re-optimise half-lives, signal weights, or the risk target from the same short history.

## 5. Signal half-lives

The original 5/15/5 choice combined signal persistence, forward-IC horizons, turnover, and a causal grid. The expanded grid favours faster decay, especially signal 3. The strongest stable alternatives were 5/8/3 and 3/8/3. The recommended challenger uses 5/8/3 because it changes fewer parameters and avoids selecting the most aggressive signal-1 decay from a small sample.

## 6. T-stat and tanh weighting

The old code did not compute a t-stat. It used `mean(IC) / standard_deviation(IC)` without multiplying by the square root of effective sample size. That is an IC information ratio.

Both the old signed tanh rule and a correctly scaled EWMA t-stat were tested. Continuous dynamic weighting underperformed fixed 0.475. Even 90%-95% prior versions did not improve the ten-fold score. The reason is procyclicality: the rule cuts or reverses alpha after losses, pays transition costs, and often misses the recovery. Fixed 0.475 acts as a strong shrinkage prior.

A signed t-stat does protect against a permanent signal reversal. In that artificial regime it changes median CAGR from negative to strongly positive. The cost is material in the empirical regime because the estimated t-stat is negative roughly 9%-10% of the time and tanh shrinks exposure whenever confidence is moderate. On the tested binary empirical-versus-permanent-reversal comparison, the insurance breaks even only if a permanent reversal has roughly a 13% prior probability. A `t < -2` gate has little normal cost but reacts too rarely to be reliable. Do not include signed tanh in the primary submission without independent evidence that sign reversal is plausible.

## Decision

The defensible candidate set is:

1. Incumbent 30/40/30, half-lives 5/15/5: best if robustness to absent or reversed alpha receives high weight.
2. Challenger 30/30/40, half-lives 5/8/3: best expected CAGR if the long out-of-sample simulation is generated by a stationary process related to training.
3. Do not submit pure PC1, timing-only signal 2, continuously tuned target risk, or signed-tanh conviction as the primary strategy.

The next decision should be explicit: choose how much prior probability to assign to structural signal failure. The data cannot estimate that probability from one 2,500-day simulation.
