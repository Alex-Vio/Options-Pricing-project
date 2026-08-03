# PM Project 4 Overnight Research Report

## Decision

Select `linear_fast_shrunk_core` for expected after-cost CAGR. Retain
`linear_fast_shrunk_core_survival` as the heavier-tail, survival-first variant
and `linear_fast_shrunk_core_tstat` as permanent-reversal insurance. Keep
`linear_prod` as the benchmark.

This is a ranking under uncertainty, not proof. The selected rule improves the
observed growth/risk trade-off consistently enough to justify one additional
core component, but its paired block-bootstrap advantage over the incumbent is
not statistically conclusive.

## Frozen selected rule

All day-t quantities below use information available by day t-1.

1. Cross-sectionally standardise each signal, lag one day, then exponentially
   smooth signals 1/2/3 with half-lives 5/8/3 days.
2. Form the score
   `s(t) = z(0.30*s1(t) + 0.40*s2(t) + 0.30*s3(t))` and the unit-gross
   long-short signal book `a(t) = s(t) / sum_i |s_i(t)|`.
3. Estimate each security's slow quality from lagged EWMA moments:
   `q_i(t) = mu_i(t) / variance_i(t)`, using a 1,000-day half-life.
   Cross-sectionally standardise q, set
   `raw_i(t) = max(1 + 0.30*z(q_i(t)), 0.25)`, then normalise raw weights to
   sum to one. Use equal weights for the first 500 days. This core is long-only
   and deliberately close to equal weight.
4. Combine and renormalise:
   `shape(t) = unit_gross(0.525*core(t) + 0.475*a(t))`.
5. Estimate downside risk from the 60-day EWMA of
   `2*min(shape(t)'*return(t), 0)^2`. Set leverage to
   `clip(0.0375/risk + 0.5*timing_signal_2, 2, 6)`.
6. Clip ideal positions to +/-1.5. Execute toward the ideal with a one-asset-
   volatility deadband, planned trades clipped to +/-0.049, and submitted gross
   capped at 9.5. The evaluator applies the official 0.05 trade cap, 5 bp cost,
   cap penalty, and post-cost drift.

The 0.30 core strength is a rounded, mild shrinkage setting inside a broad
stable region, not the exact best training point. The 1,000-day half-life and
500-day warm-up reduce estimator noise. The faster 5/8/3 signal half-lives were
chosen from persistence and horizon evidence, then checked in contiguous
folds; they are not re-estimated online.

Sensitivity supports this interpretation. Core strengths from 0.20 to 0.50
formed a flat region rather than a sharp optimum; 0.35 had the highest observed
fold score, so 0.30 was deliberately rounded away from that peak. Estimator
half-lives from 500 to 1,500 days were similarly close. The 1,000-day choice is
slow enough to suppress noise without relying on the exact best 750-day score.

## Why retain signal 2

Signal 2 is not independently convincing. Its standalone IC t-stat is weak and
its performance differs across sample halves. It is retained for two narrower
reasons:

* In the joint score, 30-40% signal-2 weight lies on a broad performance
  plateau. Removing it reduced full-sample growth substantially. A 30% weight
  produced almost the same full-sample CAGR as 40%, but had a worse worst fold,
  larger drawdown and more trading. Thus 40% is not justified by standalone
  significance; it is justified by portfolio interaction and execution.
* The cross-sectional mean of signal 2 is used separately as a small leverage
  timing term. Removing timing lowered CAGR and worsened drawdown, worst-day
  loss, turnover and cost in the frozen-core comparison. This evidence is still
  in-sample and should not be described as causal proof.

The strategy does not assume that signal 2 is a strong standalone predictor.
Its risk is explicitly acknowledged. A zero-signal-2 candidate was rejected
because it lost too much expected growth without delivering a compensating
survival improvement.

## Main evidence

### Training and resampling

Under the official evaluator, the selected rule produced 77.57% CAGR, Sharpe
1.32, 54.82% annual volatility, 57.73% maximum drawdown and 17.94% worst-day
loss. The incumbent produced 75.16% CAGR, Sharpe 1.29, 55.17% volatility,
62.53% drawdown and 18.13% worst-day loss. Neither busted or hit the trade cap.

The selected rule's worst contiguous 250-day fold CAGR was -14.95% versus
-22.59% for the incumbent's original 5/15/5 specification. It beat the
incumbent's annual log growth in 6 of 10 non-overlapping 250-day blocks and 4
of 5 non-overlapping 500-day blocks.

A paired circular block bootstrap estimated a 1.37 percentage-point annual-log-
growth advantage. Across 20/60/125/250-day blocks, 72.7-79.4% of draws were
positive, but every 95% interval included zero. The correct interpretation is
"promising and directionally stable", not "statistically established".

Leave-one-security-out checks improved both selection score and log growth for
all 20 omitted securities. This argues against the result
being driven by one lucky asset. The slow quality IC is stronger in the later
half, so time-regime dependence remains the main overfit risk.

### Long-horizon and tail stress

Matched 47,500-day tests covered empirical block bootstrap, long blocks, each
sample half, higher volatility, reduced/zero/reversed signal alpha, intermittent
reversal, fat-tail factor returns, common/reversed asset drift, asset
permutation and rotating leaders. Exact final-parameter results are included in
`overnight_final_stress_summary.csv`. None of the 240 original three-candidate
evaluations busted. The selected rule beat the incumbent's median CAGR in 15 of 16 regimes
and on every matched path within those regimes. The sole exception was the
second-half-only resample, where the incumbent won every path. This is strong
internal consistency, but the paths reuse the observed joint distribution.

An additional 100-path severe-tail suite found zero busts in ordinary empirical,
adverse-block, Student-t factor, and 1.5x-volatility Student-t(df=6.5) regimes.
In the deliberately extreme Student-t(df=4.5), 1.5x-volatility regime, the
selected, t-stat and incumbent books each busted once in 20 paths. This is
residual ruin risk, not a solved
problem. Small reductions in the nominal 6x leverage cap did not reliably
remove that failure because post-loss weight drift can dominate the target cap.

A final matched risk-target boundary test changed only the downside-volatility
target. On 20 ordinary empirical paths, median CAGR rose monotonically from
65.87% at a 3.0% target to 68.80% at 3.75%, with no busts. On 20 new extreme
Student-t(df=4.5), 1.5x-volatility paths, the 3.0% target had no busts, while
3.25%, 3.5% and 3.75% had 5%, 10% and 10% bust rates. Reconstructing the known
earlier bust path gave the same boundary: 3.0% survived, while every higher
target failed. Training CAGR falls from 77.57% to about 70.5%. Therefore 3.0%
is a material survival alternative, not the expected-CAGR selection. The test
does not prove zero ruin probability and actual weights can still explode after
near-total losses because of the evaluator's drift denominator.

### Execution sensitivity

At double transaction cost, the selected rule still beat the incumbent. At
triple cost it fell slightly behind because its faster signals trade more. A
counterfactual 0.025 daily trade cap also favoured the incumbent, but the
specified cap is 0.05. The official-cost recommendation is therefore sensitive
to a materially different execution contract, not to small perturbations.

## T-stat conclusion

The aggressive proposal was tested, not dismissed conceptually. With the final
core, the literal extrapolating `2*tanh(2t)` blend produced -2.75% training
CAGR; the more coherent additive interpretation produced 55.32%, versus 77.57%
for the fixed rule. A near-full-strength `0.95*tanh(2t)` rule without the new
core produced only 34.39% and busted one of three high-volatility paths. More
scale amplifies estimation noise and false sign changes; it does not create
evidence.

The only defensible dynamic rule was
`tilt(t) = 0.475*tanh(2*(t_stat(t)+1))`, with a correctly scaled 120-day EWMA IC
t-stat. The +1 shift encodes a prior in favour of the original signal direction.
It sacrifices about 2 percentage points of training CAGR after adding the new
core, but changes a permanent-signal-reversal stress from near-zero growth to
large positive growth. Across non-reversal scenarios it usually trails the fixed
rule slightly. Use it only if the team assigns meaningful prior probability to
a permanent sign reversal; do not average it into the primary rule by default.

With the arbitrary equal weighting of the constructed stress scenarios, its
large reversal payoff offsets its small ordinary-regime cost at only about a
0.5% permanent-reversal probability. That is a payoff break-even calculation,
not an estimated probability. If the hidden 47,500 days come from the same
stationary simulator, permanent reversal should receive near-zero prior weight;
if the generator can change sign, the insurance candidate becomes rational.

## Rejected ideas

* **PC1 core:** pure and blended covariance/correlation PC1 books did not improve
  CAGR and often increased concentration or drawdown. PC1 describes common
  variance, not expected return.
* **Remove signal 2:** reduced growth and did not buy enough robustness.
* **Nonlinear signal transforms:** clipping, ranks and tanh discarded useful
  score magnitude and underperformed linear scores.
* **Adaptive signal weights:** online slope estimates added noise and multiple-
  testing risk; strongly shrunk versions did not beat the simpler blend.
* **Return momentum:** unstable across halves and worse after costs.
* **Dual-horizon or symmetric risk:** no consistent advantage over 60-day
  downside risk.
* **Lower leverage cap alone:** clear CAGR cost and no robust elimination of
  the extreme drift-driven bust path. Lowering the downside-risk target to 3.0%
  was materially more effective and is retained as a separate finalist.
* **Model ensemble:** marginally competitive, but the extra specification set
  did not beat the single fast model consistently enough to justify complexity.

## Recommendation boundary

Submit `linear_fast_shrunk_core` if the live generator is believed to preserve
the training data-generating process broadly and expected CAGR is decisive.
Use `linear_fast_shrunk_core_survival` if the team gives material prior weight
to tails substantially heavier than the ordinary matched stresses and accepts
the CAGR cost. Use the t-stat variant only for a strong prior that signal signs
can reverse permanently. Revert to `linear_prod` if the team rejects the slow
cross-sectional mean/variance persistence assumption. No tested strategy can
honestly guarantee non-bust over 47,500 days under arbitrarily heavy tails.
