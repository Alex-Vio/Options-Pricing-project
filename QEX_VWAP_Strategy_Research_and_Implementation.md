# QEX VWAP Execution Strategy: Research and Implementation Specification

**Purpose:** handover document for implementing and testing a trading agent for the Quantedge Exchange (QEX) VWAP execution project.

**Recommended first production candidate:**  
**Adaptive volume-curve VWAP scheduler + queue-aware passive/aggressive execution + hard completion floor.**

The volume curve determines *when* to trade. The order-book policy determines *how* to trade. Short-term signals should only make bounded adjustments around the volume schedule.

---

## 1. Project formulation

Each day the agent receives one signed parent order:

\[
X \in \mathbb{Z}, \qquad
X>0 \text{ for a buy}, \quad X<0 \text{ for a sell}.
\]

Define

\[
s=\operatorname{sign}(X), \qquad N=|X|.
\]

Fills are signed in the same convention. If the agent has fills \(\tilde x_1,\ldots,\tilde x_m\), define side-adjusted executed quantity

\[
Q_t=s\sum_{i:\,t_i\le t}\tilde x_i.
\]

Thus \(Q_t\ge 0\) means progress toward the parent order regardless of whether the parent is a buy or sell. The remaining quantity is

\[
R_t=N-Q_t.
\]

The full-day benchmark is market VWAP:

\[
P_{\mathrm{VWAP}}
=
\frac{\sum_{i=1}^{n}x_iP_i}{\sum_{i=1}^{n}x_i}.
\]

The agent's realised execution price is compared with the benchmark using

\[
\mathrm{VWAPCost}
=
s\frac{\widetilde P-P_{\mathrm{VWAP}}}{P_{\mathrm{VWAP}}}.
\]

Lower is better for both sides.

### Exchange constraints

- One stock.
- Limit order book with price-time priority.
- Tick size: USD 0.01.
- Integer quantities.
- Trading day: 390 exchange minutes.
- Supported actions: market order, limit order, cancellation.
- Daily action limit: 10,000.
- Disconnect if gross transacted quantity exceeds \(3N\).
- Market orders can partially fill; unmatched quantity is cancelled.
- Limit orders may cross the spread; any unmatched quantity can rest.
- Unexecuted or overexecuted net inventory is closed using the last five-minute VWAP with a stated 100 bp adverse penalty.
- Evaluation uses 400 simulated days.
- Criteria:
  - mean VWAPCost: 3 points;
  - median absolute deviation: 6 points;
  - average of the worst 40 VWAPCost observations: 6 points.

The last criterion is the empirical upper-tail expected shortfall at the 90% level:

\[
\operatorname{CVaR}_{90\%}^{\mathrm{emp}}
=
\frac{1}{40}\sum_{d\in\text{worst 40}} C_d.
\]

Therefore 80% of the nominal points concern dispersion and tail risk, not mean performance.

---

## 2. Critical README ambiguity to verify

The displayed README formula appears to define the residual price as

\[
\widetilde P_{\mathrm{res}}
=
P_{\mathrm{last}}
\left(1-0.01\,\operatorname{sign}(\widetilde x_{\mathrm{res}})\right).
\]

This conflicts with both the word “penalty” and the displayed example.

For an unfilled buy, \(\widetilde x_{\mathrm{res}}>0\), an adverse closure should occur **above** the last VWAP, not below it. For an unfilled sell, it should occur below it. The economically consistent formula is

\[
\widetilde P_{\mathrm{res}}
=
P_{\mathrm{last}}
\left(1+0.01\,\operatorname{sign}(\widetilde x_{\mathrm{res}})\right).
\]

The example also states that an excess long position is sold 100 bp below the last VWAP, which agrees with the plus-sign formula.

**Implementation requirement:** inspect the server scoring code or run a controlled residual test. Do not design the strategy around a possible documentation sign error. Under either interpretation, the robust policy is to finish exactly and avoid residual exposure.

---

## 3. Why following the volume curve is the correct baseline

Divide the day into \(K\) bins. Let market volume and average market price in bin \(k\) be \(v_k\) and \(p_k\). Define the realised market volume share

\[
w_k=\frac{v_k}{\sum_{j=1}^{K}v_j}.
\]

Market VWAP is

\[
P_{\mathrm{VWAP}}=\sum_{k=1}^{K}w_kp_k.
\]

If the agent executes

\[
q_k=Nw_k
\]

at approximately \(p_k\), its execution price is

\[
P_{\mathrm{exec}}
=
\frac{1}{N}\sum_{k=1}^{K}q_kp_k
=
\sum_{k=1}^{K}w_kp_k
=
P_{\mathrm{VWAP}}.
\]

If the forecast volume shares are \(\widehat w_k\), then approximately

\[
P_{\mathrm{exec}}-P_{\mathrm{VWAP}}
=
\sum_{k=1}^{K}(\widehat w_k-w_k)p_k.
\]

The principal schedule risk is therefore error in the volume profile. A TWAP schedule implicitly assumes \(w_k=1/K\), which is generally not aligned with a non-uniform intraday volume pattern.

This is not a universal theorem that VWAP scheduling is always optimal. It is the correct project baseline because:

1. the benchmark itself is volume-weighted;
2. research derives VWAP execution as optimal under specific risk-neutral impact models;
3. dynamic volume models reduce VWAP tracking risk;
4. fixed historical schedules can be inferior to schedules that update with realised volume.

---

## 4. Research conclusions translated into implementation choices

### 4.1 Volume-proportional scheduling

Kato introduces stochastic market volume into an Almgren-Chriss-style model and obtains VWAP execution as the risk-neutral optimum under the model assumptions. This supports using the volume curve as the scheduler, not as a technical indicator.

**Implementation consequence:** the base target should be cumulative predicted market volume share.

### 4.2 Dynamic volume updating

Białkowski, Darolles and Le Fol separate common intraday seasonality from stock-specific dynamic volume and use intraday updating to reduce VWAP execution risk.

Mitchell, Białkowski and Tompaidis derive a dynamic VWAP tracker under general price and volume dynamics and include spread, depth and impact in empirical tests.

Kakade, Kearns, Mansour and Ortiz show theoretically that fixed schedules can be materially worse than dynamically adaptive online VWAP algorithms in worst-case settings.

**Implementation consequence:** start from a historical curve, but revise expected total and remaining volume using current-day observed volume.

### 4.3 Bounded signal overlays

Dang and Chen adjust a VWAP trading curve using short-term price signals, but constrain the modified curve inside a trading envelope to avoid extreme departures from the benchmark schedule.

**Implementation consequence:** order-book signals may accelerate or delay execution, but only within a small inventory envelope around the volume target.

### 4.4 Limit versus market order choice

Cont and Kukanov formulate the split between limit and market orders as an order-placement optimisation problem driven by queue sizes, order flow, fill risk and underfill penalties.

**Implementation consequence:** the scheduler should not directly choose order types. A separate execution layer should convert required quantity into passive and aggressive child orders using book state and schedule urgency.

### 4.5 Imbalance and adverse selection

Cont, Kukanov and Stoikov find that short-horizon price changes are more closely related to order-flow imbalance and market depth than to raw trade volume alone.

Gould and Bonart find that bid/ask queue imbalance has statistically significant one-tick-ahead predictive power, especially for large-tick stocks.

Lehalle and Mounjid show why limit-order fill probability cannot be considered alone: a passive order may be filled precisely when price is about to move against it. Queue state and adverse selection matter.

**Implementation consequence:** use imbalance primarily to decide whether to remain passive, cancel, or cross. Do not let it replace the volume schedule.

---

## 5. Recommended strategy architecture

Use four layers:

1. **Volume model:** estimates the cumulative market volume fraction.
2. **Schedule controller:** calculates target inventory and schedule deficit.
3. **Microstructure execution layer:** chooses passive versus aggressive orders.
4. **Risk and completion layer:** prevents residuals, overfills, excessive actions and gross-volume breaches.

The strategy should be implemented incrementally:

- **Version A:** static volume curve + aggressive execution.
- **Version B:** adaptive volume curve + completion floor.
- **Version C:** queue-aware passive/aggressive execution.
- **Version D:** bounded imbalance and order-flow overlay.

Do not begin with reinforcement learning or a large neural network. The supplied history is described as only a few prior days, while the number of market states and execution decisions is large. A high-capacity model would be difficult to validate and likely to overfit.

---

## 6. Historical volume-curve construction

### 6.1 Time bins

Recommended starting resolution:

\[
K=78
\]

five-minute bins over 390 minutes.

Five-minute bins are robust when the historical sample is small. One-minute bins can be added later if the historical data is sufficiently dense and stable.

For historical day \(d\), compute

\[
v_{d,k}=\text{total transaction quantity in bin }k,
\]

\[
V_d=\sum_{k=1}^{K}v_{d,k},
\]

\[
u_{d,k}=\frac{v_{d,k}}{V_d}.
\]

Each \(u_{d,\cdot}\) sums to one.

### 6.2 Robust estimate

With few historical days, use a robust blend rather than a complex model:

\[
u_k^{\mathrm{raw}}
=
\alpha\,\operatorname{median}_d(u_{d,k})
+
(1-\alpha)\,\operatorname{mean}_d(u_{d,k}),
\]

with starting value

\[
\alpha=0.5.
\]

Then:

1. smooth over neighbouring bins with a three-bin moving average;
2. apply a small positive floor \(\varepsilon\);
3. renormalise so \(\sum_k u_k=1\).

Define the cumulative baseline curve

\[
G_k=\sum_{j=1}^{k}u_j.
\]

### 6.3 What to store

Store:

- expected bin share \(u_k\);
- cumulative expected share \(G_k\);
- mean and median daily total volume;
- per-bin standard deviation or robust median absolute deviation;
- historical quantiles of cumulative volume;
- average transaction intensity by bin;
- average top-five-level book depth by bin, if reconstructible.

The curve should be computed offline and loaded by the agent at startup.

---

## 7. Adaptive intraday volume forecast

A static curve assumes today is an average day. The adaptive model should estimate whether current-day volume is running above or below normal.

At exchange time \(t\), let:

- \(V_t\): observed market transaction volume so far;
- \(G_0(t)\): baseline expected cumulative fraction at \(t\);
- \(\bar V\): robust historical expected total daily volume.

The naive total-volume estimate is

\[
\frac{V_t}{G_0(t)},
\]

but this is unstable near the open. Use shrinkage.

Define the observed volume ratio

\[
r_t
=
\frac{V_t}{\bar V\max(G_0(t),\varepsilon)}.
\]

Clip it:

\[
r_t^{c}
=
\operatorname{clip}(r_t,r_{\min},r_{\max}),
\]

with initial bounds such as

\[
r_{\min}=0.5,\qquad r_{\max}=2.0.
\]

Define a confidence weight increasing through the day:

\[
\omega_t
=
\min\left(
\omega_{\max},
\frac{G_0(t)}{g_{\mathrm{full}}}
\right),
\]

with starting values

\[
\omega_{\max}=0.8,\qquad g_{\mathrm{full}}=0.35.
\]

The current-day scale factor is

\[
\beta_t=(1-\omega_t)+\omega_t r_t^c.
\]

Forecast remaining market volume as

\[
\widehat V_{\mathrm{rem}}(t)
=
\beta_t\bar V\left(1-G_0(t)\right).
\]

Forecast total volume as

\[
\widehat V_T(t)
=
V_t+\widehat V_{\mathrm{rem}}(t).
\]

The estimated realised fraction of daily volume is

\[
\widehat G(t)
=
\frac{V_t}{\widehat V_T(t)}.
\]

The primary cumulative inventory target is

\[
Q_t^{\mathrm{vol}}
=
N\widehat G(t).
\]

### Future-bin forecast

For a future baseline bin \(j\), forecast

\[
\widehat v_j(t)=\beta_t\bar V u_j.
\]

For the next horizon \(h\), define predicted market volume

\[
\widehat V_{t,t+h}
=
\sum_{j\in(t,t+h]}\widehat v_j(t).
\]

This permits calculation of the participation rate required to reach a future target.

### Simpler fallback

If the runtime agent cannot reliably track all market trades, use the static cumulative curve \(Q_t^{\mathrm{vol}}=NG_0(t)\). Do not implement an inaccurate adaptive model from incomplete volume messages.

---

## 8. Schedule feedback controller

Define schedule deficit

\[
D_t=Q_t^{\mathrm{target}}-Q_t.
\]

- \(D_t>0\): behind schedule.
- \(D_t<0\): ahead of schedule.

Use a short control horizon \(h\), initially one five-minute bin. Let

\[
Q_{t+h}^{\mathrm{target}}
\]

be the target at the end of that horizon. Required child quantity is

\[
A_t
=
\max\left(
0,
Q_{t+h}^{\mathrm{target}}-Q_t
\right).
\]

The implied participation rate is

\[
\rho_t^{\mathrm{req}}
=
\frac{A_t}{\max(\widehat V_{t,t+h},1)}.
\]

Also calculate the minimum participation needed to finish over predicted remaining market volume:

\[
\rho_t^{\mathrm{finish}}
=
\frac{R_t}{\max(\widehat V_{\mathrm{rem}}(t),1)}.
\]

Use

\[
\rho_t
=
\operatorname{clip}
\left(
\rho_t^{\mathrm{req}}
+
k_D\frac{D_t}{N},
0,
\rho_t^{\max}
\right).
\]

A practical dynamic upper bound is

\[
\rho_t^{\max}
=
\operatorname{clip}
\left(
\max(3\rho_0,\;1.3\rho_t^{\mathrm{finish}}),
0.05,
0.60
\right),
\]

where

\[
\rho_0=\frac{N}{\widehat V_T(t)}.
\]

All constants are starting values, not established optima.

### Deadband

Avoid reacting to negligible schedule error. Use a deadband such as

\[
|D_t| < d_0N,
\qquad d_0\in[0.0025,0.005].
\]

Inside the deadband, preserve queue position rather than cancelling and replacing orders.

---

## 9. Hard completion floor

The 100 bp residual penalty is much larger than a normal spread. The agent should target exact completion before the final minutes.

Set a completion cutoff

\[
T_c=385 \text{ exchange minutes}
\]

rather than 390. The remaining five minutes provide a buffer for reconciliation and unexpected partial fills.

Start a forced completion phase at

\[
T_e\in[330,350].
\]

At \(T_e\), record remaining quantity \(R_{T_e}\). Define a maximum permitted remaining inventory:

\[
R_t^{\max}
=
R_{T_e}
\left(
\frac{T_c-t}{T_c-T_e}
\right)^p,
\qquad T_e\le t\le T_c,
\]

with \(p\in[1,2]\).

The completion-floor target is

\[
Q_t^{\mathrm{floor}}=N-R_t^{\max}.
\]

Use

\[
Q_t^{\mathrm{target}}
=
\max\left(
Q_t^{\mathrm{vol}},
Q_t^{\mathrm{floor}}
\right).
\]

This preserves volume tracking early but guarantees that the target converges to full completion.

### Suggested phases

- **0–300 min:** normal volume tracking; mostly passive if on schedule.
- **300–350 min:** gradually increase aggression; no material schedule deficit.
- **350–375 min:** mixed-to-aggressive execution.
- **375–385 min:** immediate completion dominates benchmark tracking.
- **385–390 min:** reconciliation only; correct any accidental underfill or overfill.

Use exchange time, not wall-clock time. The final simulation runs at accelerated speed.

---

## 10. Order-book features

At each decision point, construct:

### 10.1 Best quotes and spread

\[
b_t=\text{best bid},\qquad
a_t=\text{best ask},
\]

\[
m_t=\frac{a_t+b_t}{2},
\]

\[
\mathrm{spreadTicks}_t
=
\frac{a_t-b_t}{0.01}.
\]

### 10.2 Weighted multi-level imbalance

Let \(B_{\ell,t}\) and \(A_{\ell,t}\) be displayed bid and ask depth at level \(\ell\). Use decreasing weights, for example

\[
w_\ell=\frac{1}{\ell}.
\]

Define

\[
I_t
=
\frac{
\sum_{\ell=1}^{5}w_\ell B_{\ell,t}
-
\sum_{\ell=1}^{5}w_\ell A_{\ell,t}
}{
\sum_{\ell=1}^{5}w_\ell B_{\ell,t}
+
\sum_{\ell=1}^{5}w_\ell A_{\ell,t}
}.
\]

Then \(I_t\in[-1,1]\).

### 10.3 Microprice

Using top-level quantities \(B_{1,t}\) and \(A_{1,t}\),

\[
\mu_t
=
\frac{
a_t B_{1,t}+b_t A_{1,t}
}{
B_{1,t}+A_{1,t}
}.
\]

Define microprice pressure

\[
Z_t^{\mu}
=
\frac{\mu_t-m_t}{0.01}.
\]

Positive values indicate upward pressure.

### 10.4 Recent order-flow imbalance

If the event stream permits reconstruction, maintain a short-horizon signed order-flow measure based on changes at the best bid and ask. Otherwise use:

- signed transaction imbalance over the previous 5–30 exchange seconds;
- change in top-level depth;
- recent mid-price return;
- recent spread changes.

Standardise each signal using historical or rolling robust scale.

### 10.5 Parent-side adverse signal

Let \(Z_t\) be an upward-price signal. Define

\[
A_t^{\mathrm{adv}}=sZ_t.
\]

- For a buy, upward pressure is adverse, so \(A_t^{\mathrm{adv}}>0\).
- For a sell, downward pressure is adverse, also giving \(A_t^{\mathrm{adv}}>0\).

Positive adverse signal should increase urgency. Negative adverse signal may justify greater patience.

---

## 11. Bounded signal-adjusted target

Do not allow imbalance to control the entire inventory path.

Define a signal displacement

\[
\Delta Q_t^{\mathrm{sig}}
=
\operatorname{clip}
\left(
\beta_{\mathrm{sig}}A_t^{\mathrm{adv}}N,
-E_t,
E_t
\right).
\]

A simple envelope is

\[
E_t=e_0N\left(1-\frac{t}{T_c}\right),
\]

with starting value

\[
e_0\in[0.02,0.05].
\]

Then use

\[
Q_t^{\mathrm{target}}
=
\max\left[
Q_t^{\mathrm{floor}},
\;
\operatorname{clip}
\left(
Q_t^{\mathrm{vol}}+\Delta Q_t^{\mathrm{sig}},
0,
N
\right)
\right].
\]

Near the close the envelope shrinks, and completion dominates.

Only retain the signal overlay if it improves out-of-sample mean, MAD and tail cost. Statistical price-direction accuracy alone is insufficient.

---

## 12. Passive versus aggressive execution

### 12.1 Aggression score

Construct an aggression score in \([0,1]\):

\[
U_t
=
\operatorname{clip}
\left(
u_0
+
k_1\frac{D_t}{N}
+
k_2\left(\frac{t}{T_c}\right)^p
+
k_3 A_t^{\mathrm{adv}}
+
k_4 \rho_t^{\mathrm{finish}},
0,
1
\right).
\]

Interpretation:

- higher deficit increases aggression;
- later time increases aggression;
- adverse expected price movement increases aggression;
- low forecast capacity relative to remaining inventory increases aggression.

Do not assume the constants. Tune them using exchange tests.

### 12.2 Rule-based initial version

Before fitting a smooth score, use transparent rules:

| State | Suggested action |
|---|---|
| Ahead by more than 0.5% of \(N\) | Rest a small limit order or wait |
| Within ±0.5% of target | Mostly passive at best quote |
| Behind by 0.5–2% of \(N\) | Mix passive and aggressive |
| Behind by more than 2% of \(N\) | Cross the spread |
| After 350 min with material inventory | Mostly aggressive |
| After 375 min | Complete immediately |

These thresholds must scale with parent size and observed fill rate.

### 12.3 Passive order

For a buy, place at the best bid.  
For a sell, place at the best ask.

Candidate passive size:

\[
q_t^{L}
=
\min\left(
R_t,
A_t,
c_{\mathrm{depth}}D_{\mathrm{same},1},
q_{\max}^{L}
\right).
\]

Start with

\[
c_{\mathrm{depth}}\in[0.1,0.3].
\]

Do not expose the full remaining parent order at one level.

### 12.4 Aggressive order

For immediate quantity, use either:

- a market order sized to the visible eligible opposite depth; or
- a marketable limit order with an explicit worst price.

Market orders cancel the unmatched remainder, which is useful when exact immediate size is desired. Marketable limit orders provide price control but may leave a residual resting at the limit price.

For a buy, an aggressive limit price may be the best ask or a deeper ask level.  
For a sell, it may be the best bid or a deeper bid level.

Do not blindly sweep five levels. Select the minimum depth needed for the current control quantity.

### 12.5 Queue preservation

Price-time priority makes queue position valuable. Do not cancel merely because the book updated.

Cancel or replace when:

- the order is no longer at the intended price;
- adverse-selection risk crosses a threshold;
- schedule deficit requires crossing;
- order age exceeds a calibrated limit and fill probability is low;
- remaining quantity changed enough to risk overfill;
- endgame begins.

Retain the order when:

- it remains at the best quote;
- the schedule is on target;
- signal change is small;
- expected fill probability remains acceptable.

---

## 13. Fill-probability model

### Minimum viable model

Track:

- queue ahead when the order joins;
- same-price traded volume after placement;
- estimated cancellations ahead;
- elapsed order age;
- spread;
- imbalance;
- recent trade intensity.

Approximate queue depletion over horizon \(h\). A simple intensity model is

\[
P(\text{fill within }h)
=
1-\exp(-\lambda_t h),
\]

where \(\lambda_t\) is estimated from recent same-side queue depletion.

### Empirical alternative

From test logs, fit a logistic regression:

\[
\Pr(\text{fill within }h)
=
\sigma
\left(
\theta_0
+
\theta_1\log(1+\text{queueAhead})
+
\theta_2 I_t
+
\theta_3\text{tradeIntensity}_t
+
\theta_4\text{spreadTicks}_t
+
\theta_5\text{orderAge}
\right).
\]

Fit separate models for buys and sells only if symmetry fails materially.

### Important warning

High fill probability is not automatically desirable. Passive fills can be adversely selected. Record post-fill markouts:

\[
M_{\Delta}
=
s\frac{m_{t+\Delta}-P_{\mathrm{fill}}}{0.01}.
\]

For a parent buy, negative \(M_\Delta\) means the market subsequently moved lower, so waiting would have been better. Evaluate passive tactics using both fill probability and markout.

---

## 14. Exact inventory and overfill controls

Limit orders can fill asynchronously. Every child-order size must account for outstanding same-side orders.

Let

\[
L_t^{\mathrm{open}}
=
\text{total unfilled same-side resting quantity}.
\]

Maximum new same-side order size is

\[
q_t^{\mathrm{new}}
\le
\max(0,N-Q_t-L_t^{\mathrm{open}}).
\]

This prevents a fully filled open order plus a new order from exceeding \(N\).

If \(Q_t>N\), immediately trade the opposite side to return to \(N\). Do not intentionally overtrade in the baseline strategy.

Maintain:

- net executed quantity;
- gross executed quantity;
- open quantity by order ID;
- acknowledged versus pending actions;
- action count;
- cancellation status.

Set internal safety limits:

\[
\text{gross quantity}<2.8N
\]

and

\[
\text{actions}<9500,
\]

leaving buffers below the hard constraints.

---

## 15. Action-rate control

The 10,000-action limit permits active management, but not cancel-replace on every message.

Recommended starting policy:

- evaluate continuously;
- send at most one action every 5–10 exchange seconds under normal conditions;
- allow faster action only during the completion phase;
- maintain no more than one principal resting order;
- do not cancel and recreate at the same price;
- count rejected actions as actions unless server documentation proves otherwise.

At one action per five exchange seconds, the theoretical normal-day maximum is 4,680 actions, leaving room for endgame and exceptional states.

Use exchange timestamps because the final test runs at 13 times physical speed.

---

## 16. End-to-end decision pseudocode

```text
INITIALISE
    side = sign(parent_order)
    N = abs(parent_order)
    load historical volume curve and total-volume statistics
    net_filled = 0
    gross_filled = 0
    open_orders = {}
    actions = 0

ON MARKET OR FILL MESSAGE
    update exchange_time
    update five-level order book
    update observed market transactions and cumulative market volume
    reconcile fills and open orders

    Q = side * net_filled
    R = N - Q

    if Q > N:
        cancel all same-side resting orders
        submit opposite-side corrective order for Q - N
        return

    if R <= 0:
        cancel remaining same-side orders
        return

    if not action_interval_elapsed:
        return

    volume_target = adaptive_volume_target(exchange_time)
    completion_target = hard_completion_floor(exchange_time)
    adverse_signal = bounded_order_book_signal()
    signal_adjustment = trading_envelope(adverse_signal, exchange_time)

    target = max(
        completion_target,
        clip(volume_target + signal_adjustment, 0, N)
    )

    deficit = target - Q
    predicted_volume_next = forecast_market_volume(next_control_horizon)
    desired_next = max(0, target_at_horizon_end - Q)

    urgency = compute_urgency(
        deficit,
        remaining=R,
        time=exchange_time,
        adverse_signal=adverse_signal,
        predicted_remaining_volume=...
    )

    if existing_order_should_be_cancelled(...):
        cancel existing order
        return

    available_new_quantity =
        N - Q - outstanding_same_side_quantity

    child_quantity =
        min(
            desired_next,
            available_new_quantity,
            risk_and_depth_cap
        )

    if child_quantity <= 0:
        return

    if exchange_time >= hard_aggressive_time:
        submit aggressive order(child_quantity)
    elif urgency >= aggressive_threshold:
        submit aggressive order(child_quantity)
    else:
        submit passive best-quote order(child_quantity)
```

---

## 17. Suggested software structure

```text
qex_agent/
├── agent.py
├── config.py
├── models.py
├── volume_profile.py
├── volume_forecast.py
├── schedule_controller.py
├── signals.py
├── execution_policy.py
├── order_manager.py
├── risk_manager.py
├── metrics.py
└── tests/
    ├── test_inventory_accounting.py
    ├── test_volume_curve.py
    ├── test_completion_floor.py
    ├── test_order_sizing.py
    ├── test_action_limits.py
    └── test_buy_sell_symmetry.py
```

### Core state objects

```text
MarketState
    exchange_time
    bids[1:5]
    asks[1:5]
    recent_trades
    cumulative_market_volume
    last_5m_vwap
    mid
    spread
    imbalance
    microprice

ExecutionState
    parent_signed
    target_abs
    net_filled_signed
    gross_filled
    open_orders
    pending_actions
    action_count
    realised_cashflow

Decision
    action_type
    side
    quantity
    limit_price
    order_id_to_cancel
    reason
```

Every sent action should be logged with the state and reason that produced it.

---

## 18. Initial parameter ranges

These are search ranges, not final values.

| Parameter | Initial range |
|---|---:|
| Volume bins | 1, 5 or 10 min |
| Median weight in profile | 0.25–0.75 |
| Volume ratio clipping | 0.5–2.0 |
| Max adaptive weight | 0.5–0.9 |
| Schedule deadband | 0.25%–1.0% of \(N\) |
| Normal max participation | 10%–60% |
| Passive size / best depth | 10%–30% |
| Signal envelope | 0%–5% of \(N\) |
| Passive order age | 10–60 exchange sec |
| Endgame start | 330–360 min |
| Full-aggression start | 370–382 min |
| Completion cutoff | 383–387 min |
| Normal decision interval | 5–15 exchange sec |

Avoid optimising too many parameters simultaneously. Use staged ablation.

---

## 19. Testing and model-selection protocol

### 19.1 Required metrics

For every strategy version record:

\[
\text{Mean}=\frac{1}{D}\sum_d C_d,
\]

\[
\text{MAD}
=
\operatorname{median}_d
\left|
C_d-\operatorname{median}(C)
\right|,
\]

\[
\text{Tail}_{90}
=
\operatorname{mean}(\text{largest }10\%\text{ of }C_d).
\]

Also record:

- completion rate;
- mean and maximum residual quantity;
- passive fill ratio;
- aggressive fill ratio;
- average spread paid or earned;
- post-fill markouts;
- action count;
- gross volume divided by \(N\);
- maximum absolute schedule deficit;
- execution fraction by time of day.

### 19.2 Candidate comparison

Because official points are rank-based, no exact scalar backtest loss reproduces the competition score. A reasonable internal surrogate is:

\[
L
=
0.2\,z(\text{mean})
+
0.4\,z(\text{MAD})
+
0.4\,z(\text{Tail}_{90})
+
\lambda\,\text{noncompletion rate},
\]

where \(z\) standardises each metric across candidate strategies and \(\lambda\) is large.

Also inspect the three official metrics separately. A small mean improvement is not worthwhile if tail cost materially worsens.

### 19.3 Ablation sequence

Run:

1. TWAP baseline.
2. Static historical VWAP curve.
3. Adaptive VWAP curve.
4. Adaptive curve + completion floor.
5. Add passive execution.
6. Add queue-aware cancellation.
7. Add imbalance overlay.
8. Add fitted fill model.

Only keep a component if its improvement survives different days and parent sizes.

### 19.4 Stress tests

Test explicitly on:

- unusually low-volume days;
- unusually high-volume days;
- monotonic uptrends and downtrends;
- wide spreads;
- thin top-of-book depth;
- large parent orders;
- delayed acknowledgements;
- partial market-order fills;
- stale or missing data;
- late-session volume spikes;
- accidental open-order duplication.

### 19.5 Buy/sell symmetry

For every deterministic unit test, flip:

- parent sign;
- book prices around a reference price;
- imbalance sign;
- fill signs.

The resulting inventory path and cost logic should mirror. Sign errors are a major implementation risk.

---

## 20. Strategies not recommended initially

### 20.1 Pure TWAP

It ignores the volume weighting of the benchmark. Retain only as a diagnostic baseline.

### 20.2 Pure market-order VWAP

It can track the schedule but repeatedly pays the spread and may consume depth. Use it as the first reliable baseline, then improve with passive execution.

### 20.3 Pure passive execution

It risks non-completion and adverse selection. It is especially unsuitable given the 100 bp residual penalty and tail-focused score.

### 20.4 Unbounded directional timing

A model that delays a buy because price “looks high” or delays a sell because price “looks low” can diverge badly from full-day VWAP during trends. Any timing signal must remain inside a small schedule envelope.

### 20.5 Speculative round trips

The \(3N\) gross-volume limit technically permits some opposite-side trading, but round trips consume actions, spread, depth and tail-risk budget. Disable them unless a later controlled experiment demonstrates robust benefit.

### 20.6 High-capacity machine learning

The project appears to provide few historical days and limited full-speed tests. Complex models may fit historical microstructure without generalising to the 400 simulated evaluation days. Begin with interpretable models and use logs to justify added complexity.

---

## 21. Concrete implementation order for the coding model

### Milestone 1: correctness

- Parse all exchange messages.
- Maintain exact signed inventory.
- Maintain open orders by ID.
- Enforce action and gross-volume limits.
- Implement market-only schedule.
- Finish by minute 385.
- Reproduce project VWAPCost locally.
- Resolve the residual-price sign ambiguity.

### Milestone 2: volume baseline

- Build five-minute historical volume curve.
- Implement static cumulative target.
- Add schedule deficit and deadband.
- Log target versus realised execution.

### Milestone 3: adaptation

- Track current market volume.
- Implement shrinkage estimate of total volume.
- Update target using estimated realised volume fraction.
- Add completion floor.

### Milestone 4: passive execution

- Add one resting best-quote limit order.
- Cap size by depth and remaining inventory.
- Preserve queue position.
- Cancel based on schedule, quote movement and age.

### Milestone 5: microstructure overlay

- Add weighted imbalance and microprice.
- Convert to parent-side adverse signal.
- Apply only inside a small trading envelope.
- Record fill markouts.

### Milestone 6: calibration

- Sweep a small number of parameters.
- Compare mean, MAD and worst-decile cost.
- Run ablations.
- Select the simplest strategy with stable tail performance.

---

## 22. Final recommended agent

The initial competitive agent should use:

1. a robust historical five-minute volume curve;
2. an intraday shrinkage update for total and remaining market volume;
3. a cumulative target \(Q_t=N\widehat G(t)\);
4. feedback based on target deficit;
5. passive best-quote orders when on or ahead of schedule;
6. aggressive orders when behind schedule or facing adverse price pressure;
7. a bounded imbalance overlay of at most a few percent of \(N\);
8. a forced completion path ending near minute 385;
9. strict open-order, action-count and gross-volume controls;
10. direct optimisation and reporting of mean, MAD and worst-decile VWAPCost.

The most important design principle is:

> **Use the volume curve for benchmark exposure, the order book for execution quality, and the endgame controller for tail-risk protection.**

---

## 23. Research references

1. **Takashi Kato**, “VWAP Execution as an Optimal Strategy.”  
   https://arxiv.org/abs/1408.6118

2. **Jędrzej Białkowski, Serge Darolles and Gaëlle Le Fol**, “Improving VWAP Strategies: A Dynamic Volume Approach,” *Journal of Banking & Finance*, 2008.  
   https://doi.org/10.1016/j.jbankfin.2007.09.023

3. **Daniel Mitchell, Jędrzej Białkowski and Stathis Tompaidis**, “Optimal VWAP Tracking.”  
   https://ssrn.com/abstract=2333916

4. **Ngoc-Minh Dang and Yin Chen**, “Dynamic Execution of VWAP Orders with Short-Term Predictions.”  
   https://ssrn.com/abstract=2366177

5. **Sham Kakade, Michael Kearns, Yishay Mansour and Luis Ortiz**, “Competitive Algorithms for VWAP and Limit Order Trading.”  
   https://www.cis.upenn.edu/~mkearns/papers/vwap.pdf

6. **Rama Cont and Arseniy Kukanov**, “Optimal Order Placement in Limit Order Markets.”  
   https://arxiv.org/abs/1210.1625

7. **Rama Cont, Arseniy Kukanov and Sasha Stoikov**, “The Price Impact of Order Book Events.”  
   https://arxiv.org/abs/1011.6402

8. **Martin D. Gould and Julius Bonart**, “Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book.”  
   https://arxiv.org/abs/1512.03492

9. **Charles-Albert Lehalle and Othmane Mounjid**, “Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency.”  
   https://arxiv.org/abs/1610.00261
