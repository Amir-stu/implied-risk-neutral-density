# Method

This document sets out what the code computes and why each step is the way it is.
It assumes familiarity with Black-Scholes and nothing beyond that.

---

## 1. The result being used

Breeden and Litzenberger (1978) observed that the risk-neutral distribution of a
terminal price is not something to be inferred indirectly from option prices — it
is already there in them. For a European call struck at `K` expiring at `T`,

```
C(K) = e^(-rT) ∫ (S - K)+ q(S) dS
```

Differentiating twice in the strike collapses the integral:

```
∂C/∂K   = -e^(-rT) [1 - Q(K)]
∂²C/∂K² =  e^(-rT) q(K)
```

so

```
q(K) = e^(rT) ∂²C/∂K²
```

The trading intuition is cleaner than the algebra. Buy a call at `K - h`, sell two
at `K`, buy one at `K + h`. That butterfly pays out only if the price lands near
`K`, and the payoff is worth `h` there. Its price, grossed up for discounting and
divided by `h²`, is the market's probability density at `K`. **The price of the
spread is the probability.** Everything below is machinery for extracting that
number without the noise in the quotes destroying it.

---

## 2. Why the naive version fails

Two numerical derivatives of quoted prices amplify quote noise by roughly `1/h²`.
On a board with 25-point strike spacing and a two-cent tick, the rounding error
alone is the same order as the butterfly price in the wings. The output oscillates
and goes negative. `density_from_prices` implements exactly this, and
`test_density_from_prices_when_the_curve_is_noisy_goes_negative` demonstrates the
failure deliberately, so the problem this package solves is visible in the test
suite rather than only asserted in prose.

The standard fix is to smooth in implied-volatility space, where the curve is
nearly quadratic and well conditioned, then convert back. This package goes one
step further and never converts back at all.

---

## 3. Working in total variance and log-moneyness

Define

```
k = log(K / F)          (log-moneyness against the forward)
w(k) = σ(k)² T          (total implied variance)
```

Both no-arbitrage conditions and the density have clean expressions in these
coordinates, and neither does in `(K, σ)`.

Given `w`, `w'` and `w''` at a point, the density of `x = log(S_T / F)` is

```
p(k) = g(k) / sqrt(2π w(k)) · exp(-d₂(k)² / 2)

g(k) = (1 - k w'(k) / (2 w(k)))² - (w'(k)²/4)(1/w(k) + 1/4) + w''(k)/2
d₂(k) = -k/sqrt(w(k)) - sqrt(w(k))/2
```

and in strike space `q(K) = p(log(K/F)) / K`.

`g` is Durrleman's function. Three properties make it the centre of the design:

1. **It is a closed form.** No finite differences of prices anywhere. Fit a smooth
   `w`, and the density is exact to machine precision.
2. **`g(k) ≥ 0` is exactly the no-butterfly-arbitrage condition.** Because `g` is
   the numerator of the density, checking for arbitrage and checking for negative
   probability are the same test.
3. **Flat `w` gives `g ≡ 1`**, which recovers the Black-Scholes lognormal exactly.
   `test_density_from_smile_when_the_smile_is_flat_matches_the_lognormal` pins
   this to a tolerance of 1e-12.

---

## 4. The forward comes from the quotes, not from a rate

Put-call parity holds at every listed strike:

```
C(K) - P(K) = D · (F - K)
```

Regress the call-put spread on the strike across matched pairs near the money.
The slope is `-D` and the intercept is `D · F`. Both the forward and the discount
factor drop out of the option quotes themselves.

This matters more than it looks. A textbook `S · exp((r - q)T)` forward requires a
rate and a dividend estimate that nobody publishes exactly; index dividends are
forecasts, single names have borrow costs, and either can be off by tens of basis
points. A forward that is wrong by half a percent tilts the entire distribution.
The regression sidesteps all of it and prices the forward the market is trading —
including borrow and expected dividends, whatever they happen to be.

Two engineering details:

- The fit is limited to a band around the money. Parity is exact everywhere in
  theory and reliable only near the money in practice: deep in-the-money quotes
  are wide, stale and almost entirely intrinsic, so one bad print sets the forward.
- A discount factor above one is a negative implied rate. That is a real state of
  the world, so it is admitted rather than rejected.

---

## 5. Cleaning

Applied before anything is fitted:

| Rule | Damage prevented |
|---|---|
| Drop zero or missing bids | Nobody is buying, so the mid is fiction |
| Drop crossed or locked markets | Stale data from a fast tape |
| Drop relative spreads above a threshold | The quote carries less information than its own uncertainty |
| Keep only out-of-the-money options | In-the-money quotes are dominated by intrinsic value, so the same quote noise becomes a much larger volatility error |
| Drop quotes that invert to `nan` | Prices outside the no-arbitrage bounds remove themselves |

---

## 6. Fitting the smile

Two models, sharing the density code through the `Smile` interface.

### SVI (default)

Gatheral's raw parameterisation:

```
w(k) = a + b (ρ(k - m) + sqrt((k - m)² + σ²))
```

Five parameters, analytic derivatives, and enough structure to impose
no-arbitrage during the fit rather than patching it afterwards. Calibration is
penalised least squares with:

- **Box bounds** for admissibility: `b ≥ 0`, `|ρ| < 1`, `σ > 0`, and `b ≤ 2` from
  Lee's moment formula.
- **A penalty** on `min(g, 0)` over a grid, escalated through weights until a
  fine-grid audit comes back clean.
- **Multi-start**, because the objective is not convex.
- **An exact repair.** A penalty method approaches the feasible boundary from the
  wrong side: the violation shrinks with every increase in weight but never quite
  reaches zero, so a converged-looking fit can still carry a negative probability
  a fraction of a basis point deep. If the ladder leaves anything behind, `b` is
  scaled down — flattening the smile and lifting `g` toward the flat-smile value
  of one — with `a` raised to hold the at-the-money variance fixed. A scale of
  zero is a flat smile, which always satisfies the condition, so the search
  cannot fail.

The density is therefore non-negative by construction rather than by inspection.

### Weighted smoothing spline

A cubic spline through `(k, w)`, weighted by how precisely each quote pins its own
volatility: a quote determines vol to about `half_spread / vega`, and fitting `w`
rather than `σ` adds the Jacobian `2σT`. With weights as inverse standard errors,
the residual sum is a chi-square with about one degree of freedom per quote, so
the default smoothing is the sample size — the fit may miss each quote by about
its own uncertainty and no more.

The spline is the more flexible model and the less protected one. Where the two
methods disagree, the chain is saying something about its own quality.

---

## 7. Extrapolating the wings

Beyond the last quoted strike there is no information, only extrapolation, and
this is where naive implementations produce negative probabilities.

Total variance is continued **linearly** in `k`, under two rules:

1. **Lee's moment formula** caps the asymptotic slope at 2. Above that the implied
   distribution has no finite moments of any order. Substituting `w ≈ s|k|` into
   `g` gives `g → 1/4 - s²/16`, which is non-negative precisely when `s ≤ 2` —
   Lee's bound and the butterfly condition turn out to be the same statement about
   the wings.
2. **The slope is scaled back further if `g` still dips.** A linear wing has no
   curvature, so it loses the `w''/2` term holding `g` up on the spline side of
   the join. A smile bending hard at the last quoted strike goes butterfly-negative
   immediately outside it if the slope is simply carried over. The wing takes the
   steepest admissible slope found by a descending scan; a flat wing always works,
   so termination is guaranteed.

The cost is a small discontinuity in the density's curvature at the edge of the
quoted range. That is the honest place for it.

---

## 8. Checking the answer

A density that fails these is wrong no matter how convincing the chart looks.

| Check | What it catches |
|---|---|
| `total_mass ≈ 1` | Grid truncation. The grid is widened until it is consistent with the volatility at its own edges, because sizing it off the at-the-money vol alone cuts off exactly the skewed tail of interest |
| `martingale_error ≈ 0` | The single most informative test. Under the risk-neutral measure the forward *is* `E[S_T]`. A bad forward, a bad smile or a truncated grid all show up here |
| `min_density ≥ 0` | Negative probability |
| `worst_durrleman_g ≥ 0` | Butterfly arbitrage in the fitted smile |
| Static checks on the raw quotes | Bounds, monotonicity and convexity violations in the input, reported separately so a data problem is never mistaken for a modelling one |

The test suite goes further and scores the estimator against ground truth. Chains
are generated from a mixture of lognormals, which has closed-form option prices
*and* a closed-form density, so the estimate can be compared with the exact answer
in L1, in its moments, and in its tail probabilities. Recovery is within about 1%
in L1 on clean quotes.

---

## 9. What comes out

For an equity index the implied distribution is not the Black-Scholes bell. It is
left-skewed with a fat left tail: on the bundled sample chain a 20% drop carries
roughly three times the probability the lognormal assigns it, at the same forward
and the same at-the-money volatility.

That gap is the crash risk the market is paying for. Two cautions on reading it:

- **This is the risk-neutral measure, not the real-world one.** It is a probability
  weighted by risk aversion, and the two differ by a pricing kernel that this
  package makes no attempt to estimate. The density says what the market charges
  for a payoff, which is not the same as what it expects to happen. Treating a
  risk-neutral tail probability as a forecast overstates crash odds.
- **Beyond the quoted strikes it is extrapolation.** The wings are governed by the
  rules in section 7, not by data.

---

## References

- Breeden, D. and Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business* 51(4).
- Figlewski, S. (2008). Estimating the implied risk-neutral density for the US market portfolio.
- Gatheral, J. (2006). *The Volatility Surface*.
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1).
- Lee, R. (2004). The moment formula for implied volatility at extreme strikes. *Mathematical Finance* 14(3).
