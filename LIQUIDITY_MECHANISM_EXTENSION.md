# Liquidity-Mechanism Extension

## Research purpose

This extension tests whether European ML return predictability is concentrated
in difficult-to-trade securities and attenuates as the investment universe is
restricted toward institutionally investable stocks.

The design provides mechanism-consistent and limits-to-arbitrage evidence. It
does not identify a causal effect of illiquidity on expected returns.

## Frozen baseline and expanded specification

The published 18-characteristic baseline remains unchanged. The ML runner now
requires an explicit feature-set choice:

- `baseline`: the original 18 ranked characteristics;
- `expanded_liquidity`: the baseline plus two prespecified monthly liquidity
  characteristics.

The monthly extensions are:

1. `log_trading_value_eur`: log EUR market capitalisation multiplied by the
   trailing 12-month mean share-turnover ratio. This avoids comparing
   local-currency `price x volume` across European markets.
2. `turnover_volatility_12m`: trailing 12-month standard deviation of monthly
   share turnover.

Both use information available by the signal month. Baseline eligibility still
depends only on the original feature count, so the extension does not silently
change the sample. In the rebuilt panel, coverage among baseline-model-eligible
rows is 96.4% for log EUR trading value and 96.6% for turnover volatility.

Proper daily Amihud illiquidity and zero-return-day frequency remain pending.
The available daily-security pull contains only a small test batch and is not
adequate for inference. Lagged bid-ask spread remains a matched-universe
robustness because full-universe historical coverage is unavailable.

## OOS variable importance

Every fitted annual ridge, elastic-net, histogram-GBM and MLP model is evaluated
before it is discarded. For each test year, the implementation:

1. records the model's original OOS prediction;
2. sets a feature or a complete theme to zero, the monthly ranked-feature
   median;
3. holds model parameters fixed and predicts the same test observations;
4. reports the increase in OOS mean squared error, loss in zero-benchmark
   R-squared and decrease in Spearman IC.

Theme ablation is primary because correlated predictors can split
individual-feature importance. Individual ablation remains available for
within-theme interpretation. Themes are:

- price trends;
- liquidity;
- risk;
- size;
- fundamentals.

Size is kept separate from liquidity so liquidity importance cannot be inferred
mechanically from market capitalisation. Results are reported for the full test
sample and small, middle and large market-capitalisation thirds.

`oos_variable_importance.csv` contains raw and within-model/year normalized
positive importance. `oos_binned_responses.csv` and model-specific PNGs show
observed OOS prediction and realization curves for the five most important
features. These curves are descriptive, not causal partial dependence.

### Completed 2015-2026 result

The full expanded run contains 3,678,632 OOS predictions and zero training
cutoff violations. It does not support a China-style claim that liquidity is
the dominant European predictor.

Across model-years in the full cross-section:

- rank-model mean theme-level loss in zero-benchmark R-squared is 0.00361 for
  fundamentals, 0.00340 for price trends, 0.00183 for risk, 0.00102 for size
  and only 0.00024 for liquidity;
- return-model mean theme-level loss is 0.00188 for price trends and 0.00072
  for fundamentals, while removing the complete liquidity block improves
  average return-target R-squared by 0.00289;
- liquidity ablation is more damaging for rank prediction among small stocks
  than large stocks, but it remains below fundamentals, price trends and risk;
- individual liquidity variables, especially trailing turnover, can still be
  useful. Their positive individual ablations alongside weak or negative
  theme ablation show interaction and redundancy, not a dominant unified
  liquidity mechanism.

Relative to the frozen baseline, the two monthly liquidity additions change
rank R-squared by less than 0.01 percentage points for ridge, elastic net and
GBM, and reduce MLP rank R-squared by about 0.10 percentage points. Raw-return
R-squared rises slightly for the linear models and GBM but falls by about 0.06
percentage points for the MLP. No raw-return predictive-accuracy difference
survives Holm correction.

The defensible comparative conclusion is therefore that Europe differs from
the published Chinese evidence: liquidity helps selected models and segments,
but price trends and fundamentals provide the more stable predictive content.

## Predictive-accuracy inference

`predictive_accuracy_loss_tests.csv` implements the GKX monthly
cross-sectional loss comparison:

```text
d_t = cross-sectional mean(error_A^2 - error_B^2)
```

The time-series mean uses Newey-West/HAC inference with six lags and Holm
correction within target type. Raw-return models are compared only with other
raw-return models. Rank-target models are compared in rank-target units, while
`predictive_accuracy_ic_tests.csv` provides the more interpretable primary
comparison for rank signals, including momentum.

Applying the test to the existing 2015-2026 frozen forecasts finds no
Holm-significant difference among the raw-return ML models. Several nominal
differences have p-values near 5%, but their Holm-adjusted p-values are about
0.25 or larger. This supports the existing conclusion that isolated nonlinear
point estimates do not establish stable predictive dominance.

## Frozen-prediction investability ladder

The ladder never refits the forecasting model. It freezes each existing OOS
prediction and reconstructs holdings independently within five ex-ante,
nested universes:

1. standard universe excluding the bottom 5% by market capitalisation;
2. top 70% by market capitalisation;
3. largest 500 stocks;
4. largest 500 with observed bid-ask spreads;
5. the low-spread half of the preceding large-stock universe.

For every rung, model, month, weighting rule and portfolio format, holdings,
turnover, spread costs and square-root market impact are resimulated. The same
security-level cost function is used at each rung; costs differ only because
the securities and trades differ. Results are reported at EUR 10m, EUR 100m
and EUR 500m.

The completed run produces 24,660 monthly strategy rows, 540 summary rows, 45
predictive-metric rows and 1,440 paired-inference rows. At EUR 100m and value
weighting:

- MLP-return long-only net Sharpe declines from 0.822 in the standard universe
  to 0.689 in the top-500 universe and 0.705 in the large/low-spread universe.
- Momentum long-only remains stronger, with net Sharpe 0.985, 0.945 and 0.837
  in the corresponding universes.
- MLP-return long-short net Sharpe falls from 0.447 in the standard universe
  to 0.169 in the top 500 and 0.158 among large/low-spread stocks.
- Momentum long-short remains more stable at 0.507, 0.638 and 0.419.

Paired inference compares every non-momentum model with `momentum_rank` within
each rung, weighting rule, portfolio format, AUM level and bootstrap block. It
uses HAC tests for annualised net-return differences and paired stationary
bootstraps for Sharpe and certainty-equivalent differences, with Holm
correction across model comparisons in each fixed test family. In the primary
EUR 100m value-weighted long-only family, no ML signal delivers a positive
Holm-significant Sharpe or certainty-equivalent advantage over momentum on any
rung. Several broad-universe return-target models are significantly worse than
momentum. Across the full ladder, the only positive Holm-significant Sharpe
exception is the MLP-return equal-weight long-short strategy in the broadest
universe at EUR 10m with expected block length three; it is not significant by
certainty-equivalent or HAC mean-return inference and is not an implementable
headline result.

The ladder therefore supports attenuation and portfolio-design sensitivity
rather than a monotonic causal liquidity mechanism.

## Reproduction

Rebuild the feature panel:

```bash
python scripts/build_asset_pricing_panel.py
```

Run the unchanged baseline:

```bash
python scripts/run_asset_pricing_ml.py \
  --feature-set baseline \
  --output-dir results/asset_pricing_ml/baseline_with_diagnostics
```

Run the expanded monthly-liquidity specification:

```bash
python scripts/run_asset_pricing_ml.py \
  --feature-set expanded_liquidity \
  --output-dir results/asset_pricing_ml/expanded_liquidity
```

Run the frozen-signal ladder:

```bash
python scripts/run_investability_ladder.py
```

The CPZ-inspired stochastic-discount-factor/test-asset extension remains
deliberately out of scope unless requested by the supervisor. Existing
Fama-MacBeth and factor-spanning tests already provide the main asset-pricing
evidence.
