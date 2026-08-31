# Chapter 5 and 6 Result Audit

**Date:** 2026-08-05

**Files checked:**

- `/Users/ayudhya/Downloads/Chapter 5 - Empirical Results - Revised.docx`
- `/Users/ayudhya/Downloads/Chapter 6 - Discussion - Revised.docx`

**Audit rule:** factual claims were checked against live artifacts under
`results/asset_pricing_ml` and related result symlinks, not against earlier
planning notes or Markdown drafts.

---

## Executive Finding

Most Chapter 5 sections after Section 5.3 and most Chapter 6 claims are
consistent with the live result files. The main factual/structural problem is
Chapter 5 Section 5.2 and the corresponding Chapter 6 discussion of the
baseline depth result.

Chapter 5 Tables 5.1 and 5.2 match a live result file, but not the Compustat-only
baseline implied by the chapter sequence. They match:

- `results/asset_pricing_ml/dre_estimates_enriched_strict_lag1_ex_ante/model_summary.csv`
- `results/asset_pricing_ml/dre_estimates_enriched_strict_lag1_ex_ante/predictive_accuracy_ic_tests.csv`

That run uses `feature_set = estimates_enriched` and the strict-lag-1 estimates
panel. It therefore already includes analyst expectations, while Section 5.5
later introduces analyst expectations as the data-depth extension. This creates
an internal inconsistency: the baseline horse race is no longer cleanly a
Compustat/broad-characteristic baseline.

---

## Issues To Fix

### 1. Chapter 5 Section 5.2 uses estimates-enriched results as the baseline

Current Chapter 5, Table 5.1:

| Model | Mean IC | IC information ratio | VW turnover |
|---|---:|---:|---:|
| HistGBM | 0.1230 | 5.949 | 1.114 |
| MLP | 0.1194 | 6.028 | 1.080 |
| DRE | 0.1180 | 5.532 | 0.960 |
| Ridge | 0.1178 | 5.484 | 0.969 |
| Momentum | 0.0669 | 2.789 | 0.795 |

These values are live, but they come from
`dre_estimates_enriched_strict_lag1_ex_ante/model_summary.csv`, whose manifest
reports `feature_set = estimates_enriched` and the strict-lag-1 estimates panel.

If Section 5.2 is meant to be the Compustat-only baseline, use
`results/asset_pricing_ml/europe_compustat_benchmark/model_summary.csv` for the
value-weighted, standard ex-bottom-5%, long-short, 25 bps cell:

| Model | Mean IC | IC information ratio | VW turnover |
|---|---:|---:|---:|
| HistGBM | 0.1192 | 5.727 | 1.093 |
| MLP | 0.1189 | 5.633 | 0.957 |
| Ridge | 0.1162 | 5.285 | 0.962 |
| Elastic net | 0.1146 | 4.935 | 0.848 |
| Momentum | 0.0669 | 2.789 | 0.795 |

There is no DRE row in that `europe_compustat_benchmark` run. If DRE must remain
in the baseline table, use a different common-sample DRE source and state the
different sample explicitly. The older common DRE benchmark is
`results/asset_pricing_ml/deep_sequence_common_benchmark/common_model_summary.csv`
with 448,813 observations, not 459,829.

Recommended fix:

- Replace Tables 5.1 and 5.2 with the Compustat-only benchmark if the section is
  intended to precede analyst-data-depth results.
- Or explicitly relabel Section 5.2 as an estimates-enriched strict-lag-1
  benchmark and then rewrite Section 5.5 so it is not presented as adding the
  analyst layer for the first time.

The first option is cleaner for the current dissertation structure.

### 2. Chapter 5 Table 5.2 is also from the estimates-enriched strict-lag-1 DRE run

Current Chapter 5 Table 5.2 reports:

- DRE minus momentum: 0.0510, t = 7.759, Holm p < 0.001.
- Ridge minus momentum: 0.0508, t = 7.916, Holm p < 0.001.
- HistGBM minus ridge: 0.0052, t = 4.098, Holm p < 0.001.
- DRE minus ridge: 0.0002, t = 0.384, Holm p = 0.903.

These match `dre_estimates_enriched_strict_lag1_ex_ante/predictive_accuracy_ic_tests.csv`.

For the Compustat-only benchmark in `europe_compustat_benchmark`, the comparable
rank IC tests are:

| Comparison | Mean IC difference | t-statistic | Holm p-value |
|---|---:|---:|---:|
| Ridge minus momentum | 0.0492 | 7.966 | 1.31e-14 |
| HistGBM minus momentum | 0.0523 | 8.605 | 7.61e-17 |
| MLP minus momentum | 0.0519 | 7.821 | 3.66e-14 |
| HistGBM minus ridge | 0.0030 | 2.512 | 0.0614 |
| MLP minus ridge | 0.0027 | 2.568 | 0.0614 |
| HistGBM minus MLP | 0.0003 | 0.271 | 0.786 |

Implication:

- Under the Compustat-only benchmark, HistGBM is directionally higher than ridge
  but does not survive Holm adjustment at the 5 percent level in
  `europe_compustat_benchmark`.
- If the chapter keeps the current Table 5.2 values, it must disclose that the
  test is estimates-enriched, not Compustat-only.

### 3. Chapter 6 repeats the baseline-source issue

Chapter 6 Section 6.2 states that HistGBM improves on ridge by 0.0052 in
monthly rank IC. That number is live, but it is the estimates-enriched
strict-lag-1 DRE-run value.

If Chapter 6 is discussing the Compustat/broad-characteristic baseline, revise
the sentence to use the Compustat-only value:

> HistGBM improves on ridge by 0.0030 in monthly rank IC in the Compustat-only
> benchmark, but the Holm-adjusted p-value is 0.061, so the baseline IC evidence
> for model-depth gains is directional rather than formally resolved at the 5
> percent level.

Then keep the stronger model-depth result in return space:

> In the primary spanning ladder, HistGBM adds 4.28 percent per year over ridge
> and survives Holm adjustment.

---

## Sections That Check Out Against Live Results

### Chapter 5 Section 5.3: spanning ladder

Source:

- `results/asset_pricing_ml/complexity_spanning_ladder/complexity_spanning_ladder.csv`

The reported primary equal-weighted, standard ex-bottom-5%, long-short, 25 bps
non-confounded rows match the live file:

| Increment | Annual alpha | t-stat | Holm p | Adjusted R2 |
|---|---:|---:|---:|---:|
| Momentum over factors | 6.84% | 2.186 | 0.0288 | 0.531 |
| Ridge over momentum | 5.75% | 1.732 | 0.0833 | 0.512 |
| HistGBM over ridge | 4.28% | 2.991 | 0.0083 | 0.848 |
| MLP over ridge | 3.31% | 1.842 | 0.1309 | 0.907 |
| DRE over ridge | -0.60% | -0.610 | 0.5420 | 0.966 |

The prose that ridge-over-momentum is economically material but imprecise, and
that DRE is almost fully spanned by ridge, is supported.

### Chapter 5 Section 5.4: tradability gradient

Sources:

- `results/asset_pricing_ml/capacity_gradient_tests/capacity_gradient_tests.csv`
- `results/asset_pricing_ml/capacity_gradient_tests/paired_premium_by_bucket.csv`

The broad-characteristic gradients versus momentum match the live file:

- Market-cap gradients: Ridge 0.0725, HistGBM 0.0725, MLP 0.0736, DRE 0.0736.
- Trading-value gradients: Ridge 0.0777, HistGBM 0.0709, MLP 0.0801, DRE 0.0778.

The depth-premium gradients versus ridge also match: they are approximately
zero and all have Holm p-values of 1.000.

The prose correctly treats the result as associational rather than causal.

### Chapter 5 Section 5.5: analyst data depth and model depth

Source:

- `results/asset_pricing_ml/data_depth_model_depth_interaction/data_depth_model_depth_interaction.csv`

The reported data-depth effects and interactions match:

- Ridge data-depth effect: 0.0036, Holm p = 0.0227.
- HistGBM data-depth effect: 0.0070, Holm p = 0.0005.
- MLP data-depth effect: 0.0036, Holm p = 0.204.
- DRE data-depth effect: 0.0051, Holm p = 0.0054.
- HistGBM interaction: 0.0035, Holm p = 0.0375.
- MLP and DRE interactions are not Holm-significant.

The prose appropriately avoids a broad claim that deep architectures generally
become superior once analyst data are added.

### Chapter 5 Section 5.6: forecast-error mechanism

Sources:

- `results/asset_pricing_ml/forecast_error_mechanism/mechanism_fama_macbeth_summary.csv`
- `results/asset_pricing_ml/forecast_error_mechanism/mechanism_joint_signal_summary.csv`
- `results/asset_pricing_ml/forecast_error_mechanism/mechanism_sign_accuracy.csv`
- `results/asset_pricing_ml/econometric_evidence_tables/econometric_evidence_summary.csv`

The principal forecast-error rows match:

- EPS error on 3m EPS revision: coefficient 0.0233, t = 2.764, p = 0.0057.
- EPS revision robustness row: coefficient 0.0291, t = 2.544, p = 0.0110.
- Revenue error on 3m revenue revision: coefficient 0.0615, t = 5.227, p < 0.001.
- EPS error on 1m EPS revision: coefficient 0.0151, t = 11.006, p < 0.001.
- Revenue error on 1m revenue revision: coefficient 0.0664, t = 9.101, p < 0.001.

The joint-signal examples in the prose also match the live joint summary. The
sign-accuracy caveat is supported.

### Chapter 5 Section 5.7: constrained implementation

Sources:

- `results/asset_pricing_ml/constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed/constrained_summary.csv`
- `results/asset_pricing_ml/constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed/benchmark_relative_summary.csv`
- `results/asset_pricing_ml/econometric_evidence_tables/econometric_evidence_summary.csv`

The reported full-sample implementation table matches:

| Capital | Annual net return | Sharpe | Impact cost | Active return | Alpha | Alpha p |
|---|---:|---:|---:|---:|---:|---:|
| EUR 10m | 14.99% | 0.853 | 0.34% | 6.43% | 5.89% | 0.0537 |
| EUR 100m | 14.26% | 0.810 | 1.07% | 5.70% | 5.17% | 0.0944 |
| EUR 500m | 12.94% | 0.734 | 2.39% | 4.38% | 3.86% | 0.2220 |

The EUR 100m net-return CI, Sharpe CI and active-return CI also match the live
econometric evidence table. The conclusion that standalone performance is
positive but benchmark-relative outperformance is not established is supported.

### Chapter 5 Section 5.8: matched US benchmark

Sources:

- `results/asset_pricing_ml/us_comparison_exhibits/rank_model_comparison.csv`
- `results/asset_pricing_ml/market_comparison_compustat/sharpe_difference_tests.csv`
- `results/asset_pricing_ml/us_comparison_exhibits/return_target_instability.csv`
- `results/asset_pricing_ml/us_comparison_exhibits/return_correlations.csv`

The rank comparison, Sharpe-difference table, return-target instability rows and
return-correlation rows all match the live files.

The prose correctly says the US comparison excludes the analyst layer and tests
portability of broad-characteristic prediction rather than the full
decomposition.

### Chapter 6 factual consistency

Most Chapter 6 empirical claims are consistent with the live result files:

- The tradability gradient of roughly 0.07-0.08 and near-zero depth gradients
  match the capacity-gradient files.
- The limits-to-arbitrage interpretation is phrased as conditional rather than
  causal.
- The analyst-expectations mechanism discussion matches the forecast-error and
  data-depth results.
- The EUR 100m implementation numbers match the constrained implementation
  files.
- The limitations around sample length, analyst coverage, delisting-return
  coverage, observed spreads and non-causal identification are consistent with
  the current project state.

The one empirical number to adjust is the Section 6.2 statement that HistGBM
improves on ridge by 0.0052 in monthly rank IC, unless the paragraph explicitly
says it is referring to the estimates-enriched strict-lag-1 DRE run.

---

## Recommended Minimal Corrections

1. In Chapter 5 Section 5.2, decide whether the baseline is Compustat-only or
   estimates-enriched.

2. If the baseline is Compustat-only, replace Tables 5.1 and 5.2 using
   `europe_compustat_benchmark` and remove or separately caveat DRE from the
   table because that benchmark run does not include DRE.

3. If DRE must stay in Section 5.2, use the 448,813-observation common-sample
   DRE benchmark from `deep_sequence_common_benchmark`, or present DRE as a
   separately sourced model-depth comparison. Do not combine the 459,829
   Compustat benchmark language with estimates-enriched strict-lag-1 DRE rows.

4. In Chapter 6 Section 6.2, replace the 0.0052 HistGBM-minus-ridge IC statement
   with the Compustat-only 0.0030 / Holm p = 0.061 value, or explicitly label
   the 0.0052 value as estimates-enriched.

5. Keep the later Chapter 5 Sections 5.3-5.8 and Chapter 6 Sections 6.3-6.7
   largely as written; they are well aligned with the current live result files.
