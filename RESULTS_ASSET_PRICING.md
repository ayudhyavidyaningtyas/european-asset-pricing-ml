# Results - ML Asset Pricing on European Equities

## Bottom line

Machine-learning scores contain real, placebo-verified information about
next-month European stock returns. That information remains incrementally
related to returns after controlling for momentum, size, book-to-market,
rolling beta, idiosyncratic volatility, country and sector. However, it does
not produce a statistically reliable value-weighted net portfolio advantage
over 12-2 momentum.

The deeper analysis locates the wedge: ML predictability is strongest outside
large, low-idiosyncratic-risk stocks and is weaker in down markets. Nominal
factor-adjusted ML-minus-momentum alphas do not survive family-wise correction.
The defensible conclusion is therefore **incremental predictability without a
robust implementable alpha**.

Authoritative artifacts:

- Baseline: `results/asset_pricing_ml/revised_full_eur_delisting/`
- Economic-depth stage: `results/asset_pricing_ml/depth_analysis/`
- Compustat data-depth extension:
  `results/asset_pricing_ml/compustat_enriched_full_layer1_p96/`
- Compustat nonlinear rank baselines:
  `results/asset_pricing_ml/compustat_enriched_nonlinear_rank/`
- Residualized European target screen:
  `results/asset_pricing_ml/residual_target_compustat_screen/`
- Direct-utility neural SDF baseline:
  `results/asset_pricing_ml/neural_sdf_compustat_full/`
- Adversarial LSTM/GAN SDF adaptation:
  `results/asset_pricing_ml/adversarial_sdf_compustat_main/`
- AIPM linear transformer SDF adaptation:
  `results/asset_pricing_ml/aipm_linear_transformer_compustat_all53/`
- Full nonlinear AIPM transformer adaptation:
  `results/asset_pricing_ml/aipm_full_transformer_compustat_cap500_seed3/`
- Full AIPM adaptation bundle and depth scaling:
  `results/asset_pricing_ml/aipm_full_adaptation_bundle/`,
  `results/asset_pricing_ml/aipm_full_transformer_depth1_cap500_seed1/`,
  `results/asset_pricing_ml/aipm_full_transformer_depth2_cap500_seed1/`
  and `results/asset_pricing_ml/aipm_full_transformer_depth4_cap500_seed1/`
- Conditional autoencoder asset-pricing adaptation:
  `results/asset_pricing_ml/autoencoder_compustat_k5_main/`
- Autoencoder/AIPM cost-assumption sensitivity:
  `results/asset_pricing_ml/cost_assumption_sensitivity/`
- Peer-implied fundamental mispricing adaptation:
  `results/asset_pricing_ml/fundamental_mispricing_hkr_market_share_main/`
- Deep sequence modelling full runs:
  `results/asset_pricing_ml/deep_sequence_compustat_full_seq12/` and
  `results/asset_pricing_ml/deep_sequence_compustat_full_seq24/`
- Deep sequence common benchmark:
  `results/asset_pricing_ml/deep_sequence_common_benchmark/`
- Deep sequence blend/complementarity test:
  `results/asset_pricing_ml/deep_sequence_blend_experiment/`
- Deep sequence/blend investability ladder:
  `results/asset_pricing_ml/deep_sequence_blend_investability_ladder/`
- Deep sequence/blend regime diagnostics:
  `results/asset_pricing_ml/deep_sequence_regime_diagnostics/`
- Turnover-aware signal smoothing:
  `results/asset_pricing_ml/turnover_aware_signal_smoothing/`
- Rolling validation-selected implementable strategy:
  `results/asset_pricing_ml/validation_selected_implementable_strategy/`
- Fixed-rung validation-selected robustness:
  `results/asset_pricing_ml/validation_selected_top500_observed/`,
  `results/asset_pricing_ml/validation_selected_large_low_spread/` and
  `results/asset_pricing_ml/validation_selected_deep_hybrid_liquid/`
- Frozen deep/hybrid long-only robustness:
  `results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness/`
- Constrained deep/hybrid long-only construction:
  `results/asset_pricing_ml/constrained_deep_hybrid_long_only/`
- Refinitiv refreshed-liquidity validation reruns:
  `results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness_refinitiv_refresh/`,
  `results/asset_pricing_ml/constrained_deep_hybrid_long_only_refinitiv_refresh/`
  and `results/asset_pricing_ml/refinitiv_refresh_comparison/`
- Broader Refinitiv liquidity-universe stress:
  `results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness_top1000/`,
  `results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness_top2000/`,
  `results/asset_pricing_ml/constrained_deep_hybrid_long_only_top1000/`,
  `results/asset_pricing_ml/constrained_deep_hybrid_long_only_top2000/`
  and `results/asset_pricing_ml/broader_liquidity_universe_comparison/`
- Closure experiments before new ML:
  `results/asset_pricing_ml/closure_costing_top2000_liquidity/`,
  `results/asset_pricing_ml/validation_selected_constrained_deep_hybrid/`
  and `results/asset_pricing_ml/closure_experiment_summary/`

The subsequent liquidity-mechanism design, diagnostics and descriptive
investability-ladder results are documented in
[`LIQUIDITY_MECHANISM_EXTENSION.md`](LIQUIDITY_MECHANISM_EXTENSION.md).

## Reproduce

```bash
python scripts/build_asset_pricing_panel.py

python scripts/run_asset_pricing_ml.py \
    --models momentum ridge elastic_net hist_gbm mlp \
    --targets rank return \
    --first-test-year 2015 --last-test-year 2026 \
    --mlp-epochs 20 --placebo-repetitions 199 \
    --output-dir results/asset_pricing_ml/revised_full_eur_delisting

python scripts/run_asset_pricing_depth.py
python scripts/run_external_factor_robustness.py

python scripts/build_compustat_enriched_panel.py

python scripts/run_asset_pricing_ml.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --models momentum ridge dre \
    --targets rank return \
    --first-test-year 2015 --last-test-year 2026 \
    --no-tuning --placebo-repetitions 0 \
    --skip-delisting-scenarios --skip-importance \
    --dre-layers 1 --dre-features-per-block 96 \
    --output-dir results/asset_pricing_ml/compustat_enriched_full_layer1_p96

python scripts/run_asset_pricing_ml.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --models hist_gbm mlp \
    --targets rank \
    --first-test-year 2015 --last-test-year 2026 \
    --no-tuning --placebo-repetitions 0 \
    --skip-delisting-scenarios --skip-importance \
    --mlp-epochs 20 \
    --output-dir results/asset_pricing_ml/compustat_enriched_nonlinear_rank

python scripts/run_asset_pricing_ml.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --models momentum ridge hist_gbm \
    --targets rank residual_rank \
    --first-test-year 2015 --last-test-year 2026 \
    --max-training-rows 250000 \
    --no-tuning --placebo-repetitions 0 \
    --skip-delisting-scenarios --skip-importance \
    --output-dir results/asset_pricing_ml/residual_target_compustat_screen

python scripts/run_neural_sdf.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --epochs 30 --patience 8 \
    --hidden-sizes 16 8 --learning-rate 0.002 \
    --min-training-months 72 --validation-months 24 \
    --cost-grid-bps 0 10 25 50 \
    --output-dir results/asset_pricing_ml/neural_sdf_compustat_full

python scripts/run_adversarial_sdf.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --epochs 20 --patience 5 \
    --sequence-length 12 --state-hidden-size 8 \
    --sdf-hidden-sizes 32 16 --adversary-hidden-sizes 32 16 \
    --test-assets 4 \
    --min-training-months 72 --validation-months 24 \
    --device cpu \
    --output-dir results/asset_pricing_ml/adversarial_sdf_compustat_main

python scripts/run_aipm_linear_transformer_sdf.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --max-attention-features 0 \
    --min-training-months 72 --validation-months 24 \
    --output-dir results/asset_pricing_ml/aipm_linear_transformer_compustat_all53

python scripts/run_aipm_full_transformer_sdf.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --min-training-months 60 --validation-months 12 \
    --training-window-months 72 --refit-frequency annual \
    --max-monthly-stocks 500 \
    --random-feature-count 256 \
    --transformer-blocks 2 --attention-heads 1 \
    --feedforward-width 64 \
    --epochs 8 --patience 3 \
    --seeds 0 1 2 --device mps \
    --output-dir results/asset_pricing_ml/aipm_full_transformer_compustat_cap500_seed3

for blocks in 1 2 4; do
  python scripts/run_aipm_full_transformer_sdf.py \
      --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
      --feature-set compustat_enriched \
      --first-test-year 2015 --last-test-year 2026 \
      --min-training-months 60 --validation-months 12 \
      --training-window-months 72 --refit-frequency annual \
      --max-monthly-stocks 500 \
      --models own_asset_mlp nonlinear_transformer \
      --random-feature-count 256 \
      --transformer-blocks "${blocks}" --attention-heads 1 \
      --feedforward-width 64 \
      --epochs 4 --patience 2 \
      --seeds 0 --device mps \
      --output-dir "results/asset_pricing_ml/aipm_full_transformer_depth${blocks}_cap500_seed1"
done

python scripts/run_aipm_post_analysis.py \
    --run-dir results/asset_pricing_ml/aipm_full_transformer_compustat_cap500_seed3 \
    --output-dir results/asset_pricing_ml/aipm_post_analysis

python scripts/summarize_aipm_full_adaptation.py \
    --output-dir results/asset_pricing_ml/aipm_full_adaptation_bundle \
    --aum-label 100m

python scripts/run_autoencoder_asset_pricing.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --n-factors 5 --hidden-sizes 16 \
    --epochs 10 --patience 3 \
    --min-training-months 72 --validation-months 24 \
    --device cpu \
    --output-dir results/asset_pricing_ml/autoencoder_compustat_k5_main

python scripts/run_fundamental_mispricing.py \
    --output-dir results/asset_pricing_ml/fundamental_mispricing_hkr_market_share_main

python scripts/run_deep_sequence_models.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --models last_mlp lstm gru attention_lstm \
    --targets rank \
    --sequence-length 12 \
    --max-training-rows 150000 --max-validation-rows 60000 \
    --epochs 15 --patience 4 \
    --skip-delisting-scenarios \
    --output-dir results/asset_pricing_ml/deep_sequence_compustat_full_seq12

python scripts/run_deep_sequence_models.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --feature-set compustat_enriched \
    --first-test-year 2015 --last-test-year 2026 \
    --models last_mlp lstm gru attention_lstm \
    --targets rank \
    --sequence-length 24 \
    --max-training-rows 150000 --max-validation-rows 60000 \
    --epochs 15 --patience 4 \
    --skip-delisting-scenarios \
    --output-dir results/asset_pricing_ml/deep_sequence_compustat_full_seq24

python scripts/run_deep_sequence_benchmark_comparison.py \
    --output-dir results/asset_pricing_ml/deep_sequence_common_benchmark \
    --significance-bootstraps 2000

python scripts/run_deep_sequence_blend_experiment.py \
    --output-dir results/asset_pricing_ml/deep_sequence_blend_experiment \
    --significance-bootstraps 2000

python scripts/run_investability_ladder.py \
    --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
    --predictions results/asset_pricing_ml/deep_sequence_blend_experiment/blend_ladder_subset_predictions.parquet \
    --output-dir results/asset_pricing_ml/deep_sequence_blend_investability_ladder \
    --baseline-model momentum_rank \
    --bootstrap-repetitions 1000

python scripts/run_deep_sequence_regime_diagnostics.py \
    --models-from-manifest results/asset_pricing_ml/deep_sequence_blend_experiment/manifest.json \
    --output-dir results/asset_pricing_ml/deep_sequence_regime_diagnostics

python scripts/run_turnover_aware_signal_smoothing.py \
    --output-dir results/asset_pricing_ml/turnover_aware_signal_smoothing \
    --significance-bootstraps 1000

python scripts/run_validation_selected_implementable_strategy.py \
    --output-dir results/asset_pricing_ml/validation_selected_implementable_strategy \
    --bootstrap-repetitions 2000

python scripts/run_validation_selected_implementable_strategy.py \
    --candidate-ladder-monthly results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_monthly.parquet \
    --candidate-ladder-summary results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_summary.csv \
    --rungs top_500_observed_spread \
    --output-dir results/asset_pricing_ml/validation_selected_top500_observed \
    --bootstrap-repetitions 2000

python scripts/run_validation_selected_implementable_strategy.py \
    --candidate-ladder-monthly results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_monthly.parquet \
    --candidate-ladder-summary results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_summary.csv \
    --rungs large_low_spread \
    --output-dir results/asset_pricing_ml/validation_selected_large_low_spread \
    --bootstrap-repetitions 2000

python scripts/run_validation_selected_implementable_strategy.py \
    --candidate-ladder-monthly results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_monthly.parquet \
    --candidate-ladder-summary results/asset_pricing_ml/validation_selected_implementable_strategy/candidate_ladder_summary.csv \
    --rungs top_500_observed_spread large_low_spread \
    --candidate-models attention_lstm_seq12_rank attention_lstm_seq24_rank \
    gru_seq12_rank gru_seq24_rank blend90_mlp_attn_seq12_rank \
    blend90_mlp_attn_seq24_rank blend90_gbm_attn_seq24_rank \
    blend50_mom_gru_seq12_rank smooth50_attn_seq12_rank \
    smooth75_attn_seq12_rank smooth50_attn_seq24_rank \
    smooth75_attn_seq24_rank smooth50_gru_seq12_rank \
    smooth75_gru_seq12_rank smooth50_gru_seq24_rank \
    smooth75_gru_seq24_rank smooth50_blend90_mlp_attn_seq12_rank \
    smooth75_blend90_mlp_attn_seq12_rank \
    smooth50_blend90_mlp_attn_seq24_rank \
    smooth75_blend90_mlp_attn_seq24_rank \
    smooth50_blend90_gbm_attn_seq24_rank \
    smooth75_blend90_gbm_attn_seq24_rank \
    smooth50_blend50_mom_gru_seq12_rank \
    smooth75_blend50_mom_gru_seq12_rank \
    --output-dir results/asset_pricing_ml/validation_selected_deep_hybrid_liquid \
    --bootstrap-repetitions 2000

python scripts/run_frozen_deep_hybrid_robustness.py \
    --output-dir results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness \
    --bootstrap-repetitions 2000

python scripts/run_constrained_deep_hybrid_long_only.py \
    --output-dir results/asset_pricing_ml/constrained_deep_hybrid_long_only \
    --bootstrap-repetitions 2000

python scripts/run_frozen_deep_hybrid_robustness.py \
    --output-dir results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness_refinitiv_refresh \
    --bootstrap-repetitions 2000

python scripts/run_constrained_deep_hybrid_long_only.py \
    --output-dir results/asset_pricing_ml/constrained_deep_hybrid_long_only_refinitiv_refresh \
    --bootstrap-repetitions 2000

./scripts/run_broader_liquidity_universe_pipeline.sh

python scripts/run_cost_assumption_sensitivity.py \
    --weights \
      results/asset_pricing_ml/autoencoder_compustat_k5_costed/autoencoder_weights.parquet \
      results/asset_pricing_ml/autoencoder_grid/k10_h32_expanding_capall/autoencoder_weights.parquet \
      results/asset_pricing_ml/ipca_compustat_k5_main/ipca_weights.parquet \
      results/asset_pricing_ml/aipm_post_analysis/principal_portfolio_weights.parquet \
    --labels autoencoder_k5 autoencoder_k10_validation_all \
      ipca_k5 principal_portfolios \
    --liquidity \
      data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period_top2000 \
    --output-dir results/asset_pricing_ml/closure_costing_top2000_liquidity

python scripts/run_validation_selected_constrained_deep_hybrid.py \
    --output-dir results/asset_pricing_ml/validation_selected_constrained_deep_hybrid \
    --bootstrap-repetitions 2000
```

## Design

The compact GKX-style panel contains 18 ranked characteristics. Separate models
predict next-month return rank and raw return. Models use expanding annual
walk-forward estimation, a trailing 24-month validation window and a training
rule requiring every admitted target date to be no later than the previous
31 December.

The 2015-2026 OOS experiment contains 459,829 labelled stock-months per model,
4,138,461 labelled predictions and 810 additional flagged predictions for 90
scoreable missing retirement returns. Automated checks find zero
training-cutoff violations, zero retirement candidates used for training and
zero duplicate predictions.

Returns, market capitalisation and primary accounting values use EUR. Annual
fundamentals are lagged six months after fiscal year-end. The standard
investable universe excludes the bottom 5% by market capitalisation.

## Prediction

| model | rank IC | rank R2 | return R2 |
|---|---:|---:|---:|
| histogram GBM | 0.115 | 1.64% | -0.68% |
| MLP | 0.115 | 1.59% | **0.31%** |
| ridge | 0.113 | 1.51% | 0.24% |
| elastic net | 0.111 | 1.45% | 0.22% |
| momentum | 0.067 | -80.05% | - |

The ridge placebo retrains the model after shuffling labels within each month.
Across 199 repetitions, mean placebo IC is 0.0008 versus actual IC of 0.1129;
the empirical two-sided `p=0.005`.

The MLP now has the strongest positive raw-return R2. It is therefore too
strong to claim that nonlinear models add nothing. The accurate statement is
that nonlinear gains appear in selected point estimates but are not uniformly
stable across models or portfolio formats.

## Portfolios

Net Sharpe at 25 bps, standard universe (long-only uses EUR excess returns):

| model | equal-weight L/S | value-weight L/S | value-weight long-only |
|---|---:|---:|---:|
| momentum | 1.107 | 0.611 | **0.955** |
| ridge rank | 0.844 | 0.594 | 0.684 |
| histogram GBM rank | 1.388 | 0.524 | 0.725 |
| MLP rank | 1.358 | 0.788 | 0.541 |
| histogram GBM return | **2.958** | 0.832 | 0.883 |
| MLP return | 2.382 | **0.950** | 0.833 |

The primary value-weighted long-short MLP-return Sharpe difference versus
momentum is +0.339, with paired stationary-bootstrap 95% interval
[-0.230, 0.946] and `p=0.252`. Momentum remains best in the more implementable
value-weighted long-only comparison.

Holm correction is applied across all eight model-target comparisons within
each fixed weighting, universe, format, cost and block-length family. Every
adjusted p-value in the primary value-weighted long-short family equals 1.0.

Return-target nonlinear models produce significant equal-weight Sharpe
differences. These are not headline evidence: the extreme performance is
concentrated in diversified small-stock books, collapses under value weighting
and depends on a short leg with incomplete delisting returns. It is evidence of
portfolio-design sensitivity, not a general tradable ML premium.

## Deep Regression Ensemble Extension

A niche extension implements the Deep Regression Ensemble of Didisheim, Kelly
and Malamud (2022). The model is not another GKX tree or MLP: it draws
random-feature blocks, fits a grid of myopic ridge regressions, stacks their
predictions and combines the ensemble with a final ridge regression. The
implemented specification uses one DRE layer, 96 random features per block,
four random-feature scales and four ridge penalties.

The full-history DRE run uses the same European stock-month panel, annual
expanding walk-forward splits and net portfolio tests as the main ML exercise:

```bash
python scripts/run_asset_pricing_ml.py \
  --models momentum ridge dre \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --no-tuning --placebo-repetitions 0 \
  --skip-delisting-scenarios --skip-importance \
  --dre-layers 1 --dre-features-per-block 96 \
  --output-dir results/asset_pricing_ml/dre_full_layer1_p96_closed
```

The run produces 2,299,145 labelled predictions across the two DRE, two ridge
and momentum signals, with zero training-cutoff violations and zero duplicate
model-security-month predictions.

At 25 bps in the standard value-weighted long-short comparison, DRE is
descriptively stronger than momentum but not statistically reliable:

| model | annual return | volatility | Sharpe |
|---|---:|---:|---:|
| DRE return | 12.5% | 15.7% | **0.797** |
| DRE rank | 24.2% | 33.4% | 0.723 |
| ridge return | 9.4% | 14.2% | 0.661 |
| momentum | 17.2% | 28.2% | 0.611 |
| ridge rank | 19.5% | 32.5% | 0.600 |

The primary paired stationary-bootstrap Sharpe differences versus momentum are
+0.187 for DRE-return (`p=0.515`, Holm `p=1.0`) and +0.113 for DRE-rank
(`p=0.658`, Holm `p=1.0`). DRE therefore supplies a useful niche extension and
a better descriptive long-short point estimate, but it does not overturn the
main implementability conclusion.

In the more implementable value-weighted long-only comparison, momentum remains
stronger: Sharpe 0.955 versus 0.781 for DRE-return and 0.750 for DRE-rank.
Predictively, DRE-rank has slightly higher mean IC than ridge-rank
(0.114 versus 0.113), but the HAC IC difference is only marginal
(`p=0.057`). The rank-loss comparison favours DRE over ridge (`p=0.012`), while
the raw-return predictive comparison does not.

## Compustat Data-Depth Extension

The Chen-Pelger-Zhu data-depth route is implemented as an additive Compustat
Global extension rather than a replacement for the Refinitiv/Datastream return
panel. The enriched file keeps the original 1,596,754 stock-month rows and adds
33 Compustat-derived characteristics: 27 annual accounting/valuation ratios
and six monthly market variables. The ML feature set therefore rises from 20
expanded Refinitiv/liquidity ranks to 53 ranked predictors.

The Compustat build has annual coverage for 714,656 panel rows and monthly
security coverage for 819,116 rows. In model-eligible rows, the mean deep
feature count is 42.7 of 53. The primary artifacts are
`data/processed/asset_pricing/monthly_feature_panel_compustat.parquet`,
`data/processed/asset_pricing/compustat_enrichment_audit.json` and
`data/processed/asset_pricing/compustat_feature_dictionary.csv`.

The full no-tuning benchmark produces 2,299,145 labelled predictions and zero
training-cutoff violations. Prediction improves relative to the compact DRE
run:

| model | compact IC/R2 | Compustat IC/R2 |
|---|---:|---:|
| DRE rank IC | 0.114 | **0.117** |
| ridge rank IC | 0.113 | **0.116** |
| DRE return R2 | 0.14% | **0.25%** |
| ridge return R2 | 0.24% | **0.29%** |

At 25 bps in the standard value-weighted long-short family, the point estimates
also improve for return-target models:

| model | annual return | volatility | Sharpe |
|---|---:|---:|---:|
| ridge return | 12.9% | 13.5% | **0.958** |
| DRE return | 12.6% | 13.7% | 0.922 |
| ridge rank | 22.8% | 32.5% | 0.703 |
| momentum | 17.2% | 28.2% | 0.611 |
| DRE rank | 18.7% | 33.5% | 0.558 |

However, the primary paired Sharpe differences versus momentum remain
statistically unreliable after family-wise correction. The ridge-return
difference is +0.347 with stationary-bootstrap `p=0.299` and Holm `p=1.0`;
the DRE-return difference is +0.311 with `p=0.341` and Holm `p=1.0`.

In the more implementable value-weighted long-only comparison, momentum still
has the highest Sharpe: 0.955 versus 0.933 for ridge-return, 0.789 for
DRE-return, 0.666 for ridge-rank and 0.564 for DRE-rank. The data-depth
extension therefore strengthens the predictability side of the dissertation
but still supports the predictability-implementability gap.

## Residualized European Target Screen

To test whether Europe-specific market structure was hiding stock-selection
signal, the ML runner now supports residualized targets. Each month, the raw
next-month return and return-rank labels are neutralized against a deliberately
small set of broad controls: country, economic sector, size, book-to-market,
12-2 momentum, one-month reversal and 12-month volatility. Models can be
trained on `residual_rank` while portfolios are still evaluated on raw
next-month returns.

The screen uses the Compustat-enriched feature set with momentum, ridge and
histogram GBM, annual expanding walk-forward splits, a 250,000-row stratified
training cap and no hyperparameter tuning. It produces 2,758,974 predictions,
459,829 labelled observations per model and zero training-cutoff violations.

Predictively, residualization does what it is meant to do: GBM trained on the
residual target has residual IC 0.064 versus 0.047 for GBM trained on raw rank;
ridge residual IC rises to 0.057 versus 0.041. But raw-return IC falls:
GBM residual-rank IC is 0.083 versus 0.119 for raw-rank GBM, and ridge
residual-rank IC is 0.080 versus 0.116.

The portfolio result is therefore a negative but informative experiment. At
25 bps, standard value-weighted long-only Sharpe is 0.848 for raw-rank GBM,
0.648 for raw-rank ridge, 0.408 for residual-rank GBM and 0.413 for
residual-rank ridge; momentum remains 0.955. In the primary value-weighted
long-short family, residual-rank GBM has Sharpe 0.469 versus 0.767 for raw-rank
GBM and 0.611 for momentum. Its paired bootstrap Sharpe difference versus
momentum is -0.142 with Holm-adjusted `p=1.0`.

This strengthens the dissertation story: neutralized European alpha is
predictable, but the neutralized signal sacrifices economically useful broad
return structure. The current evidence does not support replacing the raw-rank
target with a residualized target for the implementable strategy.

## Direct-Utility Neural SDF Baseline

The first deep-learning baseline maps stock characteristics and month-t states
into continuous long-short SDF weights. Weights are demeaned
cross-sectionally and normalized to gross exposure two, so each month is a
self-financing long-short portfolio with long weight +1 and short weight -1.
Training maximizes monthly SDF portfolio utility and adds an Euler-equation
penalty on characteristic-managed test assets.

This is not the requested Chen-Pelger-Zhu adversarial test-asset network. It is
retained as a diagnostic because it answers a different question: whether a
cost-light neural portfolio rule can convert the enriched European predictors
into a low-volatility SDF portfolio.

The full annual expanding-window run produces 137 OOS months, 12 annual
refits, 459,829 out-of-sample security weights and zero training-cutoff or
duplicate-weight violations. It uses a deliberately compact 16/8 hidden-layer
network so the experiment is reproducible without GPU infrastructure.

Net results for the standard bottom-5%-excluded universe are:

| cost | annual return | volatility | Sharpe | turnover |
|---:|---:|---:|---:|---:|
| 0 bps | 16.1% | 5.5% | 2.932 | 0.675 |
| 10 bps | 15.3% | 5.5% | 2.784 | 0.675 |
| 25 bps | 14.1% | 5.5% | 2.561 | 0.675 |
| 50 bps | 12.1% | 5.5% | 2.190 | 0.675 |

At 25 bps, the neural SDF Sharpe exceeds the existing value-weighted
long-short Compustat ML portfolios on the same 137 months. Against
ridge-return, the paired stationary-bootstrap Sharpe difference is +1.603
with 95% interval [0.878, 2.344] at six-month expected blocks. Against
DRE-return it is +1.640 with interval [1.005, 2.277]. The annual return is
only slightly above ridge-return and DRE-return, but volatility is much lower
because the neural SDF uses diffuse continuous weights rather than decile
membership.

This is a useful high-Sharpe diagnostic, but it should not be used as the main
Chen-Pelger-Zhu result. It is a direct utility SDF with a turnover-only cost
model: it does not impose position caps, borrow limits, AUM-dependent impact or
capacity constraints.

## Adversarial LSTM/GAN SDF Adaptation

The requested Chen-Pelger-Zhu-style route is implemented separately as an
adversarial LSTM/GAN SDF. Chen, Pelger and Zhu (2024, Management Science
70(2), 714-750) estimate no-arbitrage asset-pricing models using a stochastic
discount factor network, an LSTM state network and an adversarial network that
constructs informative test assets. The European adaptation here keeps that
minimax logic but uses the available European data: 53 ranked stock
characteristics from the Compustat-enriched panel plus causal market-state
variables.

The implementation uses separate SDF and adversary networks. Each network has
its own 12-month LSTM state encoder and a one-hidden-layer characteristic
network. The SDF network produces cross-sectional SDF portfolio weights
`omega_it`; the adversary produces self-financing characteristic-managed test
asset weights `g_it,k`. For each month, the SDF factor is
`F_{t+1} = sum_i omega_it R_{i,t+1}` and the pricing-kernel proxy is
`M_{t+1} = 1 - F_{t+1}`. The minimax loss is the squared sample mean of
adversarial pricing moments,
`E[M_{t+1} sum_i g_it,k R_{i,t+1}]`, with the adversary maximizing the moment
violation and the SDF network minimizing it.

The main annual expanding-window run uses 12-month state sequences, LSTM hidden
size eight, 32/16 feed-forward layers in both the SDF and adversary networks,
four adversarial test assets, 20 epochs, one adversary step and one SDF step per
epoch. Every training target used by a given annual refit is known no later than
the previous 31 December.

The corrected full run produces 137 OOS months, 12 annual refits and 459,829
out-of-sample security weights, with zero training-cutoff violations and zero
duplicate security-month weights.

| metric | value |
|---|---:|
| input rows | 759,662 |
| OOS months | 137 |
| annual refits | 12 |
| OOS security weights | 459,829 |
| annualized SDF return | -1.9% |
| annualized SDF volatility | 13.6% |
| SDF Sharpe | -0.136 |
| average adversarial pricing-moment L2 | 0.0137 |
| max absolute adversarial pricing moment | 0.0691 |
| average test stocks per month | 3,356 |
| average gross weight | 1.000 |
| average net weight | 0.145 |

The interpretation is different from the direct-utility neural SDF baseline.
The adversarial model is the methodologically closer CPZ adaptation, but its
first-pass European economic payoff is weak. The result therefore supports the
dissertation's gap narrative: the no-arbitrage deep-learning architecture can
be implemented on European equities, but the first-pass adversarial SDF does not
yet deliver an economically strong out-of-sample pricing-kernel portfolio.
A smaller two-test-asset reproducibility check in
`results/asset_pricing_ml/adversarial_sdf_compustat_full/` also remains weak
(Sharpe 0.067), so the conclusion is not driven by a single larger
configuration.

## AIPM Linear Transformer SDF

The Bryan Kelly, Kuznetsov, Malamud and Xu (2026) artificial-intelligence asset
pricing route is implemented through their interpretable linear portfolio
transformer. The model replaces the no-attention BSV characteristic SDF
`w_it = X_it lambda` with a cross-asset attention SDF,
`w_t = N_t^{-1}(X_t W X_t')X_t lambda`. This keeps the core AIPM idea of
cross-asset information sharing while retaining a closed-form
ridge-penalized MSRR estimator.

The European implementation uses all 53 Compustat-enriched ranked
characteristics, producing 148,877 linear-attention parameters versus 53 BSV
parameters. To avoid interpreting unconstrained MSRR leverage as economic
performance, the saved results report both raw MSRR returns and gross-normalized
OOS SDF returns. The headline comparison uses the gross-normalized weights with
gross exposure one.

The full run produces 137 OOS months, 12 annual refits for each model,
919,658 OOS security weights across BSV and attention, and zero training-cutoff,
validation-cutoff or duplicate-weight violations.

| model | annual return | volatility | Sharpe | turnover |
|---|---:|---:|---:|---:|
| BSV no attention | 7.8% | 2.6% | 3.043 | 0.270 |
| Linear attention | 7.9% | 2.4% | 3.258 | 0.290 |

The attention model is descriptively stronger than BSV, but only by 0.13% per
year. The paired attention-minus-BSV annualized mean difference has HAC
`t=0.25`, `p=0.802`, and the two return series have correlation 0.87. The
important result is therefore not that the transformer attention mechanism
dominates in Europe. Rather, the broader characteristic-managed SDF is strong
after gross normalization, while the incremental cross-asset attention channel
is not statistically distinguishable from the no-attention BSV benchmark in
this first European test.

## Full Nonlinear AIPM Transformer

The full Kelly, Kuznetsov, Malamud and Xu AIPM route is now implemented beyond
the closed-form linear surrogate. The implemented model family contains the
paper's core hierarchy: BSV own-asset linear SDF, saturated linear attention,
a DKKM-style random-feature SDF, an own-asset residual MLP benchmark and the
nonlinear portfolio transformer. The transformer uses multi-head softmax
attention, residual attention and feed-forward sublayers, stacked transformer
blocks and a final linear SDF layer. Neural models are trained directly on the
MSRR objective `(1 - w(X_t)'R_{t+1})^2`.

The reported European run uses the 53 Compustat-enriched ranked predictors,
top-500 market-cap stocks each month, two transformer blocks, one attention
head, feed-forward width 64, eight training epochs, three random seeds averaged
for the neural models, 72-month rolling training windows and annual refits.
The implementation also supports the paper-style monthly refit mode, all-stock
dense attention and more seeds, but those settings are much heavier.

The run produces 137 OOS months per model, 60 annual model refits, 342,500
stock-month weights, 82,200 saved attention examples and zero training-cutoff,
validation-cutoff or duplicate-weight violations.

| model | annual return | volatility | Sharpe | normalized HJD error |
|---|---:|---:|---:|---:|
| BSV | 3.9% | 7.2% | 0.534 | 0.238 |
| Linear attention | 2.2% | 4.4% | 0.486 | 0.245 |
| DKKM random features | 5.7% | 10.2% | 0.557 | 0.233 |
| Own-asset MLP | 8.2% | 13.2% | **0.622** | 0.224 |
| Nonlinear transformer | 7.7% | 13.8% | 0.557 | **0.224** |

The transformer beats BSV by 4.7 percentage points per year in mean return,
but the rescaled SDF alpha is not statistically reliable (`t=0.81`, `p=0.416`).
Against the no-attention own-asset MLP, the transformer does not dominate:
annual mean difference is -0.5 percentage points, alpha `t=-1.63` and
`p=0.103`. The two neural SDFs are almost identical in return space
(correlation 0.993), so the European evidence does not reproduce the paper's
large nonlinear-transformer-over-MLP gain in this first tractable top-500 run.

The useful dissertation interpretation is therefore sharp: the full AIPM
architecture is feasible on European equities and cross-asset attention can be
estimated directly, but the incremental learned-attention payoff is weak once
compared with an equally deep own-asset neural SDF.

### Full AIPM Top-1000 Robustness

A larger-universe robustness check reruns the core AIPM hierarchy on the top
1,000 market-cap stocks per month, keeping annual refits, 72-month rolling
training windows, two transformer blocks, one attention head, feed-forward width
64, four epochs and one neural seed. This is a tractable robustness check, not a
replacement for the three-seed top-500 headline run.

The top-1000 run produces 137 OOS months, 36 annual model refits, 411,000
stock-month weights, 27,400 saved attention examples and zero training-cutoff,
validation-cutoff or duplicate-weight violations.

| model | annual return | volatility | Sharpe | normalized HJD error |
|---|---:|---:|---:|---:|
| BSV | 5.7% | 6.7% | **0.854** | 0.272 |
| Own-asset MLP | 9.2% | 14.2% | 0.652 | **0.259** |
| Nonlinear transformer | 8.3% | 13.9% | 0.598 | 0.261 |

The larger universe strengthens the no-attention baselines rather than the
attention result. The transformer beats BSV in mean return by 2.6 percentage
points per year, but this is not statistically reliable (`t=-0.09`, `p=0.930`
in the rescaled alpha regression). Against the own-asset MLP, the transformer
again does not dominate: annual mean difference is -0.9 percentage points and
HAC alpha `t=-0.56`, `p=0.579`.

Applying the same Refinitiv half-spread plus square-root impact cost model at
EUR 100m AUM gives:

| model | net annual return, EUR 100m | net vol | net Sharpe | annual cost |
|---|---:|---:|---:|---:|
| BSV | 5.0% | 6.7% | **0.747** | 0.72% |
| Own-asset MLP | 8.8% | 14.2% | 0.618 | 0.48% |
| Nonlinear transformer | 7.9% | 13.9% | 0.563 | 0.48% |

The robustness check therefore supports the main interpretation: increasing
cross-sectional breadth does not create a European AIPM attention premium over
an equally deep own-asset neural SDF.

### Full AIPM Monthly-Refit and Implementability-Aware Training

Two additional 2020-2026 checks address whether the annual-refit design is too
slow and whether implementation costs should be embedded directly in the neural
objective. Both checks keep causal rolling windows, a separate validation block
and fixed hyperparameters chosen before seeing the test results. They should be
read as robustness evidence rather than a new tuned headline.

The monthly-refit check uses top-500 stocks, 72-month rolling training windows,
12-month validation windows, two transformer blocks, one attention head,
feed-forward width 64, three epochs and one seed. It produces 77 OOS months,
231 monthly model refits, 115,500 stock-month weights and zero training-cutoff,
validation-cutoff or duplicate-weight violations.

| model | gross annual return | gross Sharpe | EUR 100m net return | EUR 100m net Sharpe |
|---|---:|---:|---:|---:|
| BSV | 2.0% | 0.242 | 1.5% | 0.185 |
| Own-asset MLP | 7.9% | 0.495 | 7.6% | 0.480 |
| Nonlinear transformer | 8.5% | **0.518** | 8.3% | **0.506** |

Monthly refitting therefore slightly helps the transformer in the later
subperiod, but not enough to create reliable evidence. Transformer-minus-MLP
gross mean difference is +0.7 percentage points per year with HAC alpha
`t=0.40`, `p=0.690`; after EUR 100m spread and impact costs, the alpha is
`t=0.44`, `p=0.661`.

The implementability-aware training experiment adds constraints directly to the
neural loss: gross-normalized portfolio weights, a 2% single-name cap, a net
exposure penalty, a turnover penalty, and differentiable spread/impact costs at
EUR 100m AUM. To avoid making the design artificially harsh, two fixed penalty
levels are tested: mild and standard.

| training objective | model | gross Sharpe | EUR 100m net return | EUR 100m net Sharpe |
|---|---|---:|---:|---:|
| unconstrained monthly | Own-asset MLP | 0.495 | 7.6% | 0.480 |
| unconstrained monthly | Nonlinear transformer | **0.518** | **8.3%** | **0.506** |
| mild implementability-aware | Own-asset MLP | 0.439 | 6.4% | 0.412 |
| mild implementability-aware | Nonlinear transformer | 0.267 | 3.5% | 0.225 |
| standard implementability-aware | Own-asset MLP | 0.061 | 0.2% | 0.024 |
| standard implementability-aware | Nonlinear transformer | 0.063 | 0.2% | 0.016 |

This is a negative result. Directly penalizing implementation frictions inside
the short-window neural MSRR objective does not improve the European AIPM
portfolio; it degrades the learned SDFs and can raise turnover. The safer
interpretation is not that implementation-aware learning is impossible, but
that this dissertation should report costs out of sample and avoid selecting
penalty strengths on the test set.

### Full AIPM Post-Estimation Tests

The post-estimation extension now implements the three recommended follow-up
tests around the full AIPM run: scaling and ablation collation, attention
mechanism diagnostics, and direct implementability costs for the learned SDF
weights. It also adds a Kelly-style characteristic-space principal-portfolio
benchmark inspired by Kelly, Malamud and Pedersen's Principal Portfolios
framework. The output directory is
`results/asset_pricing_ml/aipm_post_analysis/`.

Scaling and ablation collation does not overturn the headline result. The
single-seed, four-epoch top-500 transformer had Sharpe 0.606, almost identical
to the own-asset MLP at 0.608. The heavier three-seed, eight-epoch run has
transformer Sharpe 0.557 versus own-asset MLP Sharpe 0.622. More training and
seed averaging therefore stabilizes the result but does not create a robust
European transformer advantage.

The attention diagnostics show what the transformer actually attends to. Top
attention links are much more likely than random same-month links to share TRBC
economic sector: 17.7% observed weighted share versus 11.6% null. Business
sector and industry links are also above null. The strongest mechanism is size
matching rather than country matching: weighted absolute log-size-rank distance
is 0.094 for observed links versus 0.885 under the null, and weighted market-cap
percentile distance is 0.047 versus 0.445. Same-country lift is small. The
learned attention layer is therefore mostly forming economically similar
large-cap peer sets, not a distinct country-allocation channel.

The implementability test applies the same observed Refinitiv half-spread plus
AUM-sensitive square-root impact model used in the investability frontier. At
EUR 100m AUM, spread coverage is effectively complete in weight terms and costs
do not destroy the neural SDFs, but they also do not rescue the transformer.

| model | net annual return, EUR 100m | net vol | net Sharpe | annual cost | turnover |
|---|---:|---:|---:|---:|---:|
| BSV | 3.5% | 7.2% | 0.480 | 0.38% | 0.252 |
| Linear attention | 1.6% | 4.4% | 0.351 | 0.60% | 0.431 |
| DKKM random features | 4.8% | 10.2% | 0.476 | 0.83% | 0.338 |
| Own-asset MLP | 8.0% | 13.2% | **0.606** | 0.21% | 0.167 |
| Nonlinear transformer | 7.5% | 13.8% | 0.543 | 0.21% | 0.132 |

The transformer has lower turnover than the own-asset MLP, so its annualized
cost is slightly lower, but its net alpha remains below the MLP. At EUR 100m,
transformer-minus-MLP net mean difference is -0.5 percentage points per year
with HAC alpha `t=-1.62`, `p=0.105`.

The principal-portfolio benchmark is a characteristic-space adaptation rather
than the exact fixed-universe prediction-matrix implementation in Kelly,
Malamud and Pedersen. It estimates dominant return-weighted characteristic
directions in each causal training window and turns the first 1, 3 or 5
directions into gross-normalized SDF weights. The European result is weak:
gross Sharpe is -0.055, 0.106 and 0.073 for 1, 3 and 5 components. At EUR 100m
AUM, the 3-component version has only 0.24% annualized net return and Sharpe
0.038; the 1- and 5-component versions are net negative. This is useful as a
negative niche test: a transparent principal-predictability direction does not
solve the European implementability problem.

### Full AIPM Adaptation Bundle

The full AIPM adaptation is now bundled as a single evidence package in
`results/asset_pricing_ml/aipm_full_adaptation_bundle/`. The bundle maps the
paper's components to the local implementation, collates all completed AIPM
runs, and provides one short markdown brief. The component map covers BSV,
linear portfolio transformer, MSRR estimation, DKKM-style random features,
own-asset MLP ablation, nonlinear portfolio transformer, OOS Sharpe and HJD
pricing errors, pairwise alpha comparisons, depth scaling, attention
diagnostics, monthly refits and the dissertation-specific implementability
extension.

The added depth-scaling grid reruns the top-500 European AIPM with one, two and
four transformer blocks, one seed, four epochs and the same 72-month rolling
training window. This directly tests the paper's scaling channel in a tractable
European setting:

| transformer blocks | transformer Sharpe | own-asset MLP Sharpe | transformer-minus-MLP return | transformer-minus-MLP Sharpe |
|---:|---:|---:|---:|---:|
| 1 | 0.592 | 0.654 | -0.5% | -0.062 |
| 2 | 0.606 | 0.608 | +0.2% | -0.003 |
| 4 | 0.575 | 0.615 | -0.1% | -0.040 |

All three depth runs produce zero training-cutoff, validation-cutoff and
duplicate-weight violations. The European depth grid does not reproduce the
paper's monotone transformer-depth improvement. Depth two is marginally best
for the transformer, but the no-attention MLP remains essentially tied or
better. This is the most complete current interpretation of the Kelly et al.
AIPM adaptation: the architecture is fully feasible, the attention links are
economically interpretable, but cross-asset attention does not deliver a robust
European premium over an equally deep own-asset neural SDF.

## Conditional Autoencoder Asset Pricing

The Gu, Kelly and Xiu autoencoder route is implemented as a conditional
latent-factor asset-pricing model. Stock characteristics enter a nonlinear beta
network, monthly latent factor returns are recovered from the zero-intercept
return reconstruction
`r_{i,t+1} = beta(X_{i,t})' f_{t+1} + u_{i,t+1}`, and predictive returns use
the training-sample factor premia. The zero-intercept structure is the
no-arbitrage restriction, so this is not just another return-forecasting
regression.

The European adaptation uses all 53 Compustat-enriched ranked predictors,
five latent factors, one 16-unit hidden layer, annual expanding-window refits,
a trailing 24-month validation window and the same bottom-5% size exclusion as
the other ML asset-pricing tests. Every training target used by an annual refit
is known by the previous 31 December.

The full run produces 137 OOS months, 12 annual refits, 459,829
out-of-sample stock-month predictions, 137 monthly factor realizations and
zero training-cutoff or duplicate-prediction violations.

| metric | value |
|---|---:|
| total reconstruction R2 | 2.59% |
| predictive R2 | 0.12% |
| mean monthly total R2 | 2.47% |
| mean monthly predictive R2 | 0.19% |
| factor-SDF Sharpe | 3.015 |
| average pricing-moment L2 | 0.0512 |
| max absolute pricing moment | 0.0956 |
| average test stocks per month | 3,356 |

The result is methodologically useful. It shows that a conditional
no-arbitrage latent-factor model can be trained causally on European equities
and produces positive OOS reconstruction and predictive R2. But the expected
return component is economically modest: predictive R2 is only 0.12%. The high
factor-SDF Sharpe comes from the latent factor representation rather than a
directly costed stock portfolio.

### Costed Autoencoder Stock-SDF Portfolio

The autoencoder has now been converted into stock-level SDF weights. For each
test month, the learned beta network and factor-SDF coefficients imply stock
weights through the latent-factor normal equations; those weights are then
gross-normalized and passed through the same Refinitiv half-spread plus
square-root impact model used for AIPM.

The rerun writes 459,829 stock-level autoencoder SDF weights, preserves the
same 137 OOS months and 12 annual refits, and has zero training-cutoff or
duplicate-weight violations. The gross-normalized stock-SDF has annual return
7.6%, annual volatility 2.5% and Sharpe 3.016 before costs. It is very diffuse:
average weight HHI is 0.00055 and average net exposure is effectively zero.

Implementation costs are economically material:

| AUM | net annual return | net vol | net Sharpe | annual cost |
|---|---:|---:|---:|---:|
| EUR 10m | 4.9% | 2.6% | **1.900** | 2.78% |
| EUR 100m | 2.9% | 2.6% | 1.139 | 4.68% |
| EUR 500m | -0.5% | 2.7% | -0.183 | 8.12% |

Only 16.7% of the autoencoder's gross weight has observed spread coverage in
the current liquidity pull, so 83.3% of gross weight is priced from an assumed
fallback half-spread rather than a measured quote. The average weight-weighted
half-spread is 22.4 bps and the average monthly turnover is 0.694. The current
capacity numbers are therefore not fully identified for the autoencoder: they
are assumption-sensitive costed scenarios, not measured execution estimates.
This is autoencoder-specific rather than systemic, because the AIPM top-500
models have 99.5-99.9% spread coverage in weight terms.

A corrected cost-assumption sensitivity rebuilds the execution panel separately
for each fallback spread. Under the earlier implementation, all constant-spread
scenarios were silently pinned to the 25 bps fallback. The corrected curve now
shows the autoencoder's dependence on uncovered-name assumptions:

| model | AUM | net Sharpe at 25 bps | break-even assumed half-spread |
|---|---:|---:|---:|
| autoencoder K5 baseline | EUR 10m | 1.900 | 93.7 bps |
| autoencoder K5 baseline | EUR 100m | 1.139 | 66.8 bps |
| autoencoder K5 baseline | EUR 500m | -0.183 | 18.1 bps |
| autoencoder K10 validation-selected | EUR 10m | 3.227 | 112.5 bps |
| autoencoder K10 validation-selected | EUR 100m | 2.171 | 84.6 bps |
| autoencoder K10 validation-selected | EUR 500m | 0.322 | 34.1 bps |

The K10 validation-selected specification has stronger gross economics, but it
also has slightly lower spread coverage (15.7%) and a larger ADV-floored weight
(21.9%). The defensible interpretation is that the autoencoder is a promising
Kelly-style no-arbitrage model, but the current full-universe stock-SDF result
derives most of its costed performance from names whose spreads and impact are
imputed. A top-size rerun is required before citing it as an implementable
capacity result.

## Fundamental Mispricing

The Hanauer-Kononova-Rapp European fundamental-analysis route is now
implemented as a separate peer-implied fair-value experiment. The model uses
23 Compustat accounting items, made available only after the report date or a
six-month fiscal lag, rank-transforms them cross-sectionally and predicts each
stock's market value deflated by the month-t sample market value. The trading
signal is the log ratio of peer-implied market share to observed market share,
so higher values mean the model views the stock as undervalued.

The main run uses a rolling 48-month training window, annual walk-forward
refits, linear ridge, random forest, histogram GBM and RF/GBM ensemble fair
value models. Momentum is added only as a benchmark on the exact same
fundamental-scoreable universe. The run covers 553,388 fair-value eligible
stock-months, writes 1,691,970 model-security-month predictions and has zero
train-cutoff or duplicate-prediction violations.

Predictive ICs are weak relative to momentum:

| model | mean monthly IC | IC IR | positive IC months |
|---|---:|---:|---:|
| momentum | 0.067 | 2.98 | 86.1% |
| linear fair value | 0.012 | 1.16 | 67.9% |
| random forest fair value | -0.033 | -1.80 | 28.5% |
| ensemble fair value | -0.069 | -2.75 | 19.0% |
| histogram GBM fair value | -0.071 | -2.74 | 20.4% |

At 25 bps in the value-weighted standard universe:

| model | long-short Sharpe | long-only top-decile Sharpe | long-short turnover |
|---|---:|---:|---:|
| momentum | 0.427 | 0.944 | 0.796 |
| linear fair value | 0.277 | 0.529 | 0.499 |
| random forest fair value | 0.234 | 0.639 | 0.559 |
| histogram GBM fair value | 0.134 | 0.569 | 0.419 |
| ensemble fair value | -0.005 | 0.415 | 0.438 |

No fair-value model significantly beats momentum in the primary
value-weighted long-short comparison; all Holm-adjusted bootstrap p-values are
1.0. The implementation is therefore useful but not supportive of a new
tradable European alpha. It strengthens the dissertation story by showing that
a niche European fundamental ML signal can be built cleanly, yet still fails
to displace a simpler momentum benchmark once the same walk-forward and
portfolio rules are imposed.

## Deep Sequence Modelling

The deep-sequence extension adapts the stock-characteristic sequence idea from
the deep sequence modelling paper. Each stock-month is represented by a
trailing sequence of ranked characteristics ending at the signal month. The
implementation compares four models on the same European walk-forward and
portfolio rules: `last_mlp` as a current-month static neural baseline, `lstm`,
`gru` and `attention_lstm`.

The full runs use the Compustat-enriched 53-feature panel over 2015-2026, with
annual expanding refits, a trailing validation window, 150,000 deterministically
sampled training rows and 60,000 validation rows per refit. Two pre-specified
sequence lengths are tested: 12 and 24 months. Each run writes 1,795,252
model-security-month predictions across 48 annual fits, with zero
training-cutoff violations and zero duplicate model-security-month predictions.

Predictive ICs show a different ranking from implementable portfolio returns.
The plain LSTM has the best rank IC at both sequence lengths, while the static
current-month MLP remains negative:

| sequence | model | mean monthly IC | IC IR | positive IC months | rank R2 |
|---|---|---:|---:|---:|---:|
| 12m | LSTM | **0.005** | 0.54 | 59.1% | -0.001 |
| 12m | GRU | 0.001 | 0.07 | 55.5% | -0.002 |
| 12m | attention LSTM | -0.005 | -0.47 | 48.9% | -0.001 |
| 12m | last-month MLP | -0.010 | -1.02 | 37.2% | -0.005 |
| 24m | LSTM | **0.005** | 0.52 | 58.4% | -0.001 |
| 24m | GRU | 0.002 | 0.24 | 59.1% | -0.002 |
| 24m | attention LSTM | -0.004 | -0.39 | 51.8% | -0.001 |
| 24m | last-month MLP | -0.010 | -1.02 | 37.2% | -0.005 |

The LSTM-minus-static IC difference is positive and HAC-significant in both
pre-specified runs: 12-month sequences have `p=0.000063`,
Holm-adjusted `p=0.000381`; 24-month sequences have `p=0.000071`,
Holm-adjusted `p=0.000426`.

The costed value-weighted portfolio read-through is more favourable to
attention and GRU than to the plain LSTM. In the primary 25 bps,
value-weighted, standard-ex-bottom-5% long-short book:

| sequence | model | annual net return | Sharpe | turnover | gross annual return |
|---|---|---:|---:|---:|---:|
| 12m | attention LSTM | 2.9% | **0.339** | 0.389 | 4.1% |
| 12m | GRU | 2.6% | 0.291 | 0.755 | 4.9% |
| 12m | LSTM | 0.6% | 0.064 | 0.701 | 2.7% |
| 12m | last-month MLP | -5.1% | -0.580 | 1.086 | -1.9% |
| 24m | attention LSTM | 4.9% | **0.538** | 0.298 | 5.8% |
| 24m | GRU | 4.3% | 0.480 | 0.776 | 6.6% |
| 24m | LSTM | -0.1% | -0.006 | 0.704 | 2.1% |
| 24m | last-month MLP | -5.1% | -0.580 | 1.086 | -1.9% |

The paired stationary-bootstrap comparison against the static MLP is
significant for attention LSTM in both sequence lengths. In the primary family,
12-month attention LSTM has Sharpe difference `+0.919` with 95% interval
`[0.269, 1.623]` and Holm-adjusted `p=0.006`; 24-month attention LSTM has
Sharpe difference `+1.118` with interval `[0.482, 1.792]` and adjusted
`p<0.001`. The 24-month GRU also beats the static MLP after Holm correction
(`+1.060`, interval `[0.238, 1.851]`, adjusted `p=0.014`).

The interpretation is therefore nuanced. Ordered sequence learning adds
statistically detectable predictive information relative to a static neural
baseline, but the best rank-IC model is not the best implementable portfolio.
Attention appears to reduce turnover and improve costed value-weighted
performance even though its average monthly IC is slightly negative. That makes
deep sequence modelling useful for the dissertation, but it still supports the
broader message that prediction quality and implementable portfolio quality are
not the same object.

### Common Benchmark Test

The decisive follow-up compares the sequence models against the real
Compustat-enriched baselines on the exact same 448,813 stock-month common
sample. The comparison combines momentum, ridge, DRE, histogram GBM, MLP and
the 12-/24-month sequence models, then reconstructs all portfolios under the
same 10% long-short rule and 25 bps cost assumption. The output contains
5,385,756 common-sample model-security-month predictions and zero duplicate
model-security-month observations.

On predictive IC, the standard Compustat ML baselines dominate the sequence
models:

| model | mean monthly IC | IC IR | positive IC months |
|---|---:|---:|---:|
| histogram GBM | **0.118** | 5.55 | 93.4% |
| MLP | 0.117 | 5.53 | 93.4% |
| DRE | 0.115 | 5.23 | 92.7% |
| ridge | 0.115 | 5.17 | 92.7% |
| momentum | 0.066 | 2.72 | 82.5% |
| best sequence IC: 12m LSTM | 0.005 | 0.54 | 59.1% |
| best sequence portfolio: 24m attention LSTM | -0.004 | -0.39 | 51.8% |

In the primary 25 bps value-weighted long-short comparison:

| model | annual net return | Sharpe | turnover |
|---|---:|---:|---:|
| ridge | 26.8% | **0.780** | 0.950 |
| MLP | 25.9% | 0.772 | 0.961 |
| momentum | 16.3% | 0.577 | 0.802 |
| DRE | 18.5% | 0.555 | 0.975 |
| 24m attention LSTM | 4.9% | 0.538 | 0.298 |
| histogram GBM | 13.7% | 0.529 | 1.049 |
| 24m GRU | 4.3% | 0.480 | 0.776 |

The 24-month attention LSTM is therefore economically interesting because it
nearly matches DRE, histogram GBM and momentum Sharpe with much lower turnover,
but it does not significantly beat any real benchmark in the primary family.
Its paired Sharpe differences are `-0.039` versus momentum, `-0.017` versus
DRE, `+0.009` versus histogram GBM, `-0.242` versus ridge and `-0.233` versus
MLP; all Holm-adjusted p-values equal or exceed 0.780. The correct conclusion
is not that deep sequence modelling dominates the existing ML stack. It is that
attention-LSTM improves implementation characteristics relative to static
neural sequence baselines, but classical Compustat ML models still carry far
stronger cross-sectional predictive content.

### Blend and Implementability Checks

A fixed complementarity test blends the common-sample Compustat baselines with
sequence-model scores after within-month rank normalization. The grid uses
90/10, 75/25 and 50/50 baseline/sequence weights across momentum, ridge, DRE,
histogram GBM, MLP and six sequence models. This produces 90 blend
specifications, 45,330,113 evaluated prediction rows and zero duplicate
model-security-month predictions. The test is fixed-weight rather than
validation-selected, so it screens complementarity without fitting a
test-period blend.

The best primary value-weighted long-short point estimates are:

| model | annual net return | Sharpe | turnover | mean IC |
|---|---:|---:|---:|---:|
| 90% MLP + 10% 12m attention-LSTM | 23.1% | **1.007** | 0.979 | 0.117 |
| 90% GBM + 10% 24m attention-LSTM | 21.6% | 0.978 | 1.067 | 0.117 |
| 90% MLP + 10% 24m attention-LSTM | 29.0% | 0.908 | 0.968 | 0.117 |
| ridge | 26.8% | 0.780 | 0.950 | 0.115 |
| MLP | 25.9% | 0.772 | 0.961 | 0.117 |

The strongest blend-vs-parent improvement is the 90% GBM + 10% 24-month
attention-LSTM blend, with Sharpe difference `+0.449` versus GBM at six-month
bootstrap blocks. The raw p-value is `0.005`, but Holm-adjusted `p=0.090`, so
the result is economically interesting but not multiplicity-robust. No
blend-vs-parent improvement survives Holm correction in the primary family.

The liquidity ladder then carries all parent models, all sequence models and
the top 12 blend candidates into the same Refinitiv spread plus square-root
impact cost model. The standard broad universe still has weak spread coverage
for this selected panel, but the `top_500` rung has 99.8% observed-spread
coverage and median half-spread of about 3.43 bps. At EUR 100m AUM in the
value-weighted long-short book:

| model | broad Sharpe | top-500 Sharpe | observed-spread top-500 Sharpe | large-low-spread Sharpe |
|---|---:|---:|---:|---:|
| momentum | 0.473 | 0.630 | 0.628 | 0.425 |
| 50% momentum + 50% 12m GRU | **0.722** | **0.732** | **0.729** | 0.429 |
| 90% GBM + 10% 24m attention-LSTM | -0.050 | 0.660 | 0.654 | **0.654** |
| 24m attention-LSTM | 0.538 | 0.425 | 0.419 | 0.627 |
| MLP | 0.197 | 0.304 | 0.314 | 0.293 |
| ridge | 0.131 | 0.198 | 0.177 | 0.206 |

The long-short ladder inference versus momentum finds no Holm-significant
Sharpe improvement. The interpretation is again implementability-focused:
attention and attention-blend signals become more attractive inside measured,
large, low-spread universes, but the evidence is not strong enough to declare
an attention premium over momentum.

Regime diagnostics reuse the ex-ante market states from the depth analysis.
Attention sequence portfolios behave differently from their ICs: in
high-volatility months, 12-month attention-LSTM has negative mean IC
(-0.012) but a value-weighted long-short Sharpe of 1.255; 24-month
attention-LSTM has negative mean IC (-0.011) but Sharpe 0.890. In down-trend
months, the same models again have negative ICs but positive Sharpe. This is
useful evidence that prediction metrics and portfolio objectives can diverge,
especially for low-turnover neural signals.

### Turnover-Aware Signal Smoothing

A final implementability test applies a pre-specified ex-ante score-inertia
rule to the selected parent, sequence and blend predictions. This is not neural
objective retraining; it is a frozen-signal turnover surrogate:
`smoothed_score_t = (1 - rho) current_rank_score_t + rho smoothed_score_{t-1}`.
The grid uses `rho` values of 0.25, 0.50 and 0.75. It produces 69 smoothing
variants, 41,290,796 evaluated prediction rows and zero duplicate
model-security-month predictions.

The strongest primary value-weighted long-short point estimates at 25 bps are:

| model | annual net return | Sharpe | turnover | mean IC |
|---|---:|---:|---:|---:|
| smoothed 90% GBM + 10% 24m attention-LSTM, rho=0.75 | 25.3% | **1.207** | 0.448 | 0.115 |
| smoothed ridge, rho=0.75 | 28.5% | 1.187 | 0.393 | 0.112 |
| smoothed 90% MLP + 10% 12m attention-LSTM, rho=0.75 | 24.3% | 1.110 | 0.352 | 0.113 |
| smoothed 90% MLP + 10% 24m attention-LSTM, rho=0.50 | 24.6% | 1.089 | 0.518 | 0.116 |
| unsmoothed 90% MLP + 10% 12m attention-LSTM | 23.1% | 1.007 | 0.979 | 0.117 |

The rule cuts turnover by roughly 58-64% for several strong models. It also
raises Sharpe versus the parent for smoothed DRE after Holm correction within
the parent comparison family; smoothed ridge is raw-significant but marginal
after Holm correction (`p=0.078`). However, when the top 15 smoothed candidates
are tested against momentum, none survives Holm correction: the best point
estimate is `+0.630` Sharpe for smoothed 90% GBM + 10% 24-month attention-LSTM
versus momentum, with raw `p=0.148` and Holm `p=1.0`.

This is the best current implementability improvement: simple, ex-ante signal
inertia materially lowers turnover and improves several net Sharpe point
estimates without adding fitted test-period parameters. It strengthens the
dissertation's practical contribution, but it still does not invalidate the
main claim that a robust momentum-beating European ML portfolio has not yet
been statistically identified.

### Rolling Validation-Selected Strategy

The next robustness step turns the blend/smoothing experiment into a rolling
selection rule. A 39-model candidate set is formed from 13 core signals
(`momentum`, `ridge`, `DRE`, `GBM`, `MLP`, attention-LSTM, GRU and selected
deep hybrids) plus their 50% and 75% smoothed variants. The script first builds
a costed investability ladder for this frozen candidate pool, then selects the
model/rung cell each month using only prior completed returns. The selection
criterion is 36-month validation certainty equivalent with at least 24
validation months, EUR 100m AUM, value weighting and security-level
spread/impact costs.

The full candidate ladder contains 17,503,707 prediction rows, 39 candidate
models, 106,860 monthly ladder rows and zero duplicate prediction keys. The
main selector can choose among `top_500`, `top_500_observed_spread` and
`large_low_spread`; two fixed-rung robustness runs restrict selection to only
`top_500_observed_spread` or only `large_low_spread`. A final deep/hybrid-only
run excludes the plain non-deep baselines from the selection pool but still
benchmarks the selected strategy against momentum.

The selected strategy results are:

| selector | portfolio | annual net return | Sharpe | turnover | spread coverage |
|---|---|---:|---:|---:|---:|
| deep/hybrid liquid selector | long-only | **15.5%** | **0.976** | 0.643 | 100.0% |
| large-low-spread selector | long-only | 13.9% | 0.937 | 0.692 | 100.0% |
| top-500 observed-spread selector | long-only | 13.4% | 0.855 | 0.503 | 100.0% |
| multi-rung selector | long-only | 12.2% | 0.790 | 0.683 | 100.0% |
| large-low-spread selector | long-short | 5.4% | 0.324 | 1.245 | 100.0% |
| top-500 observed-spread selector | long-short | 5.2% | 0.309 | 1.069 | 100.0% |
| deep/hybrid liquid selector | long-short | 3.1% | 0.203 | 1.242 | 100.0% |
| multi-rung selector | long-short | 3.1% | 0.191 | 1.128 | 100.0% |

The deep/hybrid-only long-only selector is the most promising version. It
selects smoothed 24-month GRU for 28 months, 90% GBM + 10% 24-month
attention-LSTM for 21 months and smoothed 90% MLP + 10% 24-month attention-LSTM
for 16 months. Its annual net return is 15.5%, slightly above fixed
top-500-observed momentum over the matched 113 selected months, but the paired
Sharpe difference versus momentum is not significant after Holm correction
(`delta Sharpe=-0.026`, Holm `p=1.0` at six-month blocks). The HAC annualized
return difference versus fixed ridge is positive and Holm-significant
(`+7.1%`, Holm `p=0.038`), so the selector adds value versus ridge but not
versus momentum.

The long-short selector is not attractive. Validation frequently chooses
attention-LSTM and deep-hybrid large-low-spread cells, but realized long-short
returns underperform momentum. This is a useful negative result because it
shows that allowing experimentation through a validation rule does not
automatically manufacture a momentum-beating strategy.

The practical implication is to shift the next deep-learning work toward
long-only, liquid-universe, deep/hybrid signal construction rather than
high-turnover long-short selection. Deep learning remains central, but its most
defensible current role is as a low-turnover implementability overlay inside a
long-only European equity process.

### Frozen Deep/Hybrid Robustness

The deep/hybrid liquid long-only selector is then frozen and stress-tested
without changing its selected model/rung/month choices. The robustness runner
re-simulates the same selected choices at EUR 10m, 100m and 500m AUM, compares
them with fixed top-500-observed momentum, fixed top-500-observed ridge,
smoothed ridge and the static 90% GBM + 10% 24-month attention-LSTM large-low
spread hybrid, and reconstructs portfolio holdings for concentration checks.

The full-period AUM stress is stable:

| strategy | AUM | annual net return | Sharpe | turnover | spread cost | impact cost |
|---|---:|---:|---:|---:|---:|---:|
| frozen deep/hybrid selector | EUR 10m | 15.7% | 0.987 | 0.643 | 0.15% | 0.08% |
| frozen deep/hybrid selector | EUR 100m | 15.5% | 0.976 | 0.643 | 0.15% | 0.25% |
| frozen deep/hybrid selector | EUR 500m | 15.2% | 0.957 | 0.643 | 0.15% | 0.55% |
| fixed momentum top-500 observed | EUR 100m | 15.0% | **1.002** | 0.683 | 0.33% | 0.35% |
| fixed 90% GBM + 10% 24m attention-LSTM | EUR 100m | 15.1% | 0.926 | 1.204 | 0.29% | 0.60% |

The selected strategy is fully covered by observed spreads, so fallback
half-spread assumptions of 10, 25 and 50 bps have no effect on its reported
net returns. This is a material improvement over the autoencoder capacity
problem: for the frozen deep/hybrid long-only rule, spread-cost identification
is not the weak link.

Subperiod performance is uneven but not fragile:

| subperiod | AUM | annual net return | Sharpe | drawdown |
|---|---:|---:|---:|---:|
| 2017-2019 | EUR 100m | 15.9% | 1.244 | -12.9% |
| 2020-2022 | EUR 100m | 7.5% | 0.364 | -25.1% |
| 2023-2026 | EUR 100m | 22.0% | 1.619 | -11.4% |

The paired full-period inference remains conservative. At six-month bootstrap
blocks, the frozen selector has almost the same annual return as fixed
top-500-observed momentum (`+0.5%` per year) and a slightly lower Sharpe
(`delta Sharpe=-0.026`, Holm `p=1.0`). It beats fixed ridge by about `+7.1%`
per year before correction, but that return difference is not Holm-significant
in the robustness family (`p=0.115` at EUR 100m). The correct interpretation is
therefore not a momentum-beating claim. It is a capacity-clean, deep/hybrid
long-only candidate that is competitive with momentum and less sensitive to
market impact than higher-turnover hybrid alternatives.

The new weakness is concentration. The selected long-only portfolio has an
average effective number of only 12.5 stocks. The average maximum single-name
weight is 17.5%, the average top-five name weight is 53.8%, the average maximum
country weight is 33.0% and the average maximum sector weight is 39.0%; in the
worst month, one sector reaches 95.4% of the portfolio. This makes the next
deep-learning improvement clear: add position, country and sector constraints
directly to the long-only objective or post-processing layer.

### Constrained Long-Only Construction

The concentration issue is then tested directly while keeping the deep/hybrid
signal path fixed. The constrained construction replaces value-weighted top
decile holdings with a fully invested long-only optimizer that maximizes the
cross-sectional signal rank subject to hard single-name, country and sector
caps, with optional turnover penalty. It is run on the same frozen deep/hybrid
monthly choices and on fixed momentum under identical constraints.

All constraint cells are feasible and all selected portfolio weight has
observed spread coverage. At EUR 100m, the mild capped rule preserves most of
the frozen deep/hybrid result while materially improving diversification:

| construction | annual net return | Sharpe | turnover | effective N | top-five weight | max country | max sector |
|---|---:|---:|---:|---:|---:|---:|---:|
| unconstrained frozen deep/hybrid | 15.5% | 0.976 | 0.643 | 12.5 | 53.8% | 33.0% | 39.0% |
| 5% name / 40% country / 40% sector + turnover | 14.8% | 0.958 | 0.781 | 20.0 | 25.0% | 29.5% | 32.4% |
| 3% name / 30% country / 30% sector + turnover | 13.0% | 0.850 | 0.762 | 33.6 | 15.0% | 25.3% | 26.8% |
| 3% name / 25% country / 25% sector + turnover | 12.0% | 0.792 | 0.763 | 33.7 | 15.0% | 23.6% | 24.0% |

The same constraint set also changes the benchmark comparison. Under the
5%/40%/40% turnover rule, frozen deep/hybrid earns 14.8% net versus 14.0% for
constrained momentum, with a higher Sharpe (`0.958` versus `0.720`). The paired
difference is still not statistically reliable after correction
(`delta return=+0.8%`, six-month block Sharpe-difference Holm `p=0.965`).
Under stricter 3% name caps, the deep/hybrid return edge disappears. The
economic interpretation is therefore sharper: concentration can be fixed with
existing data, but diversification converts some of the original signal into a
portfolio-construction cost.

The refreshed Refinitiv BID/ASK pull validates these cost results. The
code-based LSEG download covers 810 of the 812 requested implementable-frontier
RICs, with zero failed identifier jobs and 123,286 monthly BID/ASK rows from
2013-2026. Rerunning both the frozen robustness and constrained construction on
the refreshed liquidity directory produces the same EUR 100m full-period
metrics to numerical precision: maximum absolute differences are below
`6e-11` for the frozen summary and below `5e-10` for the constrained summary.
The refreshed data therefore strengthen the audit trail but do not change the
empirical conclusion.

### Broader Liquidity Universe Stress

The broader Refinitiv robustness run extends measured BID/ASK coverage beyond
the original top-500 implementable frontier. The automated LSEG pull requested
3,159 top-2000 candidate RICs, completed all 32 monthly batches with zero failed
identifier jobs and produced 451,887 BID/ASK rows for 3,138 RICs from
2013-01-31 to 2026-07-31. Spread coverage is broad but the liquidity tail is
expensive: median half-spread is 28.5 bps, the 95th percentile is 243.9 bps and
the 99th percentile is 389.6 bps.

The unconstrained frozen deep/hybrid selector is robust to this broader
coverage. At EUR 100m, increasing the maximum monthly universe from top 500 to
top 1,000 and top 2,000 leaves the net result close to unchanged:

| maximum universe | annual net return | Sharpe | turnover | spread cost | impact cost | average universe |
|---:|---:|---:|---:|---:|---:|---:|
| top 500 | 15.5% | 0.976 | 0.643 | 0.15% | 0.25% | 272 |
| top 1,000 | 15.4% | 0.994 | 0.576 | 0.18% | 0.20% | 543 |
| top 2,000 | 14.8% | 0.959 | 0.569 | 0.25% | 0.22% | 1,083 |

This does not create a statistically reliable momentum-beating result: at
six-month bootstrap blocks, the full-period EUR 100m Sharpe differences versus
fixed top-500 observed-spread momentum remain Holm-insignificant for top 500,
top 1,000 and top 2,000.

The constrained portfolios tell a different story. When the same frozen
deep/hybrid signal is forced into more diversified long-only portfolios, broader
universes expose the cost of using the liquidity tail:

| constraint | maximum universe | annual net return | Sharpe | turnover | spread cost | impact cost | effective N |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% name / 40% country / 40% sector + turnover | top 500 | 14.8% | 0.958 | 0.781 | 0.23% | 0.58% | 20.0 |
| 5% name / 40% country / 40% sector + turnover | top 1,000 | 10.8% | 0.675 | 0.783 | 0.36% | 1.13% | 20.0 |
| 5% name / 40% country / 40% sector + turnover | top 2,000 | 7.9% | 0.485 | 0.820 | 0.94% | 3.81% | 20.0 |
| 3% name / 30% country / 30% sector + turnover | top 500 | 13.0% | 0.850 | 0.762 | 0.23% | 0.47% | 33.6 |
| 3% name / 30% country / 30% sector + turnover | top 1,000 | 11.1% | 0.718 | 0.761 | 0.35% | 0.82% | 33.6 |
| 3% name / 30% country / 30% sector + turnover | top 2,000 | 7.9% | 0.505 | 0.783 | 0.92% | 2.81% | 33.6 |
| 3% name / 25% country / 25% sector + turnover | top 500 | 12.0% | 0.792 | 0.763 | 0.23% | 0.47% | 33.7 |
| 3% name / 25% country / 25% sector + turnover | top 1,000 | 11.3% | 0.745 | 0.764 | 0.36% | 0.82% | 33.7 |
| 3% name / 25% country / 25% sector + turnover | top 2,000 | 8.2% | 0.536 | 0.797 | 0.91% | 2.79% | 33.9 |

The interpretation is important for the dissertation. More liquidity data
improves cost identification, but a broader tradable universe is not a free
performance improvement. The unconstrained signal naturally remains close to
larger liquid names, while hard diversification constraints pull capital into
weaker and more expensive tail names. This strengthens the
predictability-implementability gap: concentration can be reduced, and costs
can now be measured for the broader universe, but the act of making the
portfolio institutionally diversified materially weakens the realized
deep/hybrid edge.

### Closure Experiments Before Novel ML

Two closure checks were run before opening another model family. The first
re-costs the Kelly-style stock-SDF outputs using the new top-2000 Refinitiv
BID/ASK panel. This directly tests whether the earlier autoencoder capacity
problem was only missing liquidity data. The answer is no: coverage improves
materially, but the newly observed liquidity tail is expensive.

At EUR 100m, with the same constant 25 bps fallback only where top-2000
Refinitiv spreads remain unavailable:

| construction | spread coverage | ADV-floored weight | realized half-spread | gross return | net return | Sharpe | break-even half-spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| autoencoder K=5 | 70.8% | 21.2% | 53.9 bps | 7.6% | 0.2% | 0.071 | 31.8 bps |
| autoencoder K=10 validation-grid winner | 70.2% | 21.9% | 55.5 bps | 9.6% | 1.4% | 0.668 | 72.9 bps |
| IPCA K=5 | 74.0% | 20.5% | 52.3 bps | 7.9% | -0.4% | -0.141 | 8.8 bps |
| principal portfolio h=1 | 99.9% | 0.5% | 6.2 bps | -0.3% | -0.8% | -0.120 | n/a |
| principal portfolio h=3 | 99.9% | 0.4% | 6.1 bps | 0.7% | 0.2% | 0.037 | n/a |
| principal portfolio h=5 | 99.9% | 0.5% | 6.2 bps | 0.4% | -0.0% | -0.003 | n/a |

This changes the autoencoder interpretation. The K=10 autoencoder remains the
strongest Kelly-style stock-SDF variant after the broader liquidity pull, but
its EUR 100m net Sharpe is modest once actual top-2000 spreads replace the old
assumptions. IPCA does not rescue the result, and principal portfolios are well
covered by measured spreads but economically weak. The closure result is
therefore a stronger version of the same dissertation claim: deeper data
identifies costs rather than making the capacity problem disappear.

The second closure check selects the constrained deep/hybrid universe cap and
constraint rule using only trailing validation performance. For each month, the
selector chooses among top-500, top-1000 and top-2000 candidate constructions
using a 36-month validation window, a 24-month minimum history and a
certainty-equivalent objective. This guards against choosing constraints after
seeing the test sample.

| validation-selected rule | annual net return | volatility | Sharpe | max drawdown | turnover | spread cost | impact cost | effective N | top-five weight | average selected universe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen deep/hybrid selector | 13.6% | 16.1% | 0.846 | -29.0% | 0.823 | 0.39% | 1.27% | 21.4 | 24.0% | 713 |
| fixed momentum top-500 observed | 14.4% | 22.0% | 0.655 | -38.6% | 0.543 | 1.57% | 4.35% | 28.3 | 18.9% | 1,140 |

The validation-selected deep/hybrid rule has a better risk profile than the
validation-selected momentum benchmark, but not a statistically reliable
advantage. Relative to validation-selected momentum, its annual net return is
0.8 percentage points lower (`HAC p=0.859`) and its Sharpe is 0.191 higher
(six-month block bootstrap Holm `p=0.753`). It also does not improve on the
fixed top-500 5%/40%/40% turnover-constrained deep/hybrid construction: the
annual return is 1.2 percentage points lower (`HAC p=0.205`) with no reliable
Sharpe gain.

This answers the concern that the constrained experiments may be too strict.
The selector was allowed to use broader top-1000 and top-2000 universes and
milder or stricter cap rules. For the deep/hybrid signal, it still selected
top-500 constructions in 67 of 89 months, with the mild
5%/40%/40% turnover rule chosen in 47 months. The fixed top-500 mild-cap rule
therefore remains the primary implementable construction, not because it was
picked after the fact, but because a validation-only procedure usually returns
to it.

The closure decision is to keep the current implementable deep/hybrid result
as the benchmark for the dissertation and let the next novel ML experiment be
explicitly cost-aware at training time. Adding another unconstrained predictor
would be less informative than testing whether a model can internalize the
spread, impact and concentration penalties that currently create the wedge
between predictability and implementability.

## Incremental Information

Monthly Fama-MacBeth regressions use cross-sectionally ranked OOS scores. The
full specification controls for momentum, size, book-to-market, rolling beta,
idiosyncratic volatility, country fixed effects and sector fixed effects.

All ML score slopes remain positive. Annualised slopes range from approximately
7.9% to 12.8%, and remain significant after HAC inference and Holm correction.
This demonstrates incremental stock-level information beyond the included
controls. The slope is the return associated with a unit change in the scaled
score and must not be interpreted as an attainable portfolio return.

## Factor Spanning

The depth stage constructs internally consistent EUR MKT, SMB, HML, RMW, CMA
and MOM factors from time-t characteristics and next-month returns. Net-at-25
bps portfolios and ML-minus-momentum differences are regressed on those
factors with HAC standard errors.

For the primary value-weighted long-short comparison:

| model minus momentum | annualised alpha | nominal p | Holm p |
|---|---:|---:|---:|
| MLP rank | 21.5% | 0.023 | 0.185 |
| histogram GBM return | 22.6% | 0.034 | 0.236 |
| elastic-net rank | 15.3% | 0.092 | 0.550 |
| ridge rank | 13.3% | 0.120 | 0.598 |
| MLP return | 8.5% | 0.349 | 1.000 |

No factor-adjusted ML-minus-momentum alpha survives correction across the fixed
model family.

An external robustness check uses USD-denominated Kenneth French European five
factors and momentum. EUR portfolio returns are converted to USD with the
FRED USD-per-EUR spot return before regression. MLP-rank minus momentum has
nominal annualised alpha of 17.0% (`p=0.047`), but this does not survive Holm
correction (`p=0.377`). No external-factor result changes the conclusion.

## Missing Retirement Returns

Of 131 eligible inactive securities without a retirement-month return, 92 have
an investable pre-retirement signal row. Two retire after the panel's final
realized target month, leaving 90 that can be scored without a label.
Applying -30% and -100% stress returns barely changes the primary
value-weighted results. Under -100%, the MLP-return Sharpe rises from 0.950 to
0.956, while momentum falls from 0.611 to 0.606. The direction reflects the
fact that a negative retirement payoff benefits a short position. All primary
ML-minus-momentum differences remain insignificant after Holm correction.

These are deliberately severe sensitivity scenarios rather than estimates of
actual payoffs; retirement can result from acquisitions and restructuring as
well as distress.

## Conditional LambdaRank and Extended OOS Period

The extended experiment begins in January 2008, making the Global Financial
Crisis genuinely OOS. The initial model has 112,502 labelled observations
across 72 months from 2002-2007. Monthly eligibility never falls below 1,249
stocks, mean feature coverage rises smoothly from 16.5 to 17.4 of 18, and all
horizon-specific training labels are fully realized before each annual cutoff.

Cost-aware LambdaRank models estimate 1/3/6/12-month residual-return relevance.
The conditional version adds lagged market trend, volatility, EUR rates,
aggregate turnover, country and sector; the matched unconditional version
removes only those states. Horizon blends are selected from trailing OOS net
utility.

Conditional information raises one-month gross residual IC from 0.076 to
0.081, but does not improve the 3/6/12-month models consistently. One-month net
IC is approximately 0.16 because the target also contains a predictable
implementation-cost penalty; it is not a pure return-predictability statistic.

At EUR 100m:

| signal | format | excess return | volatility | Sharpe | CE |
|---|---|---:|---:|---:|---:|
| Conditional LambdaRank | long-only | 7.09% | 9.97% | 0.711 | 5.60% |
| Unconditional LambdaRank | long-only | 7.26% | 10.45% | 0.695 | 5.62% |
| Momentum | long-only | 6.82% | 10.51% | 0.649 | 5.16% |
| Conditional LambdaRank | dollar-neutral | 1.90% | 5.34% | 0.356 | 1.47% |
| Unconditional LambdaRank | dollar-neutral | 1.73% | 5.70% | 0.304 | 1.25% |
| Momentum | dollar-neutral | 5.23% | 8.20% | 0.638 | 4.22% |

Conditional long-only LambdaRank is descriptively competitive, but its Sharpe
advantage over momentum is insignificant (`p=0.466`, Holm `p=1.0`). Its direct
Sharpe difference from unconditional LambdaRank is only 0.016 (`p=0.788`,
Holm `p=1.0`). Both rankers fail economically in dollar-neutral form.

The 2010 and 2015 common-window checks preserve this pattern: conditional
long-only results remain competitive but insignificant, while dollar-neutral
results remain below momentum. The defensible conclusion is that LambdaRank
finds a defensive long-only tilt; market-state conditioning does not provide
reliable incremental information.

The Jensen-Kelly-Malamud-Pedersen theme-permutation procedure was also
implemented. Its underlying one-month residual long-short strategy has
negative net utility, so the theme ranking is retained as a diagnostic and not
interpreted economically.

Full details are in
`results/asset_pricing_ml/conditional_lambdarank_2008/README.md`.

## Implementable Efficient Frontier

The final economic-value stage freezes the prediction models and asks whether
their signals expand the net-of-cost opportunity set. It compares MLP
raw-return forecasts, momentum and an unfitted value-momentum-profitability
composite in the same top-500 investable universe.

Monthly aim portfolios solve a constrained mean-variance problem using causal
36-month one-factor risk estimates. Dollar-neutral portfolios impose zero net
exposure, gross exposure at most one, 2% position limits and absolute beta
below 0.05. Long-only portfolios impose 3% position limits and invest residual
cash at the EUR short rate. Risk aversion and partial-adjustment speed are
selected annually from the trailing 36 months only.

Execution costs combine observed Refinitiv half bid-ask spreads with
AUM-sensitive square-root impact. The dedicated pull contains 123,449 monthly
quotes for 811 relevant securities; 99.92% of rows have valid spreads. Missing
spreads receive a conservative 25 bps half-spread. Results are evaluated at
EUR 10m, EUR 100m and EUR 500m, with -100% assigned to scoreable missing
retirement returns.

At EUR 100m, the causally selected results are:

| signal | portfolio | excess return | volatility | Sharpe | CE |
|---|---|---:|---:|---:|---:|
| ML return | long-only | 6.31% | 10.46% | 0.603 | 4.67% |
| Momentum | long-only | 6.74% | 10.66% | 0.632 | 5.03% |
| Sparse 3-char. | long-only | 6.44% | 10.29% | 0.626 | 4.85% |
| ML return | dollar-neutral | 3.65% | 4.71% | 0.775 | 3.32% |
| Momentum | dollar-neutral | 5.24% | 7.61% | 0.689 | 4.37% |
| Sparse 3-char. | dollar-neutral | 3.31% | 4.03% | 0.821 | 3.07% |

Momentum dominates the ML long-only frontier at every matched risk level. ML
adds at most about 0.8-0.9 percentage points of annual return at matched risk
in some low-volatility dollar-neutral settings, but not uniformly. Every
causally selected ML-minus-momentum Sharpe comparison is insignificant, and
all Holm-adjusted p-values equal 1.0.

The deeper conclusion is therefore not merely that ML predicts returns. The
signal sometimes changes the shape of the low-risk dollar-neutral frontier,
but it does not robustly expand the implementable opportunity set; momentum
retains higher certainty-equivalent return and dominates long-only
implementation.

Full methodology and outputs are in
`results/asset_pricing_ml/implementable_frontier/README.md`.

## Where Predictability Lives

The clearest robust IC patterns are:

- Ridge, GBM and MLP rank ICs are higher in small than large stocks.
- Their ICs are higher in high- than low-idiosyncratic-volatility stocks.
- These contrasts survive Holm correction.
- Most corresponding gross portfolio-spread contrasts do not survive
  correction.

The raw-return MLP is the strongest limits-to-arbitrage case. Its
high-minus-low idiosyncratic-volatility gross spread difference is about 22.0%
annualised, and its small-minus-large difference is about 19.4%; both survive
Holm correction. These are equal-weight gross diagnostic spreads, not
implementable net returns.

## Liquidity-Mechanism Extension

The expanded specification adds log EUR trading value and turnover volatility
without changing the frozen 18-feature baseline or its eligibility rules.
Theme-level fixed-model OOS ablation does not reproduce the China result that
liquidity dominates. Fundamentals and price trends are the leading rank-model
themes, while price trends dominate raw-return models. Liquidity contributes
to selected individual models and is more useful for rank prediction among
small stocks, but its aggregate theme importance is weak and unstable.

A five-rung frozen-prediction ladder independently reconstructs portfolios from
the standard universe through large, low-spread stocks. ML predictability and
portfolio performance attenuate, but not monotonically, and momentum generally
remains stronger in the most investable long-only universes. New paired ladder
inference confirms that no ML signal has a positive Holm-significant Sharpe or
certainty-equivalent advantage over momentum in the primary EUR 100m
value-weighted long-only family. Full definitions, results and reproduction
commands are in
`LIQUIDITY_MECHANISM_EXTENSION.md`.

## State Dependence

Market states use only information available at month t. High/low volatility
uses a trailing 12-month volatility measure against an expanding historical
median shifted one month. Trend states use the trailing 12-month market return.

Ridge, GBM and MLP rank ICs are significantly lower in down markets after Holm
correction. Net value-weighted return differences between states do not survive
family-wise correction. ML predictability therefore appears procyclical rather
than uniquely valuable during stress.

## Limitations

- Retirement-month returns are present for only 10 of 141 eligible inactive
  securities. The -30%/-100% stress tests bound the mechanical impact but do
  not identify actual merger, distress or bankruptcy payoffs.
- Rolling beta and idiosyncratic volatility cover about 68% of OOS panel rows,
  so full-control Fama-MacBeth tests exclude young or sparse securities.
- The primary EUR factors are constructed from the study universe rather than
  being an independent external benchmark.
- The EUR cash-rate proxy currently leaves 132 of 137 OOS months in regressions
  requiring market excess returns or long-only excess returns.
- Conditional subgroup spreads are equal-weight gross diagnostics without
  turnover, market impact or borrow costs.
- The baseline study is a compact 18-characteristic European application. The
  Compustat extension raises the ML feature set to 53 ranked predictors. The
  adversarial LSTM/GAN SDF is a compact European Chen-Pelger-Zhu adaptation,
  not an exact replication of their U.S. data-depth design: it uses European
  Compustat/Refinitiv characteristics, four adversarial test assets and a small
  market-state LSTM for tractability.
- The AIPM linear transformer section remains a closed-form precursor. The
  later full AIPM post-estimation section is the costed transformer evidence.
- The full nonlinear AIPM implementation includes the paper's transformer
  architecture and benchmarks, but the reported European run is a tractable
  top-500, annual-refit, three-seed version rather than the paper's full
  all-stock, monthly-refit, ten-seed U.S. experiment.
- The principal-portfolio result is a characteristic-space European adaptation
  of Kelly, Malamud and Pedersen, not the exact fixed-universe
  all-security-signal prediction matrix from their U.S. setting.
- The conditional autoencoder is now converted into a gross-normalized
  stock-weight SDF portfolio and charged turnover, spread and market-impact
  costs. It still does not impose explicit borrow availability, short-sale
  fees or hard position limits during autoencoder training.
- The fundamental-mispricing adaptation follows the paper's deflated market
  value target and lagged accounting availability, but it uses the currently
  available Compustat export and rank-transformed accounting items rather than
  a hand-rebuilt Worldscope/Datastream 21-item raw-currency valuation dataset.
- The deep sequence modelling result is a full 2015-2026 Compustat-enriched
  annual-refit experiment with 12- and 24-month histories, but it is still a
  capped training-row adaptation rather than a full all-row deep sequence grid.
  The common benchmark comparison is on a strict common stock-month sample, but
  the sequence models are trained on capped rows while the classical Compustat
  baselines use their full available training histories.
- The deep-sequence blend grid and signal-smoothing grid are fixed ex-ante
  robustness experiments. The reported top point estimates should not be
  reinterpreted as a selected live strategy without a separate validation rule
  or a fresh holdout period.
- Turnover-aware signal smoothing is an implementable frozen-signal surrogate,
  not a fully retrained neural objective with position caps, spreads and market
  impact inside the loss function.
- The rolling validation selector is causal with respect to completed returns,
  but it is still an adaptive model-selection layer estimated on the same
  2015-2026 OOS era. Its positive long-only deep/hybrid result should be treated
  as a robustness direction unless confirmed in a fresh holdout or by
  pre-registering the selector before the final dissertation run.
- The frozen deep/hybrid long-only robustness check fixes the selected
  model/rung/month choices and has complete observed spread coverage, so its
  transaction-cost result is much better identified than the autoencoder
  capacity test. The unconstrained selected portfolio is materially
  concentrated.
- The constrained long-only construction imposes hard single-name, country and
  sector caps after prediction and verifies that the concentration issue can be
  reduced with current data. It is still a portfolio-construction overlay, not
  a neural network trained end-to-end with benchmark weights, free-float
  constraints or borrow-cost data.

## Conclusion

ML meaningfully predicts the European equity cross-section and contains
information beyond momentum and conventional risk controls. But its strongest
signals occur where implementation is hardest, weaken in down markets and do
not deliver a multiplicity-robust factor-adjusted advantage over momentum.
The adversarial LSTM/GAN SDF extension reinforces rather than overturns this:
the CPZ-style minimax architecture is feasible on the European panel, but its
first-pass out-of-sample SDF Sharpe is economically weak. The AIPM linear
transformer adds a newer Bryan Kelly-style architecture: its gross-normalized
SDF is strong, but the incremental attention channel is not significant versus
the no-attention characteristic SDF. The full nonlinear AIPM implementation
extends this test to softmax attention and stacked transformer blocks; in the
first European top-500 run, the transformer is feasible but does not dominate an
equally deep own-asset MLP before or after spread and market-impact costs.
Monthly refitting helps the transformer in 2020-2026 but not significantly, and
direct implementation penalties inside the neural objective degrade rather than
improve the learned portfolios. The principal-portfolio adaptation adds a
transparent cross-predictability benchmark, but its European gross and net
Sharpe ratios are weak. The conditional autoencoder adds a separate
no-arbitrage latent-factor result: positive total R2 and small positive
predictive R2, plus a high-Sharpe full-universe stock-SDF portfolio. However,
the autoencoder's execution-cost evidence is not identified with the current
liquidity pull because most of its gross weight sits outside measured spread
coverage. The peer-implied fundamental-mispricing experiment adds a separate
European accounting-based ML signal; it is feasible and leakage-safe, but it
does not beat momentum in implementable value-weighted portfolios. The full
deep-sequence experiment adds a cleaner exception to the weak-neural evidence:
LSTM improves rank IC versus a static neural sequence baseline, while
attention-LSTM and GRU improve costed value-weighted portfolios versus that
static neural baseline. But the common benchmark test shows that sequence
models do not beat momentum, ridge, DRE, histogram GBM or MLP in the primary
value-weighted long-short family. Fixed sequence-baseline blends and
liquidity-rung tests identify economically interesting attention combinations,
especially in measured large low-spread universes, but not a Holm-robust
attention premium. The turnover-aware smoothing experiment is the strongest
practical improvement: ex-ante score inertia roughly halves turnover and raises
several net Sharpe point estimates, with the clearest parent-family inference
for smoothed DRE and a marginal result for smoothed ridge. Yet the top smoothed
candidates still do not beat momentum after family-wise correction. The rolling
validation-selected experiment is the most realistic next step: it lets the
model choose deep/hybrid and smoothed signals using only prior completed
returns. Its long-only deep/hybrid liquid version is competitive with momentum
and beats ridge on annualized return, but it still does not deliver a
multiplicity-robust Sharpe advantage over momentum; the long-short selected
versions are weaker. Freezing that deep/hybrid liquid long-only rule and
re-simulating it at EUR 10m, 100m and 500m confirms that its cost estimates are
not driven by fallback spread assumptions and that impact sensitivity is
moderate. The remaining implementability bottleneck moves from missing cost
coverage to concentration: the portfolio is capacity-clean but too concentrated
without explicit name, country and sector constraints. The constrained
long-only construction shows that this weakness can be reduced with current
data: a 5% name / 40% country / 40% sector capped version preserves most of the
deep/hybrid Sharpe while cutting top-five concentration sharply. But stricter
caps weaken returns and the capped deep/hybrid advantage over capped momentum
is not statistically reliable. Even here, the best predictor, the
lowest-turnover signal and the best implementable benchmark are different
objects. The predictability-to-implementability wedge remains the
dissertation's central economic result.
