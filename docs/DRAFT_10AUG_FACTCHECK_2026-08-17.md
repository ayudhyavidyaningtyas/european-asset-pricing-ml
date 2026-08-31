# Fact-check: "Draft 10Aug ver1.docx" vs on-disk artifacts (2026-08-17)

Method: seven parallel verification passes over the pandoc extract of the
draft, one per chapter block. Every numeric claim was checked against the
actual artifact files (CSV / JSON / parquet) under `results/asset_pricing_ml/`
and `data/processed/asset_pricing/`, and against code defaults and run
manifests where the claim describes configuration. Earlier planning notes and
markdown summaries were not used as evidence.

Coverage: ~950 numeric claims, including 100% cell coverage of Appendix
Tables A.1–A.12 (~700 cells) and Appendix B.

## Verdict

Zero transcription errors: every number in the draft matches a real artifact.
The defects are (1) one stale results block — the draft cites the July Test B
generation, superseded by the canonical 16-Aug refresh on the cleaned
64-feature set — and (2) two methodology descriptions in Chapter 4 that do not
match what the code actually does.

Clean sections (no action): abstract and Ch1–2 previews (11/11), Ch3 data
chapter (45/46 hard numbers exact), Ch5.2–5.4 tables (all cells), Ch5.7
constrained implementation, Ch5.8 US comparison, Appendix Tables A.1–A.4 and
A.6–A.12, Appendix B hyperparameters.

## 1. Stale Test B block (July run -> refresh_20260816)

Canonical artifact: `results/asset_pricing_ml/data_depth_model_depth_interaction_refresh_20260816/data_depth_model_depth_interaction.csv`
(cleaned 64-feature estimates_enriched set; identical 254,600 stock-months).
The draft's Table 5.6/5.7 and Appendix A.5 match the superseded July run
`data_depth_model_depth_interaction/` instead.

| Quantity | Draft (July) | Refresh (canonical) | Status |
|---|---|---|---|
| Ridge data-depth | +.00356, t 2.53, Holm .0227 | same estimate, Holm .0340 | cosmetic |
| HistGBM data-depth | +.00702, t 3.82, Holm <.001 | identical | unchanged |
| MLP data-depth | +.00360, t 1.27, p .204 | +.00047, t 0.19, Holm .852 | estimate ~8x smaller |
| DRE data-depth | +.00511, t 3.12, Holm .0054 (sig) | +.00264, t 1.95, Holm .1028 (NOT sig) | SIGNIFICANCE REVERSED |
| HistGBM interaction | +.00346, t 2.50, Holm .0375 | identical | unchanged |
| MLP interaction | +.00005, p .982 | −.00309, Holm .382 | sign flip |
| DRE interaction | +.00155, t 1.91, Holm .113 | −.00092, t −1.09, Holm .382 | sign flip |
| HistGBM premium w/ estimates Holm | .4508 | .3060 | cosmetic |

Required edits:
- §5.5 (~line 2340 of extract) and §6.3 (~line 2709): "ridge, HistGBM and DRE
  ... surviving Holm" -> "ridge and HistGBM". DRE must be moved to the
  unresolved column.
- §5.5 (~2346): drop "MLP estimate is similar in magnitude to ridge".
- Table 5.6, Table 5.7, Appendix Table A.5: replace with refresh values
  (8 of 13 rows change; HistGBM and Compustat-only rows unchanged).
- The §5.5/§6.3 framing "complementarity is model-specific / confined to
  HistGBM" survives unchanged — the HistGBM rows are identical across
  generations.

### Related: §5.6 revisions IC-increment sentence (~2447)

Draft: "revisions raise ridge IC by 0.0069, Holm .0022" — this is the July
covered-panel ablation (`estimates_family_ablation/`), a different experiment
from the refreshed strict-lag-1 matched-sample ablation
(`estimates_family_ablation_refresh_20260816/`), where the standalone ridge
revisions delta is +.0028 and does not survive its (conservative, 39-test)
Holm family (raw p ≈ .018, Holm .42). In the refresh, the robust statement is:
revisions are the strongest standalone group (hist_gbm/mlp +.0057, ~80% of the
full-layer effect for HistGBM) and the ONLY Holm-significant marginal
contributor (full vs ex_revisions, hist_gbm +.0048, Holm .0079). Re-anchor the
sentence to those artifacts (see
`estimates_identification_evidence_20260816/identification_evidence.md`
Panels C/D).

NOTE: `econometric_evidence_tables/econometric_evidence_summary.csv` row 4
also embeds the stale July ablation value — regenerate that table before
citing its neighbours again.

## 2. Chapter 4 methodology descriptions vs code

1. HIGH — §4.9 (~1802–1810) and Table 4.6 (~1925–1930): the forecast-error
   Fama–MacBeth is described as "annual cross-sectional regressions ...
   Newey–West HAC at lag one on the annual coefficient series". The actual
   implementation (`scripts/run_forecast_error_mechanism_tests.py:473`,
   manifest config) estimates cross-sections by ANNOUNCEMENT MONTH
   (collapsed firm-year sample: 167–175 monthly periods) and applies HAC with
   lags = 15. Same in the US variant. The 5.6/A.6 numbers are correct; the
   described estimator is not the one that produced them. Rewrite the two
   passages to "monthly announcement-date cross-sections ... HAC lag 15".
2. MEDIUM — §4.9 (~1799–1801): "the final eligible pre-realisation snapshot is
   retained for each firm-fiscal-year" -> the code retains the snapshot whose
   lead is CLOSEST TO SIX MONTHS before realisation
   (`collapsed_lead_months = 6`, `build_collapsed_sample`,
   run_forecast_error_mechanism_tests.py:409–430). "One observation per
   firm-fiscal-year" is correct.
3. LOW — §4.7 (~1741–1742): "elastic net remains a second regularised linear
   specification" — elastic net is absent from the spanning ladder
   (`complexity_spanning_ladder.csv` has no elastic_net rows). Either delete
   or re-scope the sentence to the Table 5.1 baseline family.

## 3. Wording-level items

- §5.4 (~2222): "trading value shows the same monotonic pattern" — it is NOT
  monotonic: mid-liquidity premium exceeds low for ridge (.0624 vs .0498) and
  HistGBM (.0669 vs .0546) before collapsing in high/top-500
  (`capacity_gradient_tests/paired_premium_by_bucket.csv`). Use "the same
  contraction at the tradable end".
- §5.6 (~2441): "placebo variants also produce sign accuracy above 0.55" — one
  collapsed placebo (epsfr 1m) is 0.542. Use "around 0.55" or "0.54–0.59".
- §5.4: "every market-capitalisation gradient is approximately 0.001" —
  HistGBM's is 0.00002; "at most 0.0011" is tighter (table prints 0.0000, so
  defensible as-is).
- §3.6 (~1246): the 812-RIC bid-ask request set as "union of monthly top-500
  sets over 2015 to 2026" — the 812-RIC file is verified but no script
  reproduces the union construction or the window; soften or document.
- Consistency guard (abstract, Table 4.1, Table 5.9): the €100m headline
  (14.26% net, 0.81) is the strict-lag-1 pure-revisions "fixed" run
  (`constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed/`,
  137 months). The non-"fixed" constrained runs cover 113 months (2017–2026)
  and give different numbers (e.g. 12.5%/0.79 at 100m). The draft currently
  cites the fixed run consistently — keep it that way.
- §5.5/§6.2 dependence on the "primary comparison" qualifier: HistGBM−ridge IC
  is Holm .0614 (unresolved) in the primary europe_compustat_benchmark family
  but Holm .0365 (resolved) in the appendix common-benchmark family (Table
  A.2). The draft's qualifiers currently make this internally consistent —
  preserve them in any rewrite.

## 4. New material to add: identification evidence from the 2026-08-16/17 runs

The draft predates the identification extension and contains none of it.
Recommended placement: new §5.6 "Identification and robustness of the
analyst-data effect" directly after §5.5 (renumber current 5.6/5.7/5.8 to
5.7/5.8/5.9); a compact design block in Ch4 (extend §4.8 or new §4.9) plus one
row in the §4.13 mapping table; the four-panel exhibit as a new Appendix Table
A.13. Combined exhibit:
`results/asset_pricing_ml/estimates_identification_evidence_20260816/`
(identification_evidence.md, identification_evidence_panels.csv). All numbers
below were read directly from the artifacts on 2026-08-17.

### 4.1 Panel A — coverage selection (dir: estimates_coverage_selection_20260816/)

Design: monthly cross-sectional L2 logit of coverage on size, trading value,
turnover, volatility, book-to-market, momentum, country, sector; 137 OOS
months; covered rows inverse-propensity-weighted back to the full eligible
universe (459,829 rows, 254,600 covered); weights floored and normalised to
mean one per month.

- Propensity model: mean AUC 0.872, median 0.865; mean coverage rate 0.553.
  Coverage is strongly selected on observables.
- Balance (standardised mean difference, covered vs universe): log size
  0.526 -> 0.135 after weighting; trading value 0.476 -> 0.082; turnover
  0.184 -> -0.012; all country/sector levels |SMD| < 0.07 weighted.
  Mean effective sample share 0.534 (min month ~0.19; p99 weight ~5; max ~57).

Data-depth effect (IC, estimates cell minus Compustat cell), Holm within
weighting family:

| Model | Unweighted | Holm p | IPW | Holm p |
|---|---|---|---|---|
| Ridge | +0.00356 | .034 | +0.00437 | .0006 |
| HistGBM | +0.00702 | .0005 | +0.00853 | <.0001 |
| MLP | +0.00047 | .852 | +0.00104 | .657 |
| DRE | +0.00264 | .103 | +0.00264 | .186 |

- Floor sensitivity (min propensity .01/.02/.05): estimates stable to the
  third decimal (ridge .0044/.0043/.0044; HistGBM .0085/.0085/.0084).
- Propensity strata (unweighted, within-stratum): the effect lives where
  coverage is thin. Ridge: +0.0059 (stratum 0, mean propensity .36, Holm
  <.0001), +0.0059 (s1, .0028), +0.0054 (s2, .057), +0.0019 (s3, p .43),
  -0.0018 (s4, mean propensity .96). HistGBM: +0.0084/+0.0092/+0.0084 then
  +0.0018/+0.0007 (top strata ns).

Initial analysis: the unweighted column reproduces Test B exactly
(cross-check); reweighting to the universe strengthens rather than kills the
ridge/HistGBM effects; and the stratum gradient is the sharpest single fact
against "covered stocks are just easier to predict" — the lift is
concentrated in the least-covered names and absent in the most-covered.
Phrasing rule (agreed): a selection-robustness diagnostic, not causal
identification — the propensity is estimated and then treated as fixed in the
HAC inference; unobservable selection is outside the design.

### 4.2 Panel B — signal-lag ladder (dir: estimates_lag_ladder_20260816/)

Design: coverage-matched {Compustat-only, +analyst} cell pairs re-estimated
with the analyst snapshot stale-dated 1/2/3/6 months; primary scope =
stock-months common to every lag (248,339), so decay is not confounded by the
sample shifting; own-lag matched samples as robustness (nearly identical
estimates throughout). Holm within scope x model.

| Model | lag 1 | lag 2 | lag 3 | lag 6 |
|---|---|---|---|---|
| Ridge | +0.0038 (Holm .026) | +0.0022 (.191) | +0.0010 (.569) | -0.0006 |
| HistGBM | +0.0073 (.0004) | +0.0017 (.620) | +0.0035 (.024) | +0.0004 (.855) |
| DRE | +0.0029 (.138) | +0.0013 | +0.0005 | -0.0010 |
| MLP | +0.0008 | +0.0004 | -0.0025 | -0.0045 (.381) |

Initial analysis: ridge decays monotonically to zero; DRE likewise (never
resolved); the six-month boundary is clean for every model — nothing
survives, consistent with the analyst layer carrying news rather than
proxying a slow-moving characteristic. HistGBM is non-monotone (lag-2 dip
below a still-significant lag-3): write "gone by six months, irregular in
between", never "smooth decay".

### 4.3 Panels C/D — attribution (dir: estimates_family_ablation_refresh_20260816/)

Design: cleaned 11-feature analyst layer decomposed two ways (source family:
EPS/revenue/price target; information type: levels/revisions/dispersion),
each an exact partition; all cells on the identical 254,600 stock-months
(ablation_sample_checks.csv: 19 comparisons, 0 unmatched). Panel C =
standalone ("X_only" vs Compustat; Holm family spans all 13 variant
comparisons x 3 models = conservative — table-note this); Panel D = marginal
(full vs "ex_X").

Standalone (Panel C), delta IC vs Compustat baseline:

- revisions_only: HistGBM +0.0057 (t 2.49), MLP +0.0057 (t 3.02, Holm .088),
  Ridge +0.0028 (t 2.38) — for HistGBM ~80% of the full-layer effect
  (+0.0070, Holm .0050).
- eps_only: ridge +0.0030 (t 2.77), HistGBM +0.0041 (t 2.58), MLP +0.0050
  (t 2.33) — best source family, none survive the 39-test Holm family.
- levels_only, dispersion_only, revenue_only: all ~0 (|delta| <= 0.0018,
  t < 1.6).

Marginal (Panel D), full minus leave-one-group-out:

- revisions: HistGBM +0.0048 (t 3.52, Holm .0079) — the ONLY Holm-significant
  marginal contributor anywhere; Ridge +0.0022 (t 1.99, Holm .655);
  MLP -0.0042 (t -2.67) — adding revisions HURTS the MLP.
- eps: HistGBM +0.0034 (t 2.73, Holm .108); everything else unresolved;
  all MLP marginals negative.

Initial analysis: both decompositions agree — revisions carry the layer;
levels and dispersion are inert; the two panels are not interchangeable
(correlated groups can be standalone-informative yet marginally redundant)
and must be reported separately. This section REPLACES the stale July
"+0.0069, Holm .0022" sentence (see section 1 above) with a
better-identified, matched-sample claim.

### 4.4 Panel E — missingness negative control (dir: estimates_missingness_control_20260816/)

Design: third cell (Compustat + estimates_feature_count rank, no analyst
values) interposed between the Test B pair on identical 254,600 stock-months;
the data-depth effect splits into a missingness increment and an
analyst-value increment. Holm within quantity.

| Quantity | Ridge | HistGBM | MLP |
|---|---|---|---|
| Missingness increment | -0.00005 (ns; MDE .0002) | -0.0021 (ns) | -0.0036 (ns) |
| Analyst-value increment | +0.0036 (Holm .025) | +0.0092 (Holm .035) | +0.0040 (Holm .035) |
| Data-depth (cross-check) | +0.0036 = Test B | +0.0070 = Test B | +0.0005 = Test B |

Initial analysis: coverage counts alone explain none of the effect (the ridge
bound is tight: +/-0.0005 detectable); the analyst values carry everything.
Footnote required: MLP's positive value increment only offsets its negative
missingness increment — MLP stays in the unresolved column.

### 4.5 Combined chapter-ready claim (agreed wording basis)

Analyst expectations add incremental European return-predictive information
after the Compustat/price baseline. The effect is revisions-driven, survives
observable coverage-selection diagnostics (and strengthens under
reweighting), decays as analyst snapshots become stale and vanishes at a
six-month falsification boundary, is concentrated where coverage is thinner,
and is not a coverage-count/missingness artifact. The gains accrue to linear
breadth and shallow nonlinear interaction learning rather than to deeper
architecture, and remain a predictability result rather than proof of robust
implementable alpha. Restraint rules: ridge and HistGBM are the robust
beneficiaries; DRE directionally positive but never resolved; MLP flat or
negative; never write "deep models exploit analyst data".

### 4.6 seq24 uncapped run completed (2026-08-17, supersedes the exclusion note)

`deep_sequence_compustat_full_seq24_uncapped/` now exists (137 months, 0
causality violations, common 448,813-obs basis). Rank ICs: GRU .1128, LSTM
.1124, last-MLP .1101, attention-LSTM .1044 — indistinguishable from the
seq12 uncapped band (.1049-.1126) and still below the shallow models
(.115-.118). Draft edits: (a) drop "uncapped 24-month specification is
incomplete and excluded" (~lines 1535-37, 2155-56) and report the result;
(b) Table A.1's capped seq24 rows can be annotated or replaced. Note
train_rows_used < train_rows_available in the fit log reflects the 24-month
history-eligibility requirement, not a training-budget cap.

## 5. Confirmations worth keeping

- Draft bootstrap CI for the 100m constrained cell [3.76%, 24.17%] is CORRECT
  per `econometric_evidence_tables` bootstrap rows; an earlier working note's
  [3.97, 24.14] was wrong.
- All Ch5.2–5.4 table cells match `europe_compustat_benchmark/`,
  `complexity_spanning_ladder/`, `capacity_gradient_tests/` exactly.
- All US tables (5.10, 5.11, A.9–A.12) match `market_comparison_compustat/`,
  `us_comparison_exhibits/`, `us_usd_test_b_paired_tests/`,
  `us_forecast_error_mechanism/` exactly.
- Appendix B hyperparameters match `src/asset_pricing_ml.py` WalkForwardConfig,
  `src/deep_sequence_models.py`, and the cited run manifests exactly.
