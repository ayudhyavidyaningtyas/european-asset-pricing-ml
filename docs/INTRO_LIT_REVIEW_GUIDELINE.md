# Introduction and Literature Review Writing Guideline

Use this guide for the dissertation sections:

- Introduction
- Background and motivation
- Problem statement and rationale
- Aim, objectives, and research questions
- Significance and contributions
- Dissertation outline
- Literature Review
- Literature Review introduction
- Theoretical framework
- Empirical literature
- Research gap
- Hypothesis development
- Literature Review summary

Working title:

Machine Learning, Deep Asset Pricing, and the Predictability-Implementability Gap in European Equities

Core claim:

Machine learning contains real out-of-sample information about European equity returns, but the current evidence does not identify a robust value-weighted, net-of-cost, capacity-aware alpha once turnover, liquidity, market impact, concentration, and multiple-testing correction are made explicit.

Avoid the stronger claim that ML cannot beat momentum. The correct claim is that the current evidence does not identify a statistically robust implementable advantage over momentum.

## Framing Guardrails

Use these rules in the abstract, introduction, conclusion, and any section summary. They prevent the thesis from becoming stronger than the evidence.

- Say "no robust implementable advantage is identified", not "ML does not translate into alpha" or "ML cannot beat momentum".
- Say "placebo evidence supports the ridge baseline", unless additional placebo tests are confirmed for every model family.
- Say "costs combine observed Refinitiv half-spreads with a square-root impact model", not "costs are measured rather than assumed".
- Say "predictions are formed strictly out of sample from 2008" for the LambdaRank extension, not "the GFC is genuinely out of sample" in a design-stage sense.
- Say "internal leakage checks show zero cutoff and duplicate-prediction violations", not "there is no leakage"; survivorship, vendor backfill, restatement timing, and point-in-time analyst coverage remain data limitations.
- Say "attention weights concentrate on same-size and same-sector peers", not "attention is interpretable" without qualification.
- Say "momentum dominates in point estimates" or "momentum remains the stronger benchmark in this comparison", not "momentum dominates" without a significance qualifier.
- Link Fama-MacBeth evidence to implementability: equal-weighted cross-sectional slopes show incremental information, but that information is concentrated where monetisation is hardest.
- Treat smoothing and validation-selected results as practical candidates with selection/multiplicity caveats unless every searched variant is inside the stated Holm family.
- Keep external validity modest: the main 2015-2026 OOS window is one European decade with limited stress episodes.

## 1. Introduction

### Purpose

Open the dissertation by moving from the broad promise of machine learning in asset pricing to the narrower practical question: whether predicted returns can be implemented in European equities after realistic frictions.

### Recommended structure

1. Start with the growth of machine learning in empirical asset pricing and portfolio construction.
2. Explain why equity return prediction is not enough: economic value depends on trading, costs, capacity, and portfolio constraints.
3. Position Europe as an important but less-studied market setting relative to the U.S. literature.
4. Introduce the dissertation's central concept: the predictability-implementability gap.
5. Preview the empirical result: prediction is real, but robust implementable alpha over momentum is not established.

### Papers to cite here

Primary:

- Gu, Kelly and Xiu, empirical machine learning asset pricing work.
- Kelly and Xiu, Financial Machine Learning.
- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.
- Chen, Pelger and Zhu (2024), Deep Learning in Asset Pricing.
- Kelly, Kuznetsov, Malamud and Xu (2026), Artificial Intelligence Asset Pricing Models.

Supporting:

- Fama and French (2004), The Capital Asset Pricing Model: Theory and Evidence.
- Pedersen, Big Data Asset Pricing lectures: empirical asset pricing primer, machine learning, factor zoo, and frictions.
- Emerson, Kennedy, O'Shea and O'Brien, Trends and Applications of Machine Learning in Quantitative Finance.
- Routledge (2019), Machine Learning and Asset Allocation.
- Jurczenko (2020), Machine Learning Optimization Algorithms and Portfolio Allocation.

### What not to do

- Do not begin with a generic AI hype paragraph.
- Do not claim that the dissertation proposes a new model architecture.
- Do not overstate the empirical finding as a trading strategy victory.

## 2. Background and Motivation

### Purpose

Explain why the research question matters academically and practically.

### Content to include

- Classical asset pricing asks whether expected returns compensate for systematic risk or reflect mispricing.
- Modern empirical asset pricing has many characteristics and nonlinear interactions, making ML attractive.
- However, predictability can be strongest in small, volatile, illiquid, or costly-to-trade stocks.
- European markets add fragmentation, multiple countries, sector composition, currency issues, and heterogeneous liquidity.
- A model that predicts returns but requires excessive turnover, shorting, or illiquid small-cap exposure may have little implementable value.

### Papers to cite here

Asset pricing and factors:

- Fama and French (2004), CAPM theory and evidence.
- Pedersen, Big Data Asset Pricing, Lecture 2: empirical asset pricing primer.
- Pedersen, Big Data Asset Pricing, Lecture 4: factor zoo and replication.
- Asness, Frazzini and Pedersen (2012), Leverage Aversion and Risk Parity.
- Frazzini and Pedersen (2022), Embedded Leverage.
- Swade, Lohre, Nolte and Shackleton (2024), A Century of Macro Factor Investing.
- Bauer, Haerden and Molenaar, Asset Allocation in Stable and Unstable Times.

Implementation and market frictions:

- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.
- Pedersen, Big Data Asset Pricing, Lecture 6: Asset Pricing with Frictions.
- Kyle and Obizhaeva (2016), Market Microstructure Invariance.
- Boyd, Market Making and Risk Management in Options Markets.

State dependence and stress motivation:

- Ang and Bekaert (2002), International Asset Allocation with Regime Shifts.
- Ang and Bekaert (2004), How Regimes Affect Asset Allocation.
- Guidolin and Timmermann (2007), Asset Allocation under Multivariate Regime Switching.
- Guidolin and Timmermann (2008), International Asset Allocation under Regime Switching, Skew and Kurtosis.
- Kritzman, Page and Turkington (2012), Regime Shifts and Dynamic Strategies.
- Galvao and Owyang (2018), Financial Stress Regimes and the Macroeconomy.

### Bridge to this dissertation

End this subsection by saying that the dissertation therefore evaluates ML not only by IC and R2, but also by net portfolio performance, turnover, bid-ask spread, market impact, AUM sensitivity, liquidity rungs, and diversification constraints.

## 3. Problem Statement and Rationale

### Purpose

State the precise problem.

### Suggested problem statement

Existing ML asset-pricing studies show that flexible models can forecast cross-sectional equity returns, but less is known about whether these signals generate economically reliable portfolios in European equities after implementation frictions. This dissertation addresses that problem by testing whether European ML predictability survives the transition from forecasts to investable portfolios.

### Rationale

Use three points:

1. Prediction metrics and portfolio utility are different objects.
2. European market structure may make implementation harder than in broad U.S. studies.
3. Advanced architectures are increasingly feasible, but their economic value must be judged against simple benchmarks such as momentum.

### Papers to cite here

- Gu, Kelly and Xiu, ML return prediction.
- Jensen, Kelly, Malamud and Pedersen (2022), implementable frontier.
- Chen, Pelger and Zhu (2024), deep no-arbitrage SDFs.
- Kelly, Kuznetsov, Malamud and Xu (2026), AIPM.
- Hanauer, Kononova and Rapp (2022), Boosting agnostic fundamental analysis: using machine learning to identify mispricing in European stock markets.
- Kelly, Pruitt and Su, Characteristics are Covariances (IPCA).
- Gu, Kelly and Xiu, Autoencoder Asset Pricing Models.

## 4. Aim, Objectives, and Research Questions

### Aim

To evaluate whether machine learning and deep asset-pricing models generate implementable economic value in European equity selection after realistic transaction costs, liquidity, capacity, and portfolio constraints.

### Objectives

1. Build a leakage-safe European equity panel using Refinitiv/Datastream, Compustat Global, and liquidity data.
2. Estimate classical ML models using annual walk-forward out-of-sample splits.
3. Test whether ML scores contain incremental predictive information beyond conventional controls.
4. Compare ML signals with momentum in equal-weighted, value-weighted, long-short, and long-only portfolios.
5. Implement advanced deep and Kelly-style asset-pricing models, including DRE, neural SDFs, adversarial SDFs, AIPM transformers, autoencoder, IPCA, and deep sequence models.
6. Measure implementability through turnover, spread costs, market impact, AUM stress, liquidity ladders, and concentration constraints.
7. Identify where the gap between statistical predictability and implementable performance arises.

### Research questions

RQ1. Do machine-learning models predict next-month European equity returns out of sample?

RQ2. Do ML signals contain incremental information beyond momentum, size, value, beta, idiosyncratic volatility, country, and sector controls?

RQ3. Do ML signals generate value-weighted net portfolio performance that is statistically superior to momentum after transaction costs and multiple-testing correction?

RQ4. Are advanced deep asset-pricing architectures economically superior to simpler ML or momentum benchmarks in Europe?

RQ5. How do liquidity, turnover, capacity, and diversification constraints affect the conversion of predictive signals into implementable portfolios?

### Papers to align with each question

- RQ1: Gu, Kelly and Xiu; Kelly and Xiu; Pedersen ML lecture.
- RQ2: Fama-French style factor literature; Pedersen factor zoo lecture; Fama and French (2004).
- RQ3: Jensen, Kelly, Malamud and Pedersen; Kyle and Obizhaeva; Pedersen frictions lecture.
- RQ4: Chen, Pelger and Zhu; Kelly, Kuznetsov, Malamud and Xu; Gu, Kelly and Xiu autoencoder; Kelly, Pruitt and Su; Didisheim, Kelly and Malamud.
- RQ5: Jensen, Kelly, Malamud and Pedersen; Kyle and Obizhaeva; liquidity and regime/stress literature.

## 5. Significance and Contributions

### Purpose

State what is new about the dissertation without overstating originality.

### Contributions to claim

1. European evidence: applies modern ML and deep asset-pricing tools to European equities rather than relying on U.S.-centric evidence.
2. Data depth: extends a compact characteristic panel with Compustat Global and Refinitiv liquidity information.
3. Method comparison: evaluates classical ML, DRE, neural SDFs, adversarial SDFs, AIPM transformers, autoencoder, IPCA, fundamental mispricing, LambdaRank, and deep sequence models in one coherent empirical design.
4. Implementability: tests signals under bid-ask spreads, square-root impact, turnover, AUM stress, liquidity universes, and portfolio concentration constraints.
5. Main empirical contribution: documents a predictability-implementability gap in European equities.

### Contributions not to claim

- Do not claim a new theoretical asset-pricing model.
- Do not claim a new neural architecture.
- Do not claim robust ML dominance over momentum.
- Do not claim liquidity causes predictability.

## 6. Dissertation Outline

Suggested outline paragraph:

Chapter 1 introduces the research question and the predictability-implementability gap. Chapter 2 reviews the literature on empirical asset pricing, machine learning, deep asset-pricing models, market frictions, and regime dependence. Chapter 3 describes the European data, feature construction, leakage controls, and walk-forward design. Chapter 4 presents the baseline ML prediction and portfolio results. Chapter 5 develops the implementability analysis, including liquidity ladders, transaction costs, capacity, and constrained long-only construction. Chapter 6 reports advanced model extensions, including SDFs, AIPM, autoencoder/IPCA, fundamental mispricing, LambdaRank, and deep sequence models. Chapter 7 concludes and discusses limitations and future research.

## 7. Literature Review - Introduction

### Purpose

Explain how the literature review is organised and why.

### Recommended framing

The literature review should not be a chronological list. Organise it around the dissertation's causal chain:

1. Expected returns and factor structure.
2. Machine learning return prediction.
3. Deep and no-arbitrage asset-pricing models.
4. Portfolio implementation under frictions.
5. State dependence, volatility, and stress.
6. The unresolved gap: European implementability.

## 8. Literature Review - Theoretical Framework

### Purpose

Define the conceptual framework used in the dissertation.

### Core framework

The dissertation combines:

- Empirical asset pricing: characteristics may forecast returns because they proxy for risk or mispricing.
- Statistical learning: nonlinear models may capture interactions among characteristics.
- No-arbitrage/SDF thinking: expected returns can be represented through pricing kernels or factor structures.
- Implementation frictions: gross expected returns are not equivalent to attainable investor returns.
- Limits to arbitrage: signals may be strongest where arbitrage is costly.

### Papers to cite here

Asset-pricing foundations:

- Fama and French (2004), CAPM theory and evidence.
- Pedersen, Big Data Asset Pricing, empirical primer and factor zoo lectures.
- Merton (1974), On the Pricing of Corporate Debt, if discussing structural risk/default motivation only.
- Kelly, Pruitt and Su, Characteristics are Covariances (IPCA).

No-arbitrage and SDF/deep framework:

- Gu, Kelly and Xiu, Autoencoder Asset Pricing Models.
- Chen, Pelger and Zhu (2024), Deep Learning in Asset Pricing.
- Kelly, Kuznetsov, Malamud and Xu (2026), Artificial Intelligence Asset Pricing Models.

Frictions:

- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.
- Kyle and Obizhaeva (2016), Market Microstructure Invariance.
- Pedersen, Asset Pricing with Frictions.

## 9. Literature Review - Empirical Literature

Organise this subsection by themes.

### 9.1 Classical empirical asset pricing and factor investing

Use this to establish the benchmark tradition and why momentum is a serious comparator.

Papers:

- Fama and French (2004), The Capital Asset Pricing Model: Theory and Evidence.
- Pedersen, Big Data Asset Pricing, empirical asset pricing primer.
- Pedersen, Factor Zoo and Replication.
- Asness, Frazzini and Pedersen (2012), Leverage Aversion and Risk Parity.
- Frazzini and Pedersen (2022), Embedded Leverage.
- Swade, Lohre, Nolte and Shackleton (2024), A Century of Macro Factor Investing.
- Bauer, Haerden and Molenaar, Asset Allocation in Stable and Unstable Times.

### 9.2 Machine learning in asset pricing and allocation

Use this to justify nonlinear models and broad predictor sets.

Papers:

- Gu, Kelly and Xiu, empirical machine learning asset pricing work.
- Kelly and Xiu, Financial Machine Learning.
- Pedersen, Machine Learning in Asset Pricing.
- Emerson, Kennedy, O'Shea and O'Brien, Trends and Applications of Machine Learning in Quantitative Finance.
- Routledge (2019), Machine Learning and Asset Allocation.
- Jurczenko (2020), Machine Learning Optimization Algorithms and Portfolio Allocation.
- Cho (2025), Novel approach for deep learning-based market forecasting and portfolio selection incorporating market efficiency.
- Konstantinov, Chorus and Rebmann, A Network and Machine Learning Approach to Factor, Asset, and Blended Allocation.
- Machine Learning for Recession Prediction and Dynamic Asset Allocation.
- Tzikas et al., Enhancing a Risk Model by Adding Transient Statistical Factors.

### 9.3 Deep, no-arbitrage, and Kelly-style asset pricing

Use this as the direct literature foundation for the advanced model chapters.

Papers:

- Didisheim, Kelly and Malamud (2022), deep regression ensemble literature.
- Chen, Pelger and Zhu (2024), Deep Learning in Asset Pricing.
- Kelly, Kuznetsov, Malamud and Xu (2026), Artificial Intelligence Asset Pricing Models.
- Gu, Kelly and Xiu, Autoencoder Asset Pricing Models.
- Kelly, Pruitt and Su, Characteristics are Covariances (IPCA).
- Kelly, Malamud and Pedersen, Principal Portfolios.
- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.

### 9.4 European fundamental mispricing and accounting signals

Use this to support the fundamental-mispricing extension and the role of accounting data depth.

Papers:

- Hanauer, Kononova and Rapp (2022), Boosting agnostic fundamental analysis: using machine learning to identify mispricing in European stock markets.
- Compustat/Worldscope accounting feature literature should be linked to the GKX data-depth tradition.
- Gomez-Cram, Guo, Jensen and Kung (2026), Financial Prediction Markets: A New Measure of Earnings Expectations, only if analyst-expectations features are discussed.

### 9.5 Implementation, market microstructure, and capacity

Use this to motivate spreads, market impact, AUM stress, and turnover-aware construction.

Papers:

- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.
- Kyle and Obizhaeva (2016), Market Microstructure Invariance.
- Pedersen, Asset Pricing with Frictions.
- Boyd, Market Making and Risk Management in Options Markets.
- Lo and Medda, Uniswap and the Emergence of the Decentralized Exchange, only as peripheral digital-market contrast.

### 9.6 Volatility, state dependence, and stress

Use this as supporting context for regime diagnostics and the finding that ML predictability weakens in down markets. Keep it concise because the dissertation is no longer a regime-switching dissertation.

Papers:

- Bollerslev, Tauchen and Zhou (2009), Expected Stock Returns and Variance Risk Premia.
- Paye (2012), Deja Vol: Predictive Regressions for Aggregate Stock Market Volatility.
- Christiansen, Schmeling and Schrimpf (2012), A Comprehensive Look at Financial Volatility Prediction by Economic Variables.
- Bollerslev, Patton and Quaedvlieg (2020), Realized Semibetas: Signs of Things to Come.
- Ang and Bekaert (2002), International Asset Allocation with Regime Shifts.
- Ang and Bekaert (2004), How Regimes Affect Asset Allocation.
- Guidolin and Timmermann (2007), Asset Allocation under Multivariate Regime Switching.
- Guidolin and Timmermann (2008), International Asset Allocation under Regime Switching, Skew and Kurtosis.
- Kritzman, Page and Turkington (2012), Regime Shifts and Dynamic Strategies.
- Galvao and Owyang (2018), Financial Stress Regimes and the Macroeconomy.
- Lai (2022), Detecting Stock Market Regimes from Option Prices.
- Delatte, Fouquau and Portes (2017), Regime-Dependent Sovereign Risk Pricing During the Euro Crisis.
- Abdymomunov (2013), Regime-Switching Measure of Systemic Financial Stress.
- Semmler and Chen (2014), Financial Stress, Regime Switching and Macrodynamics.
- Zhang and Yi, Explainable Machine Learning for Regime-Based Asset Allocation.
- Shu and Mulvey (2024), Dynamic Factor Allocation via Regime-Switching Signals.
- Aydinhan, Kolm, Mulvey and Shu (2024), Identifying Patterns in Financial Markets: Extending the Statistical Jump Model for Regime Identification.
- Shu, Yu and Mulvey (2024), Downside Risk Reduction Using Regime-Switching Signals.
- Shu, Yu and Mulvey (2024), Dynamic Asset Allocation with Asset-Specific Regime Forecasts.
- Xiong, Shu, Zhou, Ding and Li (2026), Regime-Aware Allocation for Robust Multi-Asset Portfolio Management.

### 9.7 Cross-asset correlation and stress testing

Use only if the introduction motivates why stress states matter. Do not let this become the centre of the review.

Papers:

- Bansal, Connolly and Stivers (2009), Regime-Switching in Stock Index and Treasury Futures Returns and Measures of Stock Market Stress.
- Ommen (2026), When Bonds Stop Hedging Stocks: Time-Varying Correlation and the Inflation Regime of 2022.
- Bernhart et al. (2011), Asset Correlations in Turbulent Markets and the Impact of Different Regimes on Asset Management.
- De Santis and Stein (2014), Financial Indicators Signalling Correlation Changes in Sovereign Bond Markets.
- Piplack and Straetmans (2010), Comovements of Different Asset Classes During Market Stress.
- Campbell, Sunderam and Viceira (2007), Inflation Bets or Deflation Hedges.
- Acharya, Engle and Pierret (2014), Testing Macroprudential Stress Tests.
- Garcia-de-Andoain and Kremer (2017), Beyond Spreads: Measuring Sovereign Market Stress in the Euro Area.
- Haerdle, Wang and Yu (2016), TENET: Tail-Event Driven Network Risk.
- Sorge (2004), Stress-Testing Financial Systems: An Overview of Current Methodologies.
- Bhansali and Wise (2001), Forecasting Portfolio Risk in Normal and Stressed Markets.
- Bangia, Diebold, Kronimus, Schagen and Schuermann (2002), Ratings Migration and the Business Cycle.

### 9.8 Credit, default, and digital-finance adjacent papers

These are peripheral for the current dissertation. Mention only if needed to motivate broader financial risk modelling, not in the main empirical literature.

Papers:

- Merton (1974), On the Pricing of Corporate Debt: The Risk Structure of Interest Rates.
- Shumway, Forecasting Default with the Merton Distance to Default Model.
- Das, Duffie, Kapadia and Saita (2007), Common Failings: How Corporate Defaults Are Correlated.
- Anginer and Yildizhan (2018), systematic default risk and equity returns.
- Mertens, Mota and Nobrega (2026), Betting on Credit Betas.
- Cong and He (2026), Heterogeneity, Tokenization, and Wealth Dynamics in the Digital Economy.
- Li, Oh and Ricciardi (2026), Rating Without Market Discipline.
- Aldridge et al. (2025/2026), Agentic Artificial Intelligence in Finance: A Comprehensive Survey.

### 9.9 Other reviewed risk-model papers

Use only if a specific paragraph needs them.

- Alexander and Fabozzi (2026), On the Structure of Risk Contribution.
- Byun, Loudis and Schmidt (2026), A Tale of Two Market Returns.
- Bollerslev, Patton and Quaedvlieg (2020), Realized Semibetas.

## 10. Literature Review - Research Gap

### Gap statement

The literature has developed strong evidence that ML can predict equity returns and that deep asset-pricing architectures can be estimated. A separate literature studies portfolio frictions and implementable efficient frontiers. The gap is that less evidence exists on whether modern ML and deep asset-pricing signals remain economically useful in European equities after those frictions are integrated into the empirical design.

### Specific gaps

1. Geographic gap: U.S.-centric ML asset-pricing evidence dominates; Europe is less studied.
2. Implementation gap: prediction papers often report IC/R2 or gross portfolios without full capacity and liquidity diagnostics.
3. Architecture gap: advanced models such as CPZ-style SDFs, AIPM transformers, autoencoders, IPCA, and DRE have not been jointly stress-tested in a European implementability framework.
4. Portfolio-construction gap: less attention is paid to how turnover smoothing, liquidity rungs, AUM, and concentration caps mediate model performance.
5. Inference gap: impressive point estimates often weaken under common-sample tests, paired bootstrap inference, and Holm correction.

### Link to dissertation contribution

This dissertation addresses the gap by treating implementability as the object of study rather than as an afterthought.

## 11. Hypothesis Development

The hypotheses should be directional but not too strong.

### H1: Predictive information

H1. Machine-learning scores are positively associated with next-month European equity returns out of sample.

Expected support:

- Positive ICs.
- Placebo rejection.
- Positive Fama-MacBeth slopes.

Literature:

- Gu, Kelly and Xiu.
- Kelly and Xiu.
- Pedersen ML lecture.

### H2: Incremental information

H2. ML scores contain incremental information beyond conventional characteristics and risk controls.

Expected support:

- Positive Fama-MacBeth slopes after controlling for momentum, size, value, beta, idiosyncratic volatility, country, and sector.

Literature:

- Fama and French.
- Pedersen factor zoo.
- GKX-style ML asset pricing.

### H3: Implementable performance

H3. The current empirical design does not identify a statistically robust value-weighted net advantage for ML portfolios over momentum after costs, liquidity, and multiple-testing correction.

This is phrased as the dissertation's expected empirical tension. If the university prefers alternative hypotheses in positive form, write:

H3a. ML portfolios generate higher gross returns than momentum in selected portfolio formats.

H3b. The nominal ML-minus-momentum advantage attenuates after value weighting, transaction costs, liquidity restrictions, and multiple-testing correction.

Literature:

- Jensen, Kelly, Malamud and Pedersen.
- Kyle and Obizhaeva.
- Pedersen frictions.

### H4: Data-depth and architecture

H4. Data-depth and advanced deep architectures improve selected predictive or gross-performance metrics but do not necessarily improve implementable net performance.

Literature:

- Chen, Pelger and Zhu.
- Kelly, Kuznetsov, Malamud and Xu.
- Gu, Kelly and Xiu autoencoder.
- Kelly, Pruitt and Su.
- Didisheim, Kelly and Malamud.

### H5: Limits-to-arbitrage channel

H5. Predictability is stronger in less implementable segments, such as smaller, more volatile, or less liquid stocks, and weakens when portfolios are constrained toward larger, lower-spread, diversified universes.

Literature:

- Jensen, Kelly, Malamud and Pedersen.
- Pedersen frictions.
- Kyle and Obizhaeva.
- Regime/stress and volatility papers as support for state dependence.

## 12. Literature Review - Summary

The summary should do three things:

1. State that the literature supports using ML and deep models for return prediction.
2. State that the literature also shows implementation frictions and state dependence can materially change economic value.
3. State that the dissertation combines these strands in a European setting and tests the full path from prediction to implementable portfolio performance.

Suggested closing sentence:

The literature therefore motivates a research design in which the success of machine learning is judged not only by out-of-sample forecast accuracy, but by whether forecasts survive the economic discipline of liquidity, turnover, transaction costs, capacity, diversification, and benchmark comparison.

## 13. Full Reviewed-Paper Inventory

This inventory combines the papers visible in the project folder and the core papers embedded in the current code/results notes. Use the primary/core papers heavily; use older regime, stress, credit, and digital-finance papers selectively.

### Core current-dissertation papers

- Gu, Kelly and Xiu, empirical machine learning asset pricing work.
- Gu, Kelly and Xiu, Autoencoder Asset Pricing Models.
- Kelly and Xiu, Financial Machine Learning.
- Jensen, Kelly, Malamud and Pedersen (2022), Machine Learning and the Implementable Efficient Frontier.
- Chen, Pelger and Zhu (2024), Deep Learning in Asset Pricing.
- Kelly, Kuznetsov, Malamud and Xu (2026), Artificial Intelligence Asset Pricing Models.
- Didisheim, Kelly and Malamud (2022), deep regression ensemble literature.
- Kelly, Pruitt and Su, Characteristics are Covariances (IPCA).
- Kelly, Malamud and Pedersen, Principal Portfolios.
- Hanauer, Kononova and Rapp (2022), Boosting agnostic fundamental analysis.
- Pedersen, Big Data Asset Pricing lectures: empirical primer, factor zoo and replication, machine learning, frictions.
- Fama and French (2004), The Capital Asset Pricing Model: Theory and Evidence.

### ML, allocation, and portfolio construction papers

- Emerson, Kennedy, O'Shea and O'Brien, Trends and Applications of Machine Learning in Quantitative Finance.
- Routledge (2019), Machine Learning and Asset Allocation.
- Jurczenko (2020), Machine Learning Optimization Algorithms and Portfolio Allocation.
- Cho (2025), Novel approach for deep learning-based market forecasting and portfolio selection incorporating market efficiency.
- Konstantinov, Chorus and Rebmann, A Network and Machine Learning Approach to Factor, Asset, and Blended Allocation.
- Machine Learning for Recession Prediction and Dynamic Asset Allocation.
- Jensen and Pedersen (2026), Big Data Asset Pricing Exercises.
- Tzikas et al., Enhancing a Risk Model by Adding Transient Statistical Factors.
- Gomez-Cram, Guo, Jensen and Kung (2026), Financial Prediction Markets: A New Measure of Earnings Expectations.

### Factor investing and allocation papers

- Asness, Frazzini and Pedersen (2012), Leverage Aversion and Risk Parity.
- Frazzini and Pedersen (2022), Embedded Leverage.
- Swade, Lohre, Nolte and Shackleton (2024), A Century of Macro Factor Investing.
- Bauer, Haerden and Molenaar, Asset Allocation in Stable and Unstable Times.
- Alexander and Fabozzi (2026), On the Structure of Risk Contribution.
- Byun, Loudis and Schmidt (2026), A Tale of Two Market Returns.

### Volatility and variance-risk papers

- Bollerslev, Tauchen and Zhou (2009), Expected Stock Returns and Variance Risk Premia.
- Paye (2012), Deja Vol: Predictive Regressions for Aggregate Stock Market Volatility.
- Christiansen, Schmeling and Schrimpf (2012), A Comprehensive Look at Financial Volatility Prediction by Economic Variables.
- Bollerslev, Patton and Quaedvlieg (2020), Realized Semibetas: Signs of Things to Come.

### Regime-switching and market-regime papers

- Ang and Bekaert (2002), International Asset Allocation with Regime Shifts.
- Ang and Bekaert (2004), How Regimes Affect Asset Allocation.
- Guidolin and Timmermann (2007), Asset Allocation under Multivariate Regime Switching.
- Guidolin and Timmermann (2008), International Asset Allocation under Regime Switching, Skew and Kurtosis.
- Galvao and Owyang (2018), Financial Stress Regimes and the Macroeconomy.
- Lai (2022), Detecting Stock Market Regimes from Option Prices.
- Delatte, Fouquau and Portes (2017), Regime-Dependent Sovereign Risk Pricing During the Euro Crisis.
- Abdymomunov (2013), Regime-Switching Measure of Systemic Financial Stress.
- Kritzman, Page and Turkington (2012), Regime Shifts and Dynamic Strategies.
- Semmler and Chen (2014), Financial Stress, Regime Switching and Macrodynamics.
- Zhang and Yi, Explainable Machine Learning for Regime-Based Asset Allocation.
- Shu and Mulvey (2024), Dynamic Factor Allocation via Regime-Switching Signals.
- Aydinhan, Kolm, Mulvey and Shu (2024), Identifying Patterns in Financial Markets: Extending the Statistical Jump Model for Regime Identification.
- Shu, Yu and Mulvey (2024), Downside Risk Reduction Using Regime-Switching Signals.
- Shu, Yu and Mulvey (2024), Dynamic Asset Allocation with Asset-Specific Regime Forecasts.
- Xiong, Shu, Zhou, Ding and Li (2026), Regime-Aware Allocation for Robust Multi-Asset Portfolio Management.

### Cross-asset correlation and stress-testing papers

- Bansal, Connolly and Stivers (2009), Regime-Switching in Stock Index and Treasury Futures Returns and Measures of Stock Market Stress.
- Ommen (2026), When Bonds Stop Hedging Stocks: Time-Varying Correlation and the Inflation Regime of 2022.
- Bernhart et al. (2011), Asset Correlations in Turbulent Markets and the Impact of Different Regimes on Asset Management.
- De Santis and Stein (2014), Financial Indicators Signalling Correlation Changes in Sovereign Bond Markets.
- Piplack and Straetmans (2010), Comovements of Different Asset Classes During Market Stress.
- Campbell, Sunderam and Viceira (2007), Inflation Bets or Deflation Hedges.
- Acharya, Engle and Pierret (2014), Testing Macroprudential Stress Tests.
- Garcia-de-Andoain and Kremer (2017), Beyond Spreads: Measuring Sovereign Market Stress in the Euro Area.
- Haerdle, Wang and Yu (2016), TENET: Tail-Event Driven Network Risk.
- Sorge (2004), Stress-Testing Financial Systems: An Overview of Current Methodologies.
- Bhansali and Wise (2001), Forecasting Portfolio Risk in Normal and Stressed Markets.
- Bangia, Diebold, Kronimus, Schagen and Schuermann (2002), Ratings Migration and the Business Cycle.

### Market microstructure, credit, and digital-finance adjacent papers

- Kyle and Obizhaeva (2016), Market Microstructure Invariance.
- Boyd, Market Making and Risk Management in Options Markets.
- Lo and Medda, Uniswap and the Emergence of the Decentralized Exchange.
- Merton (1974), On the Pricing of Corporate Debt: The Risk Structure of Interest Rates.
- Shumway, Forecasting Default with the Merton Distance to Default Model.
- Das, Duffie, Kapadia and Saita (2007), Common Failings: How Corporate Defaults Are Correlated.
- Anginer and Yildizhan (2018), systematic default risk and equity returns.
- Mertens, Mota and Nobrega (2026), Betting on Credit Betas.
- Cong and He (2026), Heterogeneity, Tokenization, and Wealth Dynamics in the Digital Economy.
- Li, Oh and Ricciardi (2026), Rating Without Market Discipline.
- Aldridge et al., Agentic Artificial Intelligence in Finance: A Comprehensive Survey.
