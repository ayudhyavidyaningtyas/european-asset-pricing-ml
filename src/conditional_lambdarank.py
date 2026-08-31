"""Conditional multi-horizon LambdaRank for implementable European equities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from asset_pricing import RAW_FEATURES
from implementable_frontier import (
    FrontierConfig,
    attach_execution_inputs,
    load_eur_risk_free,
    load_monthly_liquidity,
)


RANK_FEATURES = [f"{feature}_rank" for feature in RAW_FEATURES]
STATE_FEATURES = [
    "market_return_eur",
    "market_trend_12m",
    "market_volatility_12m",
    "eur_short_rate",
    "eur_rate_change_3m",
    "aggregate_turnover",
]
CATEGORICAL_FEATURES = ["country_code", "sector_code"]
HORIZONS = (1, 3, 6, 12)
THEMES = {
    "size": ["log_size_rank", "market_cap_growth_12m_rank"],
    "value": ["book_to_market_rank"],
    "momentum_reversal": [
        "return_1m_rank",
        "momentum_6_2_rank",
        "momentum_12_2_rank",
        "max_return_12m_rank",
    ],
    "volatility": ["volatility_12m_rank"],
    "liquidity": ["turnover_1m_rank", "turnover_12m_rank"],
    "investment_growth": [
        "asset_growth_rank",
        "sales_growth_rank",
        "capex_to_assets_rank",
    ],
    "profitability_quality": [
        "profitability_roa_rank",
        "operating_profitability_rank",
        "cashflow_to_assets_rank",
    ],
    "balance_sheet": ["leverage_rank", "accruals_rank"],
}
BLENDS = {
    "near_term": np.array([1.0, 0.0, 0.0, 0.0]),
    "balanced": np.array([0.4, 0.3, 0.2, 0.1]),
    "medium": np.array([0.2, 0.4, 0.3, 0.1]),
    "persistent": np.array([0.1, 0.2, 0.3, 0.4]),
    "equal": np.array([0.25, 0.25, 0.25, 0.25]),
}


@dataclass(frozen=True)
class LambdaRankConfig:
    first_test_year: int = 2008
    last_test_year: int = 2026
    validation_months: int = 24
    blend_validation_months: int = 36
    relevance_bins: int = 10
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 200
    reg_lambda: float = 10.0
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    bagging_freq: int = 1
    random_state: int = 42
    minimum_training_rows: int = 10_000
    hypothetical_trade_weight: float = 0.01
    cost_aum_eur: float = 100_000_000.0
    impact_coefficient: float = 0.10
    utility_risk_aversion: float = 3.0


def _month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def build_state_panel(
    panel: pd.DataFrame,
    market_states_path: Path,
    eur_rate_path: Path,
) -> pd.DataFrame:
    states = pd.read_csv(market_states_path, parse_dates=["signal_date"]).rename(
        columns={"signal_date": "date"}
    )
    states["date"] = _month_end(states["date"])
    rate = load_eur_risk_free(eur_rate_path)
    annualized_rate = ((1.0 + rate) ** 12 - 1.0).rename("eur_short_rate")
    state_dates = pd.DatetimeIndex(states["date"].dropna().sort_values().unique())
    annualized_rate = annualized_rate.reindex(state_dates).ffill(limit=6)
    states = states.merge(
        annualized_rate.reset_index().rename(columns={"index": "date"}),
        on="date",
        how="left",
    )
    states["eur_rate_change_3m"] = states["eur_short_rate"].diff(3)
    liquidity = (
        panel[panel["model_eligible"].fillna(False)]
        .assign(log_turnover=lambda frame: np.log1p(frame["turnover_12m"].clip(lower=0)))
        .groupby("date")["log_turnover"]
        .median()
        .rename("aggregate_turnover")
    )
    return states.merge(liquidity.reset_index(), on="date", how="left")


def _forward_sum(
    values: pd.Series,
    dates: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    reversed_sum = values.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    end_date = dates.shift(-(horizon - 1))
    expected = dates + pd.offsets.MonthEnd(horizon - 1)
    reversed_sum = reversed_sum.where(end_date.eq(expected))
    return reversed_sum, end_date


def _within_month_relevance(values: pd.Series, dates: pd.Series, bins: int) -> pd.Series:
    percentile = values.groupby(dates).rank(method="average", pct=True)
    relevance = np.floor(percentile * bins).clip(0, bins - 1)
    return relevance.where(values.notna()).astype("Int16")


def prepare_ranking_panel(
    panel_path: Path,
    risk_path: Path,
    market_states_path: Path,
    market_return_path: Path,
    eur_rate_path: Path,
    liquidity_path: Path | None,
    config: LambdaRankConfig,
) -> pd.DataFrame:
    columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "company_market_cap",
        "market_cap_percentile",
        "turnover_12m",
        "model_eligible",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        *RANK_FEATURES,
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = _month_end(panel["date"])
    panel["target_date"] = _month_end(panel["target_date"])
    risk = pd.read_parquet(
        risk_path,
        columns=["date", "ric", "beta_36m", "idio_vol_36m", "risk_nobs"],
    )
    risk["date"] = _month_end(risk["date"])
    panel = panel.merge(risk, on=["date", "ric"], how="left", validate="one_to_one")

    states = build_state_panel(panel, market_states_path, eur_rate_path)
    panel = panel.merge(states[["date", *STATE_FEATURES]], on="date", how="left")
    market = pd.read_csv(market_return_path, parse_dates=["date"])
    market["target_date"] = _month_end(market["date"])
    panel = panel.merge(
        market[["target_date", "market_return_eur"]].rename(
            columns={"market_return_eur": "target_market_return"}
        ),
        on="target_date",
        how="left",
    )

    liquidity = load_monthly_liquidity(liquidity_path)
    execution_config = FrontierConfig(
        fallback_half_spread_bps=25.0,
        impact_coefficient=config.impact_coefficient,
    )
    panel = attach_execution_inputs(panel, liquidity, execution_config)
    participation = (
        config.hypothetical_trade_weight
        * config.cost_aum_eur
        / panel["adv_eur"].clip(lower=1.0)
    )
    impact_bps = (
        config.impact_coefficient
        * panel["idio_vol_36m"].div(np.sqrt(21.0))
        * np.sqrt(participation.clip(lower=0))
        * 10_000.0
    )
    panel["expected_trade_cost"] = (
        panel["half_spread_bps"] + impact_bps
    ) / 10_000.0

    market_residual = (
        panel["target_return_1m"]
        - panel["beta_36m"] * panel["target_market_return"]
    )
    overall = market_residual.groupby(panel["target_date"]).transform("mean")
    sector = market_residual.groupby(
        [panel["target_date"], panel["TR.TRBCECONOMICSECTOR"]]
    ).transform("mean")
    country = market_residual.groupby(
        [panel["target_date"], panel["screen_country"]]
    ).transform("mean")
    panel["residual_return_1m"] = market_residual - sector - country + overall
    panel["net_residual_1m"] = (
        panel["residual_return_1m"] - panel["expected_trade_cost"]
    )

    panel["country_code"] = panel["screen_country"].fillna("Unknown").astype("category")
    panel["sector_code"] = (
        panel["TR.TRBCECONOMICSECTOR"].fillna("Unknown").astype("category")
    )
    panel = panel.sort_values(["ric", "date"]).reset_index(drop=True)
    for horizon in HORIZONS:
        values = []
        end_dates = []
        for _, group in panel.groupby("ric", sort=False):
            forward, end_date = _forward_sum(
                group["residual_return_1m"], group["target_date"], horizon
            )
            values.append(forward)
            end_dates.append(end_date)
        panel[f"residual_return_{horizon}m"] = pd.concat(values).sort_index()
        panel[f"target_end_date_{horizon}m"] = pd.concat(end_dates).sort_index()
        panel[f"net_residual_{horizon}m"] = (
            panel[f"residual_return_{horizon}m"] - panel["expected_trade_cost"]
        ) / horizon
        panel[f"relevance_{horizon}m"] = _within_month_relevance(
            panel[f"net_residual_{horizon}m"], panel["date"], config.relevance_bins
        )
    return panel.sort_values(["date", "ric"]).reset_index(drop=True)


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("date", sort=False).size().astype(int).tolist()


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    log_cap = np.log(frame["company_market_cap"].clip(lower=1.0))
    weight = log_cap.groupby(frame["date"]).rank(pct=True).pow(0.5)
    return weight.clip(lower=0.25, upper=1.0).to_numpy(dtype=float)


def _feature_names(conditional: bool) -> list[str]:
    return [
        *RANK_FEATURES,
        *(STATE_FEATURES if conditional else []),
        *CATEGORICAL_FEATURES,
    ]


def fit_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    label: str,
    config: LambdaRankConfig,
) -> tuple[lgb.LGBMRanker, int]:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(config.relevance_bins)),
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        reg_lambda=config.reg_lambda,
        feature_fraction=config.feature_fraction,
        bagging_fraction=config.bagging_fraction,
        bagging_freq=config.bagging_freq,
        random_state=config.random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        train[features],
        train[label].astype(int),
        group=_group_sizes(train),
        sample_weight=_sample_weights(train),
        categorical_feature=categorical,
        eval_set=[(validation[features], validation[label].astype(int))],
        eval_group=[_group_sizes(validation)],
        eval_at=[10, 50],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    best_iteration = int(model.best_iteration_ or config.n_estimators)
    final = model.set_params(n_estimators=best_iteration)
    full = pd.concat([train, validation], ignore_index=True).sort_values(
        ["date", "ric"]
    )
    final.fit(
        full[features],
        full[label].astype(int),
        group=_group_sizes(full),
        sample_weight=_sample_weights(full),
        categorical_feature=categorical,
    )
    return final, best_iteration


def run_walk_forward_rankers(
    panel: pd.DataFrame,
    config: LambdaRankConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple, lgb.LGBMRanker],
    pd.DataFrame,
]:
    predictions = []
    logs = []
    theme_predictions = []
    fitted: dict[tuple, lgb.LGBMRanker] = {}
    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(year - 1, 12, 31)
        test = panel[
            panel["target_date"].dt.year.eq(year)
            & panel["model_eligible"].fillna(False)
        ].copy()
        if test.empty:
            continue
        for horizon in HORIZONS:
            label = f"relevance_{horizon}m"
            eligible = panel[
                panel["model_eligible"].fillna(False)
                & panel[f"target_end_date_{horizon}m"].le(cutoff)
                & panel[label].notna()
            ].copy()
            if len(eligible) < config.minimum_training_rows:
                raise ValueError(
                    f"Insufficient {horizon}m training rows for {year}: {len(eligible)}"
                )
            dates = sorted(eligible["date"].drop_duplicates())
            validation_dates = dates[-config.validation_months :]
            train = eligible[~eligible["date"].isin(validation_dates)].sort_values(
                ["date", "ric"]
            )
            validation = eligible[
                eligible["date"].isin(validation_dates)
            ].sort_values(["date", "ric"])
            for conditional in [False, True]:
                features = _feature_names(conditional)
                complete_train = train.dropna(subset=features)
                complete_validation = validation.dropna(subset=features)
                complete_test = test.dropna(subset=features)
                model, best_iteration = fit_ranker(
                    complete_train,
                    complete_validation,
                    features,
                    label,
                    config,
                )
                name = "conditional" if conditional else "unconditional"
                scored = complete_test[
                    [
                        "date",
                        "target_date",
                        "ric",
                        "target_return_1m",
                        "residual_return_1m",
                        "net_residual_1m",
                        "expected_trade_cost",
                        "company_market_cap",
                    ]
                ].copy()
                scored["horizon"] = horizon
                scored["model_variant"] = name
                scored["prediction"] = model.predict(
                    complete_test[features], num_iteration=best_iteration
                )
                predictions.append(scored)
                if conditional and horizon == 1:
                    baseline_prediction = scored["prediction"].to_numpy()
                    for theme_index, (theme, theme_features) in enumerate(
                        THEMES.items()
                    ):
                        permuted = complete_test[features].copy()
                        rng = np.random.default_rng(
                            config.random_state + year * 100 + theme_index
                        )
                        for feature in theme_features:
                            permuted[feature] = permuted.groupby(
                                complete_test["date"], observed=True
                            )[feature].transform(
                                lambda values: rng.permutation(values.to_numpy())
                            )
                        theme_frame = scored[
                            [
                                "date",
                                "target_date",
                                "ric",
                                "residual_return_1m",
                                "expected_trade_cost",
                            ]
                        ].copy()
                        theme_frame["theme"] = theme
                        theme_frame["baseline_prediction"] = baseline_prediction
                        theme_frame["permuted_prediction"] = model.predict(
                            permuted, num_iteration=best_iteration
                        )
                        theme_predictions.append(theme_frame)
                logs.append(
                    {
                        "test_year": year,
                        "horizon": horizon,
                        "model_variant": name,
                        "train_rows": len(complete_train) + len(complete_validation),
                        "train_months": eligible["date"].nunique(),
                        "validation_months": len(validation_dates),
                        "test_rows": len(complete_test),
                        "label_cutoff": cutoff,
                        "maximum_train_target_end": eligible[
                            f"target_end_date_{horizon}m"
                        ].max(),
                        "best_iteration": best_iteration,
                    }
                )
                fitted[(year, horizon, name)] = model
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(logs),
        fitted,
        pd.concat(theme_predictions, ignore_index=True),
    )


def _rank_portfolio_utility(
    frame: pd.DataFrame,
    score_column: str,
    gamma: float,
) -> tuple[float, float, float, int]:
    prior: dict[str, float] = {}
    net_returns = []
    turnovers = []
    for _, month in frame.groupby("target_date", sort=True):
        month = month.dropna(
            subset=[score_column, "residual_return_1m", "expected_trade_cost"]
        ).copy()
        if len(month) < 100:
            continue
        rank = month[score_column].rank(pct=True)
        long_rics = month.loc[rank.ge(0.9), "ric"].astype(str)
        short_rics = month.loc[rank.le(0.1), "ric"].astype(str)
        weights = {
            **{ric: 0.5 / len(long_rics) for ric in long_rics},
            **{ric: -0.5 / len(short_rics) for ric in short_rics},
        }
        all_rics = set(prior) | set(weights)
        delta = {ric: weights.get(ric, 0.0) - prior.get(ric, 0.0) for ric in all_rics}
        cost_map = month.set_index("ric")["expected_trade_cost"].to_dict()
        median_cost = float(month["expected_trade_cost"].median())
        trading_cost = sum(
            abs(change) * float(cost_map.get(ric, median_cost))
            for ric, change in delta.items()
        )
        return_map = month.set_index("ric")["residual_return_1m"].to_dict()
        gross_return = sum(
            weight * float(return_map.get(ric, 0.0))
            for ric, weight in weights.items()
        )
        net_returns.append(gross_return - trading_cost)
        turnovers.append(sum(abs(change) for change in delta.values()))
        prior = weights
    values = pd.Series(net_returns)
    annual_mean = float(values.mean() * 12)
    annual_variance = float(values.var(ddof=1) * 12)
    utility = annual_mean - 0.5 * gamma * annual_variance
    return utility, annual_mean, float(np.mean(turnovers)), len(values)


def economic_theme_importance(
    theme_predictions: pd.DataFrame,
    config: LambdaRankConfig,
) -> pd.DataFrame:
    records = []
    for start_year in [2008, 2010, 2015]:
        sample = theme_predictions[
            theme_predictions["target_date"].dt.year.ge(start_year)
        ]
        for theme, group in sample.groupby("theme"):
            baseline = _rank_portfolio_utility(
                group,
                "baseline_prediction",
                config.utility_risk_aversion,
            )
            permuted = _rank_portfolio_utility(
                group,
                "permuted_prediction",
                config.utility_risk_aversion,
            )
            records.append(
                {
                    "start_year": start_year,
                    "theme": theme,
                    "months": baseline[3],
                    "baseline_utility": baseline[0],
                    "permuted_utility": permuted[0],
                    "economic_importance": baseline[0] - permuted[0],
                    "baseline_annual_net_return": baseline[1],
                    "permuted_annual_net_return": permuted[1],
                    "baseline_monthly_turnover": baseline[2],
                    "permuted_monthly_turnover": permuted[2],
                }
            )
    return pd.DataFrame(records)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in predictions.groupby(["model_variant", "horizon"]):
        net_monthly = group.groupby("target_date").apply(
            lambda frame: spearmanr(
                frame["prediction"], frame["net_residual_1m"], nan_policy="omit"
            ).statistic,
            include_groups=False,
        )
        gross_monthly = group.groupby("target_date").apply(
            lambda frame: spearmanr(
                frame["prediction"], frame["residual_return_1m"], nan_policy="omit"
            ).statistic,
            include_groups=False,
        )
        records.append(
            {
                "model_variant": keys[0],
                "horizon": keys[1],
                "months": net_monthly.notna().sum(),
                "mean_monthly_gross_residual_ic": gross_monthly.mean(),
                "mean_monthly_net_residual_ic": net_monthly.mean(),
                "net_ic_information_ratio": net_monthly.mean()
                / net_monthly.std(ddof=1)
                * np.sqrt(12),
                "positive_net_ic_share": net_monthly.gt(0).mean(),
            }
        )
    return pd.DataFrame(records)


def _rank_predictions(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("target_date")["prediction"].rank(pct=True)


def _blend_utility(frame: pd.DataFrame, weights: np.ndarray, gamma: float) -> float:
    work = frame.copy()
    horizon_columns = [f"score_{horizon}m" for horizon in HORIZONS]
    work["score"] = work[horizon_columns].to_numpy() @ weights
    returns = []
    for _, month in work.groupby("target_date"):
        quantile = month["score"].rank(pct=True)
        long = month[quantile.ge(0.9)]["net_residual_1m"].mean()
        short = month[quantile.le(0.1)]["net_residual_1m"].mean()
        returns.append(long - short)
    values = pd.Series(returns).dropna()
    if len(values) < 12:
        return -np.inf
    return float(values.mean() * 12 - 0.5 * gamma * values.var(ddof=1) * 12)


def combine_horizons(
    predictions: pd.DataFrame,
    config: LambdaRankConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    choices = []
    for variant, variant_data in predictions.groupby("model_variant"):
        keys = ["date", "target_date", "ric"]
        metadata_columns = [
            "target_return_1m",
            "residual_return_1m",
            "net_residual_1m",
            "expected_trade_cost",
            "company_market_cap",
        ]
        metadata = variant_data.groupby(keys, as_index=False)[metadata_columns].first()
        wide = variant_data.pivot_table(
            index=keys,
            columns="horizon",
            values="prediction",
        ).reset_index()
        wide = wide.merge(metadata, on=keys, how="left", validate="one_to_one")
        wide = wide.rename(columns={h: f"score_{h}m" for h in HORIZONS})
        for horizon in HORIZONS:
            column = f"score_{horizon}m"
            wide[column] = wide.groupby("target_date")[column].rank(pct=True)
        for year in sorted(wide["target_date"].dt.year.unique()):
            history = wide[wide["target_date"].dt.year.lt(year)]
            history_dates = sorted(history["target_date"].unique())
            if len(history_dates) >= config.blend_validation_months:
                dates = history_dates[-config.blend_validation_months :]
                validation = history[
                    history["target_date"].isin(dates)
                ].dropna(subset=[f"score_{h}m" for h in HORIZONS])
                utilities = {
                    name: _blend_utility(
                        validation, weights, config.utility_risk_aversion
                    )
                    for name, weights in BLENDS.items()
                }
                blend_name = max(utilities, key=utilities.get)
                source = "trailing_oos_utility"
            else:
                blend_name = "balanced"
                utilities = {}
                source = "preregistered_default"
            test = wide[wide["target_date"].dt.year.eq(year)].dropna(
                subset=[f"score_{h}m" for h in HORIZONS]
            ).copy()
            test["prediction"] = (
                test[[f"score_{h}m" for h in HORIZONS]].to_numpy()
                @ BLENDS[blend_name]
            )
            test["model_variant"] = f"{variant}_multihorizon"
            test["blend"] = blend_name
            outputs.append(test)
            choices.append(
                {
                    "model_variant": variant,
                    "test_year": year,
                    "history_months": len(history_dates),
                    "blend": blend_name,
                    "selection_source": source,
                    "validation_utility": utilities.get(blend_name, np.nan),
                }
            )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(choices)


def write_outputs(
    output_dir: Path,
    config: LambdaRankConfig,
    predictions: pd.DataFrame,
    combined: pd.DataFrame,
    fit_log: pd.DataFrame,
    metrics: pd.DataFrame,
    blend_choices: pd.DataFrame,
    theme_importance: pd.DataFrame,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "horizon_predictions.parquet", index=False)
    combined.to_parquet(output_dir / "combined_predictions.parquet", index=False)
    fit_log.to_csv(output_dir / "fit_log.csv", index=False)
    metrics.to_csv(output_dir / "prediction_metrics.csv", index=False)
    blend_choices.to_csv(output_dir / "blend_choices.csv", index=False)
    theme_importance.to_csv(
        output_dir / "economic_theme_importance.csv", index=False
    )
    manifest = {
        "config": asdict(config),
        "features": {
            "characteristics": RANK_FEATURES,
            "states": STATE_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "horizons": HORIZONS,
        "blends": {name: values.tolist() for name, values in BLENDS.items()},
        "rows": {
            "horizon_predictions": len(predictions),
            "combined_predictions": len(combined),
            "fit_log": len(fit_log),
        },
        "causality": {
            "train_target_after_cutoff": int(
                (
                    pd.to_datetime(fit_log["maximum_train_target_end"])
                    > pd.to_datetime(fit_log["label_cutoff"])
                ).sum()
            )
        },
    }
    (output_dir / "conditional_lambdarank_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest
