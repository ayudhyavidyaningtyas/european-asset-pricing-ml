"""Cost-aware implementable frontiers for frozen asset-pricing signals."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import stats as project_stats


SIGNALS = ("ml_return", "momentum", "sparse3")
PORTFOLIOS = ("long_short", "long_only")


@dataclass(frozen=True)
class FrontierConfig:
    signals: tuple[str, ...] = SIGNALS
    minimum_market_cap_percentile: float = 0.20
    maximum_assets: int = 500
    alpha_scale_monthly: float = 0.01
    risk_aversions: tuple[float, ...] = (5.0, 20.0, 80.0)
    adjustment_speeds: tuple[float, ...] = (1.0, 0.5, 0.25)
    long_short_gross_limit: float = 1.0
    long_short_position_limit: float = 0.02
    long_only_position_limit: float = 0.03
    beta_tolerance: float = 0.05
    fallback_half_spread_bps: float = 25.0
    impact_coefficient: float = 0.10
    aum_eur: tuple[float, ...] = (
        10_000_000.0,
        100_000_000.0,
        500_000_000.0,
    )
    selection_lookback_months: int = 36
    selection_ce_risk_aversion: float = 3.0
    default_risk_aversion: float = 20.0
    default_adjustment_speed: float = 0.5
    bootstrap_repetitions: int = 5_000
    bootstrap_block: int = 6
    random_state: int = 42


def _month_end(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.to_period("M").dt.to_timestamp("M")


def load_frontier_panel(
    panel_path: Path,
    predictions_path: Path,
    risk_path: Path,
) -> pd.DataFrame:
    prediction_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "is_delisting_candidate",
        "model",
        "prediction",
    ]
    predictions = pd.read_parquet(predictions_path, columns=prediction_columns)
    predictions = predictions[
        predictions["model"].isin(["mlp_return", "momentum_rank"])
    ].copy()
    predictions["signal"] = predictions["model"].map(
        {"mlp_return": "ml_return", "momentum_rank": "momentum"}
    )
    signals = predictions.pivot_table(
        index=["date", "target_date", "ric"],
        columns="signal",
        values="prediction",
        aggfunc="last",
    ).reset_index()
    candidate = (
        predictions.groupby(["date", "target_date", "ric"], as_index=False)[
            "is_delisting_candidate"
        ]
        .max()
    )
    outcomes = (
        predictions.groupby(["date", "target_date", "ric"], as_index=False)[
            "target_return_1m"
        ]
        .last()
    )
    signals = signals.merge(candidate, on=["date", "target_date", "ric"], how="left")
    signals = signals.merge(outcomes, on=["date", "target_date", "ric"], how="left")

    panel_columns = [
        "date",
        "ric",
        "company_market_cap",
        "market_cap_percentile",
        "turnover_12m",
        "volatility_12m",
        "book_to_market_rank",
        "momentum_12_2_rank",
        "operating_profitability_rank",
    ]
    panel = pd.read_parquet(panel_path, columns=panel_columns)
    risk = pd.read_parquet(
        risk_path,
        columns=["date", "ric", "beta_36m", "idio_vol_36m", "risk_nobs"],
    )
    result = signals.merge(panel, on=["date", "ric"], how="left", validate="one_to_one")
    result = result.merge(risk, on=["date", "ric"], how="left", validate="one_to_one")
    result["sparse3"] = result[
        [
            "book_to_market_rank",
            "momentum_12_2_rank",
            "operating_profitability_rank",
        ]
    ].mean(axis=1, skipna=False)
    result["date"] = _month_end(result["date"])
    result["target_date"] = _month_end(result["target_date"])
    result["is_delisting_candidate"] = result["is_delisting_candidate"].fillna(False)
    result.loc[
        result["is_delisting_candidate"] & result["target_return_1m"].isna(),
        "target_return_1m",
    ] = -1.0
    return result.sort_values(["date", "ric"]).reset_index(drop=True)


def load_monthly_liquidity(dataset: Path | None) -> pd.DataFrame:
    columns = ["date", "ric", "half_spread_bps", "spread_observed"]
    if dataset is None or not dataset.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_parquet(dataset)
    except (ValueError, OSError):
        files = sorted(dataset.glob("year=*/batch_*.parquet"))
        if not files:
            return pd.DataFrame(columns=columns)
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {"date", "ric", "bid", "ask"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    frame["date"] = _month_end(frame["date"])
    bid = pd.to_numeric(frame["bid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    midpoint = (bid + ask) / 2.0
    valid = bid.gt(0) & ask.ge(bid) & midpoint.gt(0)
    frame["half_spread_bps"] = (
        0.5 * (ask - bid).div(midpoint) * 10_000.0
    ).where(valid)
    frame["half_spread_bps"] = frame["half_spread_bps"].where(
        frame["half_spread_bps"].between(0.1, 500.0)
    )
    monthly = (
        frame.groupby(["date", "ric"], as_index=False)["half_spread_bps"]
        .median()
        .sort_values(["ric", "date"])
    )
    monthly["half_spread_bps"] = monthly.groupby("ric")[
        "half_spread_bps"
    ].transform(lambda values: values.rolling(3, min_periods=1).median())
    monthly["spread_observed"] = monthly["half_spread_bps"].notna()
    return monthly[columns]


def attach_execution_inputs(
    panel: pd.DataFrame,
    liquidity: pd.DataFrame,
    config: FrontierConfig,
) -> pd.DataFrame:
    result = panel.merge(liquidity, on=["date", "ric"], how="left")
    result["spread_observed"] = (
        result["spread_observed"].astype("boolean").fillna(False).astype(bool)
    )
    result["half_spread_bps"] = pd.to_numeric(
        result["half_spread_bps"], errors="coerce"
    ).fillna(
        config.fallback_half_spread_bps
    )
    turnover = pd.to_numeric(result["turnover_12m"], errors="coerce")
    market_cap = pd.to_numeric(result["company_market_cap"], errors="coerce")
    result["adv_eur"] = market_cap * turnover
    fallback_adv = market_cap * 0.0005
    result["adv_eur"] = result["adv_eur"].where(result["adv_eur"].gt(0), fallback_adv)
    result["adv_eur"] = result["adv_eur"].clip(lower=10_000.0)
    result["idio_vol_36m"] = pd.to_numeric(
        result["idio_vol_36m"], errors="coerce"
    ).clip(lower=0.02, upper=0.75)
    result["beta_36m"] = pd.to_numeric(result["beta_36m"], errors="coerce")
    return result


def build_market_volatility(market_path: Path) -> pd.Series:
    market = pd.read_csv(market_path, parse_dates=["date"])
    market["date"] = _month_end(market["date"])
    volatility = (
        market.set_index("date")["market_return_eur"]
        .rolling(36, min_periods=24)
        .std()
        .clip(lower=0.02, upper=0.20)
    )
    volatility.name = "market_vol_36m"
    return volatility


def load_eur_risk_free(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    date_column = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    value_column = [column for column in frame.columns if column != date_column][0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    annual_percent = pd.to_numeric(frame[value_column], errors="coerce")
    monthly = (1.0 + annual_percent / 100.0).pow(1.0 / 12.0) - 1.0
    result = pd.Series(monthly.to_numpy(), index=_month_end(frame[date_column]))
    result = result.groupby(level=0).last().sort_index()
    complete_index = pd.date_range(result.index.min(), result.index.max(), freq="ME")
    return result.reindex(complete_index).ffill(limit=3).rename("rf_eur")


def _rank_signal(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average", pct=True)
    return (2.0 * ranks - 1.0).to_numpy(dtype=float)


def prepare_month(
    group: pd.DataFrame,
    config: FrontierConfig,
) -> pd.DataFrame:
    required = [
        "company_market_cap",
        "market_cap_percentile",
        "beta_36m",
        "idio_vol_36m",
        "target_return_1m",
        *config.signals,
    ]
    eligible = group[
        group["market_cap_percentile"].ge(config.minimum_market_cap_percentile)
        & group["risk_nobs"].ge(24)
    ].dropna(subset=required)
    eligible = eligible.nlargest(config.maximum_assets, "company_market_cap").copy()
    for signal in config.signals:
        eligible[f"{signal}_score"] = _rank_signal(eligible[signal])
    return eligible.sort_values("ric").reset_index(drop=True)


def solve_aim_weights(
    alpha: np.ndarray,
    beta: np.ndarray,
    idio_vol: np.ndarray,
    market_vol: float,
    risk_aversion: float,
    portfolio: str,
    config: FrontierConfig,
) -> np.ndarray:
    n_assets = len(alpha)
    if n_assets == 0:
        return np.array([], dtype=float)
    weights = cp.Variable(n_assets)
    factor_risk = cp.square(beta @ weights) * float(market_vol**2)
    idiosyncratic_risk = cp.sum_squares(cp.multiply(idio_vol, weights))
    objective = cp.Maximize(
        alpha @ weights
        - 0.5 * float(risk_aversion) * (factor_risk + idiosyncratic_risk)
    )
    if portfolio == "long_short":
        constraints = [
            cp.sum(weights) == 0,
            cp.norm1(weights) <= config.long_short_gross_limit,
            cp.abs(beta @ weights) <= config.beta_tolerance,
            cp.abs(weights) <= config.long_short_position_limit,
        ]
    elif portfolio == "long_only":
        constraints = [
            weights >= 0,
            cp.sum(weights) <= 1.0,
            weights <= config.long_only_position_limit,
        ]
    else:
        raise ValueError(f"Unsupported portfolio: {portfolio}")
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=300,
            tol_gap_abs=1e-8,
            tol_feas=1e-8,
            verbose=False,
        )
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS, max_iters=5_000, eps=1e-6, verbose=False)
    if weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Portfolio optimization failed: {problem.status}")
    result = np.asarray(weights.value, dtype=float)
    result[np.abs(result) < 1e-10] = 0.0
    return result


def _project_long_short(
    weights: np.ndarray,
    beta: np.ndarray,
    config: FrontierConfig,
) -> np.ndarray:
    result = np.asarray(weights, dtype=float).copy()
    for _ in range(5):
        result -= result.mean()
        centered_beta = beta - beta.mean()
        denominator = float(centered_beta @ centered_beta)
        if denominator > 1e-12:
            result -= float(beta @ result) / denominator * centered_beta
        result = np.clip(
            result,
            -config.long_short_position_limit,
            config.long_short_position_limit,
        )
    gross = np.abs(result).sum()
    if gross > config.long_short_gross_limit:
        result *= config.long_short_gross_limit / gross
    return result


def _project_long_only(weights: np.ndarray, config: FrontierConfig) -> np.ndarray:
    result = np.clip(np.asarray(weights, dtype=float), 0.0, config.long_only_position_limit)
    total = result.sum()
    if total > 1.0:
        result /= total
    return result


def trade_toward_aim(
    prior: np.ndarray,
    aim: np.ndarray,
    beta: np.ndarray,
    adjustment_speed: float,
    portfolio: str,
    config: FrontierConfig,
) -> np.ndarray:
    traded = prior + float(adjustment_speed) * (aim - prior)
    if portfolio == "long_short":
        return _project_long_short(traded, beta, config)
    return _project_long_only(traded, config)


def execution_cost(
    delta: np.ndarray,
    half_spread_bps: np.ndarray,
    adv_eur: np.ndarray,
    idio_vol: np.ndarray,
    aum_eur: float,
    impact_coefficient: float,
) -> tuple[float, float, float]:
    absolute_trade = np.abs(np.asarray(delta, dtype=float))
    spread_cost = float(np.sum(absolute_trade * half_spread_bps / 10_000.0))
    participation = np.divide(
        absolute_trade * float(aum_eur),
        np.maximum(adv_eur, 1.0),
    )
    daily_vol = np.asarray(idio_vol, dtype=float) / np.sqrt(21.0)
    impact_bps = (
        float(impact_coefficient)
        * daily_vol
        * np.sqrt(np.maximum(participation, 0.0))
        * 10_000.0
    )
    impact_cost = float(np.sum(absolute_trade * impact_bps / 10_000.0))
    return spread_cost, impact_cost, spread_cost + impact_cost


def _aum_label(aum: float) -> str:
    return f"{int(round(aum / 1_000_000.0))}m"


def simulate_frontiers(
    panel: pd.DataFrame,
    market_volatility: pd.Series,
    risk_free: pd.Series,
    config: FrontierConfig,
) -> pd.DataFrame:
    states: dict[tuple, dict[str, float]] = {}
    latest_inputs: dict[str, tuple[float, float, float]] = {}
    records: list[dict] = []

    for signal_date, raw_month in panel.groupby("date", sort=True):
        month = prepare_month(raw_month, config)
        if len(month) < 100:
            continue
        return_date = pd.Timestamp(month["target_date"].iloc[0])
        market_vol = market_volatility.get(signal_date, np.nan)
        if not np.isfinite(market_vol):
            continue
        rf = float(risk_free.get(return_date, 0.0))
        rics = month["ric"].astype(str).to_numpy()
        beta = month["beta_36m"].to_numpy(dtype=float)
        idio_vol = month["idio_vol_36m"].to_numpy(dtype=float)
        realized = month["target_return_1m"].to_numpy(dtype=float)
        spread = month["half_spread_bps"].to_numpy(dtype=float)
        adv = month["adv_eur"].to_numpy(dtype=float)
        for ric, values in zip(rics, zip(spread, adv, idio_vol), strict=True):
            latest_inputs[ric] = values

        for signal in config.signals:
            score = month[f"{signal}_score"].to_numpy(dtype=float)
            for portfolio in PORTFOLIOS:
                alpha = config.alpha_scale_monthly * score
                if portfolio == "long_only":
                    alpha = config.alpha_scale_monthly * (score + 1.0) / 2.0
                for risk_aversion in config.risk_aversions:
                    aim = solve_aim_weights(
                        alpha,
                        beta,
                        idio_vol,
                        float(market_vol),
                        risk_aversion,
                        portfolio,
                        config,
                    )
                    for adjustment in config.adjustment_speeds:
                        key = (signal, portfolio, risk_aversion, adjustment)
                        prior_state = states.get(key, {})
                        prior = np.array(
                            [prior_state.get(ric, 0.0) for ric in rics], dtype=float
                        )
                        current = trade_toward_aim(
                            prior,
                            aim,
                            beta,
                            adjustment,
                            portfolio,
                            config,
                        )

                        dropped = {
                            ric: weight
                            for ric, weight in prior_state.items()
                            if ric not in set(rics) and abs(weight) > 1e-12
                        }
                        delta = current - prior
                        dropped_delta = np.array([-weight for weight in dropped.values()])
                        if dropped:
                            dropped_inputs = np.array(
                                [
                                    latest_inputs.get(
                                        ric,
                                        (
                                            config.fallback_half_spread_bps,
                                            10_000.0,
                                            0.20,
                                        ),
                                    )
                                    for ric in dropped
                                ],
                                dtype=float,
                            )
                            all_delta = np.concatenate([delta, dropped_delta])
                            all_spread = np.concatenate([spread, dropped_inputs[:, 0]])
                            all_adv = np.concatenate([adv, dropped_inputs[:, 1]])
                            all_idio = np.concatenate([idio_vol, dropped_inputs[:, 2]])
                        else:
                            all_delta, all_spread, all_adv, all_idio = (
                                delta,
                                spread,
                                adv,
                                idio_vol,
                            )

                        stock_return = float(current @ realized)
                        invested = float(current.sum()) if portfolio == "long_only" else 0.0
                        gross_return = (
                            stock_return + max(0.0, 1.0 - invested) * rf
                            if portfolio == "long_only"
                            else stock_return
                        )
                        row = {
                            "signal_date": signal_date,
                            "return_date": return_date,
                            "signal": signal,
                            "portfolio": portfolio,
                            "risk_aversion": risk_aversion,
                            "adjustment_speed": adjustment,
                            "turnover_penalty": (1.0 - adjustment) / adjustment,
                            "assets": len(month),
                            "gross_exposure": float(np.abs(current).sum()),
                            "net_exposure": float(current.sum()),
                            "beta_exposure": float(beta @ current),
                            "turnover": float(np.abs(all_delta).sum()),
                            "gross_return": gross_return,
                            "rf_eur": rf,
                            "spread_observed_weight": float(
                                np.abs(
                                    current[
                                        month["spread_observed"].to_numpy(bool)
                                    ]
                                ).sum()
                                / max(np.abs(current).sum(), 1e-12)
                            ),
                            "delisting_positions": int(
                                (
                                    month["is_delisting_candidate"].to_numpy(bool)
                                    & (np.abs(current) > 1e-10)
                                ).sum()
                            ),
                        }
                        for aum in config.aum_eur:
                            spread_cost, impact_cost, total_cost = execution_cost(
                                all_delta,
                                all_spread,
                                all_adv,
                                all_idio,
                                aum,
                                config.impact_coefficient,
                            )
                            label = _aum_label(aum)
                            row[f"spread_cost_{label}"] = spread_cost
                            row[f"impact_cost_{label}"] = impact_cost
                            row[f"total_cost_{label}"] = total_cost
                            row[f"net_return_{label}"] = gross_return - total_cost
                        records.append(row)

                        denominator = 1.0 + gross_return
                        if denominator <= 0:
                            drifted = np.zeros_like(current)
                        else:
                            drifted = current * (1.0 + realized) / denominator
                        states[key] = {
                            ric: float(weight)
                            for ric, weight in zip(rics, drifted, strict=True)
                            if abs(weight) > 1e-12
                        }
    return pd.DataFrame(records)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summarize_frontiers(
    monthly: pd.DataFrame,
    config: FrontierConfig,
) -> pd.DataFrame:
    records = []
    group_columns = [
        "signal",
        "portfolio",
        "risk_aversion",
        "adjustment_speed",
        "turnover_penalty",
    ]
    for keys, group in monthly.groupby(group_columns, sort=False):
        base = dict(zip(group_columns, keys, strict=True))
        for aum in config.aum_eur:
            label = _aum_label(aum)
            net = group[f"net_return_{label}"]
            excess = net - group["rf_eur"] if base["portfolio"] == "long_only" else net
            annual_mean = float(excess.mean() * 12.0)
            annual_vol = float(excess.std(ddof=1) * np.sqrt(12.0))
            records.append(
                {
                    **base,
                    "aum_eur": aum,
                    "aum_label": label,
                    "months": len(group),
                    "annualized_gross_return": float(group["gross_return"].mean() * 12.0),
                    "annualized_net_return": float(net.mean() * 12.0),
                    "annualized_excess_return": annual_mean,
                    "annualized_volatility": annual_vol,
                    "sharpe": annual_mean / annual_vol if annual_vol > 0 else np.nan,
                    "certainty_equivalent": annual_mean
                    - 0.5 * config.selection_ce_risk_aversion * annual_vol**2,
                    "max_drawdown": _max_drawdown(net),
                    "average_monthly_turnover": float(group["turnover"].mean()),
                    "annualized_spread_cost": float(
                        group[f"spread_cost_{label}"].mean() * 12.0
                    ),
                    "annualized_impact_cost": float(
                        group[f"impact_cost_{label}"].mean() * 12.0
                    ),
                    "average_gross_exposure": float(group["gross_exposure"].mean()),
                    "average_beta_exposure": float(group["beta_exposure"].mean()),
                    "spread_observed_weight": float(
                        group["spread_observed_weight"].mean()
                    ),
                }
            )
    summary = pd.DataFrame(records)
    summary["efficient"] = False
    family = ["signal", "portfolio", "aum_label"]
    for _, indices in summary.groupby(family).groups.items():
        subset = summary.loc[indices]
        for index, row in subset.iterrows():
            dominated = (
                subset["annualized_volatility"].le(row["annualized_volatility"] + 1e-12)
                & subset["annualized_excess_return"].ge(
                    row["annualized_excess_return"] - 1e-12
                )
                & (
                    subset["annualized_volatility"].lt(
                        row["annualized_volatility"] - 1e-12
                    )
                    | subset["annualized_excess_return"].gt(
                        row["annualized_excess_return"] + 1e-12
                    )
                )
            ).any()
            summary.loc[index, "efficient"] = not dominated
    return summary


def frontier_dominance(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (portfolio, aum_label), family in summary.groupby(["portfolio", "aum_label"]):
        momentum = (
            family[
                family["signal"].eq("momentum") & family["efficient"].astype(bool)
            ]
            .sort_values("annualized_volatility")
            .drop_duplicates("annualized_volatility", keep="last")
        )
        benchmark_volatility = np.r_[
            0.0, momentum["annualized_volatility"].to_numpy(dtype=float)
        ]
        benchmark_return = np.r_[
            0.0, momentum["annualized_excess_return"].to_numpy(dtype=float)
        ]
        for _, row in family[family["signal"].ne("momentum")].iterrows():
            volatility = float(row["annualized_volatility"])
            benchmark = float(
                np.interp(
                    volatility,
                    benchmark_volatility,
                    benchmark_return,
                    left=0.0,
                    right=benchmark_return[-1],
                )
            )
            records.append(
                {
                    "signal": row["signal"],
                    "portfolio": portfolio,
                    "aum_label": aum_label,
                    "risk_aversion": row["risk_aversion"],
                    "adjustment_speed": row["adjustment_speed"],
                    "annualized_volatility": row["annualized_volatility"],
                    "annualized_excess_return": row["annualized_excess_return"],
                    "momentum_return_at_same_risk": benchmark,
                    "frontier_return_improvement": row[
                        "annualized_excess_return"
                    ]
                    - benchmark,
                }
            )
    return pd.DataFrame(records)


def causal_strategy_selection(
    monthly: pd.DataFrame,
    config: FrontierConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    selection_records = []
    for signal in config.signals:
        for portfolio in PORTFOLIOS:
            family = monthly[
                monthly["signal"].eq(signal) & monthly["portfolio"].eq(portfolio)
            ]
            for aum in config.aum_eur:
                label = _aum_label(aum)
                for year in sorted(family["return_date"].dt.year.unique()):
                    history = family[family["return_date"].dt.year.lt(year)]
                    history_months = history["return_date"].nunique()
                    if history_months >= config.selection_lookback_months:
                        cutoff_months = sorted(history["return_date"].unique())[
                            -config.selection_lookback_months :
                        ]
                        trailing = history[history["return_date"].isin(cutoff_months)]
                        candidates = []
                        for keys, group in trailing.groupby(
                            ["risk_aversion", "adjustment_speed"]
                        ):
                            net = group[f"net_return_{label}"]
                            excess = net - group["rf_eur"] if portfolio == "long_only" else net
                            annual_mean = excess.mean() * 12.0
                            annual_var = excess.var(ddof=1) * 12.0
                            candidates.append(
                                (
                                    float(
                                        annual_mean
                                        - 0.5
                                        * config.selection_ce_risk_aversion
                                        * annual_var
                                    ),
                                    keys,
                                )
                            )
                        _, choice = max(candidates, key=lambda item: item[0])
                        selection_source = "trailing_validation"
                    else:
                        choice = (
                            config.default_risk_aversion,
                            config.default_adjustment_speed,
                        )
                        selection_source = "preregistered_default"
                    risk_aversion, adjustment = choice
                    applied = family[
                        family["return_date"].dt.year.eq(year)
                        & family["risk_aversion"].eq(risk_aversion)
                        & family["adjustment_speed"].eq(adjustment)
                    ].copy()
                    applied["aum_eur"] = aum
                    applied["aum_label"] = label
                    applied["selected_net_return"] = applied[f"net_return_{label}"]
                    applied["selection_source"] = selection_source
                    selected_rows.append(applied)
                    selection_records.append(
                        {
                            "signal": signal,
                            "portfolio": portfolio,
                            "aum_eur": aum,
                            "aum_label": label,
                            "test_year": year,
                            "history_months": history_months,
                            "risk_aversion": risk_aversion,
                            "adjustment_speed": adjustment,
                            "selection_source": selection_source,
                        }
                    )
    return (
        pd.concat(selected_rows, ignore_index=True),
        pd.DataFrame(selection_records),
    )


def simulate_selected_frontiers(
    panel: pd.DataFrame,
    market_volatility: pd.Series,
    risk_free: pd.Series,
    selection_log: pd.DataFrame,
    config: FrontierConfig,
) -> pd.DataFrame:
    choices = selection_log.set_index(
        ["signal", "portfolio", "aum_label", "test_year"]
    )
    states: dict[tuple, dict[str, float]] = {}
    latest_inputs: dict[str, tuple[float, float, float]] = {}
    records: list[dict] = []

    for signal_date, raw_month in panel.groupby("date", sort=True):
        month = prepare_month(raw_month, config)
        if len(month) < 100:
            continue
        return_date = pd.Timestamp(month["target_date"].iloc[0])
        test_year = return_date.year
        market_vol = market_volatility.get(signal_date, np.nan)
        if not np.isfinite(market_vol):
            continue
        rf = float(risk_free.get(return_date, 0.0))
        rics = month["ric"].astype(str).to_numpy()
        ric_set = set(rics)
        beta = month["beta_36m"].to_numpy(dtype=float)
        idio_vol = month["idio_vol_36m"].to_numpy(dtype=float)
        realized = month["target_return_1m"].to_numpy(dtype=float)
        spread = month["half_spread_bps"].to_numpy(dtype=float)
        adv = month["adv_eur"].to_numpy(dtype=float)
        observed_spread = month["spread_observed"].to_numpy(bool)
        for ric, values in zip(rics, zip(spread, adv, idio_vol), strict=True):
            latest_inputs[ric] = values

        aim_cache: dict[tuple, np.ndarray] = {}
        for signal in config.signals:
            score = month[f"{signal}_score"].to_numpy(dtype=float)
            for portfolio in PORTFOLIOS:
                alpha = config.alpha_scale_monthly * score
                if portfolio == "long_only":
                    alpha = config.alpha_scale_monthly * (score + 1.0) / 2.0
                for aum in config.aum_eur:
                    aum_label = _aum_label(aum)
                    choice = choices.loc[
                        (signal, portfolio, aum_label, test_year)
                    ]
                    risk_aversion = float(choice["risk_aversion"])
                    adjustment = float(choice["adjustment_speed"])
                    aim_key = (signal, portfolio, risk_aversion)
                    if aim_key not in aim_cache:
                        aim_cache[aim_key] = solve_aim_weights(
                            alpha,
                            beta,
                            idio_vol,
                            float(market_vol),
                            risk_aversion,
                            portfolio,
                            config,
                        )
                    aim = aim_cache[aim_key]

                    state_key = (signal, portfolio, aum_label)
                    prior_state = states.get(state_key, {})
                    prior = np.array(
                        [prior_state.get(ric, 0.0) for ric in rics], dtype=float
                    )
                    current = trade_toward_aim(
                        prior,
                        aim,
                        beta,
                        adjustment,
                        portfolio,
                        config,
                    )
                    dropped = {
                        ric: weight
                        for ric, weight in prior_state.items()
                        if ric not in ric_set and abs(weight) > 1e-12
                    }
                    delta = current - prior
                    if dropped:
                        dropped_delta = np.array(
                            [-weight for weight in dropped.values()]
                        )
                        dropped_inputs = np.array(
                            [
                                latest_inputs.get(
                                    ric,
                                    (
                                        config.fallback_half_spread_bps,
                                        10_000.0,
                                        0.20,
                                    ),
                                )
                                for ric in dropped
                            ],
                            dtype=float,
                        )
                        all_delta = np.concatenate([delta, dropped_delta])
                        all_spread = np.concatenate([spread, dropped_inputs[:, 0]])
                        all_adv = np.concatenate([adv, dropped_inputs[:, 1]])
                        all_idio = np.concatenate([idio_vol, dropped_inputs[:, 2]])
                    else:
                        all_delta, all_spread, all_adv, all_idio = (
                            delta,
                            spread,
                            adv,
                            idio_vol,
                        )

                    invested = (
                        float(current.sum()) if portfolio == "long_only" else 0.0
                    )
                    stock_return = float(current @ realized)
                    gross_return = (
                        stock_return + max(0.0, 1.0 - invested) * rf
                        if portfolio == "long_only"
                        else stock_return
                    )
                    spread_cost, impact_cost, total_cost = execution_cost(
                        all_delta,
                        all_spread,
                        all_adv,
                        all_idio,
                        aum,
                        config.impact_coefficient,
                    )
                    row = {
                        "signal_date": signal_date,
                        "return_date": return_date,
                        "signal": signal,
                        "portfolio": portfolio,
                        "aum_eur": aum,
                        "aum_label": aum_label,
                        "risk_aversion": risk_aversion,
                        "adjustment_speed": adjustment,
                        "turnover_penalty": (1.0 - adjustment) / adjustment,
                        "selection_source": choice["selection_source"],
                        "assets": len(month),
                        "gross_exposure": float(np.abs(current).sum()),
                        "net_exposure": float(current.sum()),
                        "beta_exposure": float(beta @ current),
                        "turnover": float(np.abs(all_delta).sum()),
                        "gross_return": gross_return,
                        "rf_eur": rf,
                        "spread_observed_weight": float(
                            np.abs(current[observed_spread]).sum()
                            / max(np.abs(current).sum(), 1e-12)
                        ),
                        "delisting_positions": int(
                            (
                                month["is_delisting_candidate"].to_numpy(bool)
                                & (np.abs(current) > 1e-10)
                            ).sum()
                        ),
                        "spread_cost": spread_cost,
                        "impact_cost": impact_cost,
                        "total_cost": total_cost,
                        "selected_net_return": gross_return - total_cost,
                    }
                    records.append(row)

                    denominator = 1.0 + gross_return
                    drifted = (
                        np.zeros_like(current)
                        if denominator <= 0
                        else current * (1.0 + realized) / denominator
                    )
                    states[state_key] = {
                        ric: float(weight)
                        for ric, weight in zip(rics, drifted, strict=True)
                        if abs(weight) > 1e-12
                    }
    return pd.DataFrame(records)


def summarize_selected(selected: pd.DataFrame, config: FrontierConfig) -> pd.DataFrame:
    records = []
    for keys, group in selected.groupby(["signal", "portfolio", "aum_label"]):
        signal, portfolio, aum_label = keys
        net = group["selected_net_return"]
        excess = net - group["rf_eur"] if portfolio == "long_only" else net
        annual_mean = float(excess.mean() * 12.0)
        annual_vol = float(excess.std(ddof=1) * np.sqrt(12.0))
        records.append(
            {
                "signal": signal,
                "portfolio": portfolio,
                "aum_label": aum_label,
                "months": len(group),
                "annualized_excess_return": annual_mean,
                "annualized_volatility": annual_vol,
                "sharpe": annual_mean / annual_vol if annual_vol > 0 else np.nan,
                "certainty_equivalent": annual_mean
                - 0.5 * config.selection_ce_risk_aversion * annual_vol**2,
                "max_drawdown": _max_drawdown(net),
                "average_monthly_turnover": float(group["turnover"].mean()),
                "annualized_spread_cost": float(
                    group["spread_cost"].mean() * 12.0
                ),
                "annualized_impact_cost": float(
                    group["impact_cost"].mean() * 12.0
                ),
                "spread_observed_weight": float(
                    group["spread_observed_weight"].mean()
                ),
                "average_gross_exposure": float(group["gross_exposure"].mean()),
                "average_beta_exposure": float(group["beta_exposure"].mean()),
                "average_risk_aversion": float(group["risk_aversion"].mean()),
                "average_adjustment_speed": float(group["adjustment_speed"].mean()),
            }
        )
    return pd.DataFrame(records)


def selected_sharpe_inference(
    selected: pd.DataFrame,
    config: FrontierConfig,
) -> pd.DataFrame:
    records = []
    for (portfolio, aum_label), family in selected.groupby(["portfolio", "aum_label"]):
        baseline = family[family["signal"].eq("momentum")].set_index("return_date")
        for signal in [
            value for value in config.signals if value != "momentum"
        ]:
            model = family[family["signal"].eq(signal)].set_index("return_date")
            dates = baseline.index.intersection(model.index)
            if len(dates) < 24:
                continue
            rf = (
                model.loc[dates, "rf_eur"].to_numpy()
                if portfolio == "long_only"
                else np.zeros(len(dates))
            )
            result = project_stats.bootstrap_sharpe_diff(
                model.loc[dates, "selected_net_return"],
                baseline.loc[dates, "selected_net_return"],
                rf,
                expected_block=config.bootstrap_block,
                n_boot=config.bootstrap_repetitions,
                seed=config.random_state,
            )
            records.append(
                {
                    "signal": signal,
                    "baseline": "momentum",
                    "portfolio": portfolio,
                    "aum_label": aum_label,
                    "months": len(dates),
                    **result,
                }
            )
    result = pd.DataFrame(records)
    if not result.empty:
        result["p_two_sided_holm"] = multipletests(
            result["p_two_sided"], method="holm"
        )[1]
    return result


def plot_frontier_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "ml_return": "#1f77b4",
        "momentum": "#d62728",
        "sparse3": "#2ca02c",
    }
    labels = {
        "ml_return": "ML return",
        "momentum": "Momentum",
        "sparse3": "Sparse 3-char.",
        "conditional_rank": "Conditional LambdaRank",
        "unconditional_rank": "Unconditional LambdaRank",
    }
    colors.update(
        {
            "conditional_rank": "#1f77b4",
            "unconditional_rank": "#9467bd",
        }
    )
    signal_names = list(summary["signal"].drop_duplicates())
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for index, signal in enumerate(signal_names):
        colors.setdefault(signal, palette[index % len(palette)])
        labels.setdefault(signal, signal.replace("_", " ").title())

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, portfolio in zip(axes, PORTFOLIOS, strict=True):
        family = summary[
            summary["portfolio"].eq(portfolio) & summary["aum_label"].eq("100m")
        ]
        for signal in signal_names:
            signal_rows = family[family["signal"].eq(signal)]
            axis.scatter(
                signal_rows["annualized_volatility"],
                signal_rows["annualized_excess_return"],
                color=colors[signal],
                alpha=0.25,
                s=25,
            )
            efficient = signal_rows[signal_rows["efficient"]].sort_values(
                "annualized_volatility"
            )
            axis.plot(
                np.r_[0.0, efficient["annualized_volatility"].to_numpy()],
                np.r_[0.0, efficient["annualized_excess_return"].to_numpy()],
                color=colors[signal],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=labels[signal],
            )
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_title(
            "Dollar-neutral" if portfolio == "long_short" else "Long-only"
        )
        axis.set_xlabel("Annualized volatility")
        axis.set_ylabel("Annualized net excess return")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    frontier_path = output_dir / "implementable_frontier_100m.png"
    figure.savefig(frontier_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, portfolio in zip(axes, PORTFOLIOS, strict=True):
        family = selected_summary[selected_summary["portfolio"].eq(portfolio)]
        x = np.arange(len(configured_aums := ["10m", "100m", "500m"]))
        width = 0.24
        for offset, signal in enumerate(signal_names):
            values = (
                family[family["signal"].eq(signal)]
                .set_index("aum_label")["sharpe"]
                .reindex(configured_aums)
            )
            axis.bar(
                x + (offset - (len(signal_names) - 1) / 2) * width,
                values,
                width,
                color=colors[signal],
                label=labels[signal],
            )
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_xticks(x, [f"EUR {label}" for label in configured_aums])
        axis.set_ylabel("Net excess Sharpe")
        axis.set_title(
            "Dollar-neutral" if portfolio == "long_short" else "Long-only"
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    capacity_path = output_dir / "selected_sharpe_by_capacity.png"
    figure.savefig(capacity_path, dpi=180)
    plt.close(figure)
    return [frontier_path.name, capacity_path.name]


def write_frontier_outputs(
    output_dir: Path,
    config: FrontierConfig,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    dominance: pd.DataFrame,
    selected: pd.DataFrame,
    selection_log: pd.DataFrame,
    selected_summary: pd.DataFrame,
    inference: pd.DataFrame,
    inputs: dict[str, Path | None],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "monthly": "frontier_monthly.csv",
        "summary": "frontier_summary.csv",
        "dominance": "frontier_dominance.csv",
        "selected_monthly": "selected_frontier_monthly.csv",
        "selection_log": "selected_frontier_choices.csv",
        "selected_summary": "selected_frontier_summary.csv",
        "inference": "selected_frontier_inference.csv",
    }
    monthly.to_csv(output_dir / outputs["monthly"], index=False)
    summary.to_csv(output_dir / outputs["summary"], index=False)
    dominance.to_csv(output_dir / outputs["dominance"], index=False)
    selected.to_csv(output_dir / outputs["selected_monthly"], index=False)
    selection_log.to_csv(output_dir / outputs["selection_log"], index=False)
    selected_summary.to_csv(output_dir / outputs["selected_summary"], index=False)
    inference.to_csv(output_dir / outputs["inference"], index=False)
    outputs["figures"] = plot_frontier_outputs(output_dir, summary, selected_summary)
    manifest = {
        "config": asdict(config),
        "inputs": {
            key: str(path) if path is not None else None for key, path in inputs.items()
        },
        "signals": {
            "ml_return": "Frozen annual-walk-forward MLP raw-return prediction",
            "momentum": "Frozen 12-2 momentum rank benchmark",
            "sparse3": "Unfitted equal-weight value, 12-2 momentum and profitability ranks",
        },
        "delisting_treatment": "-100% for scoreable missing retirement returns",
        "outputs": outputs,
        "rows": {
            "monthly": len(monthly),
            "summary": len(summary),
            "selected_monthly": len(selected),
            "inference": len(inference),
        },
        "checks": {
            "maximum_abs_long_short_net_exposure": float(
                monthly.loc[
                    monthly["portfolio"].eq("long_short"), "net_exposure"
                ].abs().max()
            ),
            "maximum_abs_long_short_beta_exposure": float(
                monthly.loc[
                    monthly["portfolio"].eq("long_short"), "beta_exposure"
                ].abs().max()
            ),
            "maximum_long_only_net_exposure": float(
                monthly.loc[
                    monthly["portfolio"].eq("long_only"), "net_exposure"
                ].max()
            ),
            "negative_cost_observations": int(
                (
                    monthly[
                        [
                            column
                            for column in monthly
                            if column.startswith("total_cost_")
                        ]
                    ]
                    < 0
                )
                .sum()
                .sum()
            ),
            "average_portfolio_weight_with_observed_spread": float(
                monthly["spread_observed_weight"].mean()
            ),
            "portfolio_months_with_delisting_position": int(
                monthly["delisting_positions"].gt(0).sum()
            ),
            "selected_maximum_abs_long_short_net_exposure": float(
                selected.loc[
                    selected["portfolio"].eq("long_short"), "net_exposure"
                ].abs().max()
            ),
            "selected_maximum_abs_long_short_beta_exposure": float(
                selected.loc[
                    selected["portfolio"].eq("long_short"), "beta_exposure"
                ].abs().max()
            ),
            "selected_nonfinite_returns": int(
                (~np.isfinite(selected["selected_net_return"])).sum()
            ),
        },
    }
    (output_dir / "frontier_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest
