from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoencoder_asset_pricing import build_month_batches  # noqa: E402
from ipca_asset_pricing import (  # noqa: E402
    IPCAConfig,
    build_managed_moments,
    fit_ipca,
    run_walk_forward_ipca,
    summarize_ipca,
)


FEATURES = ["x_rank", "y_rank", "z_rank"]


def synthetic_panel(periods: int = 48, stocks: int = 12) -> pd.DataFrame:
    """Returns from an exact two-factor IPCA data-generating process."""
    records = []
    for month_index, date in enumerate(
        pd.date_range("2014-01-31", periods=periods, freq="ME")
    ):
        factor_0 = 0.02 + 0.010 * np.sin(month_index / 3.0)
        factor_1 = -0.01 + 0.008 * np.cos(month_index / 4.0)
        for security in range(stocks):
            x = -1.0 + security * 2.0 / (stocks - 1)
            y = 1.0 if security % 2 == 0 else -1.0
            z = np.sin(security)
            # Loadings are linear in characteristics, which is exactly the
            # restriction IPCA imposes.
            ret = x * factor_0 + y * factor_1
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security}",
                    "target_return_1m": ret,
                    "autoencoder_target_return": ret,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": 0.2 + security / (stocks * 2.0),
                    "x_rank": x,
                    "y_rank": y,
                    "z_rank": z,
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


def _batches(panel: pd.DataFrame, config: IPCAConfig):
    return build_month_batches(
        panel,
        FEATURES,
        np.zeros(len(FEATURES)),
        np.ones(len(FEATURES)),
        config,
    )


def test_ipca_recovers_the_true_loading_subspace():
    panel = synthetic_panel(periods=36)
    config = IPCAConfig(
        n_factors=2,
        min_monthly_stocks=4,
        include_constant=False,
        tolerance=1e-12,
        max_iterations=2000,
    )
    moments = build_managed_moments(
        _batches(panel, config), config.include_constant, training_returns=True
    )

    fit = fit_ipca(moments, config)
    gamma = fit["gamma"]

    assert fit["converged"]
    # Loadings are identified only up to rotation, so compare subspaces via the
    # projection matrix. Truth loads factor 0 on x and factor 1 on y.
    truth = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    truth_projection = truth @ np.linalg.pinv(truth)
    gamma_projection = gamma @ np.linalg.pinv(gamma)
    assert np.allclose(gamma_projection, truth_projection, atol=1e-4)
    # Orthonormality is the imposed identification.
    assert np.allclose(gamma.T @ gamma, np.eye(2), atol=1e-8)


def test_ipca_fits_exactly_when_the_dgp_is_linear():
    panel = synthetic_panel(periods=36)
    config = IPCAConfig(
        n_factors=2,
        min_monthly_stocks=4,
        include_constant=False,
        tolerance=1e-12,
        max_iterations=2000,
    )
    moments = build_managed_moments(
        _batches(panel, config), config.include_constant, training_returns=True
    )

    fit = fit_ipca(moments, config)

    # The DGP has no residual, so the managed pricing errors and the
    # unrestricted intercept block should both vanish. Monthly returns are of
    # order 1e-2, so a 1e-7 bound is six orders below the signal and leaves room
    # for ALS convergence noise rather than model error.
    assert np.abs(fit["alpha_path"]).max() < 1e-7
    assert np.linalg.norm(fit["gamma_alpha"]) < 1e-7


def test_ipca_gains_nothing_from_a_redundant_third_factor():
    panel = synthetic_panel(periods=36)
    shared = {
        "min_monthly_stocks": 4,
        "include_constant": False,
        "tolerance": 1e-12,
        "max_iterations": 2000,
    }
    batches = _batches(panel, IPCAConfig(n_factors=2, **shared))

    losses = {}
    for n_factors in (1, 2, 3):
        config = IPCAConfig(n_factors=n_factors, **shared)
        moments = build_managed_moments(
            batches, config.include_constant, training_returns=True
        )
        fit = fit_ipca(moments, config)
        residual = 0.0
        for index, moment in enumerate(moments):
            fitted = moment.w @ fit["gamma"] @ fit["factors"][index]
            residual += float(np.sum(np.square(moment.x - fitted)))
        losses[n_factors] = residual

    # Two factors is the truth: going from one to two must help materially,
    # and the third factor has nothing left to explain.
    assert losses[2] < losses[1] * 1e-6
    assert losses[3] <= losses[2] + 1e-12


def test_walk_forward_ipca_is_causal_and_gross_normalized():
    panel = synthetic_panel(periods=120, stocks=12)
    config = IPCAConfig(
        first_test_year=2021,
        last_test_year=2022,
        n_factors=2,
        min_monthly_stocks=4,
        min_training_months=24,
        validation_months=6,
        include_constant=True,
        max_iterations=200,
    )

    monthly, predictions, factors, weights, fit_log = run_walk_forward_ipca(
        panel, config, FEATURES
    )
    summary = summarize_ipca(monthly)

    assert not monthly.empty
    assert not fit_log.empty
    assert set(monthly["model"]) == {"ipca"}
    # Training labels never reach past the cutoff.
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    # No security-month is predicted twice.
    assert not predictions.duplicated(subset=["signal_date", "ric"]).any()
    # Stock-level SDF weights are gross-normalised, as in the AIPM and
    # autoencoder pipelines, so cost comparisons are on the same footing.
    gross = weights.groupby("signal_date")["sdf_weight"].apply(
        lambda values: float(values.abs().sum())
    )
    assert np.allclose(gross.to_numpy(), 1.0)
    assert not summary.empty
    assert int(summary.loc[0, "months"]) == len(monthly)


def test_constant_instrument_adds_a_row_to_the_loading_matrix():
    panel = synthetic_panel(periods=24)
    with_constant = IPCAConfig(n_factors=2, min_monthly_stocks=4, include_constant=True)
    without_constant = IPCAConfig(
        n_factors=2, min_monthly_stocks=4, include_constant=False
    )
    batches = _batches(panel, with_constant)

    fit_with = fit_ipca(
        build_managed_moments(batches, True, training_returns=True), with_constant
    )
    fit_without = fit_ipca(
        build_managed_moments(batches, False, training_returns=True), without_constant
    )

    assert fit_with["gamma"].shape == (len(FEATURES) + 1, 2)
    assert fit_without["gamma"].shape == (len(FEATURES), 2)
