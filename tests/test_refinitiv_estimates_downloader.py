from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refinitiv_estimates_downloader.py"
SPEC = importlib.util.spec_from_file_location("refinitiv_estimates_downloader", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_error_classifiers_recognize_refinitiv_throttle_and_desktop_loss():
    throttle = RuntimeError("LDError: Too many requests, please try again later")
    connection_loss = RuntimeError(
        "ConnectError('[Errno 61] Connection refused') while requesting "
        "http://localhost:9000/api/udf"
    )

    assert MODULE.is_rate_limit_error(throttle)
    assert MODULE.stop_reason_for_error(throttle) == "rate_limited"
    assert MODULE.is_connection_loss_error(connection_loss)
    assert MODULE.stop_reason_for_error(connection_loss) == "desktop_session_unavailable"


def test_retry_wait_seconds_uses_rate_limit_exponential_backoff_with_cap():
    throttle = RuntimeError("LDError: Too many requests")
    transient = RuntimeError('LDError: {"code":500,"message":"Network Error"}')

    assert MODULE.retry_wait_seconds(throttle, 0, 10.0, 120.0, 900.0) == 120.0
    assert MODULE.retry_wait_seconds(throttle, 3, 10.0, 120.0, 300.0) == 300.0
    assert MODULE.retry_wait_seconds(transient, 1, 10.0, 120.0, 900.0) == 20.0


def test_download_snapshot_batch_retries_rate_limits_with_capped_sleep(monkeypatch):
    class FakeLD:
        class HeaderType:
            NAME = "name"

        def get_data(self, **_kwargs):
            raise RuntimeError("LDError: Too many requests, please try again later")

    sleeps: list[float] = []
    monkeypatch.setattr(MODULE, "ld", FakeLD())
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="Too many requests"):
        MODULE.download_snapshot_batch(
            ["VOD.L"],
            ["TR.RIC"],
            pd.Timestamp("2024-01-31"),
            {"Period": "FY1"},
            max_retries=2,
            retry_sleep=10.0,
            rate_limit_sleep=120.0,
            max_retry_sleep=200.0,
        )

    assert sleeps == [120.0, 200.0]
