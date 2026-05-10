"""Regression test for the Alpha Vantage HTTP timeout fix (security audit H2).

The original ``_make_api_request`` in alpha_vantage_common.py called
``requests.get(API_BASE_URL, params=api_params)`` with no ``timeout=``,
letting a slow/hung server (or MITM) stall the pipeline indefinitely.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test-key")

from tradingagents.dataflows import alpha_vantage_common as av


def test_make_api_request_passes_timeout_to_requests_get():
    """The fix: every requests.get must carry an explicit timeout so a
    slow upstream can't hang the whole pipeline."""
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.text = '{"Time Series (Daily)": {}}'  # any minimal valid JSON

    with patch.object(av, "requests") as mock_requests:
        mock_requests.get.return_value = fake
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})

    args, kwargs = mock_requests.get.call_args
    assert "timeout" in kwargs, "Alpha Vantage HTTP call missing timeout=  — DoS hazard"
    assert isinstance(kwargs["timeout"], (int, float))
    assert kwargs["timeout"] > 0


def test_timeout_value_is_bounded_reasonably():
    """30 seconds is the convention across the package (congress_trades,
    lambda_finance_*, etc.). Don't let this silently drift to something
    absurd."""
    assert 0 < av._TIMEOUT <= 60
