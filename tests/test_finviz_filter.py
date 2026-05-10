"""Tests for the Finviz screener wrapper's ticker-safety guard.

Regression target: a 2026-05-09 security audit flagged that Finviz-returned
tickers flowed into ``Path(results_dir) / "by_ticker" / ticker`` interpolation
with no validation. If Finviz (a third-party with no auth) ever returns or is
MITM'd to return a malicious ticker, that's a filesystem path-traversal
vector. The fix runs every Finviz-supplied ticker through
``safe_ticker_component`` before it reaches the rest of the pipeline.
"""

from __future__ import annotations

from screener.finviz_filter import _filter_safe_tickers


def test_filter_passes_normal_tickers_through():
    assert _filter_safe_tickers(["AAPL", "MSFT", "BRK.B"]) == ["AAPL", "MSFT", "BRK.B"]


def test_filter_drops_path_traversal_payloads():
    """The motivating risk: a ticker like '../../tmp/evil' would interpolate
    into a filesystem path. Must be dropped."""
    raw = ["AAPL", "../../tmp/evil", "MSFT", "..", "."]
    assert _filter_safe_tickers(raw) == ["AAPL", "MSFT"]


def test_filter_drops_shell_metachars():
    raw = ["AAPL", "MSFT;rm -rf /", "GOOG`whoami`", "NVDA|cat"]
    assert _filter_safe_tickers(raw) == ["AAPL"]


def test_filter_drops_null_bytes():
    """Null bytes aren't stripped by .strip() — must be rejected."""
    raw = ["AAPL", "MSFT\x00", "GOOG\x00BLAH"]
    assert _filter_safe_tickers(raw) == ["AAPL"]


def test_filter_normalizes_trailing_whitespace_but_keeps_clean_ticker():
    """A ticker like ``'GOOG\\n'`` is normalised by .strip() to ``'GOOG'``,
    which is valid. This is defensive normalization, not a security flaw."""
    assert _filter_safe_tickers(["GOOG\n", "MSFT\r\n"]) == ["GOOG", "MSFT"]


def test_filter_strips_whitespace_and_drops_empty():
    raw = ["  AAPL  ", "", "   ", "MSFT"]
    assert _filter_safe_tickers(raw) == ["AAPL", "MSFT"]


def test_filter_accepts_exchange_qualified_tickers():
    """The fork supports tickers like CNC.TO, 7203.T, ^GSPC — these must
    survive validation."""
    raw = ["CNC.TO", "7203.T", "0700.HK", "^GSPC"]
    assert _filter_safe_tickers(raw) == raw
