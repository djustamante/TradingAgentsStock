import os
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows import congress_trades
from tradingagents.dataflows.config import set_config


# --- Finnhub auth + token-leak hardening ----------------------------------


def test_redact_finnhub_token_replaces_value():
    """Defensive helper: ``?token=<key>`` and ``&token=<key>`` in any
    string get replaced with a placeholder so a leaked URL never logs
    the actual key."""
    url = "https://finnhub.io/api/v1/stock/x?symbol=AAPL&token=sk_secret_abc123"
    out = congress_trades._redact_finnhub_token(url)
    assert "sk_secret_abc123" not in out
    assert "&token=***REDACTED***" in out


def test_redact_finnhub_token_handles_question_mark_form():
    url = "https://example.com/x?token=mykey"
    out = congress_trades._redact_finnhub_token(url)
    assert "mykey" not in out
    assert "?token=***REDACTED***" in out


def test_redact_finnhub_token_passes_through_when_no_token():
    """No-op on strings without ``token=``."""
    s = "Finnhub: 500 Internal Server Error"
    assert congress_trades._redact_finnhub_token(s) == s


def test_redact_finnhub_token_handles_non_string():
    """The helper is sometimes called with an exception value before
    its message has been str()-ified — must not crash."""
    assert congress_trades._redact_finnhub_token(None) is None
    assert congress_trades._redact_finnhub_token("") == ""


def test_fetch_finnhub_uses_header_auth_not_query_param(monkeypatch):
    """Regression for security audit H1: the API key must travel in the
    X-Finnhub-Token header, NOT in the URL's ``?token=`` query param.
    Header-only auth eliminates the leak-in-HTTPError vector."""
    captured: dict = {}
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"data": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params or {}
        captured["headers"] = headers or {}
        return fake_resp

    monkeypatch.setattr(congress_trades.requests, "get", fake_get)
    congress_trades._fetch_finnhub("AAPL", "sk_secret_value", date.today())

    assert "token" not in captured["params"], (
        "API key still being passed as URL query param — leaks into request URLs"
    )
    assert captured["headers"].get("X-Finnhub-Token") == "sk_secret_value", (
        "Finnhub key must travel in X-Finnhub-Token header"
    )


def test_fetch_finnhub_redacts_token_from_propagated_exception(monkeypatch):
    """If requests raises with a message containing ``token=<key>``
    (e.g. from a misbehaving proxy that echoes the original URL even
    though we used header auth), the propagated exception's str() must
    have the key redacted before any logger sees it."""
    def fake_get(*args, **kwargs):
        # Simulate the kind of message requests.HTTPError emits when
        # raise_for_status() fires on a 4xx/5xx — includes the URL.
        raise RuntimeError(
            "401 Client Error: Unauthorized for url: "
            "https://finnhub.io/api/v1/x?symbol=AAPL&token=sk_THE_REAL_KEY"
        )

    monkeypatch.setattr(congress_trades.requests, "get", fake_get)
    with pytest.raises(RuntimeError) as excinfo:
        congress_trades._fetch_finnhub("AAPL", "sk_THE_REAL_KEY", date.today())

    assert "sk_THE_REAL_KEY" not in str(excinfo.value), (
        "Finnhub API key leaked through error message — defense-in-depth failed"
    )
    assert "***REDACTED***" in str(excinfo.value)


# --- Senate Stock Watcher size cap ----------------------------------------


def _streaming_resp(body: bytes, *, content_length: str = None, encoding: str = "utf-8"):
    """Minimal mock of a streaming requests.Response: iter_content + headers."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"Content-Length": content_length} if content_length else {}
    resp.encoding = encoding
    # Split into chunks roughly the size requests uses
    chunk_size = 64 * 1024
    chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
    resp.iter_content.return_value = iter(chunks)
    resp.close = MagicMock()
    return resp


def test_fetch_with_size_cap_returns_body_when_under_cap(monkeypatch):
    body = b'{"transactions": []}'
    monkeypatch.setattr(
        congress_trades.requests, "get",
        lambda *a, **kw: _streaming_resp(body, content_length=str(len(body))),
    )
    out = congress_trades._fetch_with_size_cap(
        "https://x", max_bytes=1_000_000, timeout=10,
    )
    assert out == '{"transactions": []}'


def test_fetch_with_size_cap_rejects_oversized_content_length(monkeypatch):
    """A server declaring a huge body must be rejected BEFORE we read it
    (the cheap fail-fast path)."""
    monkeypatch.setattr(
        congress_trades.requests, "get",
        lambda *a, **kw: _streaming_resp(b"unused", content_length="200000000"),
    )
    with pytest.raises(ValueError, match="exceeds .* cap"):
        congress_trades._fetch_with_size_cap(
            "https://x", max_bytes=100_000_000, timeout=10,
        )


def test_fetch_with_size_cap_rejects_streamed_overflow(monkeypatch):
    """The defense-in-depth check: server lies about / omits Content-Length
    and emits more bytes than the cap. The chunked accumulator must abort."""
    # ~200KB body, cap at 100KB
    big_body = b"x" * (200 * 1024)
    monkeypatch.setattr(
        congress_trades.requests, "get",
        lambda *a, **kw: _streaming_resp(big_body),
    )
    with pytest.raises(ValueError, match="exceeded .* cap"):
        congress_trades._fetch_with_size_cap(
            "https://x", max_bytes=100 * 1024, timeout=10,
        )


def test_fetch_with_size_cap_tolerates_bad_content_length(monkeypatch):
    """A garbage Content-Length header should not crash the helper —
    the streamed-overflow check still protects us."""
    body = b'{"transactions": []}'
    monkeypatch.setattr(
        congress_trades.requests, "get",
        lambda *a, **kw: _streaming_resp(body, content_length="not-a-number"),
    )
    out = congress_trades._fetch_with_size_cap(
        "https://x", max_bytes=1_000_000, timeout=10,
    )
    assert out == '{"transactions": []}'


# --- Pure parser unit tests -------------------------------------------------


def test_parse_finnhub_row_maps_canonical_shape():
    row = {
        "name": "Pelosi, Nancy",
        "transactionDate": "2026-04-15",
        "transactionType": "Purchase",
        "position": "House",
        "amountFrom": 1_000_001,
        "amountTo": 5_000_000,
        "filingDate": "2026-04-30",
    }
    t = congress_trades._parse_finnhub_row(row)
    assert t is not None
    assert t["filer"] == "Pelosi, Nancy"
    assert t["chamber"] == "House"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 1_000_001
    assert t["amount_max"] == 5_000_000
    assert t["filing_lag_days"] == 15
    assert "$1.0M – $5.0M" == t["amount_label"]


def test_parse_finnhub_row_returns_none_on_garbage():
    # Missing/invalid date is tolerated by the parser, but the trade is
    # filtered out later by _within_cutoff. The parser itself returns the row.
    row = {"name": "Doe", "transactionType": "Sale"}
    t = congress_trades._parse_finnhub_row(row)
    assert t is not None
    assert t["type"] == "Sale"
    assert t["date"] == "—"


def test_parse_ssw_row_normalises_mmddyyyy_dates_and_party():
    row = {
        "transaction_date": "04/15/2026",
        "senator": "Tuberville, Tommy",
        "party": "Republican",
        "state": "AL",
        "type": "purchase",
        "amount": "$15,001 - $50,000",
        "ptr_link_date": "05/02/2026",
    }
    t = congress_trades._parse_ssw_row(row)
    assert t is not None
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-05-02"
    assert t["party"] == "R"
    assert t["state"] == "AL"
    assert t["chamber"] == "Senate"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 15_001
    assert t["amount_max"] == 50_000


def test_amount_range_to_floats_handles_various_inputs():
    assert congress_trades._amount_range_to_floats("$1,001 - $15,000") == (1001, 15000)
    assert congress_trades._amount_range_to_floats("$1,000,001 - $5,000,000") == (1_000_001, 5_000_000)
    assert congress_trades._amount_range_to_floats("$50,000") == (50_000, 50_000)
    assert congress_trades._amount_range_to_floats("") == (0.0, 0.0)


def test_parse_lambda_row_maps_canonical_shape():
    row = {
        "symbol": "AAPL",
        "name": "Pelosi, Nancy",
        "chamber": "House",
        "party": "Democrat",
        "state": "CA",
        "transaction_date": "2026-04-15",
        "transaction_type": "Purchase",
        "amount_min": 1_000_001,
        "amount_max": 5_000_000,
        "filing_date": "2026-04-30",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["filer"] == "Pelosi, Nancy"
    assert t["chamber"] == "House"
    assert t["party"] == "D"
    assert t["state"] == "CA"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 1_000_001
    assert t["amount_max"] == 5_000_000
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-04-30"
    assert t["filing_lag_days"] == 15


def test_parse_lambda_row_accepts_camelcase_and_range_string():
    """Tolerate camelCase fields and an ``amount`` range string in lieu of
    paired ``amount_min``/``amount_max`` — the live response shape isn't
    pinned in the public docs."""
    row = {
        "name": "Tuberville, Tommy",
        "chamber": "Senate",
        "party": "Republican",
        "state": "AL",
        "tradeDate": "04/15/2026",
        "type": "sell",
        "amount": "$15,001 - $50,000",
        "disclosure_date": "05/02/2026",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["chamber"] == "Senate"
    assert t["party"] == "R"
    assert t["type"] == "Sale"
    assert t["amount_min"] == 15_001
    assert t["amount_max"] == 50_000
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-05-02"


def test_parse_lambda_row_matches_live_api_shape():
    """Field names + shape captured from a real
    /api/congressional/trades response (AAPL, 2026-05). Older Senate rows
    have null party/state and lowercase chamber — verify we normalise."""
    row = {
        "symbol": "AAPL",
        "representative": "Thomas R Carper",
        "transactionDate": "2020-11-20",
        "disclosureDate": None,
        "type": "Sale (Partial)",
        "amount": "$50,001 - $100,000",
        "owner": "Spouse",
        "assetDescription": "Apple Inc.",
        "party": None,
        "state": None,
        "district": None,
        "chamber": "senate",
        "ptrLink": "https://efdsearch.senate.gov/...",
        "capGainsOver200": None,
        "comment": "--",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["filer"] == "Thomas R Carper"
    assert t["date"] == "2020-11-20"
    assert t["chamber"] == "Senate"  # title-cased from lowercase "senate"
    assert t["party"] == "—"          # null → placeholder
    assert t["state"] == "—"
    assert t["type"] == "Sale"        # "Sale (Partial)" normalises
    assert t["amount_min"] == 50_001
    assert t["amount_max"] == 100_000
    assert t["filing_date"] == "—"    # null disclosureDate → placeholder
    assert t["filing_lag_days"] is None


def test_extract_lambda_rows_matches_live_envelope():
    """Live envelope is ``{"trades": [...], "total": ..., "page": ..., "limit": ..., "hasMore": ...}``
    — no top-level ``data`` wrapper."""
    body = {
        "trades": [{"symbol": "AAPL", "representative": "X"}],
        "total": 436,
        "page": 0,
        "limit": 1,
        "hasMore": True,
    }
    rows = congress_trades._extract_lambda_rows(body)
    assert rows == [{"symbol": "AAPL", "representative": "X"}]


def test_extract_lambda_rows_handles_envelope_variants():
    rows = [{"symbol": "AAPL", "name": "X"}]
    assert congress_trades._extract_lambda_rows({"status": "ok", "data": rows}) == rows
    assert congress_trades._extract_lambda_rows({"data": {"trades": rows}}) == rows
    assert congress_trades._extract_lambda_rows({"data": {"results": rows}}) == rows
    assert congress_trades._extract_lambda_rows(rows) == rows
    assert congress_trades._extract_lambda_rows({"status": "ok"}) == []
    assert congress_trades._extract_lambda_rows("garbage") == []


def test_lambda_amounts_falls_back_through_field_variants():
    assert congress_trades._lambda_amounts({"amount_min": 1000, "amount_max": 5000}) == (1000, 5000)
    assert congress_trades._lambda_amounts({"amountFrom": 1000, "amountTo": 5000}) == (1000, 5000)
    assert congress_trades._lambda_amounts({"amount": "$1,001 - $15,000"}) == (1001, 15000)
    assert congress_trades._lambda_amounts({"amount": 50_000}) == (50_000, 50_000)
    assert congress_trades._lambda_amounts({}) == (0.0, 0.0)


def test_normalise_type_canonicalises_variants():
    assert congress_trades._normalise_type("Purchase") == "Purchase"
    assert congress_trades._normalise_type("buy") == "Purchase"
    assert congress_trades._normalise_type("Full Sale") == "Sale"
    assert congress_trades._normalise_type("sell") == "Sale"
    assert congress_trades._normalise_type("exchange") == "Exchange"
    assert congress_trades._normalise_type("") == "—"
    # Lambda Finance abbreviated codes with qualifiers
    assert congress_trades._normalise_type("S (Partial)") == "Sale"
    assert congress_trades._normalise_type("P (Full)") == "Purchase"
    assert congress_trades._normalise_type("E (Partial)") == "Exchange"
    assert congress_trades._normalise_type("S") == "Sale"
    assert congress_trades._normalise_type("P") == "Purchase"


def test_within_cutoff_excludes_old_trades():
    cutoff = date.today() - timedelta(days=180)
    old = congress_trades._Trade(date=(cutoff - timedelta(days=10)).isoformat(), type="Purchase",
                                  filer="X", amount_min=0, amount_max=0, amount_label="—",
                                  chamber="Senate", party="—", state="—",
                                  filing_date="—", filing_lag_days=None)
    new = congress_trades._Trade(date=(cutoff + timedelta(days=5)).isoformat(), type="Purchase",
                                  filer="Y", amount_min=0, amount_max=0, amount_label="—",
                                  chamber="Senate", party="—", state="—",
                                  filing_date="—", filing_lag_days=None)
    assert congress_trades._within_cutoff(new, cutoff) is True
    assert congress_trades._within_cutoff(old, cutoff) is False


# --- Reporting --------------------------------------------------------------


def test_format_report_emits_sentiment_and_table():
    trades = [
        congress_trades._Trade(
            date="2026-04-15", filer="Pelosi, Nancy", chamber="House", party="D", state="CA",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="2026-04-30", filing_lag_days=15,
        ),
        congress_trades._Trade(
            date="2026-04-10", filer="Tuberville, Tommy", chamber="Senate", party="R", state="AL",
            type="Sale", amount_min=15_001, amount_max=50_000,
            amount_label="$15k – $50k", filing_date="2026-04-25", filing_lag_days=15,
        ),
    ]
    report = congress_trades._format_report("AAPL", trades, lookback_days=180,
                                             source_label="Finnhub (House + Senate)")
    assert "AAPL" in report
    assert "1 unique buyer" in report and "1 unique seller" in report
    assert "net +0 buyers" in report
    assert "Pelosi, Nancy" in report and "Tuberville, Tommy" in report
    assert "House/D/CA" in report and "Senate/R/AL" in report
    assert "Finnhub (House + Senate)" in report


# --- Source-fallback chain --------------------------------------------------


def test_get_congress_trades_falls_through_to_senate_when_no_finnhub_key(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": "",
    })
    captured = {}

    def fake_ssw(ticker, cutoff):
        captured["called_with"] = ticker
        return [
            congress_trades._Trade(
                date=(date.today() - timedelta(days=5)).isoformat(),
                filer="Tuberville, Tommy", chamber="Senate", party="R", state="AL",
                type="Purchase", amount_min=15_001, amount_max=50_000,
                amount_label="$15k – $50k", filing_date="—", filing_lag_days=None,
            )
        ]

    monkeypatch.setattr(congress_trades, "_fetch_senate_stock_watcher", fake_ssw)
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert captured.get("called_with") == "AAPL"
    assert "Senate Stock Watcher" in out
    assert "Tuberville" in out


def test_get_congress_trades_falls_through_when_finnhub_raises(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "",
    })
    monkeypatch.setattr(
        congress_trades, "_fetch_finnhub",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda ticker, cutoff: [],
    )
    out = congress_trades.get_congress_trades("XYZ", lookback_days=30)
    # Both sources empty → bracketed fallback that mentions the failure
    assert out.startswith("[Congressional disclosures unavailable")
    assert "boom" in out


def test_get_congress_trades_returns_finnhub_data_when_key_present(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "",
    })
    expected = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="—", state="—",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_finnhub", lambda *a, **kw: expected)
    # Ensure SSW isn't called when Finnhub succeeds
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW should not be called when Finnhub returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Finnhub (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_lambda_finance_is_tried_before_finnhub_when_key_present(monkeypatch, tmp_path):
    """Lambda is the new primary — Finnhub must not be touched if Lambda
    returns rows."""
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "test-lambda-key",
    })
    expected = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="D", state="CA",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_lambda_finance", lambda *a, **kw: expected)
    monkeypatch.setattr(
        congress_trades, "_fetch_finnhub",
        lambda *a, **kw: pytest.fail("Finnhub must not be called when Lambda returns data"),
    )
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW must not be called when Lambda returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Lambda Finance (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_falls_through_to_finnhub_when_lambda_returns_empty(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "test-lambda-key",
    })
    finnhub_rows = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="—", state="—",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_lambda_finance", lambda *a, **kw: [])
    monkeypatch.setattr(congress_trades, "_fetch_finnhub", lambda *a, **kw: finnhub_rows)
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW must not be called when Finnhub returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Finnhub (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_falls_through_when_lambda_raises(monkeypatch, tmp_path):
    """A Lambda HTTP failure must not abort the agent — chain continues."""
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": "test-lambda-key",
    })
    monkeypatch.setattr(
        congress_trades, "_fetch_lambda_finance",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("lambda 502")),
    )
    monkeypatch.setattr(congress_trades, "_fetch_senate_stock_watcher", lambda *a, **kw: [])
    out = congress_trades.get_congress_trades("XYZ", lookback_days=30)
    assert out.startswith("[Congressional disclosures unavailable")
    assert "lambda 502" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("FINNHUB_API_KEY"), reason="FINNHUB_API_KEY unset")
def test_get_congress_trades_live_finnhub_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": os.environ["FINNHUB_API_KEY"],
        "lambda_finance_api_key": "",
    })
    out = congress_trades.get_congress_trades("AAPL", lookback_days=180)
    assert isinstance(out, str) and out
    # Either real data or graceful no-data — never an unhandled crash
    assert out.startswith("##") or out.startswith("[Congressional disclosures unavailable")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LAMBDA_FINANCE_API_KEY"), reason="LAMBDA_FINANCE_API_KEY unset")
def test_get_congress_trades_live_lambda_finance_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": os.environ["LAMBDA_FINANCE_API_KEY"],
    })
    out = congress_trades.get_congress_trades("AAPL", lookback_days=180)
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[Congressional disclosures unavailable")
