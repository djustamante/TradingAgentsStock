"""Tests for the recommended-options-strategies feature.

Adds three things this test file pins:

1. ``OptionsStrategy`` Pydantic schema parses cleanly and survives a
   JSON round-trip (the structured-output path serialises it through
   the OpenAI/Anthropic/Gemini schema layer).
2. ``PortfolioDecision`` keeps backwards-compat: ``recommended_strategies``
   is optional (defaults to ``[]``) so the existing call sites and the
   no-options-data path don't break.
3. ``render_pm_decision`` emits a clean Markdown table when strategies
   are present, and the existing rendering stays unchanged when not.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.schemas import (
    OptionsStrategy,
    PortfolioDecision,
    PortfolioRating,
    render_pm_decision,
)


def _sample_strategy(**overrides) -> OptionsStrategy:
    defaults = dict(
        name="Bull Call Spread",
        legs="Buy AAPL Jun 200C, Sell AAPL Jun 210C",
        rationale="Rating is Buy and IV Rank ~25; long premium is cheap.",
        max_profit="+$8.20/share if AAPL >= 210 at June expiry",
        max_loss="-$1.80/share net premium paid",
    )
    defaults.update(overrides)
    return OptionsStrategy(**defaults)


# --- Schema -----------------------------------------------------------------


def test_options_strategy_parses_required_fields():
    s = _sample_strategy()
    assert s.name == "Bull Call Spread"
    assert "Jun 200C" in s.legs
    assert "+$8.20" in s.max_profit
    assert "-$1.80" in s.max_loss


def test_options_strategy_rejects_missing_fields():
    """Every field is required (no defaults). A model that returns a
    partial strategy must fail validation rather than render gibberish."""
    with pytest.raises(Exception):  # pydantic.ValidationError subclass
        OptionsStrategy(name="X")  # type: ignore[call-arg]


def test_options_strategy_round_trips_via_json():
    """The structured-output transport serialises the model through JSON.
    Ensure the dump-then-parse cycle preserves all fields."""
    original = _sample_strategy()
    parsed = OptionsStrategy.model_validate_json(original.model_dump_json())
    assert parsed == original


def test_portfolio_decision_strategies_default_to_empty_list():
    """Backwards compat: old call sites that don't pass strategies must
    still build a valid PortfolioDecision."""
    d = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="hold",
        investment_thesis="balanced",
    )
    assert d.recommended_strategies == []


def test_portfolio_decision_accepts_exactly_three_strategies():
    """Schema permits any list length (the prompt asks for exactly 3);
    pin the realistic 3-strategy case round-trips."""
    strats = [
        _sample_strategy(name=f"Strategy {i}") for i in range(1, 4)
    ]
    d = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="buy",
        investment_thesis="strong setup",
        recommended_strategies=strats,
    )
    assert len(d.recommended_strategies) == 3
    assert [s.name for s in d.recommended_strategies] == [
        "Strategy 1", "Strategy 2", "Strategy 3",
    ]


# --- Rendering --------------------------------------------------------------


def test_render_pm_decision_omits_table_when_no_strategies():
    """The PR must not regress the existing render for decisions without
    strategies — old reports keep their existing shape."""
    d = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="hold",
        investment_thesis="balanced",
    )
    out = render_pm_decision(d)
    assert "Recommended Options Strategies" not in out
    assert "**Rating**: Hold" in out


def test_render_pm_decision_emits_strategy_table_when_present():
    strats = [
        _sample_strategy(name="Bull Call Spread"),
        _sample_strategy(
            name="Cash-Secured Put",
            legs="Sell AAPL Jun 195P",
            max_profit="+$3.50/share premium received",
            max_loss="capped at -$195/share assignment cost minus premium",
        ),
        _sample_strategy(
            name="Calendar Spread",
            legs="Buy AAPL Sep 200C, Sell AAPL Jun 200C",
            max_profit="varies with IV term structure",
            max_loss="-$2.40/share net debit",
        ),
    ]
    d = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="buy",
        investment_thesis="strong setup",
        recommended_strategies=strats,
    )
    out = render_pm_decision(d)

    assert "### Recommended Options Strategies" in out
    assert "| # | Strategy | Legs | Max Profit / Loss | Rationale |" in out
    # All three strategies appear in row order
    assert "| 1 | Bull Call Spread |" in out
    assert "| 2 | Cash-Secured Put |" in out
    assert "| 3 | Calendar Spread |" in out
    # The legs columns carry the strike numbers verbatim
    assert "Buy AAPL Jun 200C" in out
    assert "Sell AAPL Jun 195P" in out


def test_render_pm_decision_escapes_pipe_in_strategy_text():
    """A strategy description with a pipe character would otherwise break
    the markdown table layout — must be escaped to &#124;."""
    bad_strat = _sample_strategy(
        legs="Sell put | Buy further OTM put | Net credit",
    )
    d = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="hold",
        investment_thesis="x",
        recommended_strategies=[bad_strat],
    )
    out = render_pm_decision(d)
    # The literal pipe shouldn't appear inside a cell
    table_row = next(l for l in out.splitlines() if l.startswith("| 1 |"))
    # Count of unescaped | in the row equals the number of column separators
    # (5 cells = 6 separators). Embedded pipes would push that count up.
    assert table_row.count("|") == 6
    # And the user content stays readable via the HTML entity
    assert "&#124;" in out


def test_render_pm_decision_replaces_newlines_in_cells():
    """Multi-line strategy text must collapse to single-line cells so the
    markdown table doesn't break across rows."""
    strat = _sample_strategy(rationale="line one\nline two\nline three")
    d = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="hold",
        investment_thesis="x",
        recommended_strategies=[strat],
    )
    out = render_pm_decision(d)
    table_row = next(l for l in out.splitlines() if l.startswith("| 1 |"))
    assert "\n" not in table_row[2:]  # no embedded newlines in the data row
    assert "line one line two line three" in table_row


# --- Portfolio Manager prompt threading -----------------------------------


def _make_minimal_pm_state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "past_context": "",
        "risk_debate_state": {
            "history": "risk history",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "investment_plan": "research plan",
        "trader_investment_plan": "trader plan",
    }
    state.update(overrides)
    return state


def test_pm_prompt_includes_options_context_when_state_has_options_data():
    """When the state carries options_report and iv_snapshot, the PM
    prompt must include them verbatim AND include the strategies
    instruction telling the LLM to populate recommended_strategies."""
    from unittest.mock import MagicMock
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    captured = {}

    class _LLM:
        def with_structured_output(self, schema):
            structured = MagicMock()
            def _invoke(prompt):
                captured["prompt"] = prompt
                return PortfolioDecision(
                    rating=PortfolioRating.HOLD,
                    executive_summary="x",
                    investment_thesis="y",
                )
            structured.invoke = _invoke
            return structured

        def invoke(self, prompt):
            captured.setdefault("prompt", prompt)
            return MagicMock(content="**Rating**: Hold")

    state = _make_minimal_pm_state(
        options_report="Max-pain strike: $200. Call wall: $210. Put wall: $190.",
        iv_snapshot="IV Rank for AAPL: 28/100 (low; long premium is cheap).",
    )
    pm_node = create_portfolio_manager(_LLM())
    pm_node(state)

    prompt = captured["prompt"]
    # Options-market context block is present + carries real strikes
    assert "Options-Market Context" in prompt
    assert "Max-pain strike: $200" in prompt
    assert "IV Rank for AAPL: 28/100" in prompt
    # Strategies instruction is present + names the direction × IV map
    assert "RECOMMENDED OPTIONS STRATEGIES" in prompt
    assert "EXACTLY THREE" in prompt
    assert "DO NOT invent strikes" in prompt


def test_pm_prompt_omits_strategies_when_no_options_data():
    """When state has neither options_report nor iv_snapshot, the PM
    must be instructed to return an empty list — never fabricate."""
    from unittest.mock import MagicMock
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    captured = {}

    class _LLM:
        def with_structured_output(self, schema):
            structured = MagicMock()
            def _invoke(prompt):
                captured["prompt"] = prompt
                return PortfolioDecision(
                    rating=PortfolioRating.HOLD,
                    executive_summary="x",
                    investment_thesis="y",
                )
            structured.invoke = _invoke
            return structured

        def invoke(self, prompt):
            captured.setdefault("prompt", prompt)
            return MagicMock(content="**Rating**: Hold")

    state = _make_minimal_pm_state()  # no options_report / iv_snapshot
    pm_node = create_portfolio_manager(_LLM())
    pm_node(state)

    prompt = captured["prompt"]
    assert "Options-Market Context" not in prompt
    # The instruction still exists, but in its empty-list form
    assert "no options report is available" in prompt.lower()
    assert "empty list" in prompt.lower()
