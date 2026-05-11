"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config


def _count_in_words(n: int) -> str:
    """Render a small integer as English so the LLM picks it up as the
    primary instruction (numerals embedded in long prompts get glossed
    over by some free-tier models)."""
    words = {
        0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
        6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN",
    }
    return words.get(n, str(n))


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        # Options data feeds the recommended_strategies field. Both are
        # optional — empty values short-circuit the strategy-picking
        # instruction so the LLM doesn't hallucinate strikes when no
        # options data is available (e.g. an instrument with no listed
        # options, or an upstream tool failure).
        options_report = state.get("options_report") or ""
        iv_snapshot = state.get("iv_snapshot") or ""

        # How many strategies to pick. Sourced from runtime config so the
        # pipeline ``--strategies N`` CLI flag and direct config overrides
        # both reach the PM through the same path. 0 disables the feature.
        try:
            strategy_count = int(get_config().get("options_strategies_count", 3))
        except (TypeError, ValueError):
            strategy_count = 3
        strategy_count = max(0, min(strategy_count, 10))

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        if strategy_count == 0:
            options_block = ""
            strategies_instruction = (
                "\n**RECOMMENDED OPTIONS STRATEGIES** — the strategy-count "
                "config is set to 0 for this run. Return an EMPTY LIST for "
                "``recommended_strategies``.\n"
            )
        elif options_report or iv_snapshot:
            count_word = _count_in_words(strategy_count)
            options_block = (
                "\n**Options-Market Context** "
                "(use these strikes verbatim; do NOT invent strike values):\n"
                f"{iv_snapshot}\n\n"
                f"{options_report}\n"
            )
            strategies_instruction = (
                f"\n**RECOMMENDED OPTIONS STRATEGIES** — populate the "
                f"``recommended_strategies`` field with EXACTLY {count_word} "
                f"({strategy_count}) concrete strategies that match the rating "
                f"direction crossed with the current IV regime. Pick from this "
                f"map (strategy choice depends on direction × volatility):\n"
                "- **Buy / Overweight + low IV** (IV Rank ≲ 30): long call, "
                "bull call spread, long stock + collar with skewed wings.\n"
                "- **Buy / Overweight + high IV** (IV Rank ≳ 60): cash-secured "
                "put, covered call (if existing position), bull put spread.\n"
                "- **Hold** (any IV): iron condor between the put wall and "
                "call wall, calendar spread at ATM, covered call.\n"
                "- **Underweight / Sell + low IV**: long put, bear put spread, "
                "protective put on existing position.\n"
                "- **Underweight / Sell + high IV**: bear call spread, sell "
                "naked call (only if institutional risk limits permit), "
                "ratio call spread.\n"
                "\nEvery strike you cite in ``legs`` MUST come from the "
                "Options-Market Context above (spot, max-pain, call wall, "
                "put wall, or strikes appearing in the unusual-activity "
                "table). DO NOT invent strikes. If the options data is "
                "thin or unavailable, return AN EMPTY LIST for "
                "``recommended_strategies`` rather than hallucinating.\n"
            )
        else:
            options_block = ""
            strategies_instruction = (
                "\n**RECOMMENDED OPTIONS STRATEGIES** — no options report is "
                "available for this run. Return an EMPTY LIST for "
                "``recommended_strategies``; do not fabricate strategies.\n"
            )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}
{options_block}
---
{strategies_instruction}
Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
