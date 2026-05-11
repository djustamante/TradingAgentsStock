"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class OptionsStrategy(BaseModel):
    """One concrete options strategy attached to a Portfolio Manager verdict.

    Strategies are picked by the LLM to match the rating direction crossed
    with the implied-volatility regime (cheap premium → buy options;
    expensive premium → sell options). Every strike referenced in ``legs``
    MUST come from the options report that fed the prompt — the LLM is
    instructed not to invent strikes.
    """

    name: str = Field(
        description=(
            "Strategy name, e.g. 'Bull Call Spread', 'Cash-Secured Put', "
            "'Iron Condor', 'Covered Call', 'Calendar Spread', "
            "'Protective Put', 'Bear Put Spread'. Keep to the canonical "
            "name an options trader would recognize."
        ),
    )
    legs: str = Field(
        description=(
            "Concrete legs with real strikes and an approximate expiration. "
            "Example: 'Buy AAPL Jun 200C, Sell AAPL Jun 210C'. Strikes MUST "
            "match the options report supplied in the prompt; do not "
            "fabricate values. If only a max-pain / call-wall / put-wall "
            "level is available, anchor strikes near those."
        ),
    )
    rationale: str = Field(
        description=(
            "One or two sentences explaining why this strategy fits the "
            "rating direction crossed with the IV regime. Cite the IV rank "
            "and at least one specific options-report data point."
        ),
    )
    max_profit: str = Field(
        description=(
            "Maximum profit at expiry as a per-share or per-contract figure, "
            "e.g. '+$8.20/share if AAPL >= 210 at June expiry' or "
            "'+$1.50/share net premium received'."
        ),
    )
    max_loss: str = Field(
        description=(
            "Maximum loss at expiry, also per-share or per-contract. For "
            "naked-short legs (cash-secured put, short call) state the "
            "capital-at-risk explicitly. E.g. '-$1.80/share premium paid' "
            "or 'capped at -$195/share assignment cost minus premium'."
        ),
    )


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    recommended_strategies: List[OptionsStrategy] = Field(
        default_factory=list,
        description=(
            "Exactly THREE options strategies appropriate for the rating "
            "direction crossed with the current IV regime. Pick strategies "
            "that complement the equity rating: e.g. Buy rating + low IV "
            "favours long calls / bull call spreads; Buy + high IV favours "
            "cash-secured puts / covered calls; Hold favours iron condors / "
            "calendar spreads / covered calls; Sell + high IV favours bear "
            "call spreads; Sell + low IV favours long puts / bear put "
            "spreads. Cite real strikes from the options report — never "
            "invent values. If the options report is empty / unavailable, "
            "return an empty list rather than hallucinated strategies."
        ),
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.recommended_strategies:
        parts.append("")
        parts.append(_render_strategies_table(decision.recommended_strategies))
    return "\n".join(parts)


def _render_strategies_table(strategies: List[OptionsStrategy]) -> str:
    """Format an options-strategy list as a Markdown table.

    The table uses pipe-escaping on every cell so a strategy description
    containing a ``|`` (rare but possible — e.g. a leg specification like
    ``'Sell put | Buy further OTM put'``) doesn't corrupt the table layout.
    """
    lines = [
        "### Recommended Options Strategies",
        "",
        "| # | Strategy | Legs | Max Profit / Loss | Rationale |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(strategies, start=1):
        profit_loss = f"{_escape_pipe(s.max_profit)} / {_escape_pipe(s.max_loss)}"
        lines.append(
            f"| {i} | {_escape_pipe(s.name)} | {_escape_pipe(s.legs)} | "
            f"{profit_loss} | {_escape_pipe(s.rationale)} |"
        )
    return "\n".join(lines)


def _escape_pipe(text: str) -> str:
    """Escape ``|`` characters so they don't break markdown table cells."""
    if not isinstance(text, str):
        return str(text)
    # Replace pipe with HTML entity. Markdown renderers display it as ``|``
    # but the table parser doesn't see it as a column separator.
    return text.replace("\n", " ").replace("|", "&#124;")
