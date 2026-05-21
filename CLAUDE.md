# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TradingAgents is a multi-agent LLM trading research framework built on LangGraph. A pipeline of specialized agents (analysts → researchers → research manager → trader → risk debators → portfolio manager) produces a Buy/Hold/Sell-style decision for a given ticker + date.

This repo also hosts a thin **screener pipeline** at the project root (`pipeline.py`, `config.py`, `screener/`) that pulls candidates from Finviz and feeds each through `TradingAgentsGraph` in a batch.

## Common commands

Install (editable, into a Python ≥ 3.10 env):
```bash
pip install -e .
```

Run the interactive Typer/Rich CLI:
```bash
tradingagents analyze                    # installed entrypoint
python -m cli.main analyze               # equivalent
tradingagents analyze --checkpoint       # resume after crash on next invocation with same ticker+date
tradingagents analyze --clear-checkpoints
```

Run as a library: see `main.py` (one-shot `TradingAgentsGraph(...).propagate(ticker, date)`).

Run the screener pipeline (batch run over Finviz candidates):
```bash
python pipeline.py
```
Knobs live in `config.py` at the repo root: `finviz_filters`, `max_tickers_per_run`, `cache_ttl_hours`, plus a `tradingagents_config` block that copies `DEFAULT_CONFIG` and applies screener-level overrides (LLM provider, per-role models). The screener writes one JSON per analysis to `results/by_ticker/{TICKER}/` with a relative symlink under `results/by_run/{YYYY_MM_DD_HH_mm_ss}/` (Windows-safe stub fallback). The same per-run folder also holds `pipeline.log` when the run is launched via `run.sh` (which backgrounds analyze runs and redirects stdout+stderr there).

Tests (pytest, configured in `pyproject.toml`):
```bash
pytest                                   # full suite
pytest -k test_signal_processing         # single file/test
pytest -m unit                           # by marker (unit / integration / smoke)
pytest -m integration                    # live-network tests; some auto-skip on missing API keys
```
`tests/conftest.py` autouses a fixture that injects placeholder values for every supported LLM/data API key, so tests do not hang on missing credentials. The `mock_llm_client` fixture patches `tradingagents.llm_clients.factory.create_llm_client`.

There is no configured linter/formatter; do not invent one.

Docker:
```bash
docker compose run --rm tradingagents
docker compose --profile ollama run --rm tradingagents-ollama
```

## Architecture

### Pipeline shape (`tradingagents/graph/`)

`TradingAgentsGraph` (in `graph/trading_graph.py`) is the single orchestration entry point. On `__init__` it:
1. Calls `set_config(self.config)` so `dataflows.config` (a process-wide singleton) reflects the run's settings — agent tools read this lazily.
2. Optionally appends `"options"` to `selected_analysts` when `config["enable_options_analyst"]` is true.
3. Builds deep- and quick-thinking `BaseLLMClient`s via `llm_clients.factory.create_llm_client`.
4. Constructs a **role-keyed LLM map** (`self.role_llms`) via `_build_role_llms` — see "Per-role LLM routing" below.
5. Builds five `ToolNode`s keyed by analyst type (`market`, `social`, `news`, `fundamentals`, `options`).
6. Hands these to `GraphSetup.setup_graph(selected_analysts)` which wires the LangGraph `StateGraph` (see `graph/setup.py`). The state schema is `AgentState` from `agents/utils/agent_states.py`.

`propagate(ticker, date)` does several things in order before invoking the graph:
- **Resolves pending memory-log entries** for the ticker via `TradingMemoryLog` (fetches realised return + alpha vs SPY for prior decisions, generates reflections via `Reflector`, batch-writes back).
- **Pre-fetches macro and IV snapshots** (`_safe_macro_snapshot()` and `_safe_iv_snapshot(ticker)`), which the prompt-only risk debaters reference from `state["macro_snapshot"]` / `state["iv_snapshot"]` — they have no ToolNode of their own. Both fetches swallow errors and fall back to empty strings.
- **Optionally recompiles with a checkpointer** (`graph/checkpointer.py`) when `config["checkpoint_enabled"]` is true — per-ticker SQLite DBs live under `<data_cache_dir>/checkpoints/<TICKER>.db`. `thread_id` is `sha256(TICKER:date)[:16]`, so the same ticker+date resumes and a different date starts fresh.
- **Injects past_context, macro_snapshot, iv_snapshot** into the initial state via `Propagator.create_initial_state`.

After the graph returns, the final state is logged as JSON under `<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`, the decision is appended (pending) to the memory log, and the checkpoint is cleared.

### Agents (`tradingagents/agents/`)

Each subpackage exposes a `create_<role>(llm)` factory used by `GraphSetup`. The roles are:
- `analysts/` — market, social, news, fundamentals, **options** (call tools, produce a section report).
- `researchers/` — bull and bear (debate `max_debate_rounds` rounds).
- `managers/research_manager.py` — synthesises the debate into an investment plan.
- `trader/trader.py` — turns the plan into a Buy/Hold/Sell transaction proposal.
- `risk_mgmt/` — aggressive, neutral, conservative debators.
- `managers/portfolio_manager.py` — final decision; receives `past_context` from the memory log.

Analyst chain runs **sequentially** (not in parallel): `market → social → news → fundamentals → options → bull/bear debate → research_mgr → trader → risk debate → portfolio_mgr`. Each analyst's prompt now includes the dataflow tools relevant to its mandate (see "Data flow" below).

The Research Manager, Trader, and Portfolio Manager use **provider-native structured output** (json_schema for OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic). Schemas live in `agents/schemas.py`; `agents/utils/structured.py` renders the parsed Pydantic instance back to the markdown shape the rest of the pipeline already consumes. Other agents stay free-form.

Internal debate agents always run in English (for reasoning quality). User-facing agents respect `config["output_language"]` via `agent_utils.get_language_instruction()`.

### Per-role LLM routing

`TradingAgentsGraph._build_role_llms` constructs `self.role_llms`, a dict of role-keyed LLMs that `GraphSetup` hands to each agent factory. Role keys:

| Role | Config key | Used by |
|---|---|---|
| `deep` | `deep_think_llm` | always populated (fallback for `structured_output`) |
| `quick` | `quick_think_llm` | fundamentals, bull, bear, trader; fallback for `quant`/`light` |
| `structured_output` | `structured_output_llm` | research manager, portfolio manager |
| `quant` | `quant_llm` | market, options, all 3 risk debaters |
| `light` | `light_llm` | social, news |

Empty role-config strings fall back: `structured_output → deep`, `quant → quick`, `light → quick`. The map is memoised by `(model, fallback_tuple)` so two roles pointing at the same model+fallback chain share a single client.

Each role also accepts an optional `*_llm_fallbacks` list (e.g. `quick_think_llm_fallbacks: [...]`). When non-empty, `_build_chat_with_fallbacks` wraps the primary in a `FallbackChatModel` (see `tradingagents/llm_clients/fallback.py`) that retries against each fallback on recoverable upstream errors — `429`, `5xx`, request timeout, transport drop. Auth/schema/other-4xx errors are *not* caught (they would fail again on the next model and mask the real bug). Entries are plain strings (use the run's `llm_provider`), `(provider, model)` tuples, or `{"provider": ..., "model": ...}` dicts for cross-provider fallback.

### Data flow (`tradingagents/dataflows/`)

`@tool`-decorated wrappers in `agents/utils/*_tools.py` delegate to `dataflows.interface.route_to_vendor(method, *args)`. Routing reads `config["tool_vendors"]` first (per-tool override), then `config["data_vendors"]` (per-category default).

Active categories and vendors:

| Category | Vendors | Module(s) |
|---|---|---|
| `core_stock_apis` | yfinance, alpha_vantage | `y_finance.py`, `alpha_vantage_*.py` |
| `technical_indicators` | yfinance, alpha_vantage | same |
| `fundamental_data` | yfinance, alpha_vantage, **lambda_finance** | `lambda_finance_sec.py` (income statement + balance sheet); opt-in via `tool_vendors` |
| `news_data` | yfinance, alpha_vantage | `yfinance_news.py`, `alpha_vantage_news.py` (article bodies wrapped in `<untrusted_content>` tags before reaching the LLM) |
| `news_data` (insider) | yfinance, alpha_vantage, **sec** | `sec_insider.py` (Form 4 from SEC EDGAR; parsed with `defusedxml`) |
| `political_data` | **lambda_finance** (with Finnhub + Senate Stock Watcher fallback inside) | `congress_trades.py` — Senate Stock Watcher fetch is size-capped at 100 MB |
| `peer_comparison_data` | **lambda_finance** | `lambda_finance_compare.py` — `/api/sec/compare`; one row per ticker × N metrics |
| `options_data` | yfinance | `options_flow.py` (P/C ratios, max pain, walls, IVR) |
| `etf_holdings_data` | yfinance | `etf_holdings.py` — sector weights + top-10 + concentration via `Ticker.funds_data` |
| `etf_peer_comparison_data` | yfinance | `etf_peer_compare.py` — profile + returns + risk across peer ETFs |
| `macro_data` | fred | `macro_data.py` (yields, curve, HY spread, USD) |
| `transcript_data` | motley_fool | `earnings_transcript.py` (LLM-scored sentiment; transcript text wrapped in `<untrusted_content>` before scoring) |
| `sector_data` | yfinance | `sector_analysis.py` (RS vs SPY, inter-market correlations) |

`route_to_vendor` falls through to the next vendor only on `AlphaVantageRateLimitError` — other exceptions bubble up. Vendor adapters return strings starting with `[` to indicate graceful failure (so the LLM can keep going); they should never raise.

When adding a new data tool: write the vendor adapter in `dataflows/`, register it in `VENDOR_METHODS` and category in `TOOLS_CATEGORIES` in `dataflows/interface.py`, write a tool wrapper in `agents/utils/<topic>_tools.py` that calls `route_to_vendor`, and re-export from `agents/utils/agent_utils.py` so analyst factories can import it.

API-response cache helper at `dataflows/_cache.py` (file-based JSON, keyed by source + arbitrary dict, explicit TTL per call site). Cache directories created with mode `0o700`; cache files chmod'd to `0o600` so coresident local users can't read external content cached on disk.

### Analyst-quality safety nets (`agents/utils/quality_guard.py`)

Free-tier LLMs occasionally emit degenerate final answers (the canonical case: a Fundamentals analyst returning literally `"Call correct."` instead of a report). `invoke_chain_with_quality_retry` wraps the chain invocation with three layers of defense:

1. **Reasoning-trace strip** — known leak markers (`"We can stop."`, `"due to time limit may stop."`, `"Need atr."`-style prefixes) are removed from the head of LLM content before it lands in state.
2. **Degenerate-output retry** — `is_degenerate_report` flags content that's both short and structureless; helper retries once with a stricter user message.
3. **Cap-aware placeholder** — when the per-analyst tool-round cap (default 12, set via `max_tool_rounds_per_analyst`) is about to force-terminate the analyst, substitute `make_unavailable_report` so the empty section doesn't silently disappear from the rendered output.

Conditional-logic enforces the round cap (`graph/conditional_logic.py` — counts AIMessages with `tool_calls` in `state["messages"]` since the previous Msg-Clear node wipes them between analysts). Market, Fundamentals, and News analysts use the helper; Social and Options haven't been observed misbehaving so they call the chain directly.

### Graph instrumentation (`graph/setup.py`)

Every workflow node is wrapped by `_with_checkpoint(name, node_fn)` so the pipeline log carries `ENTER <Node Name> | <ticker>` / `EXIT  <Node Name> | <ticker> | <elapsed>s` markers (and `FAIL  ...` on exception). The wrapper detects Runnable-shaped nodes (ToolNode is a Pydantic Runnable with `invoke` but no `__call__`) and dispatches via `.invoke` for those — calling them as plain functions raises `TypeError`.

### LLM fallback chain (`llm_clients/fallback.py`)

`FallbackChatModel` wraps a primary chat model with an ordered list of fallbacks. Recoverable upstream errors (`openai.RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`, and **`NotFoundError`** for delisted models — plus the anthropic and google equivalents) trigger fallback to the next model; auth, schema, and other 4xx errors propagate immediately. `bind_tools` and `with_structured_output` preserve the chain. Used by `_build_chat_with_fallbacks` in `graph/trading_graph.py` when a role's `*_llm_fallbacks` list is non-empty.

### Structured output (`agents/schemas.py`, `agents/utils/structured.py`)

Research Manager, Trader, and Portfolio Manager use provider-native structured output. `NormalizedChatOpenAI.with_structured_output` defaults to `json_schema` for Chat Completions endpoints (OpenRouter etc.) — `function_calling` was previously the default but fails on free-tier models that emit lowercase tool names (`'researchPlan'` vs `'ResearchPlan'`). Native OpenAI (Responses API) keeps `function_calling` because the json_schema path emits noisy Pydantic warnings there.

`invoke_structured_or_freetext` falls back to free-text generation when the structured call raises OR when the rendered structured output is empty/whitespace-only. If BOTH paths produce empty content, it substitutes a bracketed `"[{agent_name} could not produce a verdict ...]"` placeholder so the failure is visible to the renderer.

The Portfolio Manager's `PortfolioDecision` schema includes a `recommended_strategies: List[OptionsStrategy]` field. Strategies are rendered as a Markdown table by `render_pm_decision`; the prompt threads `options_report` + `iv_snapshot` from state and instructs the LLM to cite real strikes (never invent). Count configured via `options_strategies_count` (default 3, range 0-10, 0 disables; CLI override: `--strategies N`).

### LLM clients (`tradingagents/llm_clients/`)

`factory.create_llm_client(provider, model, base_url, **kwargs)` lazily imports the right backend. Providers in `_OPENAI_COMPATIBLE` (`openai`, `xai`, `deepseek`, `qwen`, `glm`, `ollama`, `openrouter`) all go through `OpenAIClient` with a provider-specific base URL. `anthropic`, `google`, and `azure` have dedicated clients. Provider-specific reasoning/thinking knobs are passed through from config: `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`.

`OpenAIClient` defaults `max_retries=6` (overridable via `config["max_retries"]`). The openai SDK retries 429 and transient 5xx with exponential-backoff jitter — needed because OpenRouter free-tier models share upstream-provider quota pools that throttle aggressively.

`config["backend_url"]` defaults to **`None`** intentionally — each client falls back to its own provider default. Do not put an OpenAI URL in `DEFAULT_CONFIG`; it would leak into other providers (see commit `4016fd4`). The CLI sets it per-provider when the user picks one.

`model_catalog.py` is the single source of truth for which models appear in the CLI selector and is used by validators.

### Persistence

Everything user-state-shaped lives under `~/.tradingagents/` by default:
- `logs/` — full run state JSON (override: `TRADINGAGENTS_RESULTS_DIR`).
- `cache/checkpoints/<TICKER>.db` — LangGraph SqliteSaver (override base: `TRADINGAGENTS_CACHE_DIR`).
- `cache/api/<source>/<hash>.json` — file-based API-response cache (`dataflows/_cache.py`). Directories `0o700`, files `0o600`.
- `memory/trading_memory.md` — append-only decision log (override: `TRADINGAGENTS_MEMORY_LOG_PATH`). Same `0o700`/`0o600` mode hardening; coresident local users cannot read.

The memory log uses an HTML-comment separator (`<!-- ENTRY_END -->`) as a hard delimiter. The literal marker is **escaped on write** (`TradingMemoryLog._escape_separator` replaces it with `<!-- ENTRY_END__ESCAPED -->`) so poisoned content from news/transcripts can't terminate an entry early. Entries start as `pending` and are resolved in-place once price data is available.

The screener writes its own JSON output under `./results/by_ticker/<TICKER>/<TICKER>_<YYYYMMDD_HHMMSS>.json` with a same-basename relative symlink under `./results/by_run/<YYYY_MM_DD_HH_mm_ss>/`. When invoked through `run.sh`, the run folder also receives `pipeline.log` (full stdout+stderr capture) and the run is detached via `nohup`.

## Coding conventions specific to this repo

- **Always validate ticker symbols** with `tradingagents.dataflows.utils.safe_ticker_component(ticker)` before interpolating into a filesystem path. Tickers come from CLI input *and* LLM tool calls, so they are attacker-influenced (commit `2c97bad`). The regex allows letters, digits, `.`, `-`, `_`, `^`.
- **Configuration is read through `dataflows.config.get_config()`**, not by passing the dict around. `TradingAgentsGraph.__init__` calls `set_config()` once; tools and agents read it lazily so the call site does not need to know about config.
- **Preserve exchange-qualified tickers verbatim** through tool calls (`CNC.TO`, `7203.T`, `0700.HK`). `agent_utils.build_instrument_context` is the canonical instruction; reuse it rather than rephrasing.
- **Don't hardcode provider URLs** in `DEFAULT_CONFIG` or share `base_url` across provider clients (see "LLM clients" above).
- **Vendor adapters never raise.** Wrap your pipeline in `try/except`, log via `logging.getLogger(__name__).warning`, and return `f"[<source> unavailable: {e}. Proceed with available data.]"`. The LLM tolerates these strings; an unhandled exception aborts the agent run.
- **Wrap external content** (news bodies, earnings transcripts, social-media chatter — anything fetched from a third party) in `<untrusted_content source="...">` tags via `tradingagents.dataflows.utils.wrap_untrusted` before it reaches an LLM. The matching `get_untrusted_content_instruction()` clause in analyst system prompts tells the LLM to treat tag content as data, never instructions. Defends against prompt injection from poisoned news / transcript pages.
- **Use `defusedxml`** for any XML parsing of vendor responses (`from defusedxml.ElementTree import fromstring`). Stdlib `xml.etree` is vulnerable to XXE and billion-laughs DoS.
- **Every external HTTP call needs `timeout=`.** The `_TIMEOUT = 30` convention is used across vendor adapters; a missing timeout lets a hung server stall the entire pipeline. Streaming fetches that load full bodies should also enforce a byte-count cap (see `_fetch_with_size_cap` in `congress_trades.py` — 100 MB ceiling on Senate Stock Watcher).
- **Never put API keys in URL query strings.** Use header auth (`X-API-Key`, `X-Finnhub-Token`, ...) so a request URL leaking into an HTTPError's message doesn't leak the credential. FRED is the only known exception (API requires query-param auth); never log FRED request URLs.
- **Risk debaters are prompt-only** — no ToolNodes. To give them new context, pre-fetch in `_run_graph` and inject via `AgentState` (see how `macro_snapshot` / `iv_snapshot` are wired).
- The CLI loads `.env` and then `.env.enterprise` (without override), in that order. `main.py` and the screener `pipeline.py` only load `.env`.
- **Env vars at import time**: `default_config.py` resolves `FRED_API_KEY`, `FINNHUB_API_KEY`, `SEC_USER_AGENT` etc. at module-load time. Anything that imports it must `load_dotenv` first, or rely on the env being set externally. The screener `config.py` does this in the right order.

## Where things live (quick map)

```
pipeline.py                          # screener entry point: Finviz → TradingAgentsGraph → JSON
config.py                            # screener config (CONFIG dict + tradingagents_config overrides)
screener/
  finviz_filter.py                   # get_candidates() with TTL cache
  queue_manager.py                   # already_run_today / build_queue / mark_complete
  cache/                             # finviz_cache.json
results/                             # by_ticker/ + by_run/ — populated at runtime

main.py                              # minimal library-usage example
cli/main.py                          # `tradingagents analyze` Typer app + Rich UI loop
tradingagents/default_config.py      # DEFAULT_CONFIG + env-var overrides + per-role model keys
tradingagents/graph/
  trading_graph.py                   # TradingAgentsGraph (entry point) + _build_role_llms + macro/IV pre-fetch
  setup.py                           # GraphSetup — routes role_llms to agent factories
  checkpointer.py                    # per-ticker SQLite resume
  reflection.py / signal_processing.py / propagation.py / conditional_logic.py
tradingagents/agents/
  schemas.py                         # Pydantic structured-output schemas (incl. OptionsStrategy)
  analysts/{market,social,news,fundamentals,options}_analyst.py
  researchers/{bull,bear}_researcher.py
  managers/{research,portfolio}_manager.py
  trader/trader.py
  risk_mgmt/{aggressive,neutral,conservative}_debator.py
  utils/agent_states.py              # AgentState (TypedDict) — graph state shape
  utils/agent_utils.py               # tool re-exports, language instr, ticker context, untrusted-content instruction
  utils/memory.py                    # TradingMemoryLog — escape ENTRY_END, 0o600 mode
  utils/structured.py                # invoke_structured_or_freetext (placeholder on dual-failure)
  utils/quality_guard.py             # is_degenerate_report + invoke_chain_with_quality_retry
  utils/{core_stock,technical_indicators,fundamental_data,news_data}_tools.py
  utils/{political,options,macro,sector,transcript}_tools.py
  utils/{peer_comparison,etf_holdings,etf_peer_comparison}_tools.py
tradingagents/dataflows/
  interface.py                       # route_to_vendor dispatcher + tool catalog
  _cache.py                          # file-based JSON API-response cache (0o700/0o600)
  y_finance.py / alpha_vantage*.py   # original vendor adapters
  sec_insider.py                     # SEC EDGAR Form 4 (defusedxml-parsed)
  congress_trades.py                 # Lambda Finance → Finnhub → Senate Stock Watcher chain (size-capped)
  lambda_finance_sec.py              # Lambda /api/sec/income-statement + /balance-sheet
  lambda_finance_compare.py          # Lambda /api/sec/compare (peer comparison)
  etf_holdings.py                    # yfinance funds_data — sector weights + top-10 + concentration
  etf_peer_compare.py                # yfinance prices + info — ETF profile/returns/risk table
  options_flow.py                    # yfinance option chains
  macro_data.py                      # FRED yields / curve / credit / USD
  earnings_transcript.py             # Motley Fool scrape + sentiment (transcript wrapped in untrusted_content)
  sector_analysis.py                 # SPDR sector RS vs SPY + inter-market correlations
  utils.py                           # safe_ticker_component, wrap_untrusted
tradingagents/llm_clients/
  factory.py                         # create_llm_client (lazy provider import)
  model_catalog.py                   # MODEL_OPTIONS for the CLI
  openai_client.py                   # default max_retries=6; json_schema for Chat Completions
  fallback.py                        # FallbackChatModel — recoverable-error rotation (incl. 404 NotFoundError)
  {anthropic,google,azure}_client.py
tests/                               # pytest, with conftest.py setting placeholder API keys
```
