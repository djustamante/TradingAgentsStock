"""Tests for the memory-log permission hardening (security audit M2)."""

from __future__ import annotations

import os
import stat

from tradingagents.agents.utils.memory import TradingMemoryLog


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_memory_log_dir_created_with_0o700(tmp_path):
    log_path = tmp_path / "memory" / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(log_path)})
    # __init__ creates the parent dir
    assert log_path.parent.exists()
    assert _mode(log_path.parent) == 0o700, (
        f"Memory log dir has world-readable mode {oct(_mode(log_path.parent))}; "
        f"expected 0o700"
    )


def test_memory_log_dir_tightened_when_preexisting(tmp_path):
    log_dir = tmp_path / "memory"
    log_dir.mkdir(mode=0o755)
    os.chmod(log_dir, 0o755)
    assert _mode(log_dir) == 0o755

    log_path = log_dir / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(log_path)})

    assert _mode(log_dir) == 0o700


def test_memory_log_file_written_with_0o600(tmp_path):
    log_path = tmp_path / "memory" / "trading_memory.md"
    mem = TradingMemoryLog({"memory_log_path": str(log_path)})
    mem.store_decision("AAPL", "2026-05-09", "FINAL DECISION: HOLD")
    assert log_path.exists()
    assert _mode(log_path) == 0o600, (
        f"Memory log file has world-readable mode {oct(_mode(log_path))}; "
        f"expected 0o600"
    )
