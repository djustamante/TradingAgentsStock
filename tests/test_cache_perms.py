"""Tests for the cache file/dir permission hardening (security audit M1)."""

from __future__ import annotations

import os
import stat

from tradingagents.dataflows import _cache
from tradingagents.dataflows.config import set_config


def _mode(path) -> int:
    """Return the 9-bit permission portion of ``path``'s mode."""
    return stat.S_IMODE(os.stat(path).st_mode)


def test_cache_put_creates_dir_with_0o700_mode(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    _cache.cache_put("source_a", {"k": "v"}, "payload")

    cache_dir = tmp_path / "api" / "source_a"
    assert cache_dir.exists()
    assert _mode(cache_dir) == 0o700, (
        f"Cache directory has world-readable mode "
        f"{oct(_mode(cache_dir))}; expected 0o700"
    )


def test_cache_put_writes_file_with_0o600_mode(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    _cache.cache_put("source_b", {"k": "v"}, "payload")

    cache_file = tmp_path / "api" / "source_b"
    files = list(cache_file.glob("*.json"))
    assert files, "expected at least one cache file"
    assert _mode(files[0]) == 0o600, (
        f"Cache file has world-readable mode "
        f"{oct(_mode(files[0]))}; expected 0o600"
    )


def test_cache_put_tightens_preexisting_loose_dir(tmp_path):
    """A directory that already exists with 0o755 (default umask) should
    get tightened on the next cache_put. Defends against caches that
    were created before this hardening landed."""
    set_config({"data_cache_dir": str(tmp_path)})
    loose_dir = tmp_path / "api" / "source_c"
    loose_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    os.chmod(loose_dir, 0o755)  # force the loose mode
    assert _mode(loose_dir) == 0o755

    _cache.cache_put("source_c", {"k": "v"}, "payload")
    assert _mode(loose_dir) == 0o700


def test_cache_get_still_works_after_secure_write(tmp_path):
    """The hardening must not break the read path."""
    set_config({"data_cache_dir": str(tmp_path)})
    _cache.cache_put("source_d", {"k": "v"}, "payload-value")
    out = _cache.cache_get("source_d", {"k": "v"}, ttl_seconds=3600)
    assert out == "payload-value"
