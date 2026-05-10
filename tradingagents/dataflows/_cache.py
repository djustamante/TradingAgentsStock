"""File-based JSON cache for external API responses.

Used by the new dataflow modules (sec_insider, congress_trades, options_flow,
macro_data, earnings_transcript, sector_analysis) to avoid hammering external
APIs across repeated backtest runs for the same (ticker, date) combination.

Cache location: ``<data_cache_dir>/api/<source>/<sha1(key)>.json`` where
``data_cache_dir`` is read from the runtime config (defaults to
``~/.tradingagents/cache``). Each entry stores ``{"ts": <epoch>, "value": str}``.

The cache is intentionally simple: filesystem-only, no concurrency control
beyond atomic-rename writes, and no automatic eviction (callers pass a TTL).

Cache entries can contain externally-fetched data (news bodies, financial
filings, congressional disclosures). On a shared machine, the default
``mkdir`` mode (umask-derived, typically 0775) lets other local users read
these files. We explicitly tighten cache directories to ``0o700`` and cache
files to ``0o600`` so secrets / private state never leak via file system
ACLs — the user running the screener owns the cache; nobody else.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

# Owner-only permissions on cache directories and files. The cache may
# contain raw external content (e.g. earnings transcripts) and reading
# patterns that could leak portfolio interest signals to a coresident
# attacker; explicit 0o700 / 0o600 avoids depending on the user's umask.
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _cache_root() -> Path:
    from tradingagents.dataflows.config import get_config
    base = get_config().get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "cache"
    )
    return Path(base) / "api"


def _key_hash(key: Mapping[str, Any]) -> str:
    canonical = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:24]


def _entry_path(source: str, key: Mapping[str, Any]) -> Path:
    return _cache_root() / source / f"{_key_hash(key)}.json"


def cache_get(source: str, key: Mapping[str, Any], ttl_seconds: int) -> Optional[str]:
    """Return cached value if present and not older than ``ttl_seconds``."""
    path = _entry_path(source, key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - entry.get("ts", 0) > ttl_seconds:
        return None
    value = entry.get("value")
    return value if isinstance(value, str) else None


def cache_put(source: str, key: Mapping[str, Any], value: str) -> None:
    """Write ``value`` to cache under ``(source, key)`` atomically.

    Directories are created with mode 0o700 and the final cache file is
    chmod'd to 0o600 so a coresident local user can't read cached
    external content / leak portfolio interest signals.
    """
    path = _entry_path(source, key)
    _ensure_secure_dir(path.parent)
    payload = json.dumps({"ts": time.time(), "value": value})
    fd, tmp = tempfile.mkstemp(prefix=".cache_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _ensure_secure_dir(directory: Path) -> None:
    """Create ``directory`` with mode 0o700 if missing; tighten its mode
    if it already exists with looser permissions.

    ``mkdir(..., mode=0o700)`` only honours the mode when the directory is
    actually created — pre-existing dirs keep whatever mode they had (from
    the umask at creation time, often 0o775). ``chmod`` after-the-fact
    closes that gap. Applied to every ancestor up to ``api/`` so the
    whole subtree under the cache root is owner-only.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    # Walk from the cache-root upward isn't needed — we only need to
    # tighten the leaves we created. But pre-existing dirs (from older
    # cache state pre-this-fix) may be 0o775. Tighten the leaf and its
    # immediate ancestors up to the data_cache_dir root.
    cache_root = _cache_root()
    current = directory
    while True:
        try:
            os.chmod(current, _DIR_MODE)
        except OSError:
            break
        if current == cache_root or current == cache_root.parent:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
