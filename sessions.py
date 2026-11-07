"""Parked conversations: what /resume needs to carry on where you left off.

When a chat ends, the conversation is written to `paths.root/sessions/KEY.json`
and llama-server is asked to save its KV cache for the chat's slot to
`paths.root/kv/KEY.bin` (the server's --slot-save-path). /resume reads both
back: the messages go into the chat, and the cache goes back into the slot, so
the next reply only processes your new message instead of the whole history.

There is one parked conversation per folder (and per mode: a --no-tools chat
parks separately from a chat with tools). The next one parked in that folder
replaces it.

The cache is only an optimisation. llama-server compares the tokens it holds
with each new prompt and reuses the longest common prefix, so a cache that no
longer matches (another model, a changed system prompt) costs a re-read, never
a wrong answer. Even so, arborist only restores one when the model, cache
settings and system prompt are the same as when it was parked, to avoid
loading gigabytes for nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

VERSION = 1
SLOT = 0  # the chat's slot (with server.parallel = 1, the only one)


def key_for(folder: Path | None, tools: bool) -> str:
    """A file-name-safe key for a folder and mode."""
    where = str(folder) if folder is not None else "(no folder)"
    mode = "tools" if tools else "plain"
    return hashlib.sha256(f"{mode}:{where}".encode()).hexdigest()[:20]


def kv_file(key: str) -> str:
    """The cache's file name inside the server's slot-save folder."""
    return f"{key}.bin"


@dataclass
class Store:
    session_dir: Path
    kv_dir: Path

    @classmethod
    def from_config(cls, cfg: dict) -> "Store":
        d = cfg["derived"]
        return cls(Path(d["session_dir"]), Path(d["kv_dir"]))

    def path(self, key: str) -> Path:
        return self.session_dir / f"{key}.json"

    def kv_path(self, key: str) -> Path:
        return self.kv_dir / kv_file(key)

    def load(self, key: str) -> dict | None:
        try:
            data = json.loads(self.path(key).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            raise ValueError(f"can't read {self.path(key)}: {e}") from None
        if not isinstance(data, dict) or data.get("version") != VERSION \
                or not isinstance(data.get("messages"), list):
            raise ValueError(f"{self.path(key)} isn't a parked conversation this version can read")
        return data

    def save(self, key: str, data: dict) -> Path:
        """Write atomically, readable only by you (it holds what the model read)."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.session_dir, 0o700)
        path = self.path(key)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"version": VERSION, **data}, fh, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def drop_kv(self, key: str) -> None:
        try:
            self.kv_path(key).unlink()
        except OSError:
            pass

    def secure_kv(self, key: str) -> None:
        """The server writes the cache with its own umask; tighten it."""
        try:
            os.chmod(self.kv_dir, 0o700)
            os.chmod(self.kv_path(key), 0o600)
        except OSError:
            pass

    def prune(self, days: int, now: float | None = None) -> int:
        """Delete parked conversations older than `days` (and caches with no
        conversation left). Returns how many conversations went."""
        now = time.time() if now is None else now
        gone = 0
        if days > 0 and self.session_dir.is_dir():
            for p in self.session_dir.glob("*.json"):
                try:
                    if now - p.stat().st_mtime > days * 86400:
                        p.unlink()
                        gone += 1
                except OSError:
                    pass
        if self.kv_dir.is_dir():
            for p in self.kv_dir.glob("*.bin"):
                if not (self.session_dir / f"{p.stem}.json").exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        return gone


def describe_size(n_bytes: int) -> str:
    for unit, size in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n_bytes >= size:
            return f"{n_bytes / size:.1f} {unit}"
    return f"{n_bytes} B"


def model_identity(cfg: dict) -> dict:
    """What a saved cache depends on besides the prompt itself."""
    s = cfg["server"]
    return {"model_file": cfg["derived"]["model_file"],
            "cache_type_k": s["cache_type_k"], "cache_type_v": s["cache_type_v"]}
