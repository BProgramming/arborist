#!/usr/bin/env python3
"""Which arborist sessions are attached to a server right now.

Every chat and delegated task keeps one small file in
`paths.root/active/HOST-PORT/` for as long as it runs, and holds a lock
(flock) on it. Another session finds out who else is there by trying to
take that lock: refused means the owner is alive; granted means the owner is
gone (however it went: the kernel drops a dead process's locks), and the
leftover file is deleted. So there are no stale entries after a crash, and a
reused PID can't make a dead session look alive.

It exists because sessions on one server share the model's single cache
slot. A normal chat parks that slot's cache to disk when it ends, and the
slot may be holding another session's conversation: a /private one's, which
must never reach the disk. So:

  - a session that starts while a private one is attached is warned;
  - a session that goes private while others are attached is warned, and
    again when another one arrives later;
  - no session saves the cache while another session is private;
  - the launcher leaves a server running while other sessions still use it.

Only sessions that share paths.root and the server's host and port see each
other. Another program pointed at the same server is invisible here.

    presence.py --count    print how many sessions are attached (for arborist.sh)
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path


def folder(cfg: dict) -> Path:
    """The registry for the server this config points at."""
    s = cfg["server"]
    name = re.sub(r"[^\w.-]", "_", f"{s['host']}-{s['port']}")
    return Path(cfg["derived"]["root"]) / "active" / name


def _held(path: Path) -> dict | None:
    """The entry at `path` if its owner is alive, else None (and the file,
    being left over, is removed)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass  # locked: the owner is alive
        else:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        try:
            data = json.loads(os.read(fd, 65536).decode("utf-8", "replace") or "{}")
        except ValueError:
            data = {}
        return data if isinstance(data, dict) else {}
    finally:
        os.close(fd)


def attached(directory: Path, skip: str = "") -> list[dict]:
    """Every live session in the registry, oldest first, except the one named `skip`."""
    try:
        paths = sorted(Path(directory).iterdir())
    except OSError:
        return []
    out = []
    for p in paths:
        if p.name == skip:
            continue
        entry = _held(p)
        # A dot-file is an entry being written; it's only looked at so that
        # one left by a crash gets removed.
        if entry is not None and not p.name.startswith("."):
            out.append({**entry, "id": p.name})
    return sorted(out, key=lambda e: e.get("since") or 0)


class Presence:
    """This session's entry. start() publishes it; close() takes it away."""

    def __init__(self, directory: Path, kind: str = "chat"):
        self.dir = Path(directory)
        self.kind = kind  # "chat" or "task"
        self.private = False
        self.since = time.time()
        # Unique per session, not just per process (tests run several in one).
        self.name = f"{os.getpid()}-{id(self):x}"
        self._fd: int | None = None

    def start(self) -> None:
        self._publish()

    def _publish(self) -> None:
        """Write the entry to a new file, lock it, then move it into place:
        a visible entry is always locked, so nobody mistakes a live session
        for a dead one while it's being written."""
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        tmp = self.dir / f".{self.name}.tmp"
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, json.dumps({"pid": os.getpid(), "kind": self.kind,
                                     "private": self.private, "since": self.since}).encode())
            os.replace(tmp, self.dir / self.name)
        except OSError:
            os.close(fd)
            raise
        old, self._fd = self._fd, fd
        if old is not None:
            os.close(old)

    def set_private(self, private: bool = True) -> None:
        self.private = private
        self._publish()

    def others(self) -> list[dict]:
        """The other live sessions: dicts with id, pid, kind, private, since."""
        return attached(self.dir, skip=self.name)

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            os.unlink(self.dir / self.name)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None


def describe(entries: list[dict]) -> str:
    """'a chat (pid 4312) and a task (pid 4410)'."""
    parts = [f"a {e.get('kind') or 'session'} (pid {e.get('pid', '?')})" for e in entries]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def main(argv: list[str]) -> int:
    if argv[1:] != ["--count"]:
        print(__doc__, file=sys.stderr)
        return 2
    import config
    try:
        cfg = config.load()
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    print(len(attached(folder(cfg))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
