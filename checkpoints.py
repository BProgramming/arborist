"""Workspace snapshots, so the model's changes can be undone.

Each snapshot is a commit in a private "shadow" git repository kept under the
arborist data directory, with the workspace as its work tree. The project's
own repository (if it has one) is never touched: no commits, no stashes, no
index changes. The project's .gitignore files are honoured, plus the same
directories the tools skip.

A snapshot is taken before the first change or run of each turn. Restoring
one first takes a "safety" snapshot of the current state, so an undo can be
undone too.

Old checkpoints are pruned when arborist starts (prune_store): anything
older than tools.checkpoint_days goes, and a store is deleted outright once
nothing in it is recent or its workspace no longer exists. Only directories
carrying arborist's marker file are ever touched.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

GIT_TIMEOUT = 60  # seconds; a huge workspace is better skipped than waited on
MARKER = "arborist-workspace"  # file in every store: the workspace it belongs to


class CheckpointError(Exception):
    pass


@dataclass
class Checkpoint:
    sha: str
    when: datetime
    kind: str   # "turn" (before the model changed things) or "safety" (before a restore)
    label: str
    restored: str = ""  # for a safety snapshot: the sha that restore went back to

    @property
    def short(self) -> str:
        return self.sha[:7]


@dataclass
class Change:
    status: str  # "A": exists now, not in the checkpoint; "D": the reverse; "M": modified
    path: str


class Checkpoints:
    def __init__(self, root: Path, store: Path, exclude: set[str] | frozenset[str] = frozenset()):
        self.root = Path(root).resolve()
        slug = re.sub(r"[^\w.-]", "_", self.root.name) or "root"
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:10]
        self.git_dir = Path(store) / f"{slug}-{digest}.git"
        self.exclude = set(exclude)
        self.available = shutil.which("git") is not None

    # -- plumbing -------------------------------------------------------------

    def _git(self, *args: str, stdin: bytes | None = None,
             extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_DIR=str(self.git_dir), GIT_WORK_TREE=str(self.root),
                   GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0", **(extra_env or {}))
        cmd = ["git",
               # Identity, signing and hooks from the user's global config
               # don't belong on these private commits.
               "-c", "user.name=arborist", "-c", "user.email=arborist@localhost",
               "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.git_dir / 'hooks'}",
               "-c", "core.autocrlf=false", "-c", "core.fsmonitor=false", "-c", "gc.auto=0",
               *args]
        try:
            r = subprocess.run(cmd, cwd=self.root, env=env, input=stdin, capture_output=True,
                               timeout=GIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise CheckpointError(f"git {args[0]} took over {GIT_TIMEOUT}s (is the workspace huge?)") from None
        except OSError as e:
            raise CheckpointError(f"can't run git: {e}") from None
        if r.returncode != 0:
            msg = r.stderr.decode(errors="replace").strip().splitlines()
            raise CheckpointError(f"git {args[0]} failed: {msg[-1] if msg else r.returncode}")
        return r

    def _ensure(self) -> None:
        if not self.available:
            raise CheckpointError("git isn't installed")
        if (self.git_dir / "HEAD").exists():
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git("init", "--quiet")
        info = self.git_dir / "info"
        info.mkdir(exist_ok=True)
        (info / "exclude").write_text(
            "# arborist checkpoints: never snapshot these\n"
            + "".join(f"{d}/\n" for d in sorted(self.exclude)) + ".arborist-logs/\n",
            encoding="utf-8")
        (self.git_dir / MARKER).write_text(str(self.root) + "\n", encoding="utf-8")

    def _stage(self) -> None:
        self._git("add", "--all", "--ignore-errors", "--", ".")

    def _has_commits(self) -> bool:
        try:
            self._git("rev-parse", "--verify", "--quiet", "HEAD")
            return True
        except CheckpointError:
            return False

    # -- API --------------------------------------------------------------------

    def snapshot(self, label: str, kind: str = "turn", restored: str = "") -> str:
        """Record the workspace as it is now. Returns the checkpoint's sha."""
        self._ensure()
        self._stage()
        label = " ".join(label.split())[:100] or "(no label)"
        message = ["-m", f"[{kind}] {label}"] + (["-m", f"restored: {restored}"] if restored else [])
        # --allow-empty: a turn that changes nothing still marks where it began.
        self._git("commit", "--quiet", "--no-verify", "--allow-empty", *message)
        return self._git("rev-parse", "HEAD").stdout.decode().strip()

    def list(self, limit: int = 20) -> list[Checkpoint]:
        """Newest first."""
        if not self.available or not (self.git_dir / "HEAD").exists() or not self._has_commits():
            return []
        out = self._git("log", f"-n{limit}", "--format=%H%x00%ct%x00%s%x00%b%x01").stdout
        result = []
        for record in out.decode(errors="replace").split("\x01"):
            record = record.strip("\n")
            if not record:
                continue
            sha, ts, subject, body = record.split("\0", 3)
            m = re.match(r"\[(\w+)\] (.*)", subject)
            kind, label = (m[1], m[2]) if m else ("turn", subject)
            r = re.search(r"^restored: ([0-9a-f]+)$", body, re.MULTILINE)
            result.append(Checkpoint(sha, datetime.fromtimestamp(int(ts)), kind, label,
                                     r[1] if r else ""))
        return result

    def find(self, prefix: str) -> Checkpoint | None:
        prefix = prefix.lower()
        matches = [c for c in self.list(limit=500) if c.sha.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def changes_since(self, sha: str, staged: bool = False) -> list[Change]:
        """How the workspace now differs from a checkpoint."""
        if not staged:
            self._stage()
        out = self._git("diff", "--cached", "--name-status", "--no-renames", "-z", sha).stdout
        fields = [f for f in out.decode("utf-8", "surrogateescape").split("\0") if f]
        return [Change(fields[i][0], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]

    def diff_text(self, sha: str, paths: list[str] = ()) -> str:
        """The full unified diff from a checkpoint to the workspace now
        (binary files are named, not shown). `paths` limits it, relative to
        the workspace."""
        self._stage()
        out = self._git("diff", "--cached", "--no-renames", "--no-color", "--no-ext-diff",
                        sha, "--", *(paths or ["."])).stdout
        return out.decode("utf-8", "replace")

    def latest_different(self, limit: int = 100) -> tuple[Checkpoint, list[Change]] | None:
        """What /undo restores: the newest turn checkpoint the workspace
        differs from. Right after an undo (nothing new since), only
        checkpoints older than the one restored count, so repeated /undo
        keeps going back instead of bouncing forward again."""
        history = self.list(limit)
        if history and history[0].kind == "safety" and history[0].restored:
            floor = next((i for i, cp in enumerate(history) if cp.sha == history[0].restored), None)
            if floor is None:
                return None  # restored further back than we can see
            history = history[floor + 1:]
        self._stage()
        for cp in history:
            if cp.kind != "turn":
                continue
            changes = self.changes_since(cp.sha, staged=True)
            if changes:
                return cp, changes
        return None

    def restore(self, sha: str) -> tuple[list[Change], str | None]:
        """Put the workspace back as it was at `sha`.

        Returns the changes that were reverted and the sha of the safety
        snapshot taken first (None if nothing needed restoring)."""
        changes = self.changes_since(sha)
        if not changes:
            return [], None
        safety = self.snapshot(f"before restoring {sha[:7]}", kind="safety", restored=sha)
        # Files created since the checkpoint go away...
        for c in changes:
            if c.status == "A":
                p = (self.root / c.path).resolve()
                if p.is_relative_to(self.root) and (p.is_file() or p.is_symlink()):
                    p.unlink()
                    _prune_empty_dirs(p.parent, self.root)
        # ...and modified or deleted ones come back from the checkpoint.
        back = [c.path for c in changes if c.status != "A"]
        if back:
            paths = "\0".join(back).encode("utf-8", "surrogateescape")
            self._git("checkout", sha, "--pathspec-from-file=-", "--pathspec-file-nul", stdin=paths)
        self._stage()
        return changes, safety


    # -- pruning ------------------------------------------------------------

    def prune(self, cutoff: datetime) -> int:
        """Drop checkpoints older than `cutoff`. Returns how many went.

        History is linear, so the kept checkpoints are re-chained onto a new
        root (same trees, messages and dates) and the rest is garbage
        collected. If nothing is recent, the whole store is deleted."""
        cps = self.list(limit=1_000_000)  # newest first
        keep = [c for c in cps if c.when >= cutoff]
        dropped = len(cps) - len(keep)
        if not dropped:
            return 0
        if not keep:
            shutil.rmtree(self.git_dir)
            return dropped
        new_sha: dict[str, str] = {}
        parent = None
        for cp in reversed(keep):  # oldest first
            info = self._git("log", "-1", "--format=%T%x00%ad%x00%cd%x00%B", "--date=raw",
                             cp.sha).stdout.decode(errors="replace")
            tree, author_date, commit_date, message = info.split("\0", 3)
            # A safety snapshot names the checkpoint it restored; follow the rewrite.
            message = re.sub(r"^restored: ([0-9a-f]+)$",
                             lambda m: f"restored: {new_sha.get(m[1], m[1])}",
                             message.strip(), flags=re.MULTILINE)
            args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
            parent = self._git(*args, extra_env={"GIT_AUTHOR_DATE": author_date,
                                                 "GIT_COMMITTER_DATE": commit_date}
                               ).stdout.decode().strip()
            new_sha[cp.sha] = parent
        self._git("update-ref", "HEAD", parent)
        self._git("reflog", "expire", "--expire=now", "--all")
        self._git("gc", "--prune=now", "--quiet")
        return dropped


def prune_store(store: Path, max_age_days: int, now: datetime | None = None) -> tuple[int, int]:
    """Prune every workspace's checkpoints under `store`.

    Returns (checkpoints removed, whole stores deleted). Stores whose
    workspace no longer exists are deleted. Anything without arborist's
    marker file is left alone."""
    store = Path(store)
    if max_age_days <= 0 or not store.is_dir() or shutil.which("git") is None:
        return 0, 0
    cutoff = (now or datetime.now()) - timedelta(days=max_age_days)
    removed = stores = 0
    for d in sorted(store.glob("*.git")):
        marker = d / MARKER
        if not d.is_dir() or not marker.is_file():
            continue
        root = Path(marker.read_text(encoding="utf-8").strip())
        cps = Checkpoints(root, store)
        if cps.git_dir != d:
            continue  # not laid out the way we'd have made it; don't guess
        if not root.is_dir():
            shutil.rmtree(d)
            stores += 1
            continue
        try:
            n = cps.prune(cutoff)
        except CheckpointError:
            continue  # leave a store we can't read as it is
        removed += n
        if n and not d.exists():
            stores += 1
    return removed, stores


def _prune_empty_dirs(d: Path, root: Path) -> None:
    while d != root and d.is_relative_to(root):
        try:
            d.rmdir()  # only succeeds when empty
        except OSError:
            return
        d = d.parent
