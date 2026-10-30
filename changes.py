"""What a run changed in the workspace.

snapshot() records every file's size, mtime and ctime; compare() lists what
differs. ctime is the kernel's own "last changed" time: a program can set a
file's mtime back with os.utime(), but not its ctime, so a run can't hide an
edit by restoring the old timestamp. Nothing is skipped except the folders
the sandbox keeps read-only (which a sandboxed run can't change), so build
output, caches and ignored files show up too.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

Snapshot = dict[str, tuple[int, int, int, int]]


@dataclass(frozen=True)
class Change:
    status: str  # "A" added, "M" modified, "D" deleted
    path: str    # relative to the workspace, with / separators


def snapshot(root: Path, skip: list[Path] = ()) -> Snapshot:
    root = Path(root).resolve()
    skip_set = {Path(s).resolve() for s in skip}
    out: Snapshot = {}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if (here / d) not in skip_set]
        # Symlinks to folders are recorded as entries, not followed.
        for name in filenames + [d for d in dirnames if (here / d).is_symlink()]:
            p = here / name
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns, st.st_ctime_ns,
                                                   st.st_mode)
    return out


def compare(before: Snapshot, after: Snapshot) -> list[Change]:
    found = [Change("A", p) for p in after.keys() - before.keys()]
    found += [Change("D", p) for p in before.keys() - after.keys()]
    found += [Change("M", p) for p in before.keys() & after.keys() if before[p] != after[p]]
    return sorted(found, key=lambda c: c.path)


def runs_later(found) -> list[Change]:
    """Added or changed files that something other than arborist may execute
    later, outside the sandbox: anything in a .git (a hook, or a config that
    names one) and a virtualenv's pyvenv.cfg or .pth files. The sandbox keeps
    the existing ones read-only, so in a sandboxed run these are a repo or
    venv the run created, which it could fill with whatever it liked."""
    out = []
    for c in found:
        if c.status == "D":
            continue
        parts = c.path.split("/")
        if ".git" in parts or parts[-1] == "pyvenv.cfg" or parts[-1].endswith(".pth"):
            out.append(c)
    return out


RUNS_LATER_WARNING = ("WARNING: the run created or changed files that can run code outside "
                      "the sandbox the next time you use git or Python there (git hooks, git "
                      "config, a virtualenv's .pth files). Check them first:")


def describe_runs_later(risky: list[Change]) -> list[str]:
    """One line per repo or venv, not per file: a `git init` alone writes
    dozens. Named in each line: the hooks (other than git's inert .sample
    files), config and .pth files, which are what actually run."""
    groups: dict[str, list[str]] = {}
    for c in risky:
        parts = c.path.split("/")
        if ".git" in parts:
            i = parts.index(".git")
            key, inner = "/".join(parts[: i + 1]), "/".join(parts[i + 1:])
        else:
            key, inner = "/".join(parts[:-1]) or ".", parts[-1]
        groups.setdefault(key, []).append(inner)
    lines = []
    for key, inner in sorted(groups.items()):
        if inner == [""]:
            lines.append(f"  {key} (a file pointing git at another folder)")
            continue
        notable = sorted(f for f in inner if f in ("config", "pyvenv.cfg") or f.endswith(".pth")
                         or (f.startswith("hooks/") and not f.endswith(".sample")))
        n = len(inner)
        line = f"  {key}/: {n} file{'s' * (n != 1)}"
        if notable:
            line += ", including " + ", ".join(notable[:8])
            if len(notable) > 8:
                line += f" and {len(notable) - 8} more"
        lines.append(line)
    return lines


LABELS = {"A": "added", "M": "changed", "D": "deleted"}


def format_changes(found, limit: int = 40, colour=None) -> list[str]:
    """One line per file ('  M src/app.py'). Past `limit`, the rest are
    summarised per top-level folder, so a build that writes 3,000 files
    still gets a readable list without hiding where they went."""
    colour = colour or {}
    found = list(found)
    lines = []
    for c in found[:limit]:
        paint = colour.get(c.status, str)
        lines.append(f"  {paint(c.status)} {c.path}")
    rest = found[limit:]
    if rest:
        per_dir = Counter((c.path.split("/", 1)[0] + "/" if "/" in c.path else c.path, c.status)
                          for c in rest)
        lines.append(f"  … and {len(rest)} more:")
        for (top, status), n in sorted(per_dir.items()):
            lines.append(f"      {n} {LABELS[status]} in {top}")
    return lines
