"""Tools the model can call: read, list, search, edit and write files, and
check/run a single source file.

Everything is confined to one workspace directory (where you launched the
chat). The terminal UI supplies an Approver, which is asked before any file
changes and before anything runs; tests supply a fake one.

Tool failures are returned to the model as "error: ..." text rather than
raised, so it can read the message and correct itself.
"""
from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import runners

# Directories nobody wants the model walking through.
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "target", ".venv", "venv", "__pycache__",
    ".dart_tool", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
}
MAX_TEXT_BYTES = 2 * 1024 * 1024  # files bigger than this aren't read or searched


class ToolError(Exception):
    """A problem the model caused and can fix (bad path, no match, …)."""


@dataclass
class Decision:
    approved: bool
    reason: str = ""      # sent back to the model when the user says no
    always: bool = False  # "yes to all edits this session" (ignored for runs)


class Approver(Protocol):
    def approve_change(self, rel_path: str, diff: str, is_new: bool) -> Decision: ...
    def approve_run(self, rel_path: str, steps: list[list[str]], cwd: str, timeout: int) -> Decision: ...


@dataclass
class Limits:
    approve_edits: str = "ask"      # "ask" | "auto"
    max_read_lines: int = 400
    max_output_chars: int = 20000
    max_list_entries: int = 300
    max_grep_matches: int = 100
    run_timeout: int = 120

    @classmethod
    def from_config(cls, tools_cfg: dict) -> "Limits":
        return cls(**{k: tools_cfg[k] for k in cls.__dataclass_fields__ if k in tools_cfg})


@dataclass
class ToolResult:
    content: str
    # True once code has run (or a run was refused): the agent loop lets the
    # model explain the result, then stops using tools for this turn.
    hard_stop: bool = False


# ------------------------------------------------------------- workspace ---

class Workspace:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"workspace is not a directory: {self.root}")
        # path -> mtime_ns when the model last read (or wrote) it
        self._seen: dict[Path, int] = {}

    def resolve(self, path: str) -> Path:
        """Turn a model-supplied path into an absolute path inside the workspace.

        resolve() follows symlinks, so a link pointing outside is caught too.
        """
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path is required")
        p = Path(os.path.expanduser(path.strip()))
        if not p.is_absolute():
            p = self.root / p
        resolved = p.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ToolError(f"{path} is outside the workspace ({self.root}); only paths inside it are allowed")
        return resolved

    def rel(self, p: Path) -> str:
        return "." if p == self.root else p.relative_to(self.root).as_posix()

    def mark_seen(self, p: Path) -> None:
        self._seen[p] = p.stat().st_mtime_ns

    def forget(self) -> None:
        self._seen.clear()

    def require_fresh(self, p: Path) -> None:
        """Refuse to change a file the model hasn't read, or that changed since."""
        if p not in self._seen:
            raise ToolError(f"read {self.rel(p)} with read_file before changing it")
        if p.stat().st_mtime_ns != self._seen[p]:
            raise ToolError(f"{self.rel(p)} changed on disk since you read it; read it again first")

    def walk_files(self, start: Path):
        """Yield files under start, skipping ignored and hidden-tool directories."""
        if start.is_file():
            yield start
            return
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                yield Path(dirpath) / name


def _read_text(p: Path) -> str:
    if not p.exists():
        raise ToolError(f"no such file: {p.name}")
    if p.is_dir():
        raise ToolError(f"{p.name} is a directory; use list_files")
    if p.stat().st_size > MAX_TEXT_BYTES:
        raise ToolError(f"{p.name} is larger than {MAX_TEXT_BYTES // 1024 // 1024} MB; not reading it")
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        raise ToolError(f"{p.name} looks like a binary file")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{p.name} is not UTF-8 text") from None


def _diff(rel: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3,
    ))


def _truncate(text: str, limit: int) -> str:
    """Keep the start and (mostly) the end, which is where errors usually are."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 10
    tail = limit - head
    skipped = len(text) - head - tail
    return f"{text[:head]}\n\n[... {skipped:,} characters omitted ...]\n\n{text[-tail:]}"


# ---------------------------------------------------------------- tools ---

class Toolbox:
    def __init__(self, workspace: Workspace, approver: Approver, limits: Limits | None = None,
                 log_dir: str | os.PathLike | None = None,
                 on_output: Callable[[str], None] | None = None):
        self.ws = workspace
        self.approver = approver
        self.limits = limits or Limits()
        self.log_dir = Path(log_dir) if log_dir else self.ws.root / ".arborist-logs"
        self.on_output = on_output or (lambda line: None)
        self.auto_edits = self.limits.approve_edits == "auto"
        # path -> (mtime_ns, plan) for files whose dependencies checked out
        self._checked: dict[Path, tuple[int, runners.RunPlan]] = {}

    def reset(self) -> None:
        """Forget what the model has read and checked (e.g. after /clear)."""
        self.ws.forget()
        self._checked.clear()

    # Dispatch -------------------------------------------------------------

    def call(self, name: str, raw_args: str | dict) -> ToolResult:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None or name not in TOOL_NAMES:
            return ToolResult(f"error: unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
        try:
            args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return ToolResult(f"error: couldn't parse arguments for {name}: {e}")
        try:
            result = handler(**args)
        except ToolError as e:
            return ToolResult(f"error: {e}")
        except TypeError as e:
            return ToolResult(f"error: bad arguments for {name}: {e}")
        except OSError as e:
            return ToolResult(f"error: {e.strerror or e} ({name})")
        if isinstance(result, str):
            result = ToolResult(result)
        result.content = _truncate(result.content, self.limits.max_output_chars)
        return result

    # read_file ------------------------------------------------------------

    def tool_read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        p = self.ws.resolve(path)
        text = _read_text(p)
        self.ws.mark_seen(p)
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return f"{self.ws.rel(p)} is empty"
        offset = max(1, int(offset))
        if offset > total:
            raise ToolError(f"offset {offset} is past the end ({total} lines)")
        limit = min(int(limit or self.limits.max_read_lines), self.limits.max_read_lines)
        end = min(total, offset + limit - 1)
        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(offset, end + 1))
        if offset > 1 or end < total:
            more = f"; continue with offset={end + 1}" if end < total else ""
            body += f"\n[lines {offset}-{end} of {total}{more}]"
        return body

    # list_files -----------------------------------------------------------

    def tool_list_files(self, pattern: str = "*", path: str = ".") -> str:
        start = self.ws.resolve(path)
        if not start.exists():
            raise ToolError(f"no such directory: {path}")
        matches, total = [], 0
        for f in self.ws.walk_files(start):
            rel = self.ws.rel(f)
            # '*' matches across '/', so "*.py" finds Python files at any depth.
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern) or \
               (pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])):
                total += 1
                if len(matches) < self.limits.max_list_entries:
                    matches.append(rel)
        if not matches:
            return f"no files match {pattern!r} under {self.ws.rel(start)}"
        out = "\n".join(matches)
        if total > len(matches):
            out += f"\n[{total - len(matches)} more not shown; narrow the pattern or path]"
        return out

    # grep -----------------------------------------------------------------

    def tool_grep(self, pattern: str, path: str = ".", glob: str | None = None,
                  ignore_case: bool = False) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"invalid regex {pattern!r}: {e}") from None
        start = self.ws.resolve(path)
        if not start.exists():
            raise ToolError(f"no such path: {path}")
        hits, total = [], 0
        for f in self.ws.walk_files(start):
            rel = self.ws.rel(f)
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f.name, glob)):
                continue
            try:
                text = _read_text(f)
            except (ToolError, OSError):
                continue  # binary, huge or unreadable: skip quietly
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(hits) < self.limits.max_grep_matches:
                        shown = line if len(line) <= 300 else line[:300] + "…"
                        hits.append(f"{rel}:{n}: {shown}")
        if not hits:
            return f"no matches for {pattern!r}"
        out = "\n".join(hits)
        if total > len(hits):
            out += f"\n[{total - len(hits)} more matches not shown; narrow the pattern, path or glob]"
        return out

    # edit_file / write_file -----------------------------------------------

    def _apply(self, p: Path, old: str, new: str, is_new: bool) -> str:
        rel = self.ws.rel(p)
        diff = _diff(rel, old, new)
        if not self.auto_edits:
            d = self.approver.approve_change(rel, diff, is_new)
            if not d.approved:
                why = f" Their reason: {d.reason}" if d.reason else ""
                return f"The user declined this change to {rel}; the file was not modified.{why}"
            if d.always:
                self.auto_edits = True
        p.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps the file's existing line endings as written.
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
        self.ws.mark_seen(p)
        added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
        verb = "created" if is_new else "updated"
        return f"{verb} {rel} (+{added} -{removed})"

    def tool_edit_file(self, path: str, old_string: str, new_string: str,
                       replace_all: bool = False) -> str:
        p = self.ws.resolve(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}; use write_file to create it")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical; nothing to change")
        if not old_string:
            raise ToolError("old_string is empty; give the exact text to replace")
        self.ws.require_fresh(p)
        text = _read_text(p)
        count = text.count(old_string)
        if count == 0:
            raise ToolError(
                f"old_string was not found in {self.ws.rel(p)}. It must match the file exactly, "
                "including indentation and whitespace; read the file again and copy the text")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string appears {count} times in {self.ws.rel(p)}; include more surrounding "
                "lines so it matches once, or set replace_all to true")
        new = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        return self._apply(p, text, new, is_new=False)

    def tool_write_file(self, path: str, content: str) -> str:
        p = self.ws.resolve(path)
        if p.is_dir():
            raise ToolError(f"{path} is a directory")
        if p.exists():
            self.ws.require_fresh(p)
            return self._apply(p, _read_text(p), content, is_new=False)
        return self._apply(p, "", content, is_new=True)

    # check_dependencies / run_file ----------------------------------------

    def tool_check_dependencies(self, path: str) -> str:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        plan = runners.plan_for(p, self.ws.root, self.log_dir / "build")
        rel = self.ws.rel(p)
        lines = [f"Dependency check for {rel}:"]
        lines += [f"  ok: {d}" for d in plan.checked]
        lines += [f"  note: {w}" for w in plan.warnings]
        lines += [f"  MISSING: {m}" for m in plan.problems]
        if plan.problems:
            self._checked.pop(p, None)
            lines.append("Not runnable until the missing items are installed. Don't try to run it; "
                         "tell the user what to install.")
        else:
            self._checked[p] = (p.stat().st_mtime_ns, plan)
            cmd = "  then  ".join(" ".join(s) for s in plan.steps)
            lines.append(f"Ready. run_file would execute: {cmd}  (in {plan.cwd})")
        return "\n".join(lines)

    def tool_run_file(self, path: str) -> ToolResult:
        p = self.ws.resolve(path)
        rel = self.ws.rel(p)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        if p.suffix not in runners.RUNNABLE:
            raise ToolError(f"only {', '.join(sorted(runners.RUNNABLE))} files can be run")
        checked = self._checked.get(p)
        if checked is None:
            raise ToolError(f"call check_dependencies on {rel} first, and only run it if the check passes")
        if checked[0] != p.stat().st_mtime_ns:
            raise ToolError(f"{rel} changed since its dependency check; call check_dependencies again")
        plan = checked[1]
        # Single-use: every run needs a fresh check (and a fresh approval).
        del self._checked[p]

        d = self.approver.approve_run(rel, plan.steps, str(plan.cwd), self.limits.run_timeout)
        if not d.approved:
            why = f" Their reason: {d.reason}" if d.reason else ""
            return ToolResult(f"The user declined to run {rel}; nothing was executed.{why} "
                              "Tell the user this and stop; don't try another way to run it.",
                              hard_stop=True)

        outcome = runners.execute(plan, self.log_dir, self.limits.run_timeout, self.on_output)
        log = _read_log(outcome.log_path)
        status = ("timed out after %ds" % self.limits.run_timeout) if outcome.timed_out \
            else f"exit code {outcome.exit_code}"
        header = (f"Ran {rel}: {status}, {outcome.duration:.1f}s. Log: {outcome.log_path}\n"
                  f"--- log ---\n")
        footer = ("\n--- end of log ---\nNow explain what happened for the user: whether it worked, "
                  "and what any errors or warnings mean. Don't call any more tools.")
        room = self.limits.max_output_chars - len(header) - len(footer)
        return ToolResult(header + _truncate(log, max(room, 1000)) + footer, hard_stop=True)


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(couldn't read log: {e})"


# ---------------------------------------------------------- tool schemas ---

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


_PATH = {"type": "string", "description": "File path, relative to the workspace root"}

TOOL_SCHEMAS = [
    _fn("read_file",
        "Read a text file. Returns numbered lines (the numbers are not part of the file). "
        "Long files come back in pages; use offset to continue.",
        {"path": _PATH,
         "offset": {"type": "integer", "description": "First line to return, 1-based (default 1)"},
         "limit": {"type": "integer", "description": "Maximum lines to return"}},
        ["path"]),
    _fn("list_files",
        "List files in the workspace whose path matches a glob pattern, e.g. '*.py' or 'src/*.rs'. "
        "'*' also matches '/', so '*.py' finds Python files at any depth.",
        {"pattern": {"type": "string", "description": "Glob pattern (default '*', everything)"},
         "path": {"type": "string", "description": "Directory to search in (default: workspace root)"}},
        []),
    _fn("grep",
        "Search file contents with a regular expression. Returns path:line: text for each match.",
        {"pattern": {"type": "string", "description": "Python regular expression"},
         "path": {"type": "string", "description": "File or directory to search (default: workspace root)"},
         "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'"},
         "ignore_case": {"type": "boolean"}},
        ["pattern"]),
    _fn("edit_file",
        "Replace exact text in an existing file. Read the file first. old_string must match the file "
        "exactly (indentation included, without line numbers) and appear exactly once unless "
        "replace_all is true. The user approves each change.",
        {"path": _PATH,
         "old_string": {"type": "string", "description": "Exact text to replace"},
         "new_string": {"type": "string", "description": "Replacement text"},
         "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"}},
        ["path", "old_string", "new_string"]),
    _fn("write_file",
        "Create a new file, or replace an existing file's whole content (read it first). Prefer "
        "edit_file for changes to existing files. The user approves each write.",
        {"path": _PATH, "content": {"type": "string", "description": "The complete file content"}},
        ["path", "content"]),
    _fn("check_dependencies",
        "Check that a .py, .rs, .dart, .js or .ts file can run: the interpreter or compiler is "
        "installed and its imports/packages are available. Required before run_file.",
        {"path": _PATH},
        ["path"]),
    _fn("run_file",
        "Run a .py, .rs, .dart, .js or .ts file after check_dependencies passed. The user must approve. "
        "Output is logged and returned. After this you must explain the result and stop: no further "
        "tool calls in this turn.",
        {"path": _PATH},
        ["path"]),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]
