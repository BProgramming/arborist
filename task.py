#!/usr/bin/env python3
"""Delegated tasks: hand the model one job, let it work on its own, review the result.

    ./arborist.sh task run --task task.md [--mode auto|interactive] [--workspace DIR]
    ./arborist.sh task list
    ./arborist.sh task clean ID… | --all

The model never works on your project directly. Each task gets a disposable
copy of the repository under paths.root/tasks/ID/work: your commits, plus
whatever was uncommitted (staged, unstaged and untracked files, minus what
.gitignore excludes) at the moment the task started, committed there as the
task's base. When the model finishes, everything it changed relative to that
base is written to changes.patch, and report.json says how it went. Nothing
reaches your project until someone applies the patch.

Two modes:

  auto         Nobody watches. Edits in the copy apply without asking and
               runs go ahead, but only inside the bwrap sandbox, with every
               opening ([tools] run_writable, run_env_keep, run_display,
               run_gpu) forced off. Web and memory tools are off. The review
               happens afterwards, on the patch.
  interactive  Same copy and report, but you approve every edit and run in
               the terminal as it happens, as in the chat. Needs a terminal.

Between turns the harness keeps the model going: after a run it gets the
go-ahead to continue (or to fix failing tests), up to --max-turns turns. The
stuck detection and step limit from the chat still apply, and end the task.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import agent
import chat
import config as model_config
import sandbox
from tools import Decision, Limits, MemoryChoice, ToolResult, Toolbox, Workspace

MODES = ("auto", "interactive")
STATUSES = ("done", "tests_failing", "blocked", "out_of_steps", "out_of_turns", "stopped", "interrupted",
            "timed_out", "error")

GIT_IDENTITY = ["-c", "user.name=arborist", "-c", "user.email=arborist@localhost"]

TASK_GUIDE = """\
This is a delegated task. {who} You're working in a disposable copy of the \
project at {root}; nothing you change reaches the real project until it has been reviewed.
- Do the task as written, and nothing more. Don't tidy, reformat or rename things it doesn't mention.
- If something is unclear, or you're stuck, don't guess: stop and report RESULT: BLOCKED, \
saying exactly what you need.
- If the task names tests, run them before you finish. When they fail, fix the cause and \
run them again.
- When you've finished, reply with this report and nothing after it:
RESULT: DONE or BLOCKED
CHANGED: each file you changed, one per line, with what changed
CHECKED: the tests you ran and their result, or "not checked"
NOTES: anything the reviewer should look at closely, or "none"\
"""

WHO = {
    "auto": ("Another assistant assigned it and will review your work afterwards. Nobody is "
             "watching while you work, so you can't ask questions until you've finished."),
    "interactive": ("The user approves each edit and run as it happens, and the work is "
                    "reviewed again afterwards."),
}

TASK_MESSAGE = "Here is your task.\n\n<task>\n{task}\n</task>"
CONTINUE = ("(You may carry on with the task now: edit and run as you need to. When it's "
            "finished, reply with the report.)")
FIX = ("(Fix the failures you found, then run the tests again. When the task is finished, "
       "reply with the report.)")


class TaskError(Exception):
    """A task couldn't be set up (not a git repo, no sandbox, …)."""


# --------------------------------------------------------------- the copy ---

def _git(cwd: Path, *args: str, input: bytes | None = None, check: bool = True,
         timeout: int = 300) -> subprocess.CompletedProcess:
    """git with the repo's own fsmonitor and hooks switched off: a project's
    config must never get to run a command through arborist."""
    cmd = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
           "-C", str(cwd), *args]
    kw = {"input": input} if input is not None else {"stdin": subprocess.DEVNULL}
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise TaskError(f"git {args[0]} failed: {e}") from None
    if check and r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip()
        raise TaskError(f"git {args[0]} failed: {err or f'exit code {r.returncode}'}")
    return r


@dataclass
class Copy:
    repo: Path          # the real repository's top level
    work: Path          # the copy's top level
    workspace: Path     # where the model works: the copy, at the same subfolder you started in
    base: str           # the copy's commit holding your project as it was
    notes: list[str] = field(default_factory=list)  # what the copy leaves out


def make_copy(source: Path, dest: Path) -> Copy:
    """Copy the repository holding `source` to `dest`, uncommitted work included."""
    r = _git(source, "rev-parse", "--show-toplevel", check=False)
    if r.returncode != 0:
        raise TaskError(f"{source} isn't in a git repository; delegated tasks work on a copy "
                        "made with git, so run `git init` and commit first")
    repo = Path(r.stdout.decode().strip()).resolve()
    if _git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False).returncode != 0:
        raise TaskError(f"{repo} has no commits yet; commit once first")
    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()

    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(dest.parent, "clone", "--quiet", "--no-checkout", str(repo), str(dest))
    _git(dest, "checkout", "--quiet", "--detach", head)
    notes: list[str] = []

    # Staged and unstaged changes to tracked files.
    diff = _git(repo, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv").stdout
    if diff.strip():
        _git(dest, "apply", "--binary", "--whitespace=nowarn", "-", input=diff)
    # Untracked files git would show (so .gitignore'd ones stay behind).
    raw = _git(repo, "ls-files", "--others", "--exclude-standard", "-z").stdout
    for entry in raw.decode("utf-8", "surrogateescape").split("\0"):
        if not entry:
            continue
        if entry.endswith("/"):  # a nested repository
            notes.append(f"{entry} is a separate repository inside this one and wasn't copied")
            continue
        src, out = repo / entry, dest / entry
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            os.symlink(os.readlink(src), out)
        elif src.is_file():
            shutil.copy2(src, out)
    _git(dest, "add", "-A")
    _git(dest, *GIT_IDENTITY, "commit", "--quiet", "--no-verify", "--allow-empty",
         "-m", "arborist task base: the project as it was when the task started")
    base = _git(dest, "rev-parse", "HEAD").stdout.decode().strip()

    if (repo / ".gitmodules").exists():
        notes.append("submodules weren't copied")
    if (repo / "node_modules").is_dir():
        notes.append("node_modules wasn't copied (it's ignored), so JS code needing packages won't run")
    # An empty folder isn't in git, so the copy may not have the one you started in.
    workspace = dest / source.resolve().relative_to(repo)
    workspace.mkdir(parents=True, exist_ok=True)
    return Copy(repo, dest, workspace, base, notes)


def project_venv(source: Path, repo: Path) -> Path | None:
    """The project's own virtualenv. It's .gitignore'd, so the copy hasn't
    got one; runs use the original instead, which the sandbox shows read-only
    because it's outside the copy."""
    d = source.resolve()
    while True:
        for name in (".venv", "venv"):
            if (d / name / "bin" / "python").exists():
                return d / name
        if d == repo or d.parent == d:
            return None
        d = d.parent


def collect_changes(c: Copy, patch_path: Path) -> tuple[list[dict], str]:
    """Everything the task changed relative to its base: (files, diffstat).
    Writes the patch (relative to the repository's top level) when there's
    anything in it."""
    _git(c.work, "add", "-A")
    common = ["--cached", "--no-ext-diff", "--no-textconv", "--no-renames", c.base]
    names = _git(c.work, "diff", "--name-status", "-z", *common).stdout.decode("utf-8", "replace")
    parts = [p for p in names.split("\0") if p]
    files = [{"status": {"A": "added", "M": "modified", "D": "deleted"}.get(parts[i], parts[i]),
              "path": parts[i + 1]} for i in range(0, len(parts) - 1, 2)]
    stat = _git(c.work, "diff", "--stat", *common).stdout.decode("utf-8", "replace").strip()
    patch = _git(c.work, "diff", "--binary", *common).stdout
    if patch.strip():
        patch_path.write_bytes(patch)
    return files, stat


# -------------------------------------------------------------- the model ---

class AutoApprover:
    """Auto mode's answers. Edits and runs only happen in the copy, and runs
    only in the sandbox (TaskChat refuses to start auto mode without one);
    the web and memory are off, so those prompts are refusals."""

    def __init__(self, log):
        self.log = log

    def approve_change(self, rel_path, diff, is_new):
        return Decision(True)

    def show_change(self, rel_path, diff, is_new):
        self.log(f"{'created' if is_new else 'edited'} {rel_path}")

    def approve_run(self, rel_path, steps, cwd, timeout, notes=(), view=None):
        self.log(f"running {rel_path}")
        return Decision(True)

    def approve_fetch(self, host, url, reason="", scope=""):
        return Decision(False, "web access is off for delegated tasks")

    def approve_memory(self, fact, block, similar, web_tainted):
        return MemoryChoice("skip", reason="memory is off for delegated tasks")


def task_tools(cfg_tools: dict, mode: str) -> dict:
    """The [tools] settings a task runs with."""
    t = dict(cfg_tools)
    t["checkpoints"] = False  # the copy is the checkpoint
    if mode == "auto":
        t.update(approve_edits="auto", web_enabled=False, run_sandbox=True, run_writable=[],
                 run_env_keep=[], run_display=False, run_gpu=False)
    else:
        t["approve_edits"] = "ask"
    return t


class TaskUI(chat.TerminalUI):
    """The chat's display, plus a note of every test result."""

    def __init__(self, show_thinking: bool):
        super().__init__(show_thinking)
        self.tests: list[str] = []  # "passed" / "failed" / "timed out", in order

    def tool_result(self, name: str, result: ToolResult) -> None:
        super().tool_result(name, result)
        if name == "run_tests":
            m = re.match(r"Tests for .* (passed|failed|timed out)", result.content)
            if m:
                self.tests.append(m[1])


class TaskChat(chat.Chat):
    """A chat session over the task's copy, with no memory tool."""

    def __init__(self, cfg: dict, c: Copy, source: Path, mode: str, task_dir: Path):
        cfg = copy.deepcopy(cfg)
        cfg["tools"] = task_tools(cfg["tools"], mode)
        self.mode = mode
        super().__init__(cfg, c.workspace, tools_enabled=True)
        self.ui = TaskUI(cfg["chat"]["show_thinking"])
        self.run_changes: list[str] = []
        self.warnings: list[str] = []
        approver = AutoApprover(self.ui.notice) if mode == "auto" else chat.TerminalApprover()
        # The copy lives under paths.root, which the chat's toolbox hides, so
        # this one has its own: the whole copy is the model's, arborist's code
        # included when that's the project (changes to it are reviewed like any).
        self.toolbox = Toolbox(Workspace(c.workspace), approver, Limits.from_config(cfg["tools"]),
                               log_dir=task_dir / "logs", on_output=self.ui.run_output,
                               on_notice=self.ui.notice, checkpoints=None,
                               on_changes=self._record_run_changes)
        self.compactor.forget_file = self.toolbox.ws.unsee
        # The project's memories, as a chat started in the real folder sees them.
        self.memory = chat.Memory(cfg["derived"]["memory_path"], source)
        self.memory_block = self._memory_block()

    @property
    def agent_tools(self):
        return self.toolbox if self.tools_enabled else None

    def _system(self) -> str:
        parts = [self.system_prompt] if self.system_prompt else []
        parts.append(agent.TOOL_GUIDE.format(root=self.toolbox.ws.root,
                                             sandbox=agent.sandbox_rule(self.toolbox.limits)))
        parts.append(TASK_GUIDE.format(who=WHO[self.mode], root=self.toolbox.ws.root))
        if self._env_facts is None:
            with self.toolbox._probing():
                self._env_facts = chat.environment_facts(self.toolbox.ws.root,
                                                         self.cfg["chat"]["env_vars"])
        parts.append("Environment:\n" + "\n".join([self.date_line] + self._env_facts))
        if self.memory_block:
            parts.append(self.memory_block)
        return "\n\n".join(parts)

    def _record_run_changes(self, found: list) -> None:
        self._show_run_changes(found)
        self.run_changes += chat.changes_mod.format_changes(found)
        risky = chat.changes_mod.runs_later(found)
        if risky:
            self.warnings.append(chat.changes_mod.RUNS_LATER_WARNING)
            self.warnings += chat.changes_mod.describe_runs_later(risky)


# ------------------------------------------------------------------- run ---

def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(words)[:40] or "task"


def model_result(text: str) -> str | None:
    m = re.search(r"RESULT:\s*\**\s*(DONE|BLOCKED)\b", text, re.IGNORECASE)
    return m[1].upper() if m else None


def run_task(cfg: dict, source: Path, task_text: str, mode: str = "auto", max_turns: int = 6,
             timeout: int = 1800, effort: str | None = None, client=None,
             ask=chat.ask) -> dict:
    """Run one task to the end and write its report. Returns the report."""
    if mode not in MODES:
        raise TaskError(f"mode must be one of {', '.join(MODES)}")
    if not task_text.strip():
        raise TaskError("the task is empty")
    if mode == "auto":
        if not cfg["tools"]["run_sandbox"]:
            raise TaskError("auto mode needs the sandbox, and [tools] run_sandbox is false; "
                            "turn it on, or use --mode interactive")
        ok, why = sandbox.status()
        if not ok:
            raise TaskError(f"auto mode needs the bwrap sandbox, which isn't working: {why}")
    elif not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise TaskError("interactive mode asks you about every edit and run, so it needs a "
                        "terminal; run it yourself, or use --mode auto")
    if effort and effort not in cfg["model"]["efforts"]:
        raise TaskError(f"effort must be one of {cfg['model']['efforts']}")

    source = source.resolve()
    task_id = f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(task_text)}"
    task_dir = Path(cfg["derived"]["root"]) / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text(task_text, encoding="utf-8")
    report: dict = {"id": task_id, "mode": mode, "status": "error", "task_dir": str(task_dir),
                    "source": str(source)}
    try:
        c = make_copy(source, task_dir / "work")
    except TaskError as e:
        report["error"] = str(e)
        _write_report(task_dir, report)
        raise
    report.update(repo=str(c.repo), work=str(c.workspace), base=c.base, notes=list(c.notes))
    venv = project_venv(source, c.repo)
    if venv:
        os.environ["VIRTUAL_ENV"] = str(venv)
        report["notes"].append(f"runs use the project's virtualenv {venv}, read-only")

    session = TaskChat(cfg, c, source, mode, task_dir)
    if client is not None:
        session.client = client
    if effort:
        session.effort = effort
    t = session.cfg["tools"]
    messages = session.messages
    messages.append({"role": "user", "content": TASK_MESSAGE.format(task=task_text.strip())})

    started = time.monotonic()
    turns = steps = 0
    final = ""
    status = "error"

    def on_signal(signum, frame):
        raise chat.Interrupted()

    old_int = signal.signal(signal.SIGINT, on_signal)
    old_alarm = signal.signal(signal.SIGALRM, on_signal)
    if timeout:
        signal.alarm(timeout)
    try:
        while True:
            turns += 1
            session.toolbox.begin_turn(f"task {task_id} turn {turns}")
            result = agent.run_turn(messages, session._body, session.client.stream_chat,
                                    session.agent_tools, session.ui, max_steps=t["max_steps"],
                                    compactor=session.compactor,
                                    max_consecutive_errors=t["max_consecutive_errors"],
                                    max_repeat_calls=t["max_repeat_calls"])
            steps += result.steps
            final = result.text
            if result.paused:
                status = "blocked"
                report["harness"] = f"paused: {result.paused}"
                break
            if not result.hard_stopped:
                if result.steps >= t["max_steps"]:
                    status = "out_of_steps"
                else:
                    status = "blocked" if model_result(final) == "BLOCKED" else "done"
                break
            if turns >= max_turns:
                status = "out_of_turns"
                break
            if mode == "interactive":
                ans = ask(chat.bold(chat.yellow("Let the model carry on? [Y/n]: "))).strip().lower()
                if ans in ("n", "no"):
                    status = "stopped"
                    break
            messages.append({"role": "user",
                             "content": FIX if session.toolbox.tests_failed else CONTINUE})
    except chat.Interrupted:
        session.ui.end_message()
        timed_out = timeout and time.monotonic() - started >= timeout - 1
        status = "timed_out" if timed_out else "interrupted"
    except (RuntimeError, OSError) as e:
        session.ui.end_message()
        report["error"] = str(e)
        if not session.client.healthy():
            report["error"] += " (the model server isn't responding)"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGINT, old_int)

    patch = task_dir / "changes.patch"
    try:
        files, stat = collect_changes(c, patch)
    except TaskError as e:
        files, stat = [], ""
        report.setdefault("error", str(e))
    session._save([str(task_dir / "transcript.md")])
    tests = session.ui.tests
    if status == "done" and tests and tests[-1] != "passed":
        # A small model will say it's done with the tests still failing.
        status = "tests_failing"
    report.update(
        status=status,
        model_result=model_result(final),
        final_message=final.strip(),
        files=files,
        diffstat=stat,
        patch=str(patch) if patch.exists() else None,
        apply=(f"git -C {c.repo} apply --3way {patch}" if patch.exists() else None),
        tests={"runs": len(tests), "last": tests[-1] if tests else "not run"},
        run_changes=session.run_changes,
        warnings=session.warnings,
        turns=turns,
        steps=steps,
        seconds=round(time.monotonic() - started),
        transcript=str(task_dir / "transcript.md"),
        logs=str(task_dir / "logs"),
    )
    _write_report(task_dir, report)
    return report


def _write_report(task_dir: Path, report: dict) -> None:
    (task_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------- commands ---

def tasks_dir(cfg: dict) -> Path:
    return Path(cfg["derived"]["root"]) / "tasks"


def list_tasks(cfg: dict) -> list[dict]:
    out = []
    root = tasks_dir(cfg)
    for d in sorted(root.iterdir() if root.is_dir() else [], reverse=True):
        try:
            r = json.loads((d / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            r = {"id": d.name, "status": "running or unfinished"}
        out.append(r)
    return out


def clean_tasks(cfg: dict, ids: list[str], everything: bool) -> list[str]:
    root = tasks_dir(cfg)
    names = [d.name for d in root.iterdir()] if everything and root.is_dir() else ids
    removed = []
    for name in names:
        d = (root / name).resolve()
        if d.parent != root.resolve() or not d.is_dir():
            raise TaskError(f"no task {name!r} in {root}")
        shutil.rmtree(d)
        removed.append(name)
    return removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arborist.sh task",
                                 description="Delegate a task to the local model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a task in a copy of the project and write a report")
    r.add_argument("--task", "-t", required=True, help="file holding the task ('-' for stdin)")
    r.add_argument("--mode", "-m", choices=MODES, default="auto",
                   help="auto: no prompts, sandboxed runs (default); interactive: you approve "
                        "each edit and run")
    r.add_argument("--workspace", "-w", help="project folder (default: current directory)")
    r.add_argument("--max-turns", type=int, default=6,
                   help="times the model may be told to carry on after a run (default 6)")
    r.add_argument("--timeout", type=int, default=1800,
                   help="seconds before the task is stopped (default 1800; 0 = none)")
    r.add_argument("--effort", help="reasoning effort for this task (from model.efforts)")
    sub.add_parser("list", help="list tasks, newest first")
    c = sub.add_parser("clean", help="delete tasks' copies, patches and reports")
    c.add_argument("ids", nargs="*")
    c.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if not isinstance(sys.stdout, chat.SafeStdout):
        sys.stdout, sys.stderr = chat.SafeStdout(sys.stdout), chat.SafeStdout(sys.stderr)

    try:
        cfg = model_config.load()
    except model_config.ConfigError as e:
        print(chat.red(f"config error: {e}"), file=sys.stderr)
        return 1

    try:
        if args.cmd == "list":
            for t in list_tasks(cfg):
                files = len(t.get("files") or [])
                print(f"{t['id']}  {t.get('status', '?'):<13} {t.get('mode', ''):<11} "
                      f"{files} file{'s' * (files != 1)}  {t.get('source', '')}")
            return 0
        if args.cmd == "clean":
            if not args.ids and not args.all:
                ap.error("name tasks to delete, or pass --all")
            for name in clean_tasks(cfg, args.ids, args.all):
                print(f"deleted {name}")
            return 0

        text = sys.stdin.read() if args.task == "-" else Path(args.task).read_text(encoding="utf-8")
        source = chat.resolve_workspace(args.workspace)
        client = chat.Client(cfg["derived"]["base_url"])
        if not client.healthy():
            raise TaskError(f"no server at {cfg['derived']['base_url']}; start it with ./arborist.sh task")
        report = run_task(cfg, source, text, mode=args.mode, max_turns=args.max_turns,
                          timeout=args.timeout, effort=args.effort)
    except (TaskError, OSError) as e:
        print(chat.red(f"task error: {e}"), file=sys.stderr)
        return 1
    print()
    print(chat.bold(f"task {report['id']}: {report['status']}") +
          f" ({report['turns']} turn(s), {report['steps']} step(s), {report['seconds']}s)")
    print(report["diffstat"] or "no changes")
    print(f"report: {Path(report['task_dir']) / 'report.json'}")
    return 0 if report["status"] == "done" else 2


if __name__ == "__main__":
    sys.exit(main())
