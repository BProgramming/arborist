"""Shared fixtures: a temporary workspace and a scripted approver."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import sandbox

from tools import Decision, Limits, MemoryChoice, Toolbox, Workspace


class FakeApprover:
    """Records every approval request and answers from a script.

    `change`, `run` and `fetch` are lists of Decisions consumed in order;
    when a list runs out, the default is to approve. `memory` is a list of
    MemoryChoices; the default is to keep both notes.
    """

    def __init__(self, change=None, run=None, fetch=None, memory=None):
        self.memory_answers = list(memory or [])
        self.memory_requests: list[tuple[str, str, list[str], bool]] = []
        self.change_answers = list(change or [])
        self.run_answers = list(run or [])
        self.fetch_answers = list(fetch or [])
        self.change_requests: list[tuple[str, str, bool]] = []
        self.run_requests: list[tuple[str, list[list[str]], str, int]] = []
        self.run_notes: list[list[str]] = []
        self.run_views: list[str | None] = []
        self.shown_changes: list[tuple[str, str, bool]] = []
        self.fetch_requests: list[tuple[str, str]] = []
        self.fetch_details: list[tuple[str, str]] = []  # (reason, scope)

    def approve_change(self, rel_path, diff, is_new):
        self.change_requests.append((rel_path, diff, is_new))
        return self.change_answers.pop(0) if self.change_answers else Decision(True)

    def show_change(self, rel_path, diff, is_new):
        self.shown_changes.append((rel_path, diff, is_new))

    def approve_run(self, rel_path, steps, cwd, timeout, notes=(), view=None):
        self.run_requests.append((rel_path, steps, cwd, timeout))
        self.run_notes.append(list(notes))
        self.run_views.append(view)
        return self.run_answers.pop(0) if self.run_answers else Decision(True)

    def approve_memory(self, fact, block, similar, web_tainted):
        self.memory_requests.append((fact, block, list(similar), web_tainted))
        return self.memory_answers.pop(0) if self.memory_answers else MemoryChoice("keep")

    def approve_fetch(self, host, url, reason="", scope=""):
        self.fetch_requests.append((host, url))
        self.fetch_details.append((reason, scope))
        return self.fetch_answers.pop(0) if self.fetch_answers else Decision(True)


@pytest.fixture
def ws_root(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def approver() -> FakeApprover:
    return FakeApprover()


@pytest.fixture
def make_toolbox(ws_root, tmp_path):
    """Build a Toolbox over ws_root; logs go outside the workspace."""
    def make(approver=None, **limits):
        # Runs are offline by default; on a machine that can't do that (e.g. a
        # CI runner without user namespaces), the other tests run online.
        limits.setdefault("run_sandbox", sandbox.available())
        output: list[str] = []
        notices: list[str] = []
        tb = Toolbox(Workspace(ws_root), approver or FakeApprover(), Limits(**limits),
                     log_dir=tmp_path / "logs", on_output=output.append,
                     on_notice=notices.append)
        tb.output = output    # lines streamed during runs and crawls, for assertions
        tb.notices = notices
        return tb
    return make


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def needs(*tools: str):
    """Skip a test when a toolchain isn't installed on this machine."""
    missing = [t for t in tools if shutil.which(t) is None]
    return pytest.mark.skipif(bool(missing), reason=f"not installed: {', '.join(missing)}")


def _sandbox_or_skip():
    ok, why = sandbox.status()
    if ok:
        return pytest.mark.skipif(False, reason="")
    if os.environ.get("ARBORIST_REQUIRE_SANDBOX"):
        # CI sets this: an audit that silently skips proves nothing.
        raise RuntimeError(f"ARBORIST_REQUIRE_SANDBOX is set but the sandbox is unavailable: {why}")
    return pytest.mark.skip(reason=f"sandbox unavailable: {why}")


# Tests that need a working bwrap sandbox.
sandboxed = _sandbox_or_skip()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A home directory of our own, holding things no run should see."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE-KEY-MATERIAL\n")
    (home / ".bashrc").write_text("# original\n")
    monkeypatch.setenv("HOME", str(home))
    return home
