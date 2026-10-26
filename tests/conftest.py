"""Shared fixtures: a temporary workspace and a scripted approver."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

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
        self.fetch_requests: list[tuple[str, str]] = []
        self.fetch_details: list[tuple[str, str]] = []  # (reason, scope)

    def approve_change(self, rel_path, diff, is_new):
        self.change_requests.append((rel_path, diff, is_new))
        return self.change_answers.pop(0) if self.change_answers else Decision(True)

    def approve_run(self, rel_path, steps, cwd, timeout):
        self.run_requests.append((rel_path, steps, cwd, timeout))
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
