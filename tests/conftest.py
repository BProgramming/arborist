"""Shared fixtures: a temporary workspace and a scripted approver."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools import Decision, Limits, Toolbox, Workspace


class FakeApprover:
    """Records every approval request and answers from a script.

    `change` and `run` are lists of Decisions consumed in order; when a list
    runs out, the default is to approve.
    """

    def __init__(self, change=None, run=None):
        self.change_answers = list(change or [])
        self.run_answers = list(run or [])
        self.change_requests: list[tuple[str, str, bool]] = []
        self.run_requests: list[tuple[str, list[list[str]], str, int]] = []

    def approve_change(self, rel_path, diff, is_new):
        self.change_requests.append((rel_path, diff, is_new))
        return self.change_answers.pop(0) if self.change_answers else Decision(True)

    def approve_run(self, rel_path, steps, cwd, timeout):
        self.run_requests.append((rel_path, steps, cwd, timeout))
        return self.run_answers.pop(0) if self.run_answers else Decision(True)


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
        tb = Toolbox(Workspace(ws_root), approver or FakeApprover(), Limits(**limits),
                     log_dir=tmp_path / "logs", on_output=output.append)
        tb.output = output  # lines streamed during runs, for assertions
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
