"""Workspace checkpoints: snapshots in a shadow repo, restore, and /undo."""
from __future__ import annotations

import subprocess

import pytest

from checkpoints import Checkpoints
from conftest import FakeApprover, needs, write
from tools import IGNORED_DIRS

pytestmark = needs("git")


@pytest.fixture
def cps(ws_root, tmp_path):
    return Checkpoints(ws_root, tmp_path / "store", exclude=IGNORED_DIRS)


def test_restore_reverts_edits_deletions_and_new_files(ws_root, cps):
    write(ws_root, "keep.py", "original\n")
    write(ws_root, "gone.py", "will be deleted\n")
    first = cps.snapshot("fix the parser")

    write(ws_root, "keep.py", "changed\n")
    (ws_root / "gone.py").unlink()
    write(ws_root, "pkg/new/added.py", "new\n")

    changes, safety = cps.restore(first)
    assert {(c.status, c.path) for c in changes} == {
        ("M", "keep.py"), ("D", "gone.py"), ("A", "pkg/new/added.py")}
    assert (ws_root / "keep.py").read_text() == "original\n"
    assert (ws_root / "gone.py").read_text() == "will be deleted\n"
    assert not (ws_root / "pkg").exists()  # emptied directories are cleaned up
    # The state before the restore was saved, so the undo can be undone.
    assert cps.list()[0].kind == "safety" and cps.list()[0].sha == safety
    cps.restore(safety)
    assert (ws_root / "keep.py").read_text() == "changed\n"
    assert (ws_root / "pkg/new/added.py").exists()


def test_the_projects_own_repo_is_untouched(ws_root, cps):
    subprocess.run(["git", "init", "-q", str(ws_root)], check=True)
    write(ws_root, "a.py", "x\n")
    cps.snapshot("turn")
    write(ws_root, "a.py", "y\n")
    cps.snapshot("turn 2")
    status = subprocess.run(["git", "-C", str(ws_root), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    assert status == "?? a.py\n"  # nothing staged, nothing committed
    log = subprocess.run(["git", "-C", str(ws_root), "log"], capture_output=True, text=True)
    assert log.returncode != 0  # still no commits in the user's repo
    assert not str(cps.git_dir).startswith(str(ws_root))


def test_ignored_and_noise_directories_are_not_snapshotted(ws_root, cps):
    write(ws_root, ".gitignore", "secrets.txt\n")
    write(ws_root, "secrets.txt", "v1\n")
    write(ws_root, "node_modules/x/index.js", "v1\n")
    write(ws_root, "app.py", "v1\n")
    first = cps.snapshot("start")
    write(ws_root, "secrets.txt", "v2\n")
    write(ws_root, "node_modules/x/index.js", "v2\n")
    assert cps.changes_since(first) == []
    cps.restore(first)
    assert (ws_root / "secrets.txt").read_text() == "v2\n"


def test_latest_different_walks_back_one_turn_at_a_time(ws_root, cps):
    write(ws_root, "a.py", "1\n")
    cps.snapshot("turn one")
    write(ws_root, "a.py", "2\n")
    cps.snapshot("turn two")
    write(ws_root, "a.py", "3\n")

    cp, changes = cps.latest_different()
    assert cp.label == "turn two" and [c.path for c in changes] == ["a.py"]
    cps.restore(cp.sha)
    assert (ws_root / "a.py").read_text() == "2\n"

    # The safety snapshot isn't a target, and "turn two" now matches.
    cp, _ = cps.latest_different()
    assert cp.label == "turn one"
    cps.restore(cp.sha)
    assert (ws_root / "a.py").read_text() == "1\n"
    assert cps.latest_different() is None


def test_find_by_prefix_and_empty_store(ws_root, cps):
    assert cps.list() == [] and cps.find("abc") is None
    write(ws_root, "a.py", "x\n")
    sha = cps.snapshot("only")
    assert cps.find(sha[:7]).label == "only"


def test_the_users_hooks_and_identity_dont_apply(ws_root, cps, tmp_path, monkeypatch):
    hooks = tmp_path / "evil-hooks"
    hooks.mkdir()
    marker = tmp_path / "hook-ran"
    hook = hooks / "post-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f"[core]\n\thooksPath = {hooks}\n[commit]\n\tgpgsign = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    write(ws_root, "a.py", "x\n")
    cps.snapshot("turn")
    assert not marker.exists()


# ----------------------------------------------------- toolbox integration ---

def test_one_checkpoint_per_turn_before_the_first_change(ws_root, make_toolbox, cps):
    write(ws_root, "a.py", "v1\n")
    tb = make_toolbox()
    tb.checkpoints = cps
    tb.begin_turn("make it v3")
    tb.call("read_file", {"path": "a.py"})
    assert cps.list() == []  # reading changes nothing, so no snapshot
    tb.call("edit_file", {"path": "a.py", "old_string": "v1", "new_string": "v2"})
    tb.call("edit_file", {"path": "a.py", "old_string": "v2", "new_string": "v3"})
    assert [c.label for c in cps.list()] == ["make it v3"]

    tb.begin_turn("next")
    tb.call("write_file", {"path": "b.py", "content": "new\n"})
    assert [c.label for c in cps.list()] == ["next", "make it v3"]
    cps.restore(cps.list()[1].sha)
    assert (ws_root / "a.py").read_text() == "v1\n" and not (ws_root / "b.py").exists()


def test_a_declined_edit_takes_no_checkpoint(ws_root, make_toolbox, cps):
    from tools import Decision
    write(ws_root, "a.py", "v1\n")
    tb = make_toolbox(FakeApprover(change=[Decision(False)]))
    tb.checkpoints = cps
    tb.call("read_file", {"path": "a.py"})
    tb.call("edit_file", {"path": "a.py", "old_string": "v1", "new_string": "v2"})
    assert cps.list() == []


def test_a_broken_checkpoint_never_blocks_the_edit(ws_root, make_toolbox):
    class Broken:
        def snapshot(self, label):
            raise RuntimeError("disk full")

    write(ws_root, "a.py", "v1\n")
    tb = make_toolbox()
    tb.checkpoints = Broken()
    tb.call("read_file", {"path": "a.py"})
    res = tb.call("edit_file", {"path": "a.py", "old_string": "v1", "new_string": "v2"})
    assert res.content.startswith("updated") and (ws_root / "a.py").read_text() == "v2\n"
    assert any("couldn't save a checkpoint (disk full)" in n for n in tb.notices)
