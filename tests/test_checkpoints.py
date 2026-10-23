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


# ------------------------------------------------------------------ pruning ---

from datetime import datetime, timedelta  # noqa: E402

from checkpoints import prune_store  # noqa: E402


def aged_snapshot(cps, label, days_ago, kind="turn"):
    """A checkpoint dated `days_ago` days back."""
    cps._ensure()
    cps._stage()
    when = int((datetime.now() - timedelta(days=days_ago)).timestamp())
    cps._git("commit", "--quiet", "--no-verify", "--allow-empty", "-m", f"[{kind}] {label}",
             extra_env={"GIT_COMMITTER_DATE": f"{when} +0000", "GIT_AUTHOR_DATE": f"{when} +0000"})
    return cps._git("rev-parse", "HEAD").stdout.decode().strip()


def test_old_checkpoints_are_dropped_and_recent_ones_still_restore(ws_root, cps, tmp_path):
    write(ws_root, "a.py", "v1\n")
    old1 = aged_snapshot(cps, "ancient", 30)
    write(ws_root, "a.py", "v2\n")
    aged_snapshot(cps, "old", 20)
    write(ws_root, "a.py", "v3\n")
    aged_snapshot(cps, "recent", 2)
    write(ws_root, "a.py", "v4\n")
    cps.snapshot("today")
    write(ws_root, "a.py", "v5\n")

    assert prune_store(tmp_path / "store", 14) == (2, 0)
    assert [c.label for c in cps.list()] == ["today", "recent"]
    # The kept checkpoints still hold the right files.
    cp, _ = cps.latest_different()
    cps.restore(cp.sha)
    assert (ws_root / "a.py").read_text() == "v4\n"
    cps.restore(cps.find(cps.list()[-1].short).sha)
    assert (ws_root / "a.py").read_text() == "v3\n"
    # And the old commits are really gone from the store.
    with pytest.raises(Exception):
        cps._git("cat-file", "-e", old1)


def test_undo_chain_survives_the_rewrite(ws_root, cps, tmp_path):
    write(ws_root, "a.py", "1\n")
    aged_snapshot(cps, "ancient", 30)
    write(ws_root, "a.py", "2\n")
    cps.snapshot("turn one")
    write(ws_root, "a.py", "3\n")
    cps.snapshot("turn two")
    write(ws_root, "a.py", "4\n")
    cp, _ = cps.latest_different()
    cps.restore(cp.sha)  # back to "turn two"; a safety snapshot points at it

    prune_store(tmp_path / "store", 14)
    safety = cps.list()[0]
    assert safety.kind == "safety" and safety.restored == cps.list()[1].sha
    # The next /undo still goes further back, not forward.
    cp, _ = cps.latest_different()
    assert cp.label == "turn one"


def test_a_store_with_nothing_recent_is_deleted_and_recreated(ws_root, cps, tmp_path):
    write(ws_root, "a.py", "x\n")
    aged_snapshot(cps, "old", 20)
    assert prune_store(tmp_path / "store", 14) == (1, 1)
    assert not cps.git_dir.exists()
    cps.snapshot("fresh start")
    assert [c.label for c in cps.list()] == ["fresh start"]


def test_a_deleted_workspaces_store_is_removed(ws_root, cps, tmp_path):
    import shutil
    write(ws_root, "a.py", "x\n")
    cps.snapshot("recent")
    shutil.rmtree(ws_root)
    assert prune_store(tmp_path / "store", 14) == (0, 1)
    assert not cps.git_dir.exists()


def test_pruning_never_touches_project_repos_or_unknown_dirs(ws_root, cps, tmp_path):
    # The project's own repo, with an old commit.
    subprocess.run(["git", "init", "-q", str(ws_root)], check=True)
    write(ws_root, "a.py", "x\n")
    old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S")
    env = {"GIT_COMMITTER_DATE": old, "GIT_AUTHOR_DATE": old, "GIT_AUTHOR_NAME": "me",
           "GIT_AUTHOR_EMAIL": "me@x", "GIT_COMMITTER_NAME": "me", "GIT_COMMITTER_EMAIL": "me@x",
           "PATH": __import__("os").environ["PATH"]}
    subprocess.run(["git", "-C", str(ws_root), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(ws_root), "commit", "-qm", "my work"], check=True, env=env)
    before = subprocess.run(["git", "-C", str(ws_root), "log", "--format=%H"],
                            capture_output=True, text=True).stdout
    # A stray directory in the store that isn't one of ours, and a real old store.
    stray = tmp_path / "store" / "someone-else.git"
    stray.mkdir(parents=True)
    (stray / "HEAD").write_text("ref: refs/heads/main\n")
    aged_snapshot(cps, "old", 30)

    prune_store(tmp_path / "store", 14)
    after = subprocess.run(["git", "-C", str(ws_root), "log", "--format=%H"],
                           capture_output=True, text=True).stdout
    assert after == before and "my work" in subprocess.run(
        ["git", "-C", str(ws_root), "log", "--format=%s"], capture_output=True, text=True).stdout
    assert stray.exists()
    assert not cps.git_dir.exists()


def test_zero_days_keeps_everything(ws_root, cps, tmp_path):
    write(ws_root, "a.py", "x\n")
    aged_snapshot(cps, "ancient", 400)
    assert prune_store(tmp_path / "store", 0) == (0, 0)
    assert len(cps.list()) == 1
