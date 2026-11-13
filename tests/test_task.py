"""Delegated tasks (task.py), driven by a scripted fake model (no server needed)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import config
import sandbox
import task
from conftest import sandboxed, write
from test_agent import text_reply, tool_reply

MODEL = '[model]\nrepo = "someone/Some-Model-GGUF"\nfile = "Some-Model-Q4_K_M.gguf"\n'
REPORT = "RESULT: DONE\nCHANGED: x\nCHECKED: not checked\nNOTES: none"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    def load(text: str = ""):
        p = tmp_path / "test.toml"
        p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + MODEL + text)
        monkeypatch.setenv("ARBORIST_CONFIG", str(p))
        return config.load()
    return load


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(ws_root):
    git(ws_root, "init", "-q")
    git(ws_root, "config", "user.email", "t@example.com")
    git(ws_root, "config", "user.name", "t")
    write(ws_root, ".gitignore", "secret.env\n")
    write(ws_root, "app.py", "def add(a, b):\n    return a - b\n")
    git(ws_root, "add", "-A")
    git(ws_root, "commit", "-qm", "init")
    return ws_root


class FakeClient:
    """Stands in for chat.Client: plays back one scripted reply per request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def stream_chat(self, body):
        self.requests.append(body)
        yield from (self.replies.pop(0) if self.replies else text_reply("(out of script)"))

    @staticmethod
    def healthy():
        return True


@pytest.fixture
def no_sandbox_check(monkeypatch):
    """Tests that never run code don't need bwrap to say it works."""
    monkeypatch.setattr(sandbox, "status", lambda: (True, ""))


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


# ---------------------------------------------------------------- the copy ---

def test_copy_includes_uncommitted_work_but_not_ignored_files(repo, tmp_path):
    write(repo, "app.py", "def add(a, b):\n    return a + b  # fixed, not committed\n")
    write(repo, "new.py", "x = 1\n")
    write(repo, "secret.env", "TOKEN=hunter2\n")
    c = task.make_copy(repo, tmp_path / "copy")
    assert "fixed, not committed" in (c.work / "app.py").read_text()
    assert (c.work / "new.py").read_text() == "x = 1\n"
    assert not (c.work / "secret.env").exists()
    # The base commit holds all of it, so the task's patch is only the task's.
    assert git(c.work, "status", "--porcelain") == ""
    assert git(c.work, "rev-parse", "HEAD").strip() == c.base


def test_copy_of_a_subfolder_works_in_the_same_subfolder(repo, tmp_path):
    write(repo, "pkg/mod.py", "y = 2\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "pkg")
    c = task.make_copy(repo / "pkg", tmp_path / "copy")
    assert c.repo == repo.resolve()
    assert c.workspace == c.work / "pkg"


def test_copy_of_an_empty_subfolder_still_has_it(repo, tmp_path):
    (repo / "empty").mkdir()  # git doesn't track empty folders
    c = task.make_copy(repo / "empty", tmp_path / "copy")
    assert c.workspace.is_dir()


def test_copy_needs_a_git_repository_with_a_commit(ws_root, tmp_path):
    with pytest.raises(task.TaskError, match="isn't in a git repository"):
        task.make_copy(ws_root, tmp_path / "copy")
    git(ws_root, "init", "-q")
    with pytest.raises(task.TaskError, match="no commits"):
        task.make_copy(ws_root, tmp_path / "copy2")


def test_project_venv_is_found_above_the_start_folder(repo):
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("")
    (repo / "pkg").mkdir()
    assert task.project_venv(repo / "pkg", repo) == repo / ".venv"


# ------------------------------------------------------------------ a task ---

def test_auto_task_edits_only_the_copy_and_reports_a_patch(repo, cfg, no_sandbox_check):
    before = tree(repo)
    client = FakeClient(
        tool_reply(("read_file", {"path": "app.py"})),
        tool_reply(("edit_file", {"path": "app.py", "old_string": "a - b", "new_string": "a + b"})),
        text_reply(REPORT),
    )
    report = task.run_task(cfg(), repo, "Fix add() in app.py.", client=client)

    assert tree(repo) == before  # the real project is untouched
    assert report["status"] == "done" and report["model_result"] == "DONE"
    assert report["files"] == [{"status": "modified", "path": "app.py"}]
    patch = Path(report["patch"]).read_text()
    assert "-    return a - b" in patch and "+    return a + b" in patch
    saved = json.loads((Path(report["task_dir"]) / "report.json").read_text())
    assert saved == report
    assert Path(report["transcript"]).exists()

    # The patch applies to the real project as it stands.
    subprocess.run(["git", "-C", str(repo), "apply", "--check", report["patch"]], check=True)


def test_auto_task_has_no_web_or_memory_tools(repo, cfg, no_sandbox_check):
    client = FakeClient(text_reply(REPORT))
    task.run_task(cfg('[tools]\nweb_enabled = true\n'), repo, "Say hi.", client=client)
    names = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert names.isdisjoint({"fetch_url", "crawl_docs", "remember"})
    assert {"read_file", "edit_file", "run_tests"} <= names
    system = client.requests[0]["messages"][0]["content"]
    assert "delegated task" in system and "Nobody is watching" in system


def test_auto_mode_forces_the_sandbox_openings_off():
    t = task.task_tools({"run_writable": ["~/x"], "run_env_keep": ["TOKEN"], "run_display": True,
                         "run_gpu": True, "web_enabled": True, "approve_edits": "ask",
                         "run_sandbox": True}, "auto")
    assert t["run_writable"] == [] and t["run_env_keep"] == []
    assert not t["run_display"] and not t["run_gpu"] and not t["web_enabled"]
    assert t["approve_edits"] == "auto" and t["run_sandbox"]
    assert task.task_tools({"approve_edits": "auto"}, "interactive")["approve_edits"] == "ask"


def test_auto_mode_refuses_without_the_sandbox(repo, cfg, monkeypatch):
    with pytest.raises(task.TaskError, match="run_sandbox is false"):
        task.run_task(cfg('[tools]\nrun_sandbox = false\n'), repo, "x", client=FakeClient())
    monkeypatch.setattr(sandbox, "status", lambda: (False, "bwrap is not installed"))
    with pytest.raises(task.TaskError, match="bwrap is not installed"):
        task.run_task(cfg(), repo, "x", client=FakeClient())


def test_interactive_mode_needs_a_terminal(repo, cfg):
    with pytest.raises(task.TaskError, match="needs a terminal"):
        task.run_task(cfg(), repo, "x", mode="interactive", client=FakeClient())


def test_a_stuck_model_ends_the_task_as_blocked(repo, cfg, no_sandbox_check):
    bad = tool_reply(("read_file", {"path": "nope.py"}))
    client = FakeClient(bad, bad, bad, bad, text_reply("I can't find the file."))
    report = task.run_task(cfg(), repo, "Fix nope.py.", client=client)
    assert report["status"] == "blocked"
    assert report["harness"].startswith("paused:")
    assert report["patch"] is None and report["files"] == []


def test_the_model_saying_blocked_is_reported(repo, cfg, no_sandbox_check):
    client = FakeClient(text_reply("RESULT: BLOCKED\nNOTES: which function?"))
    report = task.run_task(cfg(), repo, "Fix it.", client=client)
    assert report["status"] == "blocked" and report["model_result"] == "BLOCKED"


@sandboxed
def test_after_a_run_the_model_is_told_to_carry_on(repo, cfg):
    write(repo, "test_app.py", "from app import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "test")
    client = FakeClient(
        tool_reply(("run_tests", {"path": "test_app.py"})),
        text_reply("add() subtracts; it should add."),
        tool_reply(("read_file", {"path": "app.py"})),
        tool_reply(("edit_file", {"path": "app.py", "old_string": "a - b", "new_string": "a + b"})),
        tool_reply(("run_tests", {"path": "test_app.py"})),
        text_reply("Fixed; tests pass."),
        text_reply(REPORT),
    )
    report = task.run_task(cfg(), repo, "Make test_app.py pass.", client=client)
    users = [m["content"] for m in client.requests[-1]["messages"] if m["role"] == "user"]
    assert task.FIX in users and task.CONTINUE in users
    assert report["status"] == "done"
    assert report["tests"] == {"runs": 2, "last": "passed"}
    assert "a + b" not in (repo / "app.py").read_text()  # still only in the copy


@sandboxed
def test_done_with_failing_tests_is_not_reported_as_done(repo, cfg):
    write(repo, "test_app.py", "from app import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
    client = FakeClient(tool_reply(("run_tests", {"path": "test_app.py"})),
                        text_reply("All good."), text_reply(REPORT))
    report = task.run_task(cfg(), repo, "Run the tests.", client=client)
    assert report["model_result"] == "DONE"
    assert report["status"] == "tests_failing"
    assert report["tests"] == {"runs": 1, "last": "failed"}


@sandboxed
def test_the_task_ends_after_max_turns(repo, cfg):
    write(repo, "test_app.py", "def test_ok():\n    assert True\n")
    run = tool_reply(("run_tests", {"path": "test_app.py"}))
    client = FakeClient(run, text_reply("ok"), run, text_reply("ok"), run, text_reply("ok"))
    report = task.run_task(cfg(), repo, "Run the tests.", max_turns=2, client=client)
    assert report["status"] == "out_of_turns" and report["turns"] == 2


# ---------------------------------------------------------------- commands ---

def test_list_and_clean(repo, cfg, no_sandbox_check):
    c = cfg()
    report = task.run_task(c, repo, "Say hi.", client=FakeClient(text_reply(REPORT)))
    listed = task.list_tasks(c)
    assert [t["id"] for t in listed] == [report["id"]]
    with pytest.raises(task.TaskError):
        task.clean_tasks(c, ["../../etc"], everything=False)
    assert task.clean_tasks(c, [report["id"]], everything=False) == [report["id"]]
    assert task.list_tasks(c) == []


# ---------------------------------------------------------------- progress ---

def test_progress_is_recorded_and_status_reports_it(repo, cfg, no_sandbox_check):
    c = cfg()
    client = FakeClient(
        tool_reply(("read_file", {"path": "app.py"})),
        text_reply(REPORT),
    )
    report = task.run_task(c, repo, "Read app.py.", client=client)
    p = json.loads((Path(report["task_dir"]) / "progress.json").read_text())
    assert p["state"] == "done" and p["steps"] == 1 and p["turn"] == 1
    assert p["last"].startswith("read_file")
    out = task.task_status(c)
    assert out.startswith(f"{report['id']}: done") and "report.json" in out
    assert task.task_status(c, report["id"]) == out


def test_status_while_running_and_after_the_process_died(repo, cfg, no_sandbox_check):
    c = cfg()
    d = task.tasks_dir(c) / "20990101-000000-x"
    d.mkdir(parents=True)
    prog = task.Progress(d, d.name, "auto")
    prog.update(turn=1, steps=3, doing="run_tests(path=test_app.py)", tests=["failed"])
    out = task.task_status(c, d.name, now=prog.data["updated"] + 5)
    assert ": running," in out and "now: run_tests(path=test_app.py) (for 5s)" in out
    assert "tests: failed" in out
    assert task.list_tasks(c)[0]["status"] == "running"

    prog.update(pid=2 ** 31 - 2)  # no such process
    assert ": died," in task.task_status(c, d.name)
    assert task.list_tasks(c)[0]["status"] == "died"


def test_status_with_no_tasks_or_a_bad_id(cfg):
    c = cfg()
    with pytest.raises(task.TaskError):
        task.task_status(c)
    with pytest.raises(task.TaskError):
        task.task_status(c, "../../etc")


def test_task_compaction_notices_reach_the_tasks_own_ui(repo, cfg, tmp_path):
    c = task.make_copy(repo, tmp_path / "copy")
    session = task.TaskChat(cfg(), c, repo, "auto", tmp_path / "taskdir")
    assert session.compactor.notice == session.ui.notice
