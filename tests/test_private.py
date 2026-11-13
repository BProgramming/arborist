"""/private: a session that leaves nothing behind once it ends."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import chat
import chat_input
import config
import sessions
from tools import ToolResult

MODEL = '[model]\nrepo = "someone/Some-Model-GGUF"\nfile = "Some-Model-Q4_K_M.gguf"\n'


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "test.toml"
    p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + MODEL)
    monkeypatch.setenv("ARBORIST_CONFIG", str(p))
    return config.load()


class FakeServer:
    def __init__(self, healthy=True, fail: int | None = None):
        self.up = healthy
        self.fail = fail
        self.calls: list[tuple[int, str, str]] = []

    def healthy(self):
        return self.up

    def slot_action(self, slot, action, filename):
        self.calls.append((slot, action, filename))
        if self.fail:
            raise chat.SlotError(self.fail, "nope")
        return {"id_slot": slot}


def new_chat(cfg, ws_root, server=None):
    c = chat.Chat(cfg, ws_root, tools_enabled=True)
    c._env_facts = ["OS: test"]  # skip the toolchain probe
    c.client = server or FakeServer()
    return c


needs_git = pytest.mark.skipif(subprocess.run(["which", "git"], capture_output=True).returncode != 0,
                               reason="needs git")


def started_chat(cfg, ws_root):
    """A chat whose loop has started: the session's starting point is recorded."""
    c = new_chat(cfg, ws_root)
    c._mark_session_start()
    return c


class StubPrompt:
    def __init__(self, path):
        self.history = chat_input.SessionHistory(str(path))


def tool_names(c):
    return [t["function"]["name"] for t in c.agent_tools.schemas]


def test_private_session_parks_nothing_and_cleans_up(cfg, ws_root, capsys):
    c = new_chat(cfg, ws_root)
    c.command("/private")
    assert c.private and c.private_dir.is_dir()
    assert c.private_dir.stat().st_mode & 0o777 == 0o700
    c.messages = [{"role": "user", "content": "secret plans"},
                  {"role": "assistant", "content": "noted"}]
    c.park()
    c.end_private()
    store = sessions.Store.from_config(cfg)
    assert not store.path(c.session_key).exists()
    assert not store.kv_path(c.session_key).exists()
    assert c.client.calls == [(sessions.SLOT, "erase", "")]  # no "save"
    assert not c.private_dir.exists()
    assert "nothing kept" in capsys.readouterr().out


def test_a_normal_session_still_parks(cfg, ws_root):
    c = new_chat(cfg, ws_root)
    c.messages = [{"role": "user", "content": "hi"}]
    c.park()
    c.end_private()  # no-op
    assert sessions.Store.from_config(cfg).path(c.session_key).exists()
    assert [a for _, a, _ in c.client.calls] == ["save"]


def test_private_lasts_until_you_leave(cfg, ws_root, capsys):
    c = new_chat(cfg, ws_root)
    c.command("/private")
    folder = c.private_dir
    c.command("/private")
    c.command("/clear")
    assert c.private and c.private_dir == folder
    assert "already private" in capsys.readouterr().out
    c.end_private()


def test_the_model_cant_remember_in_private(cfg, ws_root):
    c = new_chat(cfg, ws_root)
    assert "remember" in tool_names(c)
    assert chat.MEMORY_GUIDE in c._system()
    c.command("/private")
    assert "remember" not in tool_names(c)
    system = c._system()
    assert chat.PRIVATE_GUIDE in system and chat.MEMORY_GUIDE not in system
    # Even if the model calls it anyway.
    res = c.agent_tools.call("remember", '{"memory": "x likes y", "scope": "general"}')
    assert isinstance(res, ToolResult) and "private session" in res.content
    assert "x likes y" not in c.memory.path.read_text()
    c.end_private()


def test_existing_memories_are_still_read(cfg, ws_root):
    c = new_chat(cfg, ws_root)
    c.memory.add("prefers tabs", chat.GENERAL)
    c.command("/private")
    c._clear()  # re-reads the memory block
    assert "prefers tabs" in c._system()
    c.end_private()


def test_logs_and_checkpoints_move_to_the_scratch_folder(cfg, ws_root):
    c = new_chat(cfg, ws_root)
    normal_store = c.checkpoints.git_dir
    c.command("/private")
    assert c.log_dir == c.private_dir / "logs"
    assert c.toolbox.log_dir == c.log_dir
    assert c.checkpoints is c.toolbox.checkpoints
    assert c.checkpoints.git_dir.is_relative_to(c.private_dir)
    assert c.checkpoints.git_dir != normal_store
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log = c._log_error()
    assert log == c.private_dir / "logs" / "chat-errors.log" and log.exists()
    folder = c.private_dir
    c.end_private()
    assert not folder.exists()
    assert not (Path(cfg["derived"]["log_dir"]) / "chat-errors.log").exists()


@needs_git
def test_checkpoint_labels_never_reach_the_normal_store(cfg, ws_root):
    c = new_chat(cfg, ws_root)
    c.command("/private")
    (ws_root / "a.py").write_text("x = 1\n")
    c.toolbox.begin_turn("my secret message")
    c.toolbox._checkpoint()
    assert any("my secret message" in cp.label for cp in c.checkpoints.recent(limit=5))
    c.end_private()
    normal = chat.Checkpoints(ws_root, cfg["derived"]["checkpoint_dir"])
    if normal.git_dir.exists():
        assert all("secret" not in cp.label for cp in normal.recent(limit=50))


def test_resume_is_unavailable_in_private(cfg, ws_root, capsys):
    c = new_chat(cfg, ws_root)
    c.command("/private")
    c.resume()
    assert "unavailable in a private session" in capsys.readouterr().out
    c.end_private()


def test_a_failed_erase_is_reported_and_the_folder_still_goes(cfg, ws_root, capsys):
    c = new_chat(cfg, ws_root, FakeServer(fail=501))
    c.command("/private")
    folder = c.private_dir
    c.end_private()
    assert "couldn't clear the server's cache" in capsys.readouterr().out
    assert not folder.exists()


def test_startup_deletes_folders_left_by_dead_private_sessions(cfg, capsys):
    top = chat.private_root(cfg)
    (top / str(os.getpid())).mkdir(parents=True)  # a live one: ours
    dead = subprocess.Popen(["true"])
    dead.wait()
    (top / str(dead.pid)).mkdir()
    (top / "junk").mkdir()
    chat.prune_private(cfg)
    assert sorted(p.name for p in top.iterdir()) == [str(os.getpid())]
    assert "deleted 2 folders" in capsys.readouterr().out


@pytest.mark.skipif(not chat_input.HAVE_PROMPT_TOOLKIT, reason="needs prompt_toolkit")
def test_prompt_history_stops_writing_when_private(tmp_path):
    path = tmp_path / "prompt_history"
    h = chat_input.SessionHistory(str(path))
    h.append_string("before")
    h.private = True
    h.append_string("during")
    text = path.read_text()
    assert "before" in text and "during" not in text
    assert list(h.get_strings())[-1] == "during"  # ↑ still finds it this session


def test_toolbar_says_private():
    assert chat_input.toolbar_text("M", "high", True, 5, private=True).endswith(", PRIVATE")
    assert "PRIVATE" not in chat_input.toolbar_text("M", "high", True, 5)


# -- going private partway through ----------------------------------------------

def test_mid_session_keeps_the_conversation_and_parks_nothing(cfg, ws_root):
    c = started_chat(cfg, ws_root)
    c.messages = [{"role": "user", "content": "secret plans"},
                  {"role": "assistant", "content": "noted"}]
    c.command("/private")
    assert c.private and len(c.messages) == 2
    c.park()
    c.end_private()
    assert not sessions.Store.from_config(cfg).path(c.session_key).exists()


@pytest.mark.skipif(not chat_input.HAVE_PROMPT_TOOLKIT, reason="needs prompt_toolkit")
def test_this_sessions_input_history_comes_out_of_the_file(cfg, ws_root, tmp_path):
    path = tmp_path / "prompt_history"
    other = chat_input.SessionHistory(str(path))  # another arborist, same file
    other.append_string("from another session")
    c = started_chat(cfg, ws_root)
    c.prompt = StubPrompt(path)
    c.prompt.history.append_string("my secret\nsecond line")
    other.append_string("another one, after")
    c.prompt.history.append_string("/private")
    c.command("/private")
    text = path.read_text()
    assert "secret" not in text and "/private" not in text
    assert "from another session" in text and "another one, after" in text
    c.prompt.history.append_string("said in private")
    assert "said in private" not in path.read_text()
    c.end_private()


@needs_git
def test_this_sessions_checkpoints_move_out_of_the_normal_store(cfg, ws_root):
    (ws_root / "a.py").write_text("x = 1\n")
    before = started_chat(cfg, ws_root)  # an earlier session leaves a checkpoint
    before.toolbox.begin_turn("an old message")
    before.toolbox._checkpoint()
    normal = before.checkpoints

    start = normal.head()
    c = started_chat(cfg, ws_root)  # its session-start snapshot is this session's too
    c.toolbox.begin_turn("my secret message")
    c.toolbox._checkpoint()
    (ws_root / "a.py").write_text("x = 2\n")
    secret_sha = normal.head()
    c.command("/private")

    assert normal.head() == start
    labels = [cp.label for cp in normal.recent(limit=50)]
    assert "an old message" in labels and not any("secret" in lb for lb in labels)
    gone = subprocess.run(["git", "--git-dir", str(normal.git_dir), "cat-file", "-e", secret_sha],
                          capture_output=True)
    assert gone.returncode != 0  # purged, not just unreferenced

    # /undo and /diff still reach back, from the private store.
    assert c.checkpoints.git_dir.is_relative_to(c.private_dir)
    private_labels = [cp.label for cp in c.checkpoints.recent(limit=50)]
    assert "my secret message" in private_labels and "an old message" in private_labels
    assert [ch.path for ch in c.checkpoints.changes_since(c.session_base)] == ["a.py"]
    c.end_private()
    assert normal.head() == start


@needs_git
def test_a_store_created_this_session_is_deleted(cfg, ws_root):
    (ws_root / "a.py").write_text("x = 1\n")
    c = started_chat(cfg, ws_root)  # the session-start snapshot creates the store
    normal = c.checkpoints
    assert normal.git_dir.exists()
    c.command("/private")
    assert not normal.git_dir.exists()
    assert c.checkpoints.head() is not None  # the session start, now private
    c.end_private()


def test_this_sessions_logs_are_taken_back(cfg, ws_root):
    c = started_chat(cfg, ws_root)
    log_dir = Path(cfg["derived"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "chat-errors.log").write_text("--- an older error\n")
    try:
        raise RuntimeError("boom with secret")
    except RuntimeError:
        c._log_error()
    run_log = log_dir / "runs" / "20260101-000000-a.log"
    run_log.parent.mkdir()
    run_log.write_text("output")
    c.toolbox.trail.file(run_log)
    c.toolbox._log_internal_error("edit_file", {"path": "secret.py"})
    c.command("/private")
    assert (log_dir / "chat-errors.log").read_text() == "--- an older error\n"
    assert "secret" not in (log_dir / "tool-errors.log").read_text()
    assert not run_log.exists()
    c.end_private()


def save_note(c, fact, monkeypatch, replace=False):
    tools = c.agent_tools
    if replace:
        monkeypatch.setattr(c.toolbox.approver, "approve_memory",
                            lambda *a: chat.MemoryChoice("replace", [0]), raising=False)
    res = tools.call("remember", {"memory": fact, "scope": "general"})
    assert res.content.startswith("saved"), res.content


def test_memories_saved_this_session_are_asked_about(cfg, ws_root, monkeypatch):
    c = started_chat(cfg, ws_root)
    c.memory.add("uses spaces for indents", chat.GENERAL)  # from an earlier session
    save_note(c, "uses tabs for indents now", monkeypatch, replace=True)
    save_note(c, "is planning a surprise party", monkeypatch)
    answers = iter(["", "n"])  # remove the first, keep the second
    monkeypatch.setattr(chat, "ask", lambda prompt: next(answers))
    c.command("/private")
    current = [n.entry for n in c.memory.notes()]
    assert not any("tabs" in e for e in current)
    assert any("uses spaces for indents" in e for e in current)  # current again
    assert any("surprise party" in e for e in current)
    assert not any(n.outdated for n in c.memory.notes(include_outdated=True))
    c.end_private()


def test_ctrl_c_at_the_memory_question_calls_it_off(cfg, ws_root, monkeypatch):
    c = started_chat(cfg, ws_root)
    save_note(c, "likes long walks", monkeypatch)

    def interrupted(prompt):
        raise KeyboardInterrupt
    monkeypatch.setattr(chat, "ask", interrupted)
    c.command("/private")
    assert not c.private and c.private_dir is None
    assert any("long walks" in n.entry for n in c.memory.notes())
    assert len(c.saved_notes) == 1  # still offered next time
