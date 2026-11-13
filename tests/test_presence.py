"""Sessions sharing one server: who can see whom, the warnings, and that a
private session's cache never gets saved by somebody else's park."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import chat
import config
import presence

MODEL = '[model]\nrepo = "someone/Some-Model-GGUF"\nfile = "Some-Model-Q4_K_M.gguf"\n'
HERE = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    p = tmp_path / "test.toml"
    p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + MODEL)
    monkeypatch.setenv("ARBORIST_CONFIG", str(p))
    return config.load()


class FakeServer:
    def __init__(self):
        self.calls: list[tuple[int, str, str]] = []

    @staticmethod
    def healthy():
        return True

    def slot_action(self, slot, action, filename):
        self.calls.append((slot, action, filename))
        return {"id_slot": slot, "n_saved": 10, "n_written": 100}


def new_chat(cfg, folder):
    folder.mkdir(exist_ok=True)
    c = chat.Chat(cfg, folder, tools_enabled=True)
    c._env_facts = ["OS: test"]
    c.client = FakeServer()
    return c


# ---- the registry -----------------------------------------------------------------

def test_sessions_see_each_other_until_they_close(tmp_path):
    a, b = presence.Presence(tmp_path / "reg"), presence.Presence(tmp_path / "reg", kind="task")
    a.start()
    assert a.others() == []
    b.start()
    [seen] = a.others()
    assert seen["kind"] == "task" and seen["pid"] == os.getpid() and seen["private"] is False
    assert [s["kind"] for s in b.others()] == ["chat"]
    b.close()
    assert a.others() == []
    a.close()
    assert list((tmp_path / "reg").iterdir()) == []


def test_private_flag_is_visible_to_others(tmp_path):
    a, b = presence.Presence(tmp_path), presence.Presence(tmp_path)
    a.start(), b.start()
    b.set_private()
    assert [s["private"] for s in a.others()] == [True]
    assert [s["private"] for s in b.others()] == [False]
    a.close(), b.close()


def test_entry_left_by_a_dead_session_is_removed(tmp_path):
    a = presence.Presence(tmp_path)
    a.start()
    stale = tmp_path / "99999-dead"
    stale.write_text('{"pid": 99999, "kind": "chat", "private": true}')  # nobody holds its lock
    half = tmp_path / ".99999-dead.tmp"
    half.write_text("")
    assert a.others() == []
    assert not stale.exists() and not half.exists()
    a.close()


def test_session_killed_outright_stops_counting(tmp_path):
    """The lock goes with the process, however it dies."""
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import presence\n"
            "p = presence.Presence(sys.argv[2]); p.start(); print('up', flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code, str(HERE), str(tmp_path)],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "up"
        mine = presence.Presence(tmp_path)
        mine.start()
        assert [s["pid"] for s in mine.others()] == [child.pid]
        child.kill()
        child.wait()
        assert mine.others() == []
        mine.close()
    finally:
        child.kill()


def test_registry_is_per_server(cfg):
    other = {**cfg, "server": {**cfg["server"], "port": cfg["server"]["port"] + 1}}
    assert presence.folder(cfg) != presence.folder(other)
    assert presence.folder(cfg).parent == Path(cfg["derived"]["root"]) / "active"


def test_count_for_the_launcher(cfg):
    def count():
        r = subprocess.run([sys.executable, str(HERE / "presence.py"), "--count"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return int(r.stdout)
    assert count() == 0
    p = presence.Presence(presence.folder(cfg))
    p.start()
    assert count() == 1
    p.close()
    assert count() == 0


# ---- the chat -----------------------------------------------------------------------

def test_new_session_is_warned_about_a_private_one(cfg, tmp_path, capsys):
    private = new_chat(cfg, tmp_path / "a")
    private.join_server()
    private.command("/private")
    capsys.readouterr()
    late = new_chat(cfg, tmp_path / "b")
    late.join_server()
    out = capsys.readouterr().out
    assert "warning" in out and "a private session is using this server" in out
    late.leave_server()
    private.end_private(), private.leave_server()


def test_new_session_isnt_warned_about_ordinary_ones(cfg, tmp_path, capsys):
    first = new_chat(cfg, tmp_path / "a")
    first.join_server()
    capsys.readouterr()
    second = new_chat(cfg, tmp_path / "b")
    second.join_server()
    assert "warning" not in capsys.readouterr().out
    first.leave_server(), second.leave_server()


def test_going_private_warns_about_open_sessions(cfg, tmp_path, capsys):
    other = new_chat(cfg, tmp_path / "a")
    other.join_server()
    c = new_chat(cfg, tmp_path / "b")
    c.join_server()
    capsys.readouterr()
    c.command("/private")
    out = capsys.readouterr().out
    assert "1 other session is open on this server" in out and f"pid {os.getpid()}" in out
    c._warn_shared()  # nothing new: not repeated
    assert "warning" not in capsys.readouterr().out
    other.leave_server()
    c.end_private(), c.leave_server()


def test_going_private_alone_doesnt_warn(cfg, tmp_path, capsys):
    c = new_chat(cfg, tmp_path / "a")
    c.join_server()
    c.command("/private")
    assert "warning" not in capsys.readouterr().out
    c.end_private(), c.leave_server()


def test_private_session_is_told_about_a_later_arrival_once(cfg, tmp_path, capsys):
    c = new_chat(cfg, tmp_path / "a")
    c.join_server()
    c.command("/private")
    late = new_chat(cfg, tmp_path / "b")
    late.join_server("task")
    capsys.readouterr()
    c._warn_shared()
    out = capsys.readouterr().out
    assert "1 more session has started on this server" in out and "a task" in out
    c._warn_shared()
    assert capsys.readouterr().out == ""
    late.leave_server()
    c.end_private(), c.leave_server()


def test_park_doesnt_save_the_cache_while_another_session_is_private(cfg, tmp_path, capsys):
    private = new_chat(cfg, tmp_path / "a")
    private.join_server()
    private.command("/private")
    normal = new_chat(cfg, tmp_path / "b")
    normal.join_server()
    normal.messages.append({"role": "user", "content": "hello"})
    capsys.readouterr()
    normal.park()
    out = capsys.readouterr().out
    assert normal.client.calls == []  # the shared slot was never written to disk
    assert "a private session is using this server" in out
    assert normal.store.load(normal.session_key)["kv"] is None  # the messages still park
    # Once the private session is gone, parking saves the cache as usual.
    private.end_private(), private.leave_server()
    normal.messages.append({"role": "assistant", "content": "hi"})
    normal.park()
    assert [c[1] for c in normal.client.calls] == ["save"]
    normal.leave_server()


def test_sessions_that_never_joined_behave_as_before(cfg, tmp_path):
    c = new_chat(cfg, tmp_path / "a")
    assert c.other_sessions() == []
    c.messages.append({"role": "user", "content": "hello"})
    c.park()
    assert [call[1] for call in c.client.calls] == ["save"]
