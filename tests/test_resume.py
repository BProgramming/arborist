"""Parking a conversation on exit and /resume: messages, state and the KV cache."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import chat
import config
import sessions

MODEL = '[model]\nrepo = "someone/Some-Model-GGUF"\nfile = "Some-Model-Q4_K_M.gguf"\n'


@pytest.fixture
def load_toml(tmp_path, monkeypatch):
    def load(text: str = ""):
        p = tmp_path / "test.toml"
        model = "" if "[model]" in text else MODEL
        p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + model + text)
        monkeypatch.setenv("ARBORIST_CONFIG", str(p))
        return config.load()
    return load


class FakeServer:
    """Stands in for chat.Client: records slot calls, writes a fake cache file."""

    def __init__(self, kv_dir, healthy=True, fail: int | None = None):
        self.kv_dir = kv_dir
        self.up = healthy
        self.fail = fail
        self.calls: list[tuple[int, str, str]] = []

    def healthy(self):
        return self.up

    def slot_action(self, slot, action, filename):
        self.calls.append((slot, action, filename))
        if self.fail:
            raise chat.SlotError(self.fail, "nope")
        if action == "save":
            self.kv_dir.mkdir(parents=True, exist_ok=True)
            (self.kv_dir / filename).write_bytes(b"kv" * 100)
            return {"id_slot": slot, "filename": filename, "n_saved": 1234, "n_written": 200}
        return {"id_slot": slot, "filename": filename, "n_restored": 1234, "n_read": 200}


def new_chat(cfg, ws_root, server=None):
    c = chat.Chat(cfg, ws_root, tools_enabled=True)
    c._env_facts = ["OS: test"]  # skip the toolchain probe
    c.client = server or FakeServer(sessions.Store.from_config(cfg).kv_dir)
    return c


CONVO = [{"role": "user", "content": "add a flag"},
         {"role": "assistant", "content": "done"}]


def test_park_then_resume_restores_the_same_request_prefix(load_toml, ws_root, monkeypatch):
    cfg = load_toml()
    first = new_chat(cfg, ws_root)
    first.messages = list(CONVO)
    (ws_root / "a.py").write_text("x = 1\n")
    first.toolbox.model_writes[ws_root / "a.py"] = "abc"
    first.toolbox.web_tainted = True
    before = first.request_body(first.messages, with_tools=True)
    first.park()
    assert first.client.calls == [(0, "save", sessions.kv_file(first.session_key))]

    # Later: a new day, new memories would be read, a fresh probe would run.
    monkeypatch.setattr(chat, "date_line", lambda now=None: "Today is some other day.")
    second = new_chat(cfg, ws_root)
    second._env_facts = None
    second.resume()
    after = second.request_body(second.messages, with_tools=True)

    # Byte-identical system prompt, tools and history: the cache lines up.
    assert after["messages"] == before["messages"] and after["tools"] == before["tools"]
    assert second.client.calls == [(0, "restore", sessions.kv_file(first.session_key))]
    # The new date reaches the model as a note on the next message, after the prefix.
    assert "Today is some other day." in second.pending_notes[0]
    # Safety state carries over.
    assert second.toolbox.web_tainted
    assert second.toolbox.model_writes == {(ws_root / "a.py").resolve(): "abc"}
    # Reads don't: files may have changed while it was parked.
    assert not second.toolbox.ws._seen


def test_requests_are_pinned_to_the_parked_slot(load_toml, ws_root):
    on = new_chat(load_toml(), ws_root).request_body([], with_tools=True)
    assert on["id_slot"] == sessions.SLOT
    off = new_chat(load_toml('[chat]\nresume = false\n'), ws_root).request_body([], with_tools=True)
    assert "id_slot" not in off


def test_tasks_dont_park():
    import task
    assert task.TaskChat.resumable is False


def test_server_gets_a_slot_save_path_only_when_resume_is_on(load_toml):
    cfg = load_toml()
    args = config.server_args(cfg)
    assert args[args.index("--slot-save-path") + 1] == cfg["derived"]["kv_dir"]
    assert "--slot-save-path" not in config.server_args(load_toml('[chat]\nresume = false\n'))


@pytest.mark.parametrize("snippet, message", [
    ('[chat]\nresume = "yes"\n', "chat.resume"),
    ('[chat]\nresume_days = -1\n', "chat.resume_days"),
])
def test_resume_settings_are_validated(load_toml, snippet, message):
    with pytest.raises(config.ConfigError, match=message):
        load_toml(snippet)


def test_nothing_new_since_resume_isnt_parked_again(load_toml, ws_root):
    cfg = load_toml()
    first = new_chat(cfg, ws_root)
    first.messages = list(CONVO)
    first.park()
    second = new_chat(cfg, ws_root)
    second.resume()
    second.park()
    assert [a for _, a, _ in second.client.calls] == ["restore"]
    second.messages.append({"role": "user", "content": "more"})
    second.park()
    assert [a for _, a, _ in second.client.calls] == ["restore", "save"]


def test_an_empty_chat_doesnt_replace_the_parked_one(load_toml, ws_root):
    cfg = load_toml()
    first = new_chat(cfg, ws_root)
    first.messages = list(CONVO)
    first.park()
    new_chat(cfg, ws_root).park()
    assert sessions.Store.from_config(cfg).load(first.session_key)["messages"] == CONVO


def test_server_without_cache_saving_still_parks_the_messages(load_toml, ws_root, capsys):
    cfg = load_toml()
    store = sessions.Store.from_config(cfg)
    c = new_chat(cfg, ws_root, FakeServer(store.kv_dir, fail=501))
    store.kv_dir.mkdir(parents=True)
    store.kv_path(c.session_key).write_bytes(b"stale")
    c.messages = list(CONVO)
    c.park()
    out = capsys.readouterr().out
    assert "cache not saved" in out and "agent stop" in out
    assert store.load(c.session_key)["kv"] is None
    assert not store.kv_path(c.session_key).exists()  # a stale cache is removed

    again = new_chat(cfg, ws_root)
    again.resume()
    assert again.messages == CONVO and again.client.calls == []
    assert "no cache was saved" in capsys.readouterr().out


def test_cache_isnt_loaded_for_another_model(load_toml, ws_root, capsys):
    first = new_chat(load_toml(), ws_root)
    first.messages = list(CONVO)
    first.park()
    other = new_chat(load_toml('[model]\nrepo = "a/b"\nfile = "Other.gguf"\n'), ws_root)
    other.resume()
    assert other.messages == CONVO and other.client.calls == []
    assert "another model" in capsys.readouterr().out


def test_cache_isnt_loaded_when_the_system_prompt_changed(load_toml, ws_root, capsys):
    cfg = load_toml()
    first = new_chat(cfg, ws_root)
    first.messages = list(CONVO)
    first.park()
    path = sessions.Store.from_config(cfg).path(first.session_key)
    data = json.loads(path.read_text())
    data["system"] = "something older"
    path.write_text(json.dumps(data))
    second = new_chat(cfg, ws_root)
    second.resume()
    assert second.client.calls == [] and "system prompt has changed" in capsys.readouterr().out


def test_failed_restore_keeps_the_conversation(load_toml, ws_root, capsys):
    cfg = load_toml()
    first = new_chat(cfg, ws_root)
    first.messages = list(CONVO)
    first.park()
    second = new_chat(cfg, ws_root, FakeServer(sessions.Store.from_config(cfg).kv_dir, fail=400))
    second.resume()
    assert second.messages == CONVO
    assert "couldn't load the cache" in capsys.readouterr().out


def test_resume_needs_an_empty_conversation(load_toml, ws_root, capsys):
    c = new_chat(load_toml(), ws_root)
    c.messages = list(CONVO)
    c.command("/resume")
    assert "/clear first" in capsys.readouterr().out
    c._clear()
    c.command("/resume")
    assert "nothing parked" in capsys.readouterr().out


def test_folders_and_modes_park_separately(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    keys = {sessions.key_for(a, True), sessions.key_for(b, True), sessions.key_for(a, False),
            sessions.key_for(None, False)}
    assert len(keys) == 4 and all(k.isalnum() for k in keys)


def test_parked_files_are_private(load_toml, ws_root):
    cfg = load_toml()
    c = new_chat(cfg, ws_root)
    c.messages = list(CONVO)
    c.park()
    store = sessions.Store.from_config(cfg)
    assert store.path(c.session_key).stat().st_mode & 0o777 == 0o600
    assert store.session_dir.stat().st_mode & 0o777 == 0o700
    assert store.kv_path(c.session_key).stat().st_mode & 0o777 == 0o600


def test_prune_removes_old_conversations_and_orphan_caches(tmp_path):
    store = sessions.Store(tmp_path / "sessions", tmp_path / "kv")
    store.kv_dir.mkdir()
    old, new = store.save("old", {"messages": []}), store.save("new", {"messages": []})
    for k in ("old", "new", "orphan"):
        store.kv_path(k).write_bytes(b"x")
    os.utime(old, (time.time() - 10 * 86400,) * 2)
    assert store.prune(7) == 1
    assert not old.exists() and new.exists()
    assert sorted(p.name for p in store.kv_dir.iterdir()) == ["new.bin"]
    assert store.prune(0) == 0 and new.exists()  # 0 keeps them


def test_slot_action_talks_to_the_slots_endpoint():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        # noinspection PyShadowingBuiltins
        def log_message(self, format, *args):
            pass

        # noinspection PyPep8Naming
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            if "restore" in self.path:
                payload = {"error": {"code": 501, "message": "This server does not support slots action.",
                                     "type": "not_supported_error"}}
                self.send_response(501)
            else:
                payload = {"id_slot": 0, "filename": body["filename"], "n_saved": 7, "n_written": 99}
                self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        client = chat.Client(f"http://127.0.0.1:{server.server_port}")
        assert client.slot_action(0, "save", "k.bin")["n_saved"] == 7
        with pytest.raises(chat.SlotError, match="does not support slots action.*agent stop"):
            client.slot_action(0, "restore", "k.bin")
    finally:
        server.shutdown()
    assert seen == [("/slots/0?action=save", {"filename": "k.bin"}),
                    ("/slots/0?action=restore", {"filename": "k.bin"})]


def test_stats_show_cached_tokens(load_toml, ws_root, capsys):
    import agent
    c = new_chat(load_toml(), ws_root)
    c.last = agent.TurnResult(text="hi", steps=0, hard_stopped=False, usage={},
                              timings={"prompt_n": 12, "prompt_per_second": 500.0,
                                       "cache_n": 40000, "predicted_per_second": 30.0})
    c._print_stats()
    assert "prompt 12 @ 500 tok/s, 40,000 cached" in capsys.readouterr().out
