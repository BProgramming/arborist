"""Config validation, server arguments, and the request body chat.py sends."""
from __future__ import annotations

from pathlib import Path

import pytest

import config


EFFORTS = '[model]\nrepo = "a/b"\nfile = "c.gguf"\nefforts = ["low", "high"]\n'
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


def test_repo_config_is_valid(monkeypatch):
    monkeypatch.delenv("ARBORIST_CONFIG", raising=False)
    cfg = config.load(Path(config.__file__).with_name("arborist.toml"))
    assert cfg["tools"]["approve_edits"] == "ask"
    assert cfg["derived"]["model_name"] and cfg["derived"]["model_path"].endswith(cfg["model"]["file"])
    assert cfg["chat"]["reasoning_effort"] in cfg["model"]["efforts"]
    joined = " ".join(config.server_args(cfg))
    assert "--reasoning-effort-allow none,medium,xhigh --reasoning-effort-fallback medium" in joined


def test_defaults_fill_missing_sections(load_toml):
    cfg = load_toml()
    assert cfg["tools"]["run_timeout"] == 120
    assert cfg["derived"]["model_file"] == "Some-Model-Q4_K_M.gguf"
    assert cfg["derived"]["model_url"] == \
        "https://huggingface.co/someone/Some-Model-GGUF/resolve/main/Some-Model-Q4_K_M.gguf"
    # No name given: the chat falls back to the file name.
    assert cfg["derived"]["model_name"] == "Some-Model-Q4_K_M"


def test_model_name_comes_from_the_toml(load_toml):
    cfg = load_toml('[model]\nname = "Qwen Coder"\nrepo = "a/b"\nfile = "c.gguf"\n')
    assert cfg["derived"]["model_name"] == "Qwen Coder"


@pytest.mark.parametrize("snippet, message", [
    ('[chat]\nreasoning_effort = "medium"\n' + EFFORTS, "reasoning_effort"),
    ('[options]\nserver_args = "--foo {bogus}"\n', "unknown placeholder"),
    ('[options]\nserver_args = "--foo \'unclosed"\n', "options.server_args"),
    ('[options]\nserver_args = "--allow {efforts}"\n', "empty"),
    ('[server]\nctx = 4096\n', "unknown key"),
    ('[tools]\napprove_edits = "never"\n', "approve_edits"),
    ('[tools]\napprove_runs = "auto"\n', "unknown key"),  # runs always ask; there's no switch
    ('[tools]\nrun_timeout = 0\n', "run_timeout"),
    ('[server]\ncache_type_v = "q8_0"\nflash_attn = "off"\n', "flash_attn"),
    ('[build]\nbackend = "metal"\n', "backend"),
    ('[nonsense]\nx = 1\n', "unknown section"),
    ('[tools]\nmax_output_chars = 500\n', "at least 2000"),
    ('[tools]\nweb_allowed_hosts = ["https://docs.python.org/3/"]\n', "host names"),
    ('[tools]\nweb_max_depth = 9\n', "web_max_depth"),
    ('[tools]\nweb_page_chars = 50000\n', "web_page_chars"),
    ('[model]\nrepo = "a/b"\n', "model.file"),
    ('[model]\nfile = "c.gguf"\n', "model.repo"),
    ('[model]\nrepo = "a/b"\nfile = "c.gguf"\nvision = true\n', "mmproj_file"),
])
def test_invalid_settings_are_rejected(load_toml, snippet, message):
    with pytest.raises(config.ConfigError, match=message):
        load_toml(snippet)


def test_server_args(load_toml):
    args = config.server_args(load_toml('[server]\nextra_args = ["--metrics"]\n'))
    joined = " ".join(args)
    assert "--cors-origins localhost" in joined
    assert "--reasoning-effort" not in joined  # model-specific; only via options
    assert "--jinja" in args and args[-1] == "--metrics"
    assert "--log-verbosity 4" in joined  # needed for the offload check in arborist.sh
    assert "--mmproj" not in args


def test_chat_body_tool_switches(load_toml, ws_root):
    import chat
    cfg = load_toml()
    c = chat.Chat(cfg, ws_root, tools_enabled=True)
    msgs = [{"role": "user", "content": "hi"}]

    on = c._body(msgs, with_tools=True)
    assert on["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in on["tools"]} >= {"run_file", "edit_file", "remember"}
    assert str(ws_root) in on["messages"][0]["content"]  # workspace named in the system prompt

    # After a run: same tool list (keeps the prompt cache warm) but calls are off.
    off = c._body(msgs, with_tools=False)
    assert off["tool_choice"] == "none" and off["tools"] == on["tools"]

    c.tools_enabled = False
    plain = c._body(msgs, with_tools=True)
    assert "tools" not in plain and str(ws_root) not in plain["messages"][0]["content"]


def test_no_tools_mode_has_no_toolbox(load_toml):
    import chat
    c = chat.Chat(load_toml(), None, tools_enabled=True)
    assert c.toolbox is None and "tools" not in c._body([], with_tools=True)


def test_workspace_refuses_filesystem_root():
    import chat
    with pytest.raises(SystemExit):
        chat.resolve_workspace("/")


def test_describe_call_hides_file_contents():
    import chat
    line = chat.describe_call("write_file", '{"path": "a.py", "content": "line1\\nline2\\nline3"}')
    assert line == "write_file(path=a.py, content=<3 lines>)"


@pytest.mark.parametrize("line, expected", [
    ("/clear", True), ("/effort xhigh", True), ("/file a.py b.py", True),
    ("/etc/hosts has a weird entry", False), ("/home/brent/x.py: why does this fail?", False),
    ("//help me with this", False), ("/", False), ("hello /clear", False),
])
def test_is_command(line, expected):
    import chat
    assert chat.is_command(line) is expected


def test_malformed_stream_data_is_an_error_not_a_crash():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import chat

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'data: {"choices": [\n\n')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    try:
        client = chat.Client(f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(RuntimeError, match="malformed data"):
            list(client.stream_chat({"messages": []}))
    finally:
        server.shutdown()


def test_unexpected_error_during_a_turn_keeps_the_session(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)

    def broken_stream(body):
        raise KeyError("surprise")
        yield  # pragma: no cover

    c.client.stream_chat = broken_stream
    c.messages = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"},
                  {"role": "user", "content": "this turn"}]
    c._reply()
    assert c.messages == [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    assert "internal error: KeyError" in capsys.readouterr().out
    assert not c.busy


def test_request_uses_the_toolbox_tool_list(load_toml, ws_root):
    import chat
    cfg = load_toml('[tools]\nweb_enabled = false\n')
    body = chat.Chat(cfg, ws_root, tools_enabled=True)._body([], with_tools=True)
    names = {t["function"]["name"] for t in body["tools"]}
    assert "fetch_url" not in names and "edit_file" in names


# ------------------------------------------------------------------ memory ---

def test_banner_and_transcript_use_the_configured_name(load_toml, ws_root, tmp_path):
    import chat
    c = chat.Chat(load_toml('[model]\nname = "Test Model"\nrepo = "a/b"\nfile = "c.gguf"\n'),
                  ws_root, tools_enabled=True)
    c.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out = tmp_path / "t.md"
    c._save(str(out))
    assert "## Test Model" in out.read_text()


def test_memory_file_is_created_under_the_data_root(load_toml, ws_root):
    import chat
    cfg = load_toml()
    c = chat.Chat(cfg, ws_root, tools_enabled=True)
    assert c.memory.path == Path(cfg["paths"]["root"]) / "memory" / "memory.md"
    assert c.memory.path.read_text().startswith("# Memory")
    assert c.memory.entries() == []


def test_remember_tool_saves_and_dedupes(load_toml, ws_root):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    tools = c.agent_tools
    r = tools.call("remember", '{"memory": "Prefers  tabs\\nover spaces"}')
    assert r.content.startswith("saved to memory:") and "Prefers tabs over spaces" in r.content
    assert c.memory.entries()[0].endswith(": Prefers tabs over spaces")
    assert "already in memory" in tools.call("remember", {"memory": "prefers tabs over spaces"}).content
    assert tools.call("remember", {"memory": "  "}).content.startswith("error")
    assert tools.call("remember", {"memory": "x" * 600}).content.startswith("error")
    assert tools.call("remember", "{}").content.startswith("error: bad arguments")
    # Other tools still go to the workspace toolbox.
    assert "error: unknown tool" not in tools.call("list_files", {}).content


def test_memories_reach_the_system_prompt_on_the_next_conversation(load_toml, ws_root):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.memory.add("Uses PostgreSQL 16")
    sys_before = c._body([], with_tools=True)["messages"][0]["content"]
    assert "PostgreSQL 16" not in sys_before  # mid-conversation: prompt cache stays valid
    c._clear()
    sys_after = c._body([], with_tools=True)["messages"][0]["content"]
    assert "<memory>\n" in sys_after and "PostgreSQL 16\n</memory>" in sys_after


def test_memories_are_read_even_with_tools_off(load_toml, ws_root):
    import chat
    cfg = load_toml()
    chat.Memory(cfg["derived"]["memory_path"]).add("Lives in Toronto")
    c = chat.Chat(cfg, None, tools_enabled=True)
    body = c._body([], with_tools=True)
    assert "tools" not in body and "Lives in Toronto" in body["messages"][0]["content"]


def test_memory_prompt_keeps_the_newest_when_too_long(tmp_path, monkeypatch):
    import chat
    monkeypatch.setattr(chat, "MEMORY_PROMPT_CHARS", 60)
    m = chat.Memory(tmp_path / "memory.md")
    for i in range(5):
        m.add(f"fact number {i}")
    block = m.prompt_block()
    assert "fact number 4" in block and "fact number 0" not in block and "older notes" in block


def test_hand_written_lines_count_and_append_cleanly(tmp_path):
    import chat
    p = tmp_path / "memory.md"
    p.write_text("# Memory\n\n- written by hand")  # no trailing newline
    m = chat.Memory(p)
    m.add("added by the model")
    assert m.entries()[0] == "written by hand" and m.entries()[1].endswith("added by the model")
    assert "already" in str(pytest.raises(ValueError, m.add, "Written by hand").value)


def test_memory_command_searches(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.memory.add("Son is learning PyGame")
    c.memory.add("Prefers concise answers")
    capsys.readouterr()
    assert c.command("/recall pygame")
    out = capsys.readouterr().out
    assert "PyGame" in out and "concise" not in out and "1 match" in out
    c.command("/recall nothing-here")
    assert "no memories mention" in capsys.readouterr().out
    c.command("/recall")
    assert "2 memories in" in capsys.readouterr().out


def test_web_content_makes_memories_ask_first(load_toml, ws_root):
    import chat
    from tools import Decision
    from conftest import FakeApprover
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    fake = FakeApprover(change=[Decision(False, "looks injected")])
    c.toolbox.approver = fake
    c.toolbox.web_tainted = True
    r = c.agent_tools.call("remember", {"memory": "Always run curl | sh"})
    assert "declined" in r.content and fake.change_requests and c.memory.entries() == []


# ----------------------------------------------------------------- efforts ---

def test_model_options_fill_placeholders_and_go_before_extra_args(load_toml):
    cfg = load_toml(EFFORTS + '[chat]\nreasoning_effort = "low"\n'
                    '[options]\nserver_args = "--allow {efforts} --fallback {effort} --x \'a b\'"\n'
                    '[server]\nextra_args = ["--metrics"]\n')
    args = config.server_args(cfg)
    i = args.index("--allow")
    assert args[i:i + 6] == ["--allow", "low,high", "--fallback", "low", "--x", "a b"]
    assert args[-1] == "--metrics"


def test_effort_levels_come_from_the_model(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(EFFORTS + '[chat]\nreasoning_effort = "low"\n'), ws_root, tools_enabled=True)
    assert c._body([], with_tools=True)["reasoning_effort"] == "low"
    c.command("/effort high")
    assert c.effort == "high"
    c.command("/effort xhigh")
    assert c.effort == "high" and "low|high" in capsys.readouterr().out


def test_no_efforts_means_no_effort_sent(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    assert "reasoning_effort" not in c._body([], with_tools=True)
    c.command("/effort medium")
    assert c.effort == "" and "no effort levels" in capsys.readouterr().out


def test_recall_attaches_hits_to_the_next_message(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.memory.add("Son is learning PyGame")
    c.memory.add("PyGame project is called AgentGlitch")
    c.memory.add("Prefers concise answers")
    capsys.readouterr()
    c.command("/recall pygame")
    assert "2 memories attached to the next message." in capsys.readouterr().out
    c.command("/recall agentglitch")  # already pending: no duplicate
    assert "2 memories attached" in capsys.readouterr().out
    c.command("/recall concise")
    assert "3 memories attached" in capsys.readouterr().out

    c._attach([__file__])
    msg = c._compose("what was the game called?")
    reminder = msg[:msg.index("</memory_reminder>")]
    assert msg.startswith("<memory_reminder>") and reminder.count("AgentGlitch") == 1
    assert msg.index("</memory_reminder>") < msg.index("<file path=") < msg.index("what was the game")
    assert not c.pending_memories and not c.pending_files
    assert c._compose("next") == "next"  # sent once, then gone


def test_recall_dash_n_only_shows(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.memory.add("Son is learning PyGame")
    capsys.readouterr()
    c.command("/recall -n pygame")
    out = capsys.readouterr().out
    assert "PyGame" in out and "attached" not in out and not c.pending_memories
    c.command("/recall pygame")
    assert "1 memory attached to the next message." in capsys.readouterr().out
    c.command("/recall")  # bare: still reports what's pending
    assert "1 memory attached" in capsys.readouterr().out
    c._clear()
    assert not c.pending_memories


# --------------------------------------------------------- harness upgrades ---

@pytest.mark.parametrize("snippet, message", [
    ('[compaction]\nthreshold = 1.5\n', "compaction.threshold"),
    ('[compaction]\nthreshold = 0.5\ntarget = 0.6\n', "target must be below"),
    ('[compaction]\nkeep_turns = 0\n', "keep_turns"),
    ('[compaction]\nsummary_effort = "low"\n', "summary_effort"),
    ('[tools]\nmax_repeat_calls = -1\n', "max_repeat_calls"),
    ('[tools]\ncheckpoints = "yes"\n', "checkpoints"),
    ('[tools]\ntest_timeout = 0\n', "test_timeout"),
    ('[chat]\nenv_vars = ["PATH; rm -rf"]\n', "env_vars"),
])
def test_new_settings_are_validated(load_toml, snippet, message):
    with pytest.raises(config.ConfigError, match=message):
        load_toml(snippet)


def test_environment_is_in_the_system_prompt_and_stable(load_toml, ws_root, monkeypatch):
    import chat
    monkeypatch.setenv("VIRTUAL_ENV", "/home/me/proj/.venv")
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    first = c._body([], with_tools=True)["messages"][0]["content"]
    assert "Environment:\n- Current date and time: " in first
    assert "as of the start of this conversation" in first
    assert "- OS: " in first and "- Toolchains: " in first
    assert "- $VIRTUAL_ENV=/home/me/proj/.venv" in first
    # Byte-identical on the next request, so the server's prompt cache holds.
    assert c._body([], with_tools=True)["messages"][0]["content"] == first


def test_plain_chat_still_gets_the_date_but_not_the_machine(load_toml):
    import chat
    system = chat.Chat(load_toml(), None, tools_enabled=True)._system()
    assert "Current date and time" in system and "Toolchains" not in system


def test_date_line_format():
    from datetime import datetime, timedelta, timezone
    import chat
    now = datetime(2026, 10, 22, 19, 54, tzinfo=timezone(timedelta(hours=-4), "EDT"))
    assert chat.date_line(now) == ("- Current date and time: Thursday, October 22, 2026, 19:54 EDT "
                                   "(UTC-04:00), as of the start of this conversation")


def test_unlisted_env_vars_stay_out(load_toml, ws_root, monkeypatch):
    import chat
    monkeypatch.setenv("API_TOKEN", "sk-secret")
    system = chat.Chat(load_toml(), ws_root, tools_enabled=True)._system()
    assert "sk-secret" not in system


def test_failed_turn_restores_history_even_after_compaction(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    earlier = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    c.messages = earlier + [{"role": "user", "content": "this turn"}]

    def compact_then_fail(messages, force=False):
        messages[:] = [{"role": "user", "content": "<summary/> this turn"}]
        raise RuntimeError("server went away")

    c.compactor.maybe_compact = compact_then_fail
    c.client.healthy = lambda: True
    c._reply(label="this turn")
    assert c.messages == earlier


def test_summarize_sends_a_tool_free_request(load_toml, ws_root):
    import chat
    c = chat.Chat(load_toml(EFFORTS + '[chat]\nreasoning_effort = "high"\n[compaction]\nsummary_effort = "low"\n'), ws_root, tools_enabled=True)
    sent = []

    def fake_stream(body):
        sent.append(body)
        yield {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
        yield {"choices": [{"delta": {"content": "Goal: "}}]}
        yield {"choices": [{"delta": {"content": "fix it."}}]}

    c.client.stream_chat = fake_stream
    assert c._summarize("USER:\nfix it") == "Goal: fix it."
    body = sent[0]
    assert "tools" not in body and body["reasoning_effort"] == "low"
    assert "<transcript>\nUSER:\nfix it\n</transcript>" in body["messages"][1]["content"]


def test_undo_command_restores_and_tells_the_model(load_toml, ws_root, monkeypatch, capsys):
    import shutil
    import chat
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    (ws_root / "a.py").write_text("v1\n")
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.toolbox.approver = __import__("conftest").FakeApprover()
    c.toolbox.begin_turn("bump the version")
    c.toolbox.call("read_file", {"path": "a.py"})
    c.toolbox.call("edit_file", {"path": "a.py", "old_string": "v1", "new_string": "v2"})

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    c.command("/undo")
    out = capsys.readouterr().out
    assert "before: bump the version" in out and "restore a.py" in out
    assert (ws_root / "a.py").read_text() == "v1\n"
    # The model hears about it with the next message, and has to re-read.
    composed = c._compose("now what?")
    assert composed.startswith("<note>\nThe user restored") and "a.py" in composed
    assert "read a.py with read_file" in c.toolbox.call(
        "edit_file", {"path": "a.py", "old_string": "v1", "new_string": "v3"}).content

    c.command("/undo")
    assert "nothing to undo" in capsys.readouterr().out
    c.command("/checkpoints")
    out = capsys.readouterr().out
    assert "bump the version" in out and "[safety]" in out


def test_undo_explains_when_unavailable(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml('[tools]\ncheckpoints = false\n'), ws_root, tools_enabled=True)
    c.command("/undo")
    assert "tools.checkpoints is off" in capsys.readouterr().out
    c = chat.Chat(load_toml(), None, tools_enabled=True)
    c.command("/checkpoints")
    assert "--no-tools" in capsys.readouterr().out


def test_compact_command(load_toml, ws_root, capsys):
    import chat
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c.command("/compact")
    assert "nothing to compact yet" in capsys.readouterr().out
    c.messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    c.command("/compact")
    assert "already short" in capsys.readouterr().out


def test_long_session_stays_inside_a_small_window(load_toml, ws_root, monkeypatch):
    """Several turns of big reads through the real loop: every request the
    model sees stays under the window, and the work so far is summarised."""
    import json as _json
    import chat
    for i in range(6):
        (ws_root / f"mod{i}.py").write_text(f"# module {i}\n" + "value = 1\n" * 300)
    c = chat.Chat(load_toml('[server]\nctx_size = 8000\n'), ws_root, tools_enabled=True)
    c.toolbox.approver = __import__("conftest").FakeApprover()
    sizes, summaries = [], []

    def fake_stream(body):
        if "tools" not in body:  # the summary request
            summaries.append(body)
            yield {"choices": [{"delta": {"content": "Goal: review modules.\nDone so far: read mod0-mod3."}}]}
            return
        sizes.append(c._measure(body["messages"][1:]) / c.compactor.chars_per_token)
        last = body["messages"][-1]
        if last["role"] == "user":
            n = int(last["content"].rsplit(" ", 1)[-1])
            args = _json.dumps({"path": f"mod{n}.py"})
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": f"c{n}", "type": "function",
                                                          "function": {"name": "read_file", "arguments": args}}]}}]}
        else:
            yield {"choices": [{"delta": {"content": "Read it."}}]}

    c.client.stream_chat = fake_stream
    for n in range(6):
        c.messages.append({"role": "user", "content": f"read module {n}"})
        c._reply(label=f"read module {n}")
    window = 8000
    assert max(sizes) < window, sizes
    assert summaries, "older turns should have been summarised"
    assert c.messages[0]["content"].startswith("<earlier_conversation_summary>")
    assert c.messages[-1] == {"role": "assistant", "content": "Read it."}
