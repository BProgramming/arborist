"""Config validation, server arguments, and the request body chat.py sends."""
from __future__ import annotations

from pathlib import Path

import pytest

import config


@pytest.fixture
def load_toml(tmp_path, monkeypatch):
    def load(text: str = ""):
        p = tmp_path / "test.toml"
        p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + text)
        monkeypatch.setenv("BONSAI_CONFIG", str(p))
        return config.load()
    return load


def test_repo_config_is_valid(monkeypatch):
    monkeypatch.delenv("BONSAI_CONFIG", raising=False)
    cfg = config.load(Path(config.__file__).with_name("bonsai.toml"))
    assert cfg["tools"]["approve_edits"] == "ask"
    assert cfg["chat"]["reasoning_effort"] in config.EFFORTS


def test_defaults_fill_missing_sections(load_toml):
    cfg = load_toml()
    assert cfg["tools"]["run_timeout"] == 120
    assert cfg["derived"]["model_file"] == "Ternary-Bonsai-2-27B-PQ2_0.gguf"


@pytest.mark.parametrize("snippet, message", [
    ('[chat]\nreasoning_effort = "high"\n', "reasoning_effort"),
    ('[server]\nctx = 4096\n', "unknown key"),
    ('[tools]\napprove_edits = "never"\n', "approve_edits"),
    ('[tools]\napprove_runs = "auto"\n', "unknown key"),  # runs always ask; there's no switch
    ('[tools]\nrun_timeout = 0\n', "run_timeout"),
    ('[server]\ncache_type_v = "q8_0"\nflash_attn = "off"\n', "flash_attn"),
    ('[build]\nbackend = "metal"\n', "backend"),
    ('[nonsense]\nx = 1\n', "unknown section"),
])
def test_invalid_settings_are_rejected(load_toml, snippet, message):
    with pytest.raises(config.ConfigError, match=message):
        load_toml(snippet)


def test_server_args(load_toml):
    args = config.server_args(load_toml('[server]\nextra_args = ["--metrics"]\n'))
    joined = " ".join(args)
    assert "--cors-origins localhost" in joined
    assert "--reasoning-effort-allow none,medium,xhigh" in joined
    assert "--jinja" in args and args[-1] == "--metrics"
    assert "--mmproj" not in args


def test_chat_body_tool_switches(load_toml, ws_root):
    import chat
    cfg = load_toml()
    c = chat.Chat(cfg, ws_root, tools_enabled=True)
    msgs = [{"role": "user", "content": "hi"}]

    on = c._body(msgs, with_tools=True)
    assert on["tool_choice"] == "auto" and {t["function"]["name"] for t in on["tools"]} >= {"run_file", "edit_file"}
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
