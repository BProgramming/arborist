"""Runs are cut off from the network and from arborist's environment, and
edit prompts highlight dotted names (file names, hosts) in the diff."""
from __future__ import annotations

import pytest

import chat
import runners
from conftest import write

offline = pytest.mark.skipif(not runners.offline_available(),
                             reason="unprivileged user namespaces unavailable here")


def run(tb, rel):
    tb.call("check_dependencies", {"path": rel})
    return tb.call("run_file", {"path": rel})


@offline
def test_run_cannot_reach_the_network(ws_root, make_toolbox):
    write(ws_root, "phone_home.py",
          "import socket\n"
          "s = socket.socket(); s.settimeout(3)\n"
          "try:\n    s.connect(('1.1.1.1', 443)); print('CONNECTED')\n"
          "except OSError as e:\n    print('BLOCKED', e)\n")
    out = run(make_toolbox(run_offline=True), "phone_home.py").content
    assert "BLOCKED" in out and "CONNECTED" not in out


@offline
def test_loopback_still_works_offline(ws_root, make_toolbox):
    write(ws_root, "loop.py",
          "import socket\n"
          "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen()\n"
          "c = socket.create_connection(srv.getsockname(), timeout=3)\n"
          "a, _ = srv.accept(); c.sendall(b'ping'); print(a.recv(4).decode())\n")
    out = run(make_toolbox(run_offline=True), "loop.py").content
    assert "ping" in out and "exit code 0" in out


@offline
def test_run_tests_are_offline_too(ws_root, make_toolbox):
    write(ws_root, "tests/test_net.py",
          "import socket, pytest\n"
          "def test_no_net():\n"
          "    with pytest.raises(OSError):\n"
          "        socket.create_connection(('1.1.1.1', 443), timeout=3)\n")
    out = make_toolbox(run_offline=True).call("run_tests", {"path": "tests"}).content
    assert "passed" in out


def test_secrets_in_the_environment_dont_reach_runs(ws_root, make_toolbox, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("MY_DB_URL", "postgres://kept")
    write(ws_root, "env.py",
          "import os\n"
          "for k in ('OPENAI_API_KEY', 'SSH_AUTH_SOCK', 'MY_DB_URL', 'PATH'):\n"
          "    print(k, 'SET' if k in os.environ else 'UNSET')\n")
    out = run(make_toolbox(run_env_keep=["MY_DB_URL"]), "env.py").content
    assert "OPENAI_API_KEY UNSET" in out and "SSH_AUTH_SOCK UNSET" in out
    assert "MY_DB_URL SET" in out and "PATH SET" in out


def test_refused_before_asking_when_offline_is_impossible(ws_root, make_toolbox, approver,
                                                          monkeypatch):
    monkeypatch.setattr(runners, "_offline", False)
    write(ws_root, "hello.py", "print('hi')\n")
    write(ws_root, "test_hello.py", "def test_x():\n    pass\n")
    out = run(make_toolbox(approver, run_offline=True), "hello.py")
    assert "was not run" in out.content and "run_offline" in out.content and out.hard_stop
    assert approver.run_requests == []
    tests = make_toolbox(approver, run_offline=True).call("run_tests", {"path": "."})
    assert "not run" in tests.content and approver.run_requests == []


# ------------------------------------------------------------ highlighting ---

def test_dotted_names_include_lines_past_the_cutoff():
    diff = ["+++ b/rule.py", "@@ -1 +1,3 @@", "+import os.path",
            "-old = 'gone.txt'", " ctx", "+# see notes.md",
            "+url = 'http://sanmaohe.oss-cn-beijing.aliyuncs.com/upload'"]
    shown, hidden = chat.dotted_names(diff, shown_count=6)
    assert shown == ["os.path", "notes.md"]
    assert hidden == ["sanmaohe.oss-cn-beijing.aliyuncs.com"]  # only added lines, not '-'


def test_plain_sentences_have_no_dotted_names():
    shown, hidden = chat.dotted_names(["+Done. Next step... ok", "+.hidden file"], 10)
    assert shown == [".hidden"] and hidden == []


def test_mark_dotted_keeps_the_text(monkeypatch):
    monkeypatch.setattr(chat, "USE_COLOR", True)
    line = "+fetch('x.aliyuncs.com') # foo.py"
    out = chat.mark_dotted(line, chat.green)
    assert chat.marked("x.aliyuncs.com") in out and chat.marked("foo.py") in out
    monkeypatch.setattr(chat, "USE_COLOR", False)
    assert chat.mark_dotted(line, chat.green) == line
