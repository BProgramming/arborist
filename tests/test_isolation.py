"""Runs are cut off from the network and from arborist's environment, and
edit prompts highlight dotted names (file names, hosts) in the diff."""
from __future__ import annotations

import pytest

import chat
import runners
from conftest import write

sandboxed = pytest.mark.skipif(not runners.sandbox_available(),
                             reason="unprivileged user namespaces unavailable here")


def run(tb, rel):
    tb.call("check_dependencies", {"path": rel})
    return tb.call("run_file", {"path": rel})


@sandboxed
def test_run_cannot_reach_the_network(ws_root, make_toolbox):
    write(ws_root, "phone_home.py",
          "import socket\n"
          "s = socket.socket(); s.settimeout(3)\n"
          "try:\n    s.connect(('1.1.1.1', 443)); print('CONNECTED')\n"
          "except OSError as e:\n    print('BLOCKED', e)\n")
    out = run(make_toolbox(run_sandbox=True), "phone_home.py").content
    assert "BLOCKED" in out and "CONNECTED" not in out


@sandboxed
def test_loopback_still_works_offline(ws_root, make_toolbox):
    write(ws_root, "loop.py",
          "import socket\n"
          "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen()\n"
          "c = socket.create_connection(srv.getsockname(), timeout=3)\n"
          "a, _ = srv.accept(); c.sendall(b'ping'); print(a.recv(4).decode())\n")
    out = run(make_toolbox(run_sandbox=True), "loop.py").content
    assert "ping" in out and "exit code 0" in out


@sandboxed
def test_run_tests_are_offline_too(ws_root, make_toolbox):
    write(ws_root, "tests/test_net.py",
          "import socket, pytest\n"
          "def test_no_net():\n"
          "    with pytest.raises(OSError):\n"
          "        socket.create_connection(('1.1.1.1', 443), timeout=3)\n")
    out = make_toolbox(run_sandbox=True).call("run_tests", {"path": "tests"}).content
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
    monkeypatch.setattr(runners, "_sandbox", False)
    write(ws_root, "hello.py", "print('hi')\n")
    write(ws_root, "test_hello.py", "def test_x():\n    pass\n")
    out = run(make_toolbox(approver, run_sandbox=True), "hello.py")
    assert "was not run" in out.content and "run_sandbox" in out.content and out.hard_stop
    assert approver.run_requests == []
    tests = make_toolbox(approver, run_sandbox=True).call("run_tests", {"path": "."})
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


# --------------------------------------------------------- persistence ---

@pytest.fixture
def home_around_workspace(tmp_path, monkeypatch):
    """Make tmp_path the home directory, so the workspace and logs are inside it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


WRITE_EACH = ("import sys\n"
              "for p in sys.argv[1:] or open('targets.txt').read().split():\n"
              "    try:\n        open(p, 'w').write('x'); print('WROTE', p)\n"
              "    except OSError as e:\n        print('BLOCKED', p, e.strerror)\n")


@sandboxed
def test_run_cannot_write_outside_the_workspace(ws_root, make_toolbox, home_around_workspace):
    home = home_around_workspace
    (ws_root / ".git" / "hooks").mkdir(parents=True)
    (ws_root / ".venv").mkdir()
    (ws_root / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (home / "cache").mkdir()
    targets = {
        "bashrc": home / ".bashrc",
        "hook": ws_root / ".git" / "hooks" / "pre-commit",
        "pth": ws_root / ".venv" / "evil.pth",
        "extra": home / "cache" / "ok",
        "workspace": ws_root / "out.txt",
    }
    write(ws_root, "targets.txt", "\n".join(str(p) for p in targets.values()))
    write(ws_root, "persist.py", WRITE_EACH)
    out = run(make_toolbox(run_sandbox=True, run_writable=[str(home / "cache")]), "persist.py").content
    for key in ("bashrc", "hook", "pth"):
        assert f"BLOCKED {targets[key]}" in out and not targets[key].exists(), key
    for key in ("extra", "workspace"):
        assert f"WROTE {targets[key]}" in out and targets[key].exists(), key


def test_spec_protects_the_projects_own_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    ws = tmp_path / "proj"
    venv = ws / "env311"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("")
    (ws / ".git").mkdir()
    spec = runners.sandbox_spec(ws, ws, [[str(venv / "bin" / "python"), "x.py"]], tmp_path / "build")
    assert spec["ro"] == [str(tmp_path.resolve())]
    assert str(ws.resolve()) in spec["rw"] and str((tmp_path / "build").resolve()) in spec["rw"]
    assert sorted(spec["inner_ro"]) == sorted([str(ws / ".git"), str(venv)])


def test_workspace_at_home_leaves_home_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert runners.sandbox_spec(tmp_path, tmp_path, [], tmp_path / "b")["ro"] == []


def test_display_is_opt_in(ws_root, make_toolbox, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    write(ws_root, "gui.py", "import os\nprint('DISPLAY', os.environ.get('DISPLAY'), "
                             "'RUNTIME', os.environ.get('XDG_RUNTIME_DIR'))\n")
    assert "DISPLAY None RUNTIME None" in run(make_toolbox(), "gui.py").content
    assert "DISPLAY :0 RUNTIME /run/user/1000" in run(make_toolbox(run_display=True), "gui.py").content
