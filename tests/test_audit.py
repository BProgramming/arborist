"""Two audits, as tests.

1. SHARING: nothing the model does reaches the network, another program on
   this machine, or your clipboard, unless you approve it at that moment (a
   web visit) or a config option you set allows it (web_allowed_hosts,
   run_sandbox = false, run_display, run_gpu).

2. EDITS: no file changes without being shown to you. Every edit is shown
   in full (and asked about, unless you chose auto), every run's changes are
   listed afterwards, and nothing on screen can be disguised.

Each test names the channel it covers. CI runs these with
ARBORIST_REQUIRE_SANDBOX=1, so a machine that can't sandbox fails instead of
skipping them. scripts/local-checks.sh covers what CI can't: your own
distro, desktop and GPU.
"""
from __future__ import annotations

import io
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

import changes
import chat
import runners
import sandbox
import web
from conftest import FakeApprover, needs, sandboxed, write
from tools import Decision


def run(tb, rel):
    report = tb.call("check_dependencies", {"path": rel}).content
    assert "MISSING" not in report, report
    return tb.call("run_file", {"path": rel})


@pytest.fixture
def host_service():
    """A TCP server on this machine's loopback that counts connections. It
    stands in for the llama-server, a local database or a dev server."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    srv.settimeout(0.2)
    hits: list[str] = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            hits.append("connection")
            conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield srv.getsockname()[1], hits
    stop.set()
    t.join(2)
    srv.close()


# =====================================================================
# 1. SHARING
# =====================================================================

# --- runs: network ---------------------------------------------------

@sandboxed
def test_run_has_no_internet(ws_root, make_toolbox):
    write(ws_root, "net.py",
          "import socket\n"
          "for what, fn in [('tcp', lambda: socket.create_connection(('1.1.1.1', 443), timeout=3)),\n"
          "                 ('dns', lambda: socket.getaddrinfo('example.com', 443))]:\n"
          "    try:\n        fn(); print(what, 'REACHED')\n"
          "    except OSError:\n        print(what, 'blocked')\n")
    out = run(make_toolbox(run_sandbox=True), "net.py").content
    assert "tcp blocked" in out and "dns blocked" in out and "REACHED" not in out


@sandboxed
def test_run_cannot_reach_services_on_this_machine(ws_root, make_toolbox, host_service):
    """The run's loopback is its own: the llama-server and anything else
    listening on 127.0.0.1 out here is unreachable."""
    port, hits = host_service
    write(ws_root, "local.py",
          "import socket\n"
          f"try:\n    socket.create_connection(('127.0.0.1', {port}), timeout=3); print('REACHED')\n"
          "except OSError:\n    print('blocked')\n")
    out = run(make_toolbox(run_sandbox=True), "local.py").content
    assert "blocked" in out and hits == []


@sandboxed
def test_run_cannot_reach_unix_sockets(ws_root, tmp_path, make_toolbox):
    """Docker, D-Bus and desktop sessions listen on Unix sockets: filesystem
    ones (outside the workspace, so absent) and abstract ones (scoped to the
    network namespace)."""
    path = tmp_path / "outside" / "svc.sock"
    path.parent.mkdir()
    fs = socket.socket(socket.AF_UNIX)
    fs.bind(str(path))
    fs.listen()
    abstract = socket.socket(socket.AF_UNIX)
    abstract.bind("\0arborist-audit-test")
    abstract.listen()
    try:
        write(ws_root, "unix.py",
              "import socket\n"
              f"for name, addr in [('fs', {str(path)!r}), ('abstract', '\\0arborist-audit-test')]:\n"
              "    s = socket.socket(socket.AF_UNIX)\n"
              "    try:\n        s.connect(addr); print(name, 'REACHED')\n"
              "    except OSError:\n        print(name, 'blocked')\n")
        out = run(make_toolbox(run_sandbox=True), "unix.py").content
        assert "fs blocked" in out and "abstract blocked" in out
    finally:
        fs.close()
        abstract.close()


# --- runs: what they can see -------------------------------------------

@sandboxed
def test_run_cannot_read_files_outside_the_workspace(ws_root, tmp_path, make_toolbox, fake_home):
    other = tmp_path / "other-project"
    other.mkdir()
    (other / "secrets.env").write_text("TOKEN=abc\n")
    targets = [fake_home / ".ssh" / "id_ed25519", other / "secrets.env"]
    write(ws_root, "read.py",
          "import sys\n"
          f"for p in {[str(t) for t in targets]!r}:\n"
          "    try:\n        open(p).read(); print('READ', p)\n"
          "    except OSError:\n        print('hidden', p)\n")
    out = run(make_toolbox(run_sandbox=True), "read.py").content
    assert "READ" not in out and out.count("hidden") == 2


def test_run_environment_carries_no_secrets(ws_root, make_toolbox, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("MY_DB_URL", "postgres://kept")
    write(ws_root, "env.py",
          "import os\n"
          "for k in ('OPENAI_API_KEY', 'SSH_AUTH_SOCK', 'DBUS_SESSION_BUS_ADDRESS', 'MY_DB_URL'):\n"
          "    print(k, 'SET' if k in os.environ else 'UNSET')\n")
    out = run(make_toolbox(run_env_keep=["MY_DB_URL"]), "env.py").content
    for k in ("OPENAI_API_KEY", "SSH_AUTH_SOCK", "DBUS_SESSION_BUS_ADDRESS"):
        assert f"{k} UNSET" in out
    assert "MY_DB_URL SET" in out  # only because run_env_keep lists it


@sandboxed
def test_run_cannot_see_other_processes(ws_root, make_toolbox):
    write(ws_root, "procs.py",
          "import os\n"
          f"print('HOST-PID-VISIBLE' if os.path.exists('/proc/{os.getpid()}/cmdline') else 'own pids only')\n")
    out = run(make_toolbox(run_sandbox=True), "procs.py").content
    assert "own pids only" in out


@sandboxed
def test_run_cannot_create_namespaces(ws_root, make_toolbox):
    """No nested user namespaces: the usual first step of a kernel exploit."""
    if not sandbox._supports_disable_userns():
        pytest.skip("this bwrap is too old for --disable-userns (needs 0.8+)")
    write(ws_root, "ns.py",
          "import os\n"
          "try:\n    os.unshare(os.CLONE_NEWUSER); print('CREATED')\n"
          "except (OSError, AttributeError):\n    print('refused')\n")
    assert "refused" in run(make_toolbox(run_sandbox=True), "ns.py").content


def test_sandbox_starts_a_new_terminal_session():
    """--new-session: sandboxed code can't push keystrokes into your terminal."""
    args = sandbox.wrap(["true"], sandbox.Spec(workspace=Path("/tmp"), cwd=Path("/tmp")),
                        {"PATH": "/usr/bin"})
    assert "--new-session" in args and "--unshare-all" in args and "--die-with-parent" in args


# --- dependency checks (before any approval) ---------------------------

@sandboxed
def test_dependency_checks_run_inside_the_sandbox(ws_root, make_toolbox, monkeypatch):
    """check_dependencies starts the project's interpreter before you've
    approved anything, so every probe it makes goes through bwrap."""
    seen: list[list[str]] = []
    real = subprocess.run

    def spy(cmd, *a, **kw):
        seen.append(list(cmd))
        return real(cmd, *a, **kw)
    monkeypatch.setattr(runners.subprocess, "run", spy)
    write(ws_root, "imp.py", "import json, pytest\nprint('x')\n")
    make_toolbox(run_sandbox=True).call("check_dependencies", {"path": "imp.py"})
    assert seen, "no probes ran"
    bwrap = sandbox.bwrap_path()
    assert all(c[0] == bwrap for c in seen), [c[:3] for c in seen if c[0] != bwrap]


def test_nothing_runs_when_the_sandbox_is_unavailable(ws_root, make_toolbox, approver, monkeypatch):
    monkeypatch.setattr(sandbox, "_status", (False, "bubblewrap (bwrap) isn't installed."))
    probes: list = []
    monkeypatch.setattr(runners.subprocess, "run", lambda *a, **k: probes.append(a))
    write(ws_root, "hello.py", "print('hi')\n")
    write(ws_root, "test_hello.py", "def test_x():\n    pass\n")
    tb = make_toolbox(approver, run_sandbox=True)
    check = tb.call("check_dependencies", {"path": "hello.py"}).content
    assert "bwrap" in check and probes == []  # not even a probe
    out = tb.call("run_file", {"path": "hello.py"}).content
    tests = tb.call("run_tests", {"path": "."})
    assert "bwrap" in tests.content and tests.hard_stop
    assert approver.run_requests == [] and probes == []


# --- arborist's own git calls ----------------------------------------------

@needs("git")
def test_project_git_config_cannot_make_arborist_run_commands(ws_root, make_toolbox, tmp_path):
    """A repo's .git/config can name a command (core.fsmonitor) that git
    runs on ordinary reads. arborist's own git calls switch it off."""
    marker = tmp_path / "fsmonitor-ran"
    subprocess.run(["git", "init", "-q"], cwd=ws_root, check=True)
    write(ws_root, "a.py", "x = 1\n")
    subprocess.run(["git", "config", "core.fsmonitor", f"touch {marker}; false"], cwd=ws_root,
                   check=True)
    tb = make_toolbox()
    tb.call("list_files", {})
    tb.call("grep", {"pattern": "x"})
    tb.call("view_tree", {})
    assert not marker.exists()


def test_model_cannot_edit_git_files(ws_root, make_toolbox, approver):
    (ws_root / ".git" / "hooks").mkdir(parents=True)
    write(ws_root, ".git/config", "[core]\n")
    tb = make_toolbox(approver)
    for path in (".git/config", ".git/hooks/pre-commit"):
        out = tb.call("write_file", {"path": path, "content": "x"}).content
        assert out.startswith("error:") and ".git" in out
    assert approver.change_requests == [] and (ws_root / ".git/config").read_text() == "[core]\n"


# --- web ----------------------------------------------------------------

def test_web_visit_asks_before_any_lookup(make_toolbox, monkeypatch):
    """The host name alone can carry data (a DNS lookup sends it), so the
    approval comes before the lookup."""
    lookups: list = []
    monkeypatch.setattr(web.socket, "getaddrinfo", lambda *a, **k: lookups.append(a) or [])
    approver = FakeApprover(fetch=[Decision(False)])
    tb = make_toolbox(approver)
    out = tb.call("fetch_url", {"url": "https://some-data-here.example.com/x"}).content
    assert "declined" in out and lookups == []
    assert approver.fetch_requests == [("some-data-here.example.com",
                                        "https://some-data-here.example.com/x")]
    # Redirects to a new host ask again: test_web.py::test_redirect_to_a_new_host_needs_approval


def test_only_config_skips_the_web_prompt(make_toolbox):
    assert make_toolbox().allowed_hosts == set()
    assert make_toolbox(web_allowed_hosts=["Docs.Python.org"]).allowed_hosts == {"docs.python.org"}


# --- terminal -------------------------------------------------------------

@pytest.mark.parametrize("hostile,visible", [
    ("\x1b]52;c;c2VjcmV0\x07", "␛]52;c;c2VjcmV0^G"),      # OSC 52: copy to clipboard
    ("\x1b]0;title\x07", "␛]0;title^G"),                  # set window title
    ("\x1b]8;;https://x.example\x1b\\link\x1b]8;;\x1b\\", "␛]8;;https://x.example␛\\link␛]8;;␛\\"),
])
def test_output_cannot_drive_the_terminal(hostile, visible):
    assert chat.clean(hostile) == visible


def test_safe_stdout_keeps_only_arborists_own_colours(monkeypatch):
    monkeypatch.setattr(chat, "USE_COLOR", True)
    buf = io.StringIO()
    out = chat.SafeStdout(buf)
    out.write(chat.green("ok") + " model says \x1b[31mred\x1b[0m")
    text = buf.getvalue()
    assert "\x1b[32mok" in text            # arborist's colour survives
    assert "\x1b[31mred" not in text and "␛[31mred" in text
    assert chat.TAG not in text


# --- options that open things up are explicit -------------------------------

def test_openings_are_off_by_default_and_named_on_the_prompt(ws_root, make_toolbox, approver,
                                                             monkeypatch):
    tb = make_toolbox(approver)
    lim = tb.limits
    assert lim.run_sandbox and not lim.run_display and not lim.run_gpu
    assert lim.web_allowed_hosts == [] and lim.run_env_keep == [] and lim.run_writable == []
    write(ws_root, "hello.py", "print('hi')\n")
    tb = make_toolbox(approver, run_sandbox=False)
    tb.call("check_dependencies", {"path": "hello.py"})
    tb.call("run_file", {"path": "hello.py"})
    assert any(n.startswith("NOT sandboxed") for n in approver.run_notes[-1])


# =====================================================================
# 2. EDITS
# =====================================================================

def test_every_edit_is_shown_and_asked(ws_root, make_toolbox, approver):
    write(ws_root, "a.py", "x = 1\n")
    tb = make_toolbox(approver)
    tb.call("read_file", {"path": "a.py"})
    tb.call("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
    tb.call("write_file", {"path": "b.py", "content": "y = 3\n"})
    diffs = [d for _, d, _ in approver.change_requests]
    assert len(diffs) == 2 and "+x = 2" in diffs[0] and "+y = 3" in diffs[1]


def test_auto_approved_edits_are_still_shown_in_full(ws_root, make_toolbox, approver):
    write(ws_root, "a.py", "".join(f"line {i}\n" for i in range(400)))
    tb = make_toolbox(approver, approve_edits="auto")
    tb.call("read_file", {"path": "a.py", "limit": 400})
    tb.call("edit_file", {"path": "a.py", "old_string": "line 399", "new_string": "changed"})
    assert approver.change_requests == []
    (_, diff, _), = approver.shown_changes
    assert "+changed" in diff and "-line 399" in diff


def test_hidden_diff_lines_must_be_viewed_before_yes(monkeypatch, capsys):
    ap = chat.TerminalApprover()
    ap.MAX_DIFF_LINES = 3
    diff = "\n".join(["--- a/x", "+++ b/x", "@@ -1 +1,4 @@"] + [f"+line {i}" for i in range(4)])
    answers = iter(["y", "v", "y"])  # "y" before viewing isn't accepted
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda p="": asked.append(p) or next(answers))
    assert ap.approve_change("x", diff, is_new=False).approved
    out = capsys.readouterr().out
    assert "+line 3" in out                      # every line was printed before approval
    assert "[v]iew" in asked[0] and "[v]iew" in asked[1] and "Apply?" in asked[2]


def test_diffs_cannot_disguise_their_content(monkeypatch):
    """File contents can't erase or redraw lines of the prompt that shows
    them, or reorder how code reads (bidi overrides: "Trojan Source")."""
    buf = io.StringIO()
    monkeypatch.setattr(chat.sys, "stdout", chat.SafeStdout(buf))
    diff = ("--- a/x\n+++ b/x\n@@ -1 +1 @@\n"
            "+harmless()\x1b[1A\x1b[2K\r\n"
            "+access = 'user' # \u202e } \u2066if admin\u2069 \u2066 ⁦\n")
    monkeypatch.setattr("builtins.input", lambda p="": "n")
    chat.TerminalApprover().approve_change("x", diff, is_new=False)
    text = buf.getvalue()
    assert "\x1b[1A" not in text and "␛[1A" in text and "␛[2K" in text
    assert "\u202e" not in text and "<U+202E>" in text and "<U+2066>" in text


@sandboxed
def test_every_file_a_run_touches_is_listed(ws_root, make_toolbox):
    write(ws_root, "keep.txt", "same\n")
    write(ws_root, "change.txt", "old\n")
    write(ws_root, "gone.txt", "bye\n")
    write(ws_root, "sneaky.txt", "abcd\n")
    write(ws_root, "touch.py",
          "import os\n"
          "open('new.txt', 'w').write('hi')\n"
          "open('change.txt', 'w').write('new\\n')\n"
          "os.remove('gone.txt')\n"
          "st = os.stat('sneaky.txt')\n"
          "open('sneaky.txt', 'w').write('wxyz\\n')  # same size...\n"
          "os.utime('sneaky.txt', ns=(st.st_atime_ns, st.st_mtime_ns))  # ...old timestamp\n")
    seen: list = []
    tb = make_toolbox(run_sandbox=True)
    tb.on_changes = seen.append
    out = run(tb, "touch.py").content
    got = {(c.status, c.path) for c in seen[0]}
    assert got == {("A", "new.txt"), ("M", "change.txt"), ("D", "gone.txt"), ("M", "sneaky.txt")}
    assert "M sneaky.txt" in out  # the model is told too


@sandboxed
def test_runs_cannot_write_outside_the_workspace(ws_root, tmp_path, make_toolbox, fake_home):
    other = tmp_path / "other-project"
    other.mkdir()
    (ws_root / ".git" / "hooks").mkdir(parents=True)
    (ws_root / ".venv").mkdir()
    (ws_root / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (ws_root / "node_modules").mkdir()
    targets = [other / "x.py", ws_root / ".git/hooks/pre-commit",
               ws_root / ".venv/evil.pth", ws_root / "node_modules/x.js"]
    write(ws_root, "w.py",
          f"for p in {[str(t) for t in targets]!r}:\n"
          "    try:\n        open(p, 'a').write('x'); print('WROTE', p)\n"
          "    except OSError:\n        print('blocked', p)\n")
    write(ws_root, "rc.py", "import os\nopen(os.path.expanduser('~/.bashrc'), 'a').write('evil')\n")
    tb = make_toolbox(run_sandbox=True)
    out = run(tb, "w.py").content
    assert "WROTE" not in out and out.count("blocked") == len(targets)
    # Home is a throwaway copy: the write "works" and then disappears.
    run(tb, "rc.py")
    assert (fake_home / ".bashrc").read_text() == "# original\n"


@sandboxed
def test_runs_cannot_write_nested_repos_or_venvs(ws_root, make_toolbox):
    """The same protection at any depth: a repo cloned into a subfolder, a
    venv with an unusual name, a submodule's .git pointer file."""
    (ws_root / "vendor/lib/.git/hooks").mkdir(parents=True)
    (ws_root / "tools/env").mkdir(parents=True)
    (ws_root / "tools/env/pyvenv.cfg").write_text("home = /usr/bin\n")
    (ws_root / "sub").mkdir()
    (ws_root / "sub/.git").write_text("gitdir: ../.git/modules/sub\n")
    (ws_root / "web/node_modules").mkdir(parents=True)
    targets = [ws_root / "vendor/lib/.git/hooks/pre-commit", ws_root / "tools/env/evil.pth",
               ws_root / "sub/.git", ws_root / "web/node_modules/x.js"]
    write(ws_root, "w.py",
          f"for p in {[str(t) for t in targets]!r}:\n"
          "    try:\n        open(p, 'a').write('x'); print('WROTE', p)\n"
          "    except OSError:\n        print('blocked', p)\n")
    out = run(make_toolbox(run_sandbox=True), "w.py").content
    assert "WROTE" not in out and out.count("blocked") == len(targets)


@sandboxed
def test_a_repo_the_run_creates_is_flagged(ws_root, make_toolbox):
    """Existing repos are read-only, but nothing stops a run making a new one
    and filling in its hooks. That's called out, by name, to you and the model."""
    write(ws_root, "mk.py",
          "import os\n"
          "os.makedirs('fresh/.git/hooks')\n"
          "open('fresh/.git/hooks/post-checkout', 'w').write('#!/bin/sh\\n')\n"
          "open('fresh/.git/hooks/pre-commit.sample', 'w').write('')\n"
          "open('fresh/.git/config', 'w').write('')\n")
    seen: list = []
    tb = make_toolbox(run_sandbox=True)
    tb.on_changes = seen.append
    out = run(tb, "mk.py").content
    assert changes.RUNS_LATER_WARNING in out
    # git's inert .sample hooks are counted but not named
    assert "  fresh/.git/: 3 files, including config, hooks/post-checkout\n" in out
    assert changes.runs_later(seen[0])


def test_protected_dirs_finds_repos_and_venvs_at_any_depth(ws_root, tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # would be added wherever it is
    (ws_root / ".git").mkdir()
    (ws_root / "a/b/.git").mkdir(parents=True)
    (ws_root / "a/b/.git/nested/.git").mkdir(parents=True)  # inside a protected one: not listed
    (ws_root / "wt").mkdir()
    (ws_root / "wt/.git").write_text("gitdir: /elsewhere\n")
    (ws_root / "py/myenv").mkdir(parents=True)
    (ws_root / "py/myenv/pyvenv.cfg").write_text("")
    (ws_root / "js/node_modules/dep/node_modules").mkdir(parents=True)
    (ws_root / "secret/.git").mkdir(parents=True)
    outside = tmp_path / "outside-repo"
    (outside / ".git").mkdir(parents=True)
    (ws_root / "linked").symlink_to(outside)
    got = sandbox.protected_dirs(ws_root, ws_root / "py", [], skip=[ws_root / "secret"])
    rel = sorted(p.relative_to(ws_root.resolve()).as_posix() for p in got)
    assert rel == [".git", "a/b/.git", "js/node_modules", "py/myenv", "wt/.git"]


def test_runs_later_names_what_executes():
    found = [changes.Change("A", f"r/.git/{n}") for n in
             ("HEAD", "config", "hooks/pre-push", "hooks/update.sample", "objects/ab/cd")]
    found += [changes.Change("A", "env/pyvenv.cfg"),
              changes.Change("M", "env/lib/python3.13/site-packages/x.pth"),
              changes.Change("A", "sub/.git"),
              changes.Change("D", "old/.git/config"),  # a deletion can't run anything
              changes.Change("A", "src/app.py")]
    risky = changes.runs_later(found)
    assert {c.path for c in risky} == {c.path for c in found[:8]}
    assert changes.describe_runs_later(risky) == [
        "  env/: 1 file, including pyvenv.cfg",
        "  env/lib/python3.13/site-packages/: 1 file, including x.pth",
        "  r/.git/: 5 files, including config, hooks/pre-push",
        "  sub/.git (a file pointing git at another folder)",
    ]


def test_dependency_checks_cannot_write_the_workspace(ws_root):
    spec = sandbox.Spec(workspace=ws_root, cwd=ws_root, workspace_writable=False)
    args = sandbox.wrap(["true"], spec, {"PATH": "/usr/bin"})
    i = args.index(str(ws_root.resolve()))
    assert args[i - 1] == "--ro-bind"


@sandboxed
@needs("cargo")
def test_rust_check_does_not_create_a_lockfile(ws_root, make_toolbox):
    write(ws_root, "Cargo.toml", '[package]\nname = "d"\nversion = "0.1.0"\nedition = "2021"\n')
    write(ws_root, "src/main.rs", "fn main() {}\n")
    make_toolbox(run_sandbox=True).call("check_dependencies", {"path": "src/main.rs"})
    assert not (ws_root / "Cargo.lock").exists()


def test_symlinks_cannot_point_edits_outside(ws_root, tmp_path, make_toolbox, approver):
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched\n")
    (ws_root / "link.txt").symlink_to(outside)
    out = make_toolbox(approver).call("write_file", {"path": "link.txt", "content": "x"}).content
    assert out.startswith("error:") and outside.read_text() == "untouched\n"
    assert approver.change_requests == []


def test_saved_memories_are_shown_in_full(load_toml, ws_root):
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    notices: list[str] = []
    c.toolbox.on_notice = notices.append
    fact = "Uses tabs; " + "long detail " * 15
    c.agent_tools.call("remember", {"memory": fact, "scope": "general"})
    assert any(fact.strip() in n for n in notices)


@pytest.fixture
def load_toml(tmp_path, monkeypatch):
    import config

    def load(text: str = ""):
        p = tmp_path / "test.toml"
        p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n'
                     '[model]\nrepo = "a/b"\nfile = "c.gguf"\n' + text)
        monkeypatch.setenv("ARBORIST_CONFIG", str(p))
        return config.load()
    return load


# --- /diff and /config -----------------------------------------------------

@needs("git")
def test_diff_shows_changes_since_the_session_started(load_toml, ws_root, capsys):
    write(ws_root, "a.txt", "one\n")
    write(ws_root, "sub/b.txt", "two\n")
    c = chat.Chat(load_toml(), ws_root, tools_enabled=True)
    c._mark_session_start()
    write(ws_root, "a.txt", "one changed\n")
    write(ws_root, "sub/b.txt", "two changed\n")
    write(ws_root, "c.txt", "new\n")
    capsys.readouterr()
    c.command("/diff")
    out = capsys.readouterr().out
    assert "M a.txt" in out and "M sub/b.txt" in out and "A c.txt" in out
    c.command("/diff sub")
    out = capsys.readouterr().out
    assert "M sub/b.txt" in out and "a.txt" not in out
    c.command("/diff --full a.txt")
    out = capsys.readouterr().out
    assert "+one changed" in out and "-one" in out
    c.command("/diff -x")
    assert "unknown option" in capsys.readouterr().out


def test_config_command_opens_nano_and_applies_tools(load_toml, ws_root, monkeypatch, capsys):
    c = chat.Chat(load_toml("[tools]\nrun_timeout = 120\n"), ws_root, tools_enabled=True)
    path = Path(c.cfg["derived"]["config_path"])
    opened: list = []

    def fake_nano(cmd, *a, **k):
        opened.append(cmd)
        path.write_text(path.read_text().replace("run_timeout = 120", "run_timeout = 30"))
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/nano" if name == "nano" else None)
    monkeypatch.setattr(chat.subprocess, "run", fake_nano)
    c.command("/config")
    assert opened == [["/usr/bin/nano", str(path)]]
    assert c.toolbox.limits.run_timeout == 30 and "applied" in capsys.readouterr().out

    def broken(cmd, *a, **k):
        path.write_text(path.read_text().replace("run_timeout = 30", "run_timeout = -1"))
    monkeypatch.setattr(chat.subprocess, "run", broken)
    c.command("/config")
    assert c.toolbox.limits.run_timeout == 30  # a bad edit changes nothing
    assert "nothing was applied" in capsys.readouterr().out


# --- arborist's own code and data ---------------------------------------------

@pytest.fixture
def ws_with_arborist(ws_root, tmp_path):
    """A workspace that contains an arborist install and its data folder,
    like starting arborist from ~/Documents or ~."""
    install = ws_root / "arborist"
    write(ws_root, "arborist/tools.py", "SAFE = True\n")
    write(ws_root, "arborist/.config/config", "[tools]\nrun_sandbox = true\n")
    data = ws_root / "data"
    write(ws_root, "data/memory/memory.toml", "[general]\nmemories = ['private']\n")
    from tools import Toolbox, Workspace, Limits

    def make(approver=None, **limits):
        limits.setdefault("run_sandbox", sandbox.available())
        return Toolbox(Workspace(ws_root, hidden=[data], readonly=[install]),
                       approver or FakeApprover(), Limits(**limits), log_dir=tmp_path / "logs")
    return make, install, data


def test_model_can_read_but_not_change_arborist(ws_with_arborist, ws_root):
    make, install, _ = ws_with_arborist
    approver = FakeApprover()
    tb = make(approver)
    assert "SAFE = True" in tb.call("read_file", {"path": "arborist/tools.py"}).content
    for args in ({"path": "arborist/tools.py", "content": "SAFE = False\n"},
                 {"path": "arborist/.config/config", "content": "[tools]\nrun_sandbox = false\n"},
                 {"path": "arborist/new.py", "content": "x\n"}):
        out = tb.call("write_file", args).content
        assert out.startswith("error:") and "part of arborist itself" in out
    assert approver.change_requests == []  # refused before you're even asked
    assert (install / "tools.py").read_text() == "SAFE = True\n"


def test_model_cannot_see_arborists_data(ws_with_arborist):
    make, _, data = ws_with_arborist
    tb = make()
    assert "off limits" in tb.call("read_file", {"path": "data/memory/memory.toml"}).content
    assert "memory.toml" not in tb.call("list_files", {}).content
    assert "memory.toml" not in tb.call("grep", {"pattern": "private"}).content


@sandboxed
def test_runs_see_arborist_read_only_and_its_data_not_at_all(ws_with_arborist, ws_root):
    make, install, data = ws_with_arborist
    write(ws_root, "probe.py",
          "import os\n"
          f"try:\n    open({str(install / 'tools.py')!r}, 'w').write('x'); print('WROTE-CODE')\n"
          "except OSError:\n    print('code read-only')\n"
          f"print('DATA', os.listdir({str(data)!r}))\n")
    tb = make(run_sandbox=True)
    tb.call("check_dependencies", {"path": "probe.py"})
    out = tb.call("run_file", {"path": "probe.py"}).content
    assert "code read-only" in out and "DATA []" in out
    assert (install / "tools.py").read_text() == "SAFE = True\n"


@needs("git")
def test_nested_repos_are_listed_with_their_own_gitignore(ws_root, make_toolbox):
    """A workspace inside a git repo, holding another repo: the inner repo's
    files are listed (they used to vanish), minus what its .gitignore drops."""
    subprocess.run(["git", "init", "-q", str(ws_root)], check=True)
    write(ws_root, "notes.txt", "outer\n")
    inner = ws_root / "proj"
    write(ws_root, "proj/main.py", "def needle(): pass\n")
    write(ws_root, "proj/.gitignore", "secret.env\n")
    write(ws_root, "proj/secret.env", "needle=1\n")
    subprocess.run(["git", "init", "-q", str(inner)], check=True)
    tb = make_toolbox()
    listing = tb.call("list_files", {}).content
    assert "proj/main.py" in listing and "notes.txt" in listing and "secret.env" not in listing
    assert "proj/main.py" in tb.call("grep", {"pattern": "needle"}).content
    assert "main.py" in tb.call("view_tree", {"path": "proj"}).content


def test_startup_warns_about_broad_workspaces(tmp_path, monkeypatch):
    home = tmp_path / "home"
    docs = home / "Documents"
    for repo in ("arborist", "seymour", "slimeclimb"):
        (docs / repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(chat, "HERE", docs / "arborist")
    warnings = " | ".join(chat.workspace_warnings(docs.resolve(), {}))
    assert "read everything in it" in warnings
    assert "3 separate projects" in warnings
    assert "arborist's own code" in warnings and "not change it" in warnings
    inside = chat.workspace_warnings((docs / "arborist").resolve(), {})
    assert len(inside) == 1 and "not change it" in inside[0]
    assert chat.workspace_warnings((docs / "seymour").resolve(), {}) == []


def test_chat_protects_arborist_and_hides_its_data(load_toml, tmp_path, monkeypatch):
    ws = tmp_path / "Documents"
    install = ws / "arborist"
    install.mkdir(parents=True)
    monkeypatch.setattr(chat, "HERE", install.resolve())
    c = chat.Chat(load_toml(), ws, tools_enabled=True)
    assert install.resolve() in c.toolbox.ws.readonly
    assert Path(c.cfg["derived"]["config_path"]).resolve() in c.toolbox.ws.readonly
    assert Path(c.cfg["derived"]["root"]).resolve() in c.toolbox.ws.hidden


# --- run prompts tie the run to what you approved -----------------------------

def _run_notes_for(tb, approver, rel):
    tb.call("check_dependencies", {"path": rel})
    tb.call("run_file", {"path": rel})
    return " | ".join(approver.run_notes[-1])


def test_run_prompt_says_where_the_file_came_from(ws_root, make_toolbox, approver):
    write(ws_root, "mine.py", "print('mine')\n")
    tb = make_toolbox(approver)
    assert "wasn't written by the model" in _run_notes_for(tb, approver, "mine.py")
    assert approver.run_views[-1] == "print('mine')\n"  # offered for [v]iew

    tb.call("write_file", {"path": "new.py", "content": "print('new')\n"})
    assert "unchanged since the diff you were shown" in _run_notes_for(tb, approver, "new.py")

    (ws_root / "new.py").write_text("print('swapped')\n")  # changed after the diff you saw
    notes = _run_notes_for(tb, approver, "new.py")
    assert "CHANGED: new.py" in notes and approver.run_views[-1] == "print('swapped')\n"


def test_run_prompt_flags_other_changed_model_files(ws_root, make_toolbox, approver):
    tb = make_toolbox(approver)
    tb.call("write_file", {"path": "helper.py", "content": "X = 1\n"})
    write(ws_root, "main.py", "import helper\nprint(helper.X)\n")
    (ws_root / "helper.py").write_text("X = 2\n")
    notes = _run_notes_for(tb, approver, "main.py")
    assert "CHANGED since you were shown them: helper.py" in notes


def test_run_prompt_warns_when_web_content_is_present(ws_root, make_toolbox, approver):
    write(ws_root, "a.py", "print(1)\n")
    tb = make_toolbox(approver)
    assert "WEB:" not in _run_notes_for(tb, approver, "a.py")
    tb.web_tainted = True
    assert "WEB:" in _run_notes_for(tb, approver, "a.py")


def test_terminal_run_prompt_can_show_the_file(monkeypatch, capsys):
    answers = iter(["v", "y"])
    monkeypatch.setattr("builtins.input", lambda p="": next(answers))
    d = chat.TerminalApprover().approve_run("a.py", [["python3", "a.py"]], "/w", 10,
                                            notes=["CHANGED: a.py was written by the model, then changed"],
                                            view="import os\nos.remove('x')\n")
    out = capsys.readouterr().out
    assert d.approved and "os.remove('x')" in out and "CHANGED" in out
