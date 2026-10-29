"""The bwrap sandbox: what runs can and can't do, and that toolchains still
work inside it. The security guarantees themselves are in test_audit.py."""
from __future__ import annotations

from pathlib import Path

import pytest

import chat
import runners
import sandbox
from conftest import needs, sandboxed, write


def run(tb, rel):
    report = tb.call("check_dependencies", {"path": rel}).content
    assert "MISSING" not in report and "Can't" not in report, report
    return tb.call("run_file", {"path": rel})


@sandboxed
def test_loopback_works_inside(ws_root, make_toolbox):
    write(ws_root, "loop.py",
          "import socket\n"
          "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen()\n"
          "c = socket.create_connection(srv.getsockname(), timeout=3)\n"
          "a, _ = srv.accept(); c.sendall(b'ping'); print(a.recv(4).decode())\n")
    out = run(make_toolbox(run_sandbox=True), "loop.py").content
    assert "ping" in out and "exit code 0" in out


@sandboxed
def test_home_is_empty_and_throwaway(ws_root, make_toolbox, fake_home):
    write(ws_root, "home.py",
          "import os\n"
          "home = os.path.expanduser('~')\n"
          "print('LISTING', sorted(os.listdir(home)))\n"
          "open(os.path.join(home, '.cache-file'), 'w').write('x')\n"
          "print('WROTE-HOME')\n")
    out = run(make_toolbox(run_sandbox=True), "home.py").content
    assert "WROTE-HOME" in out  # tools may write caches...
    assert not (fake_home / ".cache-file").exists()  # ...which vanish
    assert ".ssh" not in out and ".bashrc" not in out


@sandboxed
def test_run_writable_and_build_dir_stay_writable(ws_root, make_toolbox, fake_home, approver):
    extra = fake_home / "tool-cache"
    extra.mkdir()
    write(ws_root, "w.py", f"open({str(extra / 'ok')!r}, 'w').write('x'); print('OK')\n")
    out = run(make_toolbox(approver, run_sandbox=True, run_writable=[str(extra)]), "w.py").content
    assert "OK" in out and (extra / "ok").exists()
    notes = "\n".join(approver.run_notes[-1])
    assert str(extra) in notes  # every writable folder outside the workspace is on the prompt


@sandboxed
def test_run_tests_run_sandboxed(ws_root, make_toolbox):
    write(ws_root, "tests/test_net.py",
          "import socket, pytest\n"
          "def test_no_net():\n"
          "    with pytest.raises(OSError):\n"
          "        socket.create_connection(('1.1.1.1', 443), timeout=3)\n")
    out = make_toolbox(run_sandbox=True).call("run_tests", {"path": "tests"}).content
    assert "passed" in out.splitlines()[0]


@sandboxed
@needs("rustc")
def test_rust_compiles_into_the_build_dir(ws_root, make_toolbox):
    write(ws_root, "hi.rs", 'fn main() { println!("rust {}", 6 * 7); }\n')
    assert "rust 42" in run(make_toolbox(run_sandbox=True), "hi.rs").content


@sandboxed
@needs("node")
def test_node_runs(ws_root, make_toolbox):
    write(ws_root, "hi.js", "console.log('node', 6 * 7)\n")
    assert "node 42" in run(make_toolbox(run_sandbox=True), "hi.js").content


def test_tool_root_follows_symlink_chains(tmp_path):
    (tmp_path / "pkgs/node_modules/tsx/dist").mkdir(parents=True)
    real = tmp_path / "pkgs/node_modules/tsx/dist/cli.mjs"
    real.write_text("")
    (tmp_path / "pkgs/node_modules/.bin").mkdir()
    (tmp_path / "pkgs/node_modules/.bin/tsx").symlink_to("../tsx/dist/cli.mjs")
    (tmp_path / "global/bin").mkdir(parents=True)
    (tmp_path / "global/bin/tsx").symlink_to(tmp_path / "pkgs/node_modules/.bin/tsx")
    roots = sandbox.tool_root(tmp_path / "global/bin/tsx")
    assert tmp_path / "global" in roots
    assert tmp_path / "pkgs/node_modules" in roots  # the package and its siblings


def test_doctored_interpreter_cant_name_secret_folders(tmp_path, monkeypatch, fake_home):
    """An interpreter's answer about its own paths is filtered before use."""
    assert not runners._python_dir(fake_home / ".ssh")
    assert not runners._python_dir(fake_home)
    assert not runners._python_dir(Path("/"))
    site = tmp_path / "py/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    assert runners._python_dir(site) and runners._python_dir(tmp_path / "py")


def test_display_needs_wayland(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    with pytest.raises(sandbox.SandboxError, match="Wayland"):
        sandbox._display_args({})


def test_display_passes_only_the_wayland_socket(tmp_path, monkeypatch):
    runtime = tmp_path / "run"
    runtime.mkdir()
    (runtime / "wayland-1").write_text("")  # stands in for the socket
    (runtime / "bus").write_text("")        # D-Bus lives here too
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("DISPLAY", ":0")
    env = {"DISPLAY": ":0"}
    args = sandbox._display_args(env)
    assert str(runtime / "wayland-1") in args and str(runtime / "bus") not in args
    assert "DISPLAY" not in env and env["WAYLAND_DISPLAY"] == "wayland-1"


def test_gpu_binds_only_when_asked(ws_root):
    spec = sandbox.Spec(workspace=ws_root, cwd=ws_root)
    assert "/dev/dri" not in sandbox.wrap(["true"], spec, {"PATH": "/usr/bin"})
    spec.gpu = True
    args = sandbox.wrap(["true"], spec, {"PATH": "/usr/bin"})
    assert "/dev/dri" in args and "/sys/devices" in args


# ------------------------------------------------------------ highlighting ---

def test_dotted_names_include_lines_past_the_cutoff():
    diff = ["+++ b/rule.py", "@@ -1 +1,3 @@", "+import os.path",
            "-old = 'gone.txt'", " ctx", "+# see notes.md",
            "+url = 'http://sanmaohe.oss-cn-beijing.aliyuncs.com/upload'"]
    shown, hidden = chat.dotted_names(diff, shown_count=6)
    assert shown == ["os.path", "notes.md"]
    assert hidden == ["sanmaohe.oss-cn-beijing.aliyuncs.com"]


def test_mark_dotted_keeps_the_text(monkeypatch):
    monkeypatch.setattr(chat, "USE_COLOR", True)
    line = "+fetch('x.aliyuncs.com') # foo.py"
    out = chat.mark_dotted(line, chat.green)
    assert chat.marked("x.aliyuncs.com") in out and chat.marked("foo.py") in out
    monkeypatch.setattr(chat, "USE_COLOR", False)
    assert chat.mark_dotted(line, chat.green) == line
