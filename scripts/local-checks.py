#!/usr/bin/env python3
"""Checks that need your own machine, which GitHub Actions can't provide.

    python3 scripts/local-checks.py            # everything
    python3 scripts/local-checks.py --no-suite # skip the pytest suite
    python3 scripts/local-checks.py --window   # also open a test window (needs pygame)

Everything runs through arborist's real run path (Toolbox.run_file), in a
temporary workspace, with every prompt answered yes. Nothing is changed
outside that workspace, and nothing is sent anywhere: the network checks
only try to connect and report whether they could.

Results are PASS, FAIL or SKIP (not applicable here). Exit status 1 if
anything failed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import runners  # noqa: E402
import sandbox  # noqa: E402
from tools import Decision, Limits, MemoryChoice, Toolbox, Workspace  # noqa: E402

GREEN, RED, DIM, BOLD, RESET = ("\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m") \
    if sys.stdout.isatty() else ("",) * 5
results: list[tuple[str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    colour = {"PASS": GREEN, "FAIL": RED}.get(status, DIM)
    print(f"  {colour}{status}{RESET}  {name}" + (f"{DIM}  ({detail}){RESET}" if detail else ""))
    results.append((status, name))


def check(name: str, ok: bool, detail: str = "") -> None:
    report("PASS" if ok else "FAIL", name, detail)


class YesToEverything:
    def approve_change(self, *a, **k):
        return Decision(True)

    def show_change(self, *a, **k):
        pass

    def approve_run(self, *a, **k):
        return Decision(True)

    def approve_fetch(self, *a, **k):
        return Decision(False)  # these checks never fetch

    def approve_memory(self, *a, **k):
        return MemoryChoice("skip")


def run_code(code: str, name: str = "check.py", **limits) -> str:
    """Run `code` (Python unless name says otherwise) through arborist's
    sandboxed run path; returns the tool's report, log included."""
    with tempfile.TemporaryDirectory(prefix="arborist-check-") as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        (ws / name).write_text(textwrap.dedent(code))
        tb = Toolbox(Workspace(ws), YesToEverything(), Limits(run_sandbox=True, **limits),
                     log_dir=Path(tmp) / "logs")
        dep = tb.call("check_dependencies", {"path": name}).content
        if "MISSING" in dep or "Can't" in dep:
            return dep
        return tb.call("run_file", {"path": name}).content


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


# ------------------------------------------------------------------ checks ---

def check_platform() -> bool:
    section("Sandbox on this machine")
    exe = sandbox.bwrap_path()
    check("bwrap installed", exe is not None, exe or sandbox.INSTALL_HINT)
    if exe is None:
        return False
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    ok, why = sandbox.status()
    check("bwrap can create a sandbox", ok, version if ok else why)
    check("bwrap blocks nested user namespaces (--disable-userns, 0.8+)",
          sandbox._supports_disable_userns(), version)
    restrict = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    if restrict.exists():
        value = restrict.read_text().strip()
        report("SKIP", "AppArmor user-namespace restriction",
               "off here" if value == "0" else "on, but bwrap is allowed through")
    return ok


def check_secrets() -> None:
    section("Your files and services stay out of reach")
    home = Path.home()
    probes = [p for p in (home / ".ssh", home / ".aws", home / ".gnupg", home / ".config",
                          home / ".local/share/arborist") if p.exists()]
    if not probes:
        report("SKIP", "home folders hidden", "none of ~/.ssh ~/.aws ~/.gnupg ~/.config exist")
    else:
        out = run_code(f"""
            import os
            for p in {[str(p) for p in probes]!r}:
                print('VISIBLE' if os.path.exists(p) else 'hidden', p)
        """)
        check("home folders hidden (" + ", ".join(p.name for p in probes) + ")",
              "VISIBLE" not in out and "hidden" in out)

    uid = os.getuid()
    sockets = {"D-Bus session bus": Path(f"/run/user/{uid}/bus"),
               "Docker": Path("/var/run/docker.sock"),
               "systemd user manager": Path(f"/run/user/{uid}/systemd/private")}
    present = {k: v for k, v in sockets.items() if v.exists()}
    if not present:
        report("SKIP", "local sockets unreachable", "no D-Bus/Docker sockets here")
    for label, path in present.items():
        for display in (False, True):
            if display and not os.environ.get("WAYLAND_DISPLAY"):
                continue
            out = run_code(f"""
                import socket
                s = socket.socket(socket.AF_UNIX)
                try:
                    s.connect({str(path)!r}); print('REACHED')
                except OSError as e:
                    print('blocked', e)
            """, run_display=display)
            check(f"{label} unreachable" + (" (with run_display on)" if display else ""),
                  "blocked" in out and "REACHED" not in out)

    try:
        cfg = config.load()
        host, port = cfg["server"]["host"], cfg["server"]["port"]
    except Exception:
        host, port = "127.0.0.1", 8091
    listening = False
    try:
        socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1", port), timeout=1).close()
        listening = True
    except OSError:
        pass
    if not listening:
        report("SKIP", "llama-server unreachable", f"nothing listening on {port}; start arborist to test")
    else:
        out = run_code(f"""
            import socket
            try:
                socket.create_connection(('127.0.0.1', {port}), timeout=3); print('REACHED')
            except OSError:
                print('blocked')
        """)
        check(f"llama-server (port {port}) unreachable", "blocked" in out)

    out = run_code("""
        import socket
        try:
            socket.create_connection(('1.1.1.1', 443), timeout=3); print('REACHED')
        except OSError:
            print('blocked')
        try:
            socket.getaddrinfo('example.com', 443); print('DNS-REACHED')
        except OSError:
            print('dns blocked')
    """)
    check("internet and DNS unreachable", "REACHED" not in out and "blocked" in out)


def check_toolchains() -> None:
    section("Your toolchains work inside the sandbox")
    py = run_code("import sys; print('py', sys.version.split()[0])")
    check("Python", "py " in py, _first_error(py))
    if os.environ.get("VIRTUAL_ENV") or shutil.which("pyenv") or shutil.which("conda"):
        report("PASS" if "py " in py else "FAIL", "your Python manager's interpreter",
               os.environ.get("VIRTUAL_ENV") or "pyenv/conda")
    cases = [("node", "hi.js", "console.log('node', 6 * 7)", "node 42"),
             ("tsx", "hi.ts", "const n: number = 6 * 7; console.log('ts', n)", "ts 42"),
             ("rustc", "hi.rs", 'fn main() { println!("rust {}", 6 * 7); }', "rust 42"),
             ("dart", "hi.dart", "void main() { print('dart ${6 * 7}'); }", "dart 42")]
    nv = runners._node_version()
    node_strips_types = bool(nv) and nv[1] >= (22, 6)
    for tool, name, code, want in cases:
        if tool == "tsx" and not shutil.which("tsx") and not node_strips_types:
            report("SKIP", "TypeScript", "no tsx, and Node is older than 22.6 "
                   f"({'v%d.%d' % nv[1] if nv else 'no node'}); npm install -g tsx to test it")
            continue
        if tool != "tsx" and not shutil.which(tool):
            report("SKIP", tool, "not installed")
            continue
        out = run_code(code, name=name)
        check(tool, want in out, _first_error(out))
    flutter = shutil.which("flutter")
    if not flutter:
        report("SKIP", "Flutter", "not installed")
    else:
        # Flutter writes to its own cache on every command; arborist binds
        # just that folder writable. This shows whether that's enough.
        out = run_code(f"""
            import subprocess
            r = subprocess.run([{flutter!r}, '--version', '--suppress-analytics'],
                               capture_output=True, text=True)
            print('exit', r.returncode); print(r.stdout[:300]); print(r.stderr[-500:])
        """)
        check("Flutter starts", "exit 0" in out, _first_error(out))


def check_display(window: bool) -> None:
    section("Display (run_display)")
    out = run_code("import os; print('DISPLAY', os.environ.get('DISPLAY'), "
                   "'WAYLAND', os.environ.get('WAYLAND_DISPLAY'))")
    check("no display without run_display", "DISPLAY None WAYLAND None" in out)
    if not os.environ.get("WAYLAND_DISPLAY"):
        report("SKIP", "Wayland window", "not in a Wayland session")
        return
    out = run_code("""
        import os, socket
        path = os.path.join(os.environ['XDG_RUNTIME_DIR'], os.environ['WAYLAND_DISPLAY'])
        s = socket.socket(socket.AF_UNIX)
        try:
            s.connect(path); print('WAYLAND-OK')
        except OSError as e:
            print('wayland failed', e)
        print('X11', os.environ.get('DISPLAY'))
        print('RUNTIME', sorted(os.listdir(os.environ['XDG_RUNTIME_DIR'])))
    """, run_display=True)
    check("Wayland socket reachable with run_display", "WAYLAND-OK" in out, _first_error(out))
    check("X11 not passed", "X11 None" in out)
    check("nothing else from XDG_RUNTIME_DIR",
          f"RUNTIME ['{os.environ['WAYLAND_DISPLAY']}']" in out)
    if window:
        out = run_code("""
            import os
            os.environ.setdefault('SDL_VIDEODRIVER', 'wayland')
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((320, 120))
            pygame.display.set_caption('arborist sandbox check')
            screen.fill((40, 120, 60)); pygame.display.flip()
            pygame.time.wait(2000)
            print('WINDOW-OK', pygame.display.get_driver())
        """, run_display=True)
        check("pygame window opens (you should have seen a green window)", "WINDOW-OK" in out,
              _first_error(out))


def check_gpu() -> None:
    section("GPU (run_gpu)")
    if not Path("/dev/dri").exists():
        report("SKIP", "GPU", "no /dev/dri on this machine")
        return
    out = run_code("import os; print('DRI', os.path.exists('/dev/dri'))")
    check("no GPU without run_gpu", "DRI False" in out)
    out = run_code("""
        import glob, os
        nodes = sorted(glob.glob('/dev/dri/renderD*'))
        ok = []
        for n in nodes:
            try:
                os.close(os.open(n, os.O_RDWR)); ok.append(n)
            except OSError as e:
                print('cannot open', n, e)
        print('RENDER', ok)
    """, run_gpu=True)
    check("render node opens with run_gpu", "RENDER ['/dev/dri/renderD" in out, _first_error(out))
    vk = shutil.which("vulkaninfo")
    if not vk:
        report("SKIP", "Vulkan sees the real GPU", "vulkaninfo not installed (package vulkan-tools)")
        return
    out = run_code(f"""
        import subprocess
        r = subprocess.run([{vk!r}, '--summary'], capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if 'deviceName' in l:
                print('DEV', l.split('=', 1)[1].strip())
    """, run_gpu=True)
    devices = [l[4:] for l in out.splitlines() if l.startswith("DEV ")]
    real = [d for d in devices if "llvmpipe" not in d.lower()]  # llvmpipe = software rendering
    check("Vulkan sees a hardware GPU with run_gpu", bool(real), ", ".join(devices) or "no devices")


def _first_error(out: str) -> str:
    """The first error-looking line of the program's own log (or of the
    dependency check, if it never ran)."""
    if "--- log ---" in out:
        out = out.split("--- log ---", 1)[1].split("--- end of log ---", 1)[0]
    for line in out.splitlines():
        if any(w in line for w in ("Error", "error", "MISSING", "not found", "failed", "bwrap:")):
            return line.strip()[:160]
    return ""


def run_suite() -> None:
    section("Test suite (sandbox required)")
    env = dict(os.environ, ARBORIST_REQUIRE_SANDBOX="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    if "No module named pytest" in r.stderr:
        report("FAIL", "pytest", f"pytest isn't installed for {sys.executable}: "
               "sudo apt install python3-pytest, or pip install -r requirements-dev.txt in a venv")
        return
    tail = ((r.stdout.strip() or r.stderr.strip()).splitlines() or ["(no output)"])[-1]
    check("pytest", r.returncode == 0, tail)
    if r.returncode:
        print(textwrap.indent((r.stdout + r.stderr)[-3000:], "    "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-suite", action="store_true", help="skip the pytest suite")
    ap.add_argument("--window", action="store_true", help="open a pygame test window (Wayland)")
    args = ap.parse_args()
    if sys.platform != "linux":
        print("arborist's sandbox is Linux-only")
        return 1
    if check_platform():
        check_secrets()
        check_toolchains()
        check_display(args.window)
        check_gpu()
    if not args.no_suite:
        run_suite()
    failed = [n for s, n in results if s == "FAIL"]
    print(f"\n{BOLD}{len(results) - len(failed)} ok/skipped, {len(failed)} failed{RESET}")
    for n in failed:
        print(f"  {RED}✗{RESET} {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
