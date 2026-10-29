"""Run model-written code inside bubblewrap (bwrap).

Every run_file / run_tests step, and every dependency probe, runs through
wrap(). Inside, the process sees:

  - /usr and /etc, read-only, and nothing else from the system (/run, /var,
    /srv, /mnt... are absent, along with the sockets in them: Docker's, D-Bus,
    the desktop session's)
  - an empty, throwaway home directory: tools can write their caches there,
    but the writes vanish when the run ends
  - the toolchain folders the run needs (RunPlan.reads), read-only, with
    credential files in them hidden
  - the workspace, writable, except .git, virtualenvs and node_modules, which
    are read-only (git hooks and .pth files would otherwise run later, outside
    the sandbox)
  - the build directory, plus tools.run_writable and RunPlan.writable, writable
  - a fresh /tmp, /proc and minimal /dev
  - no network but loopback, no further user namespaces, its own process,
    IPC and hostname namespaces, a new session (no typing into your terminal)
  - optionally your Wayland socket (tools.run_display) and GPU device nodes
    (tools.run_gpu)

Linux only. If bwrap is missing or can't create namespaces, nothing runs.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PROBE_TIMEOUT = 20

INSTALL_HINT = ("Install bubblewrap (Debian/Ubuntu: sudo apt install bubblewrap; Fedora: "
                "sudo dnf install bubblewrap; Arch: sudo pacman -S bubblewrap)")
OFF_HINT = ("or set run_sandbox = false in .config/config to run unsandboxed (with network "
            "access and write access to your home directory)")

# Folders in a project that hold things you'll run yourself, outside arborist.
PROTECTED_NAMES = (".git", ".venv", "venv", "node_modules")
# Credential files that can sit inside toolchain folders. They're replaced
# with an empty file inside the sandbox.
CREDENTIAL_NAMES = ("credentials", "credentials.toml", "credentials.json",
                    "pub-credentials.json", ".npmrc", ".pypirc")

_status: tuple[bool, str] | None = None
_disable_userns: bool | None = None


@dataclass
class Spec:
    workspace: Path
    cwd: Path
    reads: list[Path] = field(default_factory=list)      # read-only binds
    writable: list[Path] = field(default_factory=list)   # writable binds outside the workspace
    protect: list[Path] = field(default_factory=list)    # read-only, inside writable binds
    workspace_writable: bool = True                      # False for probes
    display: bool = False
    gpu: bool = False

    def outside_writes(self) -> list[Path]:
        """Writable folders outside the workspace, for the approval prompt."""
        return [w for w in self.writable if not _within(w, self.workspace)]


def _within(p: Path, d: Path) -> bool:
    return p == d or d in p.parents


def bwrap_path() -> str | None:
    return shutil.which("bwrap")


def status() -> tuple[bool, str]:
    """(usable, explanation). Checked once: bwrap present and able to sandbox."""
    global _status
    if _status is None:
        exe = bwrap_path()
        if exe is None:
            _status = (False, f"bubblewrap (bwrap) isn't installed. {INSTALL_HINT}, {OFF_HINT}.")
        else:
            try:
                r = subprocess.run([exe, *_base_args(), "--", "/usr/bin/true"],
                                   capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                                   stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"})
                ok, err = r.returncode == 0, r.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                ok, err = False, str(e)
            _status = (ok, "" if ok else
                       f"bubblewrap is installed but can't create a sandbox here ({err or 'no reason given'}). "
                       "Unprivileged user namespaces may be disabled (sysctl "
                       f"kernel.unprivileged_userns_clone, or an AppArmor policy); fix that, {OFF_HINT}.")
    return _status


def available() -> bool:
    return status()[0]


def _supports_disable_userns() -> bool:
    global _disable_userns
    if _disable_userns is None:
        try:
            r = subprocess.run([bwrap_path() or "bwrap", "--help"], capture_output=True, text=True,
                               timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
            _disable_userns = "--disable-userns" in r.stdout
        except (OSError, subprocess.TimeoutExpired):
            _disable_userns = False
    return _disable_userns


def _base_args() -> list[str]:
    """The system part of the sandbox: everything but the per-run binds."""
    args = ["--die-with-parent", "--new-session", "--unshare-all", "--unshare-user"]
    if _supports_disable_userns():
        args.append("--disable-userns")  # no nested namespaces (the classic escalation start)
    args += ["--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
    for d in ("bin", "sbin", "lib", "lib32", "lib64", "libx32"):
        top = Path("/") / d
        if top.is_symlink():
            args += ["--symlink", os.readlink(top), str(top)]
        elif top.is_dir():
            args += ["--ro-bind", str(top), str(top)]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/var",
             "--tmpfs", "/var/tmp"]
    return args


def wrap(argv: list[str], spec: Spec, env: dict[str, str]) -> list[str]:
    """`argv` rewritten to run inside the sandbox, with exactly `env`."""
    home = Path(env.get("HOME") or Path.home())
    ws = spec.workspace.resolve()
    args = [bwrap_path() or "bwrap", *_base_args(), "--tmpfs", str(home)]

    # Reads already visible through another bind are skipped: under /usr or
    # /etc, or inside the workspace or a writable folder (where a separate
    # read-only bind would block writes the run is allowed to make, such as a
    # compiled binary's own folder). Project folders that must stay read-only
    # are in spec.protect.
    reads, protect = [], list(spec.protect)
    for r in _dedupe(spec.reads):
        if not any(_within(r, d) for d in (Path("/usr"), Path("/etc"), ws, *spec.writable)):
            reads.append(r)
    for r in reads:
        args += ["--ro-bind", str(r), str(r)]
    args += ["--bind" if spec.workspace_writable else "--ro-bind", str(ws), str(ws)]
    for w in _dedupe(spec.writable):
        args += ["--bind", str(w), str(w)]
    for p in _dedupe(protect):
        args += ["--ro-bind", str(p), str(p)]
    for f in _credential_files(reads + [ws]):
        args += ["--ro-bind", "/dev/null", str(f)]

    env = dict(env)
    env["TMPDIR"] = "/tmp"
    if spec.display:
        args += _display_args(env)
    else:
        for k in ("WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DISPLAY"):
            env.pop(k, None)
    if spec.gpu:
        args += _gpu_args()

    args.append("--clearenv")
    for k, v in sorted(env.items()):
        args += ["--setenv", k, v]
    args += ["--chdir", str(spec.cwd.resolve()), "--", *argv]
    return args


def wrap_system_readonly(argv: list[str]) -> list[str]:
    """For the one probe that has to run before a sandbox can be described
    (asking an interpreter where it loads code from): the whole filesystem
    read-only, no network, no environment beyond PATH. It can read, but not
    write or send anything; its answer is checked before it's used."""
    args = [bwrap_path() or "bwrap", "--die-with-parent", "--new-session", "--unshare-all",
            "--unshare-user"]
    if _supports_disable_userns():
        args.append("--disable-userns")
    return args + ["--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--clearenv", "--setenv", "PATH", os.environ.get("PATH", "/usr/bin:/bin"),
                   "--", *argv]


def _display_args(env: dict[str, str]) -> list[str]:
    """Just the Wayland socket, in an otherwise empty XDG_RUNTIME_DIR.

    X11 isn't offered: any X client can read every keystroke and the whole
    screen. Raises SandboxError if there's no Wayland session to give.
    """
    runtime, name = os.environ.get("XDG_RUNTIME_DIR"), os.environ.get("WAYLAND_DISPLAY")
    if not runtime or not name:
        raise SandboxError("run_display is on, but this isn't a Wayland session "
                           "(WAYLAND_DISPLAY / XDG_RUNTIME_DIR not set). X11 isn't supported.")
    sock = Path(name) if os.path.isabs(name) else Path(runtime) / name
    if not sock.exists():
        raise SandboxError(f"run_display is on, but the Wayland socket {sock} doesn't exist")
    env.pop("DISPLAY", None)
    env.update({"XDG_RUNTIME_DIR": runtime, "WAYLAND_DISPLAY": name,
                "XDG_SESSION_TYPE": "wayland"})
    return ["--dir", runtime, "--perms", "0700", "--ro-bind", str(sock), str(sock)]


def _gpu_args() -> list[str]:
    """GPU device nodes, plus the read-only /sys entries drivers use to find them."""
    args = ["--dev-bind-try", "/dev/dri", "/dev/dri", "--dev-bind-try", "/dev/kfd", "/dev/kfd"]
    for n in sorted(glob.glob("/dev/nvidia*")):
        args += ["--dev-bind", n, n]
    for d in ("/sys/dev", "/sys/devices", "/sys/class", "/sys/bus"):
        args += ["--ro-bind-try", d, d]
    return args


def _credential_files(dirs: list[Path]) -> list[Path]:
    """Known credential files directly in, or one level below, the given folders."""
    found = []
    for d in dirs:
        for name in CREDENTIAL_NAMES:
            for f in (d / name, *d.glob(f"*/{name}")):
                if f.is_file():
                    found.append(f)
    return _dedupe(found)


def _dedupe(paths) -> list[Path]:
    out: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        out.setdefault(str(p), p)
    return list(out.values())


class SandboxError(Exception):
    pass


def protected_dirs(workspace: Path, cwd: Path, steps: list[list[str]]) -> list[Path]:
    """Folders in the workspace that stay read-only: PROTECTED_NAMES at the
    workspace root and the run's folder, plus the virtualenv of any
    interpreter a step uses."""
    found = set()
    for d in {workspace.resolve(), cwd.resolve()}:
        found.update(d / n for n in PROTECTED_NAMES)
    for step in steps:
        exe = Path(step[0])
        if exe.is_absolute() and (exe.parent.parent / "pyvenv.cfg").is_file():
            found.add(exe.parent.parent)
    if os.environ.get("VIRTUAL_ENV"):
        found.add(Path(os.environ["VIRTUAL_ENV"]))
    return sorted(p.resolve() for p in found if p.is_dir())


def tool_root(exe: str | os.PathLike | None) -> list[Path]:
    """The install folder an executable belongs to, for a read-only bind.

    /opt/node22/bin/node -> /opt/node22; ~/.cargo/bin/cargo -> ~/.cargo. When
    the parent of bin/ would be too broad (/, home, ~/.local), just the
    executable's own folder. Symlink chains (rustup, pipx and npm shims) are
    followed hop by hop, keeping every folder on the way. Anything inside a
    node_modules folder brings that whole node_modules (the package's
    dependencies sit next to it).
    """
    if not exe:
        return []
    home = Path.home().resolve()
    broad = {Path("/"), home, home / ".local"}
    hops, p = [], Path(os.path.abspath(exe))
    for _ in range(40):
        hops.append(p)
        if not p.is_symlink():
            break
        p = Path(os.path.normpath(p.parent / os.readlink(p)))
    hops.append(Path(os.path.realpath(exe)))
    out = []
    for h in hops:
        if "node_modules" in h.parts[:-1]:
            out.append(Path(*h.parts[:h.parts.index("node_modules") + 1]))
            continue
        d = h.parent
        out.append(d.parent if d.name == "bin" and d.parent not in broad else d)
    return _dedupe(out)


def home_dir(env_var: str, default: str) -> list[Path]:
    """A toolchain's home (CARGO_HOME, RUSTUP_HOME, PUB_CACHE) if it exists."""
    p = Path(os.environ.get(env_var) or Path.home() / default)
    return [p.resolve()] if p.is_dir() else []
