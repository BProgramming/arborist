"""Check and run a single source file: .py, .rs, .dart, .js, .ts.

plan_for() works out how a file would be run and checks that the toolchain
and dependencies are present. It never executes the file or anything in the
project; it only runs `--version`-style probes. execute() runs a plan with a
timeout, writing combined stdout/stderr to a log file.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import sandbox

RUNNABLE = {".py", ".rs", ".dart", ".js", ".ts"}

# Python decides whether a cached .pyc is stale from the source's size and
# mtime (whole seconds). An edit that keeps the size, made within a second of
# a run (exactly what an edit-and-rerun loop does), would run the old code.
# So arborist's own Python runs don't write bytecode caches at all.
PYTHON_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}
PROBE_TIMEOUT = 20  # seconds, for version checks and dependency probes

# What a run inherits from arborist's environment. Code the model wrote runs
# with only these (plus the plan's own env and the user's tools.run_env_keep),
# so API keys, tokens, SSH_AUTH_SOCK and the D-Bus address don't reach it.
# LC_* variables are kept too.
SAFE_ENV = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ", "TMPDIR",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "CARGO_HOME", "RUSTUP_HOME", "PUB_CACHE", "FLUTTER_ROOT", "NODE_PATH",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
})
# With tools.run_display, also these, so GUI programs (pygame and the like)
# can open a window. That also hands them XDG_RUNTIME_DIR, where the D-Bus
# socket lives, so it's off by default.
DISPLAY_ENV = frozenset({"DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR",
                         "XDG_SESSION_TYPE"})


def run_env(extra: dict[str, str] | None = None, keep=(), display: bool = False) -> dict[str, str]:
    """The whole environment a run gets: SAFE_ENV, LC_*, `keep`, then `extra`."""
    env = {k: v for k, v in os.environ.items()
           if k in SAFE_ENV or k.startswith("LC_") or k in keep
           or (display and k in DISPLAY_ENV)}
    env.update(extra or {})
    return env


# Dependency probes run in the sandbox too (they execute the project's own
# interpreter, before you've approved anything). The Toolbox installs the
# wrapper with probe_sandbox(); None means probes run directly.
_probe_wrap: Callable[[list[str], Path | None], list[str]] | None = None


class probe_sandbox:
    """`with probe_sandbox(fn):` makes _probe run fn(cmd, cwd) instead of cmd."""
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        global _probe_wrap
        self.prev, _probe_wrap = _probe_wrap, self.fn

    def __exit__(self, *exc):
        global _probe_wrap
        _probe_wrap = self.prev


@dataclass
class RunPlan:
    steps: list[list[str]] = field(default_factory=list)  # run in order; stop at first failure
    cwd: Path = Path(".")
    checked: list[str] = field(default_factory=list)      # what passed, for the report
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)     # non-empty means "don't run"
    env: dict[str, str] = field(default_factory=dict)     # added to the environment for the run
    reads: list[Path] = field(default_factory=list)       # toolchain folders the sandbox shows, read-only
    writable: list[Path] = field(default_factory=list)    # folders outside the workspace it must write

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.steps)


@dataclass
class RunOutcome:
    exit_code: int | None
    timed_out: bool
    duration: float
    log_path: Path


# --------------------------------------------------------------- helpers ---

def _probe(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess | None:
    run_cwd = cwd
    if _probe_wrap is not None:
        cmd, run_cwd = _probe_wrap(cmd, cwd), None  # the sandbox sets the directory
    try:
        return subprocess.run(cmd, cwd=run_cwd, capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL, env=run_env())
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version(cmd: list[str]) -> str | None:
    r = _probe(cmd)
    if r is None or r.returncode != 0:
        return None
    out = (r.stdout or r.stderr).strip()
    return out.splitlines()[0] if out else "unknown version"


def _find_up(start: Path, root: Path, name: str) -> Path | None:
    """Nearest `name` in start or a parent, without leaving the workspace."""
    d = start
    while True:
        candidate = d / name
        if candidate.exists():
            return candidate
        if d == root or d.parent == d:
            return None
        d = d.parent


def plan_for(path: Path, root: Path, build_dir: Path) -> RunPlan:
    planner = {".py": _plan_python, ".rs": _plan_rust, ".dart": _plan_dart,
               ".js": _plan_js, ".ts": _plan_ts}.get(path.suffix)
    if planner is None:
        return RunPlan(problems=[f"{path.suffix or 'this file type'} can't be run; "
                                 f"only {', '.join(sorted(RUNNABLE))} are supported"])
    return _with_toolchain(planner(path, root, build_dir))


# ------------------------------------------------------------- toolchain ---

# Tools a step may start on its own (npm runs node, cargo runs rustc, tsx runs
# node...). Their install folders are shown to every run, read-only.
COMPANIONS = ("node", "npm", "npx", "tsx", "cargo", "rustc", "rustup", "dart", "flutter")

_PY_PATHS = r"""
import json, site, sys
paths = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, *sys.path]
paths += site.getsitepackages() + [site.getusersitepackages()]
print(json.dumps(paths))
"""
_py_paths_cache: dict[tuple[str, int], list[Path]] = {}


def _python_paths(interp: str) -> list[Path]:
    """Where an interpreter loads code from. Asked with -I -S, so no .pth
    files or site customisation run: this one probe can't be sandboxed (its
    answer is what the sandbox needs to show)."""
    real = os.path.realpath(interp)
    try:
        key = (real, os.stat(real).st_mtime_ns)
    except OSError:
        return []
    if key not in _py_paths_cache:
        cmd = [interp, "-I", "-S", "-c", _PY_PATHS]
        if _probe_wrap is not None:  # sandboxing is on
            cmd = sandbox.wrap_system_readonly(cmd)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                               stdin=subprocess.DEVNULL,
                               env={"PATH": os.environ.get("PATH", "/usr/bin")})
            found = json.loads(r.stdout) if r.returncode == 0 else []
            if not isinstance(found, list):
                found = []
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            found = []
        cfg = Path(interp).parent.parent / "pyvenv.cfg"  # a venv's base interpreter
        if cfg.is_file():
            m = re.search(r"^home\s*=\s*(.+)$", cfg.read_text(errors="replace"), re.MULTILINE)
            if m:
                found.append(m[1].strip())
        _py_paths_cache[key] = [p for p in (Path(f).resolve() for f in found
                                            if isinstance(f, str) and f) if _python_dir(p)]
    return _py_paths_cache[key]


_PYTHONISH = re.compile(r"python|site-packages|dist-packages|venv|pyenv|conda|mamba|lib-dynload",
                        re.IGNORECASE)


def _python_dir(p: Path) -> bool:
    """Whether an interpreter's answer looks like a place Python loads code
    from. A doctored interpreter could name ~/.ssh to have it shown to later
    runs; anything that doesn't look like Python's is dropped, and so is
    home itself or anything above it."""
    if not p.is_dir():
        return p.suffix == ".zip" and p.is_file()  # zipped stdlib on sys.path
    home = Path.home().resolve()
    if p == home or p in home.parents or p == Path("/"):
        return False
    return bool(_PYTHONISH.search(str(p))) or any((p / "lib").glob("python3*"))


def toolchain_folders(steps: list[list[str]]) -> tuple[list[Path], list[Path]]:
    """(read-only folders, writable folders) the sandbox must show for `steps`."""
    reads: list[Path] = []
    writable: list[Path] = []
    exes = [s[0] for s in steps if s] + [shutil.which(c) for c in COMPANIONS]
    for exe in filter(None, exes):
        if not os.path.isabs(exe):
            exe = shutil.which(exe) or exe
        reads += sandbox.tool_root(exe)
        name = Path(exe).name
        if name.startswith("python"):
            reads += _python_paths(exe)
        if name == "flutter":
            # Flutter takes a lock in, and updates, its own cache on every
            # command; it won't start without write access to it.
            for root in sandbox.tool_root(exe):
                if (root / "bin" / "cache").is_dir():
                    writable.append((root / "bin" / "cache").resolve())
    reads += sandbox.home_dir("CARGO_HOME", ".cargo") + sandbox.home_dir("RUSTUP_HOME", ".rustup")
    reads += sandbox.home_dir("PUB_CACHE", ".pub-cache")
    return sandbox._dedupe(reads), sandbox._dedupe(writable)


def _with_toolchain(plan: RunPlan) -> RunPlan:
    if plan.ok:
        reads, writable = toolchain_folders(plan.steps)
        plan.reads = sandbox._dedupe(plan.reads + reads)
        plan.writable = sandbox._dedupe(plan.writable + writable)
    return plan


# ---------------------------------------------------------------- python ---

_IMPORT_PROBE = r"""
import importlib.util, json, sys
names = json.loads(sys.argv[1])
std = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)
missing = []
for n in names:
    if n in std:
        continue
    try:
        if importlib.util.find_spec(n) is None:
            missing.append(n)
    except Exception:
        missing.append(n)
print(json.dumps(missing))
"""


def _python_interpreter(path: Path, root: Path) -> str | None:
    return python_for_dir(path.parent, root)


def python_for_dir(d: Path, root: Path) -> str | None:
    """The interpreter for code in d: the nearest .venv/venv, then $VIRTUAL_ENV,
    then python3 on PATH."""
    while True:
        for venv in (".venv", "venv"):
            exe = d / venv / "bin" / "python"
            if exe.exists():
                return str(exe)
        if d == root or d.parent == d:
            break
        d = d.parent
    if os.environ.get("VIRTUAL_ENV"):
        exe = Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"
        if exe.exists():
            return str(exe)
    return shutil.which("python3") or shutil.which("python")


def python_imports(source: str) -> tuple[set[str], set[str]]:
    """Top-level module names imported by `source`: (required, optional).

    Imports inside a try block count as optional, since code usually guards
    optional dependencies that way. Relative imports are skipped.
    """
    tree = ast.parse(source)
    guarded: set[int] = set()
    try_types = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t)
    for node in ast.walk(tree):
        if isinstance(node, try_types):
            for stmt in node.body:
                guarded.update(id(n) for n in ast.walk(stmt))
    required, optional = set(), set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        (optional if id(node) in guarded else required).update(names)
    return required - optional, optional


def _is_local_module(name: str, path: Path, root: Path) -> bool:
    for d in (path.parent, root, root / "src"):
        if (d / f"{name}.py").exists() or (d / name).is_dir():
            return True
    return False


def _plan_python(path: Path, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=path.parent)
    interp = _python_interpreter(path, root)
    version = _version([interp, "--version"]) if interp else None
    if not version:
        plan.problems.append("Python interpreter (python3) not found")
        return plan
    where = "project venv" if "/venv/" in interp or "/.venv/" in interp else "system"
    plan.checked.append(f"{version} ({where}: {interp})")

    try:
        required, optional = python_imports(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        plan.problems.append(f"syntax error at line {e.lineno}: {e.msg}")
        return plan
    required = {n for n in required if not _is_local_module(n, path, root)}
    optional = {n for n in optional if not _is_local_module(n, path, root)}

    names = sorted(required | optional)
    if names:
        # Probe from a neutral directory so a project file named like a stdlib
        # module (json.py, …) can't be imported and executed by the check.
        build_dir.mkdir(parents=True, exist_ok=True)
        r = _probe([interp, "-c", _IMPORT_PROBE, json.dumps(names)], cwd=build_dir)
        try:
            missing = set(json.loads(r.stdout)) if r and r.returncode == 0 else None
        except json.JSONDecodeError:
            missing = None
        if missing is None:
            plan.problems.append("couldn't check imports with this interpreter")
            return plan
        for n in sorted(required & missing):
            plan.problems.append(f"Python package providing '{n}' is not installed (pip install …)")
        for n in sorted(optional & missing):
            plan.warnings.append(f"optional import '{n}' is not installed (the code guards it with try)")
        found = sorted((required | optional) - missing)
        if found:
            plan.checked.append("imports available: " + ", ".join(found))
    else:
        plan.checked.append("no third-party imports")
    plan.steps = [[interp, "-u", str(path)]]
    plan.env = dict(PYTHON_ENV)
    return plan


# -------------------------------------------------------------- js / ts ---

def _node_version() -> tuple[str, tuple[int, int]] | None:
    node = shutil.which("node")
    v = _version([node, "--version"]) if node else None
    m = re.match(r"v(\d+)\.(\d+)", v or "")
    return (node, (int(m[1]), int(m[2]))) if m else None


def _check_node_packages(path: Path, root: Path, plan: RunPlan) -> Path:
    """Check package.json dependencies are installed; returns the dir to run in."""
    pkg_json = _find_up(path.parent, root, "package.json")
    if pkg_json is None:
        plan.warnings.append("no package.json; only Node built-in modules can be used")
        return path.parent
    pkg_dir = pkg_json.parent
    try:
        deps = json.loads(pkg_json.read_text(encoding="utf-8")).get("dependencies", {}) or {}
    except (OSError, json.JSONDecodeError) as e:
        plan.problems.append(f"can't read {pkg_json.name}: {e}")
        return pkg_dir
    missing = [d for d in deps if not (pkg_dir / "node_modules" / d / "package.json").exists()]
    if missing:
        plan.problems.append(f"npm packages not installed: {', '.join(missing)} (run npm install in {pkg_dir})")
    elif deps:
        plan.checked.append(f"npm dependencies installed ({len(deps)})")
    else:
        plan.checked.append("package.json has no dependencies")
    return pkg_dir


def _plan_js(path: Path, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan()
    nv = _node_version()
    if nv is None:
        plan.problems.append("Node.js (node) not found")
        return plan
    node, (major, minor) = nv
    plan.checked.append(f"Node.js v{major}.{minor} ({node})")
    plan.cwd = _check_node_packages(path, root, plan)
    plan.steps = [[node, str(path)]]
    return plan


def _plan_ts(path: Path, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan()
    nv = _node_version()
    if nv is None:
        plan.problems.append("Node.js (node) not found")
        return plan
    node, (major, minor) = nv
    plan.checked.append(f"Node.js v{major}.{minor} ({node})")
    plan.cwd = _check_node_packages(path, root, plan)

    local_tsx = _find_up(path.parent, root, "node_modules/.bin/tsx")
    tsx = str(local_tsx) if local_tsx else shutil.which("tsx")
    if tsx:
        plan.checked.append(f"tsx ({tsx})")
        plan.steps = [[tsx, str(path)]]
    elif (major, minor) >= (23, 6) or (major == 22 and minor >= 18):
        # PERF/behaviour: Node strips the types and runs the JS. Fast, but it
        # doesn't type-check, and it can't handle enums/namespaces.
        plan.checked.append("Node runs TypeScript natively (type stripping)")
        plan.warnings.append("types are stripped, not checked; run tsc separately for type errors")
        plan.steps = [[node, str(path)]]
    elif (major, minor) >= (22, 6):
        plan.checked.append("Node type stripping (--experimental-strip-types)")
        plan.warnings.append("types are stripped, not checked; run tsc separately for type errors")
        plan.steps = [[node, "--experimental-strip-types", str(path)]]
    else:
        plan.problems.append("TypeScript needs tsx (npm install -g tsx) or Node 22.6+")
    return plan


# ----------------------------------------------------------------- rust ---

def _plan_rust(path: Path, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan()
    cargo_toml = _find_up(path.parent, root, "Cargo.toml")
    if cargo_toml is not None:
        pkg = cargo_toml.parent
        plan.cwd = pkg
        rel = path.relative_to(pkg).as_posix()
        if rel == "src/main.rs":
            target = []
        elif m := re.fullmatch(r"src/bin/([^/]+)\.rs|src/bin/([^/]+)/main\.rs", rel):
            target = ["--bin", m[1] or m[2]]
        elif m := re.fullmatch(r"examples/([^/]+)\.rs", rel):
            target = ["--example", m[1]]
        else:
            plan.problems.append(f"{rel} isn't a binary target in this Cargo project; "
                                 "run src/main.rs, a src/bin/ file, or an example")
            return plan
        # Resolves the dependency graph without touching the network: fails if
        # any crate isn't already downloaded.
        cargo = _cargo_offline_ready(pkg, plan)
        if cargo is None:
            return plan
        # --offline: never fetch at run time; the check above is the contract.
        plan.steps = [[cargo, "run", "--quiet", "--offline", *target]]
        return plan

    rustc = shutil.which("rustc")
    v = _version([rustc, "--version"]) if rustc else None
    if not v:
        plan.problems.append("rustc not found (install Rust with rustup)")
        return plan
    plan.checked.append(v)
    plan.warnings.append("standalone file (no Cargo.toml): only the standard library is available")
    plan.cwd = path.parent
    out = build_dir / "rust" / path.stem
    # Debug build: compiles faster, which matters more than speed for a quick run.
    plan.steps = [[rustc, "--edition", "2021", str(path), "-o", str(out)], [str(out)]]
    return plan


# ----------------------------------------------------------------- dart ---

def _plan_dart(path: Path, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=path.parent)
    dart = shutil.which("dart")
    v = _version([dart, "--version"]) if dart else None
    if not v:
        plan.problems.append("Dart SDK (dart) not found")
        return plan
    plan.checked.append(v)
    try:
        if re.search(r"""import\s+['"]package:flutter/""", path.read_text(encoding="utf-8")):
            plan.problems.append("this is a Flutter file; it needs `flutter run`, which isn't supported here")
            return plan
    except OSError:
        pass
    pubspec = _find_up(path.parent, root, "pubspec.yaml")
    if pubspec is not None:
        pkg = pubspec.parent
        plan.cwd = pkg
        config = pkg / ".dart_tool" / "package_config.json"
        if not config.exists():
            plan.problems.append(f"packages not fetched (run `dart pub get` in {pkg})")
            return plan
        if pubspec.stat().st_mtime > config.stat().st_mtime:
            plan.problems.append("pubspec.yaml changed since the last `dart pub get`; run it again")
            return plan
        plan.checked.append("pub packages fetched")
    else:
        plan.warnings.append("no pubspec.yaml; only dart: core libraries can be used")
    plan.steps = [[dart, "run", str(path)]]
    return plan


# ----------------------------------------------------------------- tests ---

# Project files that say which test runner to use, checked from the target
# upwards; at one level, earlier entries win.
_TEST_MARKERS = [
    ("rust", ("Cargo.toml",)),
    ("dart", ("pubspec.yaml",)),
    ("python", ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py")),
    ("node", ("package.json",)),
]
_TEST_KIND_BY_SUFFIX = {".py": "python", ".rs": "rust", ".dart": "dart",
                        ".js": "node", ".mjs": "node", ".cjs": "node", ".ts": "node"}
_NPM_PLACEHOLDER = 'echo "Error: no test specified"'


def _has_python_tests(start: Path, limit: int = 5000) -> bool:
    seen = 0
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and d not in ("node_modules", "venv", "target", "__pycache__")]
        for name in filenames:
            seen += 1
            if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
                return True
            if seen > limit:
                return False
    return False


def plan_tests(target: Path, root: Path, build_dir: Path) -> RunPlan:
    """Work out how to run the tests for a file or directory, like plan_for()
    does for a single file: probes only, nothing in the project is executed."""
    start = target if target.is_dir() else target.parent
    want = None if target.is_dir() else _TEST_KIND_BY_SUFFIX.get(target.suffix)
    if target.is_file() and want is None:
        return RunPlan(problems=[f"{target.name} isn't a test file arborist knows how to run "
                                 f"(supported: {', '.join(sorted(_TEST_KIND_BY_SUFFIX))})"])
    kind, project = None, None
    d = start
    while kind is None:
        for k, names in _TEST_MARKERS:
            if (want is None or k == want) and any((d / n).exists() for n in names):
                kind, project = k, d
                break
        if d == root or d.parent == d:
            break
        d = d.parent
    if kind is None:
        if want == "python" or (want is None and _has_python_tests(start)):
            kind, project = "python", root
        elif want == "node":
            kind, project = "node", start
        elif want == "rust":
            return RunPlan(problems=["Rust tests need a Cargo project (no Cargo.toml found)"])
        elif want == "dart":
            return RunPlan(problems=["Dart tests need a pubspec.yaml with the test package"])
        else:
            return RunPlan(problems=["couldn't tell how to run tests here: no Cargo.toml, "
                                     "pubspec.yaml, pyproject.toml/setup.cfg/pytest.ini or "
                                     "package.json, and no test_*.py files. Pass a test file"])
    rel = os.path.relpath(target, project)
    planner = {"python": _tests_python, "rust": _tests_rust, "dart": _tests_dart, "node": _tests_node}[kind]
    return _with_toolchain(planner(target, project, rel, root, build_dir))


def _tests_python(target: Path, project: Path, rel: str, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=project, env=dict(PYTHON_ENV))
    interp = python_for_dir(project, root)
    version = _version([interp, "--version"]) if interp else None
    if not version:
        plan.problems.append("Python interpreter (python3) not found")
        return plan
    plan.checked.append(f"{version} ({interp})")
    build_dir.mkdir(parents=True, exist_ok=True)
    r = _probe([interp, "-c", "import pytest"], cwd=build_dir)
    if r is not None and r.returncode == 0:
        plan.checked.append("pytest")
        plan.steps = [[interp, "-m", "pytest", "-q", "--tb=short", "--color=no",
                       "-p", "no:cacheprovider", *([rel] if rel != "." else [])]]
    else:
        plan.warnings.append("pytest isn't installed for this interpreter, so unittest was used")
        if target.is_file():
            plan.steps = [[interp, "-m", "unittest", rel]]
        else:
            plan.steps = [[interp, "-m", "unittest", "discover", "-s", rel, "-t", "."]]
    return plan


def _cargo_offline_ready(pkg: Path, plan: RunPlan) -> str | None:
    """cargo's path if it's installed and every crate is already downloaded."""
    cargo = shutil.which("cargo")
    v = _version([cargo, "--version"]) if cargo else None
    if not v:
        plan.problems.append("cargo not found (install Rust with rustup)")
        return None
    plan.checked.append(v)
    if _probe_wrap is not None and not any((d / "Cargo.lock").exists()
                                           for d in (pkg, *pkg.parents)):
        # cargo metadata would write Cargo.lock, and sandboxed probes can't
        # write the workspace (nothing changes before you approve a run). The
        # run itself is --offline, so a missing crate still fails there.
        plan.warnings.append("no Cargo.lock yet, so crates weren't checked; the run creates it")
        return cargo
    r = _probe([cargo, "metadata", "--format-version", "1", "--offline", "--quiet"], cwd=pkg)
    if r is None or r.returncode != 0:
        err = (r.stderr.strip().splitlines() or ["unknown error"])[-1] if r else "timed out"
        plan.problems.append(f"crate dependencies aren't available offline (run `cargo fetch`): {err}")
        return None
    plan.checked.append("crate dependencies available offline")
    return cargo


def _tests_rust(target: Path, project: Path, rel: str, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=project)
    cargo = _cargo_offline_ready(project, plan)
    if cargo is None:
        return plan
    select = []
    if m := re.fullmatch(r"tests/([^/]+)\.rs", rel.replace(os.sep, "/")):
        select = ["--test", m[1]]
    plan.steps = [[cargo, "test", "--offline", "--quiet", "--color", "never", *select]]
    return plan


def _tests_dart(target: Path, project: Path, rel: str, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=project)
    try:
        pubspec = (project / "pubspec.yaml").read_text(encoding="utf-8")
    except OSError as e:
        plan.problems.append(f"can't read pubspec.yaml: {e}")
        return plan
    flutter = bool(re.search(r"^\s*sdk:\s*flutter\b", pubspec, re.MULTILINE))
    if flutter:
        exe = shutil.which("flutter")
        if not exe:
            plan.problems.append("this is a Flutter project and flutter isn't installed")
            return plan
        plan.checked.append(f"Flutter ({exe})")  # `flutter --version` is slow; skip it
    else:
        exe = shutil.which("dart")
        v = _version([exe, "--version"]) if exe else None
        if not v:
            plan.problems.append("Dart SDK (dart) not found")
            return plan
        plan.checked.append(v)
    config = project / ".dart_tool" / "package_config.json"
    tool = "flutter" if flutter else "dart"
    if not config.exists():
        plan.problems.append(f"packages not fetched (run `{tool} pub get` in {project})")
        return plan
    if (project / "pubspec.yaml").stat().st_mtime > config.stat().st_mtime:
        plan.problems.append(f"pubspec.yaml changed since the last `{tool} pub get`; run it again")
        return plan
    plan.checked.append("pub packages fetched")
    plan.steps = [[exe, "test", *([rel] if rel != "." else [])]]
    return plan


def _tests_node(target: Path, project: Path, rel: str, root: Path, build_dir: Path) -> RunPlan:
    plan = RunPlan(cwd=project)
    nv = _node_version()
    if nv is None:
        plan.problems.append("Node.js (node) not found")
        return plan
    node, (major, minor) = nv
    plan.checked.append(f"Node.js v{major}.{minor} ({node})")
    pkg_json = project / "package.json"
    script = ""
    if pkg_json.exists():
        _check_node_packages(pkg_json, root, plan)
        if plan.problems:
            return plan
        try:
            script = (json.loads(pkg_json.read_text(encoding="utf-8")).get("scripts") or {}).get("test") or ""
        except (OSError, json.JSONDecodeError):
            script = ""
    if target.is_dir() and script and not script.startswith(_NPM_PLACEHOLDER):
        npm = shutil.which("npm")
        if not npm:
            plan.problems.append("package.json has a test script but npm isn't installed")
            return plan
        plan.checked.append(f"npm test script: {script}")
        plan.steps = [[npm, "test", "--silent"]]
        return plan
    if major < 18:
        plan.problems.append("node --test needs Node 18 or newer")
        return plan
    if target.is_dir():
        plan.cwd = target  # node --test finds test files under the current directory
        plan.steps = [[node, "--test"]]
    elif target.suffix == ".ts":
        local_tsx = _find_up(target.parent, root, "node_modules/.bin/tsx")
        tsx = str(local_tsx) if local_tsx else shutil.which("tsx")
        if tsx:
            plan.steps = [[tsx, "--test", rel]]
        elif (major, minor) >= (22, 18) or major >= 23:
            plan.steps = [[node, "--test", rel]]
        elif (major, minor) >= (22, 6):
            plan.steps = [[node, "--experimental-strip-types", "--test", rel]]
        else:
            plan.problems.append("TypeScript tests need tsx (npm install -g tsx) or Node 22.6+")
    else:
        plan.steps = [[node, "--test", rel]]
    return plan


# Where failure detail starts in each runner's output.
_FAILURE_START = re.compile(
    r"^(=+ (FAILURES|ERRORS) =+"          # pytest
    r"|(FAIL|ERROR): "                     # unittest
    r"|failures:\s*$"                      # cargo test
    r"|error(\[E\d+\])?: "                 # rustc compile errors
    r"|not ok \d+"                         # node --test (TAP)
    r"|\s*✖ "                              # node --test (spec reporter)
    r"|.*\[E\]"                            # dart/flutter test
    r")")


def focus_test_output(log: str, passed: bool) -> str:
    """The useful part of a test log: the summary when everything passed,
    otherwise everything from the first failure on (with the command line)."""
    lines = log.splitlines()
    if passed:
        return "\n".join(lines[-15:])
    first = next((i for i, line in enumerate(lines) if _FAILURE_START.match(line)), None)
    if not first:  # None, or the failure is already at the top
        return log
    head = [lines[0]] if lines[0].startswith("$ ") else []
    return "\n".join(head + [f"[... {first - len(head)} lines before the first failure omitted ...]"]
                     + lines[first:])


# ------------------------------------------------------------- execution ---

def execute(plan: RunPlan, log_dir: Path, timeout: int,
            on_output: Callable[[str], None], name: str = "run",
            spec: sandbox.Spec | None = None, env_keep=(), display: bool = False) -> RunOutcome:
    """Run each step with a shared timeout; combined output goes to a log file.

    `name` (usually the source file's stem) labels the log file. Steps get a
    minimal environment (run_env). With a sandbox `spec`, each step runs
    inside bwrap (sandbox.wrap); the caller checks sandbox.available() first.
    """
    env = run_env(plan.env, env_keep, display and spec is None)
    runs = log_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]", "_", name) or "run"
    log_path = runs / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.log"
    # A compile step (rustc -o OUT) needs its output directory to exist.
    for step in plan.steps:
        if "-o" in step:
            Path(step[step.index("-o") + 1]).parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    exit_code, timed_out = None, False
    with open(log_path, "w", encoding="utf-8") as log:
        for step in plan.steps:
            header = f"$ {' '.join(step)}   (in {plan.cwd})"
            log.write(header + "\n")
            log.flush()
            on_output(header)
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                break
            if spec is not None:
                cmd, cwd, step_env = sandbox.wrap(step, spec, env), None, {"PATH": "/usr/bin:/bin"}
            else:
                cmd, cwd, step_env = step, plan.cwd, env
            exit_code, timed_out = _run_step(cmd, cwd, remaining, log, on_output, step_env)
            if timed_out or exit_code != 0:
                break
        footer = (f"[timed out after {timeout}s]" if timed_out else f"[exit code {exit_code}]")
        log.write(footer + "\n")
        on_output(footer)
    return RunOutcome(exit_code, timed_out, time.monotonic() - start, log_path)


def _run_step(cmd: list[str], cwd: Path, timeout: float, log, on_output,
              env: dict[str, str] | None = None) -> tuple[int | None, bool]:
    try:
        # New session: Ctrl-C at the terminal doesn't reach it directly, and on
        # timeout or interrupt we can kill the whole process group.
        proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1, start_new_session=True)
    except OSError as e:
        msg = f"failed to start: {e}"
        log.write(msg + "\n")
        on_output(msg)
        return 127, False

    def pump():
        for line in proc.stdout:
            log.write(line)
            log.flush()
            on_output(line.rstrip("\n"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        timed_out = True
    except BaseException:  # Ctrl-C while waiting: don't leave it running
        _kill_group(proc)
        reader.join(timeout=2)
        raise
    reader.join(timeout=5)
    return (None if timed_out else proc.returncode), timed_out


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue
