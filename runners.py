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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

RUNNABLE = {".py", ".rs", ".dart", ".js", ".ts"}
PROBE_TIMEOUT = 20  # seconds, for version checks and dependency probes


@dataclass
class RunPlan:
    steps: list[list[str]] = field(default_factory=list)  # run in order; stop at first failure
    cwd: Path = Path(".")
    checked: list[str] = field(default_factory=list)      # what passed, for the report
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)     # non-empty means "don't run"

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
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
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
    return planner(path, root, build_dir)


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
    d = path.parent
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
        cargo = shutil.which("cargo")
        v = _version([cargo, "--version"]) if cargo else None
        if not v:
            plan.problems.append("cargo not found (install Rust with rustup)")
            return plan
        plan.checked.append(v)
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
        r = _probe([cargo, "metadata", "--format-version", "1", "--offline", "--quiet"], cwd=pkg)
        if r is None or r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["unknown error"])[-1] if r else "timed out"
            plan.problems.append(f"crate dependencies aren't available offline (run `cargo fetch`): {err}")
            return plan
        plan.checked.append("crate dependencies available offline")
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


# ------------------------------------------------------------- execution ---

def execute(plan: RunPlan, log_dir: Path, timeout: int,
            on_output: Callable[[str], None]) -> RunOutcome:
    """Run each step with a shared timeout; combined output goes to a log file."""
    runs = log_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    name = Path(plan.steps[-1][-1]).stem if plan.steps else "run"
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
            exit_code, timed_out = _run_step(step, plan.cwd, remaining, log, on_output)
            if timed_out or exit_code != 0:
                break
        footer = (f"[timed out after {timeout}s]" if timed_out else f"[exit code {exit_code}]")
        log.write(footer + "\n")
        on_output(footer)
    return RunOutcome(exit_code, timed_out, time.monotonic() - start, log_path)


def _run_step(cmd: list[str], cwd: Path, timeout: float, log, on_output) -> tuple[int | None, bool]:
    try:
        # New session: Ctrl-C at the terminal doesn't reach it directly, and on
        # timeout or interrupt we can kill the whole process group.
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
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
