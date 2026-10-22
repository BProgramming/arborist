"""check_dependencies / run_file: the dependency gate, approval and logging.

Language-specific tests skip themselves when that toolchain isn't installed.
"""
from __future__ import annotations

import os
import textwrap

import pytest

import runners
from conftest import FakeApprover, needs, write
from tools import Decision


def check_and_run(tb, rel):
    report = tb.call("check_dependencies", {"path": rel}).content
    return report, tb.call("run_file", {"path": rel})


# ----------------------------------------------------------------- gating ---

class TestGate:
    def test_run_requires_a_passing_check_first(self, ws_root, make_toolbox, approver):
        write(ws_root, "hello.py", "print('hi')\n")
        tb = make_toolbox(approver)
        out = tb.call("run_file", {"path": "hello.py"})
        assert "call check_dependencies" in out.content and not out.hard_stop
        assert approver.run_requests == []

    @pytest.mark.parametrize("name", ["script.sh", "Main.java", "a.rb", "Makefile"])
    def test_only_allowed_languages_run(self, ws_root, make_toolbox, approver, name):
        write(ws_root, name, "echo hi\n")
        tb = make_toolbox(approver)
        assert "MISSING" in tb.call("check_dependencies", {"path": name}).content
        assert "can be run" in tb.call("run_file", {"path": name}).content
        assert approver.run_requests == []

    def test_check_is_single_use(self, ws_root, make_toolbox):
        write(ws_root, "hello.py", "print('hi')\n")
        tb = make_toolbox()
        check_and_run(tb, "hello.py")
        assert "call check_dependencies" in tb.call("run_file", {"path": "hello.py"}).content

    def test_editing_after_the_check_invalidates_it(self, ws_root, make_toolbox):
        p = write(ws_root, "hello.py", "print('hi')\n")
        tb = make_toolbox()
        tb.call("check_dependencies", {"path": "hello.py"})
        p.write_text("print('changed')\n")
        os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1_000_000))
        assert "changed since its dependency check" in tb.call("run_file", {"path": "hello.py"}).content

    def test_failed_check_blocks_the_run(self, ws_root, make_toolbox, approver):
        write(ws_root, "needs.py", "import surely_not_a_real_package_xyz\n")
        tb = make_toolbox(approver)
        report, result = check_and_run(tb, "needs.py")
        assert "MISSING" in report and "surely_not_a_real_package_xyz" in report
        assert "call check_dependencies" in result.content
        assert approver.run_requests == []

    def test_declined_run_executes_nothing_and_stops(self, ws_root, make_toolbox):
        marker = ws_root / "ran.txt"
        write(ws_root, "touch.py", f"open({str(marker)!r}, 'w').write('x')\n")
        tb = make_toolbox(FakeApprover(run=[Decision(False, "not now")]))
        _, result = check_and_run(tb, "touch.py")
        assert result.hard_stop
        assert "declined" in result.content and "not now" in result.content
        assert not marker.exists()

    def test_runs_always_ask_even_in_auto_edit_mode(self, ws_root, make_toolbox, approver):
        write(ws_root, "hello.py", "print('hi')\n")
        tb = make_toolbox(approver, approve_edits="auto")
        check_and_run(tb, "hello.py")
        assert len(approver.run_requests) == 1

    def test_missing_toolchain_is_reported(self, ws_root, make_toolbox, monkeypatch):
        write(ws_root, "a.js", "console.log(1)\n")
        monkeypatch.setattr(runners.shutil, "which", lambda name: None)
        report = make_toolbox().call("check_dependencies", {"path": "a.js"}).content
        assert "MISSING: Node.js (node) not found" in report


# ----------------------------------------------------------------- python ---

class TestPython:
    def test_import_scanner(self):
        src = textwrap.dedent("""
            import os, json
            import numpy as np
            from requests.adapters import HTTPAdapter
            from . import sibling
            try:
                import ujson
            except ImportError:
                ujson = None
        """)
        required, optional = runners.python_imports(src)
        assert required == {"os", "json", "numpy", "requests"}
        assert optional == {"ujson"}

    def test_runs_and_logs_output(self, ws_root, make_toolbox, approver):
        write(ws_root, "hello.py", "import sys\nprint('hello from python')\nprint('oops', file=sys.stderr)\n")
        tb = make_toolbox(approver)
        report, result = check_and_run(tb, "hello.py")
        assert "Ready." in report
        assert result.hard_stop
        assert "exit code 0" in result.content
        assert "hello from python" in result.content and "oops" in result.content
        assert "Don't call any more tools" in result.content
        # It asked first, showing the exact command.
        rel, steps, cwd, timeout = approver.run_requests[0]
        assert rel == "hello.py" and steps[0][-1] == str(ws_root / "hello.py")
        # Output streamed live and was written to a log outside the workspace.
        assert "hello from python" in tb.output
        log = next((tb.log_dir / "runs").glob("*-hello.log"))
        assert "hello from python" in log.read_text()
        assert not log.is_relative_to(ws_root)

    def test_failure_exit_code_and_traceback(self, ws_root, make_toolbox):
        write(ws_root, "boom.py", "raise ValueError('bad input')\n")
        _, result = check_and_run(make_toolbox(), "boom.py")
        assert "exit code 1" in result.content and "ValueError: bad input" in result.content

    def test_timeout_kills_the_process(self, ws_root, make_toolbox):
        write(ws_root, "slow.py", "import time\nprint('start', flush=True)\ntime.sleep(30)\n")
        _, result = check_and_run(make_toolbox(run_timeout=1), "slow.py")
        assert "timed out after 1s" in result.content and "start" in result.content

    def test_local_modules_are_not_treated_as_missing(self, ws_root, make_toolbox):
        write(ws_root, "helpers.py", "def greet():\n    return 'hi'\n")
        write(ws_root, "main.py", "from helpers import greet\nprint(greet())\n")
        report, result = check_and_run(make_toolbox(), "main.py")
        assert "MISSING" not in report and "hi" in result.content

    def test_the_check_never_executes_project_code(self, ws_root, make_toolbox):
        # A project file shadowing a stdlib module the probe itself imports.
        marker = ws_root / "executed.txt"
        write(ws_root, "json.py", f"open({str(marker)!r}, 'w').write('x')\n")
        write(ws_root, "main.py", "import requests_not_installed_xyz\n")
        make_toolbox().call("check_dependencies", {"path": "main.py"})
        assert not marker.exists()

    def test_syntax_error_is_reported_before_running(self, ws_root, make_toolbox, approver):
        write(ws_root, "bad.py", "def broken(:\n")
        tb = make_toolbox(approver)
        assert "syntax error at line 1" in tb.call("check_dependencies", {"path": "bad.py"}).content


# ------------------------------------------------------------- node / ts ---

@needs("node")
class TestNode:
    def test_runs_javascript(self, ws_root, make_toolbox):
        write(ws_root, "app.js", "console.log('hello from node', 1 + 1)\n")
        report, result = check_and_run(make_toolbox(), "app.js")
        assert "Node.js v" in report and "hello from node 2" in result.content

    def test_missing_npm_dependency_blocks_the_run(self, ws_root, make_toolbox):
        write(ws_root, "package.json", '{"dependencies": {"left-pad": "^1.3.0"}}')
        write(ws_root, "app.js", "require('left-pad')\n")
        report = make_toolbox().call("check_dependencies", {"path": "app.js"}).content
        assert "npm packages not installed: left-pad" in report

    def test_installed_npm_dependency_passes(self, ws_root, make_toolbox):
        write(ws_root, "package.json", '{"dependencies": {"tiny": "1.0.0"}}')
        write(ws_root, "node_modules/tiny/package.json", '{"name": "tiny", "main": "index.js"}')
        write(ws_root, "node_modules/tiny/index.js", "module.exports = () => 'tiny ok'\n")
        write(ws_root, "app.js", "console.log(require('tiny')())\n")
        report, result = check_and_run(make_toolbox(), "app.js")
        assert "npm dependencies installed (1)" in report and "tiny ok" in result.content

    def test_runs_typescript(self, ws_root, make_toolbox):
        write(ws_root, "app.ts", "const n: number = 21;\nconsole.log(`answer ${n * 2}`);\n")
        report, result = check_and_run(make_toolbox(), "app.ts")
        if "MISSING" in report:
            pytest.skip("this Node is too old for TypeScript and tsx isn't installed")
        assert "answer 42" in result.content


# ------------------------------------------------------------------- rust ---

class TestRust:
    @needs("rustc")
    def test_single_file_compiles_then_runs(self, ws_root, make_toolbox, approver):
        write(ws_root, "hello.rs", 'fn main() { println!("hello from rust {}", 6 * 7); }\n')
        tb = make_toolbox(approver, run_timeout=120)
        report, result = check_and_run(tb, "hello.rs")
        assert "only the standard library" in report
        assert len(approver.run_requests[0][1]) == 2  # compile step + run step shown
        assert "hello from rust 42" in result.content

    @needs("rustc")
    def test_compile_error_stops_before_running(self, ws_root, make_toolbox):
        write(ws_root, "bad.rs", "fn main() { let x: i32 = \"no\"; }\n")
        _, result = check_and_run(make_toolbox(run_timeout=120), "bad.rs")
        assert "mismatched types" in result.content and "exit code 1" in result.content

    @needs("cargo")
    def test_cargo_project_runs_main(self, ws_root, make_toolbox):
        write(ws_root, "Cargo.toml", '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n')
        write(ws_root, "src/main.rs", 'fn main() { println!("cargo ok"); }\n')
        write(ws_root, "src/util.rs", "pub fn x() {}\n")
        tb = make_toolbox(run_timeout=180)
        report, result = check_and_run(tb, "src/main.rs")
        assert "crate dependencies available offline" in report
        assert "cargo ok" in result.content
        assert "isn't a binary target" in tb.call("check_dependencies", {"path": "src/util.rs"}).content


# ------------------------------------------------------------------- dart ---

class TestDart:
    @needs("dart")
    def test_runs_dart(self, ws_root, make_toolbox):
        write(ws_root, "hello.dart", "void main() { print('hello from dart ${6 * 7}'); }\n")
        _, result = check_and_run(make_toolbox(run_timeout=120), "hello.dart")
        assert "hello from dart 42" in result.content

    @needs("dart")
    def test_flutter_files_are_refused(self, ws_root, make_toolbox):
        write(ws_root, "app.dart", "import 'package:flutter/material.dart';\nvoid main() {}\n")
        assert "Flutter" in make_toolbox().call("check_dependencies", {"path": "app.dart"}).content

    @needs("dart")
    def test_unfetched_packages_block_the_run(self, ws_root, make_toolbox):
        write(ws_root, "pubspec.yaml", "name: demo\nenvironment:\n  sdk: ^3.0.0\n")
        write(ws_root, "bin/main.dart", "void main() {}\n")
        report = make_toolbox().call("check_dependencies", {"path": "bin/main.dart"}).content
        assert "dart pub get" in report


def test_run_output_fits_the_limit_and_keeps_its_instructions(ws_root, make_toolbox):
    write(ws_root, "noisy.py", "for i in range(5000):\n    print('line', i)\n")
    tb = make_toolbox(max_output_chars=2000)
    _, result = check_and_run(tb, "noisy.py")
    assert len(result.content) <= 2000
    assert result.content.rstrip().endswith("Don't call any more tools.")
    assert "line 4999" in result.content  # the end of the log is kept


@needs("cargo")
def test_cargo_run_log_is_named_after_the_file(ws_root, make_toolbox):
    write(ws_root, "Cargo.toml", '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n')
    write(ws_root, "src/main.rs", 'fn main() { println!("ok"); }\n')
    tb = make_toolbox(run_timeout=180)
    check_and_run(tb, "src/main.rs")
    logs = [p.name for p in (tb.log_dir / "runs").glob("*.log")]
    assert len(logs) == 1 and logs[0].endswith("-main.log")
