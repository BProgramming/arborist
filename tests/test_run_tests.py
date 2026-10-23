"""run_tests: picking the runner, approval, and failure-focused output."""
from __future__ import annotations

import pytest

import runners
from conftest import FakeApprover, needs, write
from tools import Decision

PASSING = "def test_ok():\n    assert 1 + 1 == 2\n"
FAILING = ("def test_ok():\n    assert True\n\n"
           "def test_math():\n    expected = 5\n    assert 2 + 2 == expected\n")


# ------------------------------------------------------------- planning ---

def test_python_project_uses_pytest_from_the_project_dir(ws_root, tmp_path):
    write(ws_root, "pyproject.toml", "")
    write(ws_root, "tests/test_a.py", PASSING)
    plan = runners.plan_tests(ws_root / "tests", ws_root, tmp_path / "build")
    assert plan.ok and plan.cwd == ws_root
    step = plan.steps[0]
    assert step[1:4] == ["-m", "pytest", "-q"] and step[-1] == "tests"


def test_loose_test_files_without_a_project_file_still_run(ws_root, tmp_path):
    write(ws_root, "test_thing.py", PASSING)
    plan = runners.plan_tests(ws_root, ws_root, tmp_path / "build")
    assert plan.ok and "pytest" in plan.steps[0]


def test_nearest_project_file_wins(ws_root, tmp_path):
    write(ws_root, "package.json", '{"name": "web"}')
    write(ws_root, "tools/pyproject.toml", "")
    write(ws_root, "tools/test_x.py", PASSING)
    plan = runners.plan_tests(ws_root / "tools", ws_root, tmp_path / "build")
    assert plan.cwd == ws_root / "tools" and "pytest" in plan.steps[0]


def test_a_file_picks_its_own_language_over_a_nearer_marker(ws_root, tmp_path):
    write(ws_root, "package.json", '{"name": "x"}')
    write(ws_root, "test_a.py", PASSING)
    plan = runners.plan_tests(ws_root / "test_a.py", ws_root, tmp_path / "build")
    assert "pytest" in plan.steps[0] and plan.steps[0][-1] == "test_a.py"


def test_nothing_recognisable_is_a_problem_not_a_guess(ws_root, tmp_path):
    write(ws_root, "notes.txt", "hi")
    plan = runners.plan_tests(ws_root, ws_root, tmp_path / "build")
    assert not plan.ok and "couldn't tell how to run tests" in plan.problems[0]
    plan = runners.plan_tests(ws_root / "notes.txt", ws_root, tmp_path / "build")
    assert "isn't a test file" in plan.problems[0]


@needs("node", "npm")
def test_npm_test_script_is_used_but_not_the_placeholder(ws_root, tmp_path):
    write(ws_root, "package.json", '{"scripts": {"test": "vitest run"}}')
    plan = runners.plan_tests(ws_root, ws_root, tmp_path / "build")
    assert plan.steps[0][1:] == ["test", "--silent"]
    write(ws_root, "package.json", '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}')
    plan = runners.plan_tests(ws_root, ws_root, tmp_path / "build")
    assert plan.steps[0][1:] == ["--test"]


def test_flutter_project_needs_flutter(ws_root, tmp_path, monkeypatch):
    write(ws_root, "pubspec.yaml", "name: app\ndependencies:\n  flutter:\n    sdk: flutter\n")
    monkeypatch.setattr(runners.shutil, "which", lambda name: None)
    plan = runners.plan_tests(ws_root, ws_root, tmp_path / "build")
    assert "Flutter project and flutter isn't installed" in plan.problems[0]


# ------------------------------------------------------------ the output ---

def test_focus_keeps_failures_and_summary():
    log = "\n".join(["$ python -m pytest -q   (in /p)"] + ["."] * 50 + [
        "=================================== FAILURES ===================================",
        "_________________________________ test_math __________________________________",
        "E       assert 4 == 5",
        "1 failed, 50 passed in 0.10s", "[exit code 1]"])
    out = runners.focus_test_output(log, passed=False)
    assert out.splitlines()[0].startswith("$ python -m pytest")
    assert "[... 50 lines before the first failure omitted ...]" in out
    assert "E       assert 4 == 5" in out and "1 failed, 50 passed" in out


def test_focus_on_a_pass_keeps_only_the_summary():
    log = "\n".join([f"test {i} ... ok" for i in range(100)] + ["100 passed", "[exit code 0]"])
    out = runners.focus_test_output(log, passed=True)
    assert len(out.splitlines()) == 15 and out.endswith("[exit code 0]")


def test_focus_without_a_known_marker_keeps_everything():
    log = "$ thing\nsomething odd happened\n[exit code 2]"
    assert runners.focus_test_output(log, passed=False) == log


# -------------------------------------------------------------- end to end ---

def test_failing_pytest_run_end_to_end(ws_root, make_toolbox):
    pytest.importorskip("pytest")
    write(ws_root, "pyproject.toml", "")
    write(ws_root, "tests/test_math.py", FAILING)
    approver = FakeApprover()
    tb = make_toolbox(approver)
    res = tb.call("run_tests", {})
    assert res.hard_stop
    assert res.content.startswith("Tests for . failed (exit code 1)")
    assert "assert 2 + 2 == expected" in res.content and "1 failed, 1 passed" in res.content
    assert "Don't call any more tools" in res.content
    rel, steps, cwd, timeout = approver.run_requests[0]
    assert rel == "tests (.)" and timeout == 300 and "pytest" in steps[0]


def test_passing_run_is_short(ws_root, make_toolbox):
    pytest.importorskip("pytest")
    write(ws_root, "test_ok.py", PASSING)
    res = make_toolbox().call("run_tests", {"path": "test_ok.py"})
    assert "Tests for test_ok.py passed" in res.content and "1 passed" in res.content
    assert len(res.content) < 800


def test_declining_runs_nothing_and_ends_tool_use(ws_root, make_toolbox):
    write(ws_root, "test_ok.py", PASSING)
    tb = make_toolbox(FakeApprover(run=[Decision(False, "not now")]))
    res = tb.call("run_tests", {})
    assert res.hard_stop and "declined to run the tests" in res.content and "not now" in res.content
    assert not tb.output  # nothing executed


def test_missing_toolchain_is_reported_without_a_hard_stop(ws_root, make_toolbox, monkeypatch):
    write(ws_root, "Cargo.toml", "[package]\nname = 'x'\n")
    monkeypatch.setattr(runners.shutil, "which", lambda name: None)
    res = make_toolbox().call("run_tests", {})
    assert not res.hard_stop and "MISSING: cargo not found" in res.content


@needs("node")
def test_node_test_runner_end_to_end(ws_root, make_toolbox):
    write(ws_root, "math.test.mjs",
          "import test from 'node:test';\nimport assert from 'node:assert';\n"
          "test('adds', () => assert.strictEqual(1 + 1, 3));\n")
    res = make_toolbox().call("run_tests", {"path": "math.test.mjs"})
    assert "failed" in res.content.splitlines()[0]
    assert "adds" in res.content


@needs("cargo")
def test_cargo_test_end_to_end(ws_root, make_toolbox):
    write(ws_root, "Cargo.toml", '[package]\nname = "calc"\nversion = "0.1.0"\nedition = "2021"\n')
    write(ws_root, "src/lib.rs",
          "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
          "#[cfg(test)]\nmod tests {\n    #[test]\n    fn adds() { assert_eq!(super::add(2, 2), 5); }\n}\n")
    res = make_toolbox(run_timeout=300).call("run_tests", {})
    assert "failed" in res.content.splitlines()[0]
    assert "adds" in res.content and "assertion" in res.content
