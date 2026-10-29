"""run_tests: picking the runner, approval, and failure-focused output."""
from __future__ import annotations

import pytest

import runners
import sandbox
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
    assert "fix you propose" in res.content and "don't call any more tools" in res.content
    assert tb.tests_failed
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


def test_a_same_size_edit_right_after_a_run_is_not_stale(ws_root, make_toolbox):
    """Python's .pyc check uses size + whole-second mtime; arborist's runs
    mustn't leave caches that make a quick same-size fix invisible."""
    pytest.importorskip("pytest")
    write(ws_root, "calc.py", "def add(a, b):\n    return a - b\n")
    write(ws_root, "test_calc.py", "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
    tb = make_toolbox()
    assert "failed" in tb.call("run_tests", {}).content.splitlines()[0]
    write(ws_root, "calc.py", "def add(a, b):\n    return a + b\n")  # same size, same second
    assert "passed" in tb.call("run_tests", {}).content.splitlines()[0]
    assert not (ws_root / "__pycache__").exists()


def test_tests_failed_resets_each_turn_and_on_a_pass(ws_root, make_toolbox):
    pytest.importorskip("pytest")
    write(ws_root, "test_math.py", FAILING)
    tb = make_toolbox()
    tb.call("run_tests", {})
    assert tb.tests_failed
    tb.begin_turn("next")
    assert not tb.tests_failed
    write(ws_root, "test_math.py", PASSING)
    tb.call("run_tests", {})
    assert not tb.tests_failed


# ------------------------------------------------------ asking to fix them ---

@pytest.fixture
def chat_with_failing_tests(ws_root, tmp_path, monkeypatch):
    """A Chat whose model runs the (failing) tests, then presents the bug."""
    pytest.importorskip("pytest")
    import json
    import chat
    import config
    cfg_file = tmp_path / "t.toml"
    cfg_file.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n'
                        '[model]\nrepo = "a/b"\nfile = "c.gguf"\n')
    monkeypatch.setenv("ARBORIST_CONFIG", str(cfg_file))
    write(ws_root, "calc.py", "def add(a, b):\n    return a - b\n")
    write(ws_root, "test_calc.py", "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
    c = chat.Chat(config.load(), ws_root, tools_enabled=True)
    c.toolbox.approver = FakeApprover()
    # Like make_toolbox: offline where this machine can do it, online where it can't.
    c.toolbox.limits.run_sandbox = sandbox.available()
    requests = []

    def call(name, args, cid):
        return {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": cid, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    def text(t):
        return {"choices": [{"delta": {"content": t}}]}

    script = [
        [call("run_tests", {}, "t1")],
        [text("Bug: calc.py:2 subtracts. Proposed fix: return a + b.")],
        # After "yes": read, fix, re-run (passes), report.
        [call("read_file", {"path": "calc.py"}, "r1")],
        [call("edit_file", {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"}, "e1")],
        [call("run_tests", {}, "t2")],
        [text("Fixed; tests pass.")],
    ]

    def stream(body):
        requests.append(body)
        yield from script.pop(0)

    c.client.stream_chat = stream
    c.requests = requests
    return c


def test_failing_tests_lead_to_an_offer_and_yes_fixes(chat_with_failing_tests, ws_root, monkeypatch):
    import chat
    c = chat_with_failing_tests
    c.messages.append({"role": "user", "content": "run the tests"})
    c._reply(label="run the tests")
    assert c.fix_offer
    assert (ws_root / "calc.py").read_text().endswith("a - b\n")  # nothing changed yet

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert c.offer_fix() is True
    assert not c.fix_offer  # the re-run passed, so no second offer
    assert (ws_root / "calc.py").read_text().endswith("a + b\n")
    assert c.messages[-1]["content"] == "Fixed; tests pass."
    go_ahead = [m for m in c.messages if m["role"] == "user"][-1]["content"]
    assert go_ahead == chat.FIX_GO_AHEAD


def test_saying_no_changes_nothing(chat_with_failing_tests, ws_root, monkeypatch, capsys):
    c = chat_with_failing_tests
    c.messages.append({"role": "user", "content": "run the tests"})
    c._reply(label="run the tests")
    n = len(c.messages)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert c.offer_fix() is False
    assert len(c.messages) == n and not c.fix_offer
    assert (ws_root / "calc.py").read_text().endswith("a - b\n")
    assert "tell the model what you'd like instead" in capsys.readouterr().out


def test_ctrl_c_at_the_offer_is_a_no(chat_with_failing_tests, monkeypatch):
    c = chat_with_failing_tests
    c.fix_offer = True

    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert c.offer_fix() is False and not c.fix_offer
