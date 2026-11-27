"""The task plan (plan.py): parsing, and refusing to tick off unproven steps."""
from __future__ import annotations

import pytest

import plan
from conftest import write
from tools import TestRun, ToolError


@pytest.fixture
def tb(make_toolbox):
    return make_toolbox(approve_edits="auto")


def passed(tb, path):
    tb.test_runs.append(TestRun(path, True, tb.writes_snapshot()))


def failed(tb, path):
    tb.test_runs.append(TestRun(path, False, tb.writes_snapshot()))


def test_steps_are_parsed_leniently(tb):
    p = plan.Plan()
    out = plan.update(p, ["write the test", {"step": "  fix   add() ", "status": "In Progress"},
                          {"title": "docs", "status": "completed"}], tb)
    assert [(s.step, s.status) for s in p.steps] == [
        ("write the test", "pending"), ("fix add()", "in_progress"), ("docs", "done")]
    assert "2. [>] fix add()" in out and "3. [x] docs  (no test)" in out
    assert "Next: step 2, fix add()." in out
    assert p.unchecked() == ["docs"]


@pytest.mark.parametrize("items, why", [
    ([], "non-empty"),
    ([{"step": f"s{n}"} for n in range(plan.MAX_STEPS + 1)], "at most"),
    ([{"step": ""}], "no 'step'"),
    ([{"step": "a", "status": "maybe"}], "status 'maybe'"),
    ([{"step": "a", "status": "in_progress"}, {"step": "b", "status": "in_progress"}], "only one"),
    ([{"step": "a", "test": "../outside.py"}], "outside the workspace"),
])
def test_bad_plans_are_refused(tb, items, why):
    with pytest.raises(ToolError, match=why):
        plan.update(plan.Plan(), items, tb)


def test_a_step_with_a_test_needs_the_test_to_exist_and_pass(tb, ws_root):
    p = plan.Plan()
    item = [{"step": "fix add", "status": "done", "test": "tests/test_app.py"}]
    out = plan.update(p, item, tb)
    assert p.steps[0].status == "in_progress" and "doesn't exist yet" in out

    write(ws_root, "tests/test_app.py", "def test_x(): pass\n")
    assert "hasn't been run" in plan.update(p, item, tb)

    failed(tb, "tests/test_app.py")
    assert "didn't pass" in plan.update(p, item, tb)

    passed(tb, "tests")  # a run of the folder covers the file
    out = plan.update(p, item, tb)
    assert p.steps[0].status == "done" and p.steps[0].passed
    assert "Not marked done" not in out and "Every step is done" in out


def test_an_edit_after_the_pass_reopens_an_unproven_step(tb, ws_root):
    write(ws_root, "test_app.py", "def test_x(): pass\n")
    write(ws_root, "app.py", "x = 1\n")
    passed(tb, ".")
    tb.call("read_file", {"path": "app.py"})
    tb.call("edit_file", {"path": "app.py", "old_string": "x = 1", "new_string": "x = 2"})
    p = plan.Plan()
    out = plan.update(p, [{"step": "a", "status": "done", "test": "test_app.py"}], tb)
    assert "files changed after" in out and p.steps[0].status == "in_progress"


def test_a_proven_step_stays_done_and_goes_stale_if_later_edits_break_the_proof(tb, ws_root):
    write(ws_root, "test_a.py", "def test_a(): pass\n")
    write(ws_root, "b.py", "y = 1\n")
    p = plan.Plan()
    passed(tb, "test_a.py")
    plan.update(p, [{"step": "a", "status": "done", "test": "test_a.py"},
                    {"step": "b", "status": "in_progress"}], tb)
    tb.call("read_file", {"path": "b.py"})
    tb.call("edit_file", {"path": "b.py", "old_string": "y = 1", "new_string": "y = 2"})
    # Sent again unchanged: still done, not re-checked...
    plan.update(p, [{"step": "a", "status": "done", "test": "test_a.py"},
                    {"step": "b", "status": "done"}], tb)
    assert [s.status for s in p.steps] == ["done", "done"]
    # ...but the report says its proof no longer holds.
    assert plan.stale(p, tb) == ["a (files changed after test_a.py last passed; run it again)"]
    passed(tb, ".")
    assert plan.stale(p, tb) == []


def test_a_refused_step_doesnt_make_two_in_progress(tb):
    p = plan.Plan()
    plan.update(p, [{"step": "a", "status": "done", "test": "t.py"},
                    {"step": "b", "status": "in_progress"}], tb)
    assert [s.status for s in p.steps] == ["pending", "in_progress"]
    assert p.next_step()[0] == 2


def test_steps_removed_before_they_were_done_are_remembered(tb):
    p = plan.Plan()
    plan.update(p, [{"step": "a", "status": "done"}, {"step": "b"}, {"step": "c"}], tb)
    plan.update(p, [{"step": "a", "status": "done"}, {"step": "c"}], tb)
    assert p.dropped == ["b"]
    plan.update(p, [{"step": "b"}], tb)  # back in: no longer dropped; c is now
    assert p.dropped == ["c"]


def test_the_tool_is_only_in_tasks(tb):
    assert "update_plan" not in tb.names
