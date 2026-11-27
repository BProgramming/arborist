"""A delegated task's plan: the steps the model works through, in order.

The model writes the plan with the update_plan tool and sends the whole list
again each time something changes. A step can name a test (a test file, or a
directory of them). Such a step can only be marked done once run_tests has
passed on that test, or on a folder holding it, with no edits since. Otherwise
the harness keeps the step open and says why. A small model will happily tick
off work it hasn't done, so the plan is only worth something if the harness
checks it.

Steps without a test can be marked done freely; the report lists them as
unchecked so the reviewer knows where to look. A step taken out of the plan
before it was done is allowed (plans change), but it's remembered as dropped,
so it can't quietly disappear.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tools import ToolError, Toolbox, _fn

STATUSES = ("pending", "in_progress", "done")
MAX_STEPS = 10

# What small models write instead of the three statuses.
_ALIASES = {
    "todo": "pending", "not_started": "pending", "open": "pending",
    "in-progress": "in_progress", "inprogress": "in_progress", "active": "in_progress",
    "doing": "in_progress", "started": "in_progress", "current": "in_progress",
    "complete": "done", "completed": "done", "finished": "done",
}

_MARK = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}


@dataclass
class Step:
    step: str
    status: str = "pending"
    test: str | None = None  # workspace-relative test file or directory
    passed: bool = False     # done, with its test proven to pass at the time


class Plan:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.dropped: list[str] = []  # steps removed before they were done

    def __bool__(self) -> bool:
        return bool(self.steps)

    def open_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status != "done"]

    def next_step(self) -> tuple[int, Step] | None:
        """The step to work on: the one in progress, else the first not done."""
        for i, s in enumerate(self.steps, 1):
            if s.status == "in_progress":
                return i, s
        for i, s in enumerate(self.steps, 1):
            if s.status != "done":
                return i, s
        return None

    def unchecked(self) -> list[str]:
        """Done steps that no test backs up."""
        return [s.step for s in self.steps if s.status == "done" and not s.test]

    def render(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            note = ""
            if s.test:
                note = f"  (test: {s.test}{', passed' if s.passed else ''})"
            elif s.status == "done":
                note = "  (no test)"
            lines.append(f"{i}. {_MARK[s.status]} {s.step}{note}")
        return "\n".join(lines)

    def next_line(self) -> str:
        nxt = self.next_step()
        if nxt is None:
            return ("Every step is done. If you haven't run the tests on everything you changed, "
                    "do that; then reply with the report.")
        i, s = nxt
        how = f" It's done when run_tests on {s.test} passes." if s.test else ""
        return f"Next: step {i}, {s.step}.{how}"

    def to_json(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


def proven(test: str, tb: Toolbox) -> tuple[bool, str]:
    """Has `test` passed, with nothing edited since? (ok, why not)"""
    p = tb.ws.root / test
    if not p.exists():
        return False, f"{test} doesn't exist yet; write the test first"
    runs = [r for r in tb.test_runs if r.covers(test)]
    if not runs:
        return False, f"run_tests hasn't been run on {test} yet"
    last = runs[-1]
    if not last.passed:
        return False, f"the last run_tests covering {test} (on {last.path}) didn't pass"
    if last.writes != tb.writes_snapshot():
        return False, f"files changed after {test} last passed; run it again"
    return True, ""


def stale(plan: Plan, tb: Toolbox) -> list[str]:
    """Steps marked done on a passing test that no longer holds (files were
    edited after it passed, or a later run covering it failed)."""
    return [f"{s.step} ({proven(s.test, tb)[1]})" for s in plan.steps
            if s.status == "done" and s.test and not proven(s.test, tb)[0]]


def _parse(raw, tb: Toolbox) -> list[Step]:
    if isinstance(raw, dict):  # {"items": ...} nested once more than asked
        raw = raw.get("items", raw.get("steps"))
    if not isinstance(raw, list) or not raw:
        raise ToolError("items must be a non-empty list of steps, each "
                        '{"step": "...", "status": "pending", "test": "optional/test/path"}')
    if len(raw) > MAX_STEPS:
        raise ToolError(f"a plan has at most {MAX_STEPS} steps; combine some")
    steps = []
    for n, r in enumerate(raw, 1):
        if isinstance(r, str):
            r = {"step": r}
        if not isinstance(r, dict):
            raise ToolError(f"step {n} must be an object with a 'step' field")
        text = " ".join(str(r.get("step") or r.get("title") or "").split())
        if not text:
            raise ToolError(f"step {n} has no 'step' text")
        status = str(r.get("status") or "pending").strip().lower().replace(" ", "_")
        status = _ALIASES.get(status, status)
        if status not in STATUSES:
            raise ToolError(f"step {n} has status {r.get('status')!r}; use one of "
                            f"{', '.join(STATUSES)}")
        test = r.get("test")
        rel = None
        if test not in (None, "", "none", "null"):
            if not isinstance(test, str):
                raise ToolError(f"step {n}: test must be a path, like tests/test_x.py")
            rel = tb.ws.rel(tb.ws.resolve(test.split("::")[0]))
        steps.append(Step(text, status, rel))
    if sum(s.status == "in_progress" for s in steps) > 1:
        raise ToolError("only one step can be in_progress at a time")
    return steps


def update(plan: Plan, items, tb: Toolbox) -> str:
    """The update_plan tool: replace the plan, checking every step marked done."""
    steps = _parse(items, tb)
    # A step proven earlier stays proven when it's sent again unchanged; later
    # edits for later steps don't reopen it (stale() reports those at the end).
    before = {(s.step, s.test) for s in plan.steps if s.passed}
    refused = []
    for n, s in enumerate(steps, 1):
        if s.status != "done" or not s.test:
            continue
        if (s.step, s.test) in before:
            s.passed = True
            continue
        ok, why = proven(s.test, tb)
        if ok:
            s.passed = True
        else:
            busy = any(o.status == "in_progress" for o in steps)
            s.status = "pending" if busy else "in_progress"
            refused.append(f"step {n} is still open: {why}.")
    kept = {s.step for s in steps}
    for old in plan.steps:
        if old.status != "done" and old.step not in kept and old.step not in plan.dropped:
            plan.dropped.append(old.step)
    plan.dropped = [d for d in plan.dropped if d not in kept]  # put back
    plan.steps = steps
    out = ["Plan updated:", plan.render()]
    if refused:
        out += ["", "Not marked done:"] + [f"- {r}" for r in refused]
    out += ["", plan.next_line()]
    return "\n".join(out)


SCHEMA = _fn(
    "update_plan",
    "Write or update your plan for the task: a short ordered list of steps. Send the whole "
    "list each time. Mark a step in_progress when you start it and done when it's finished. "
    "Give a step a test (a test file) when a test can show it works; that step can only be "
    "marked done after run_tests on that file has passed, with no edits since.",
    {"items": {"type": "array", "description": f"The steps, in order (at most {MAX_STEPS})",
               "items": {"type": "object", "properties": {
                   "step": {"type": "string", "description": "What to do, in a few words"},
                   "status": {"type": "string", "enum": list(STATUSES)},
                   "test": {"type": "string", "description": "Test file that proves this step "
                                                             "works (optional)"}},
                   "required": ["step", "status"]}}},
    ["items"])
