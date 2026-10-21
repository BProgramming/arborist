"""The agent loop, driven by a scripted fake model (no server needed)."""
from __future__ import annotations

import json

import agent
from conftest import FakeApprover, write
from tools import Decision


class NullUI:
    def __init__(self):
        self.calls, self.notices, self.texts = [], [], []

    def reasoning(self, text): pass
    def text(self, text): self.texts.append(text)  # noqa: E704
    def end_message(self): pass
    def tool_call(self, name, arguments): self.calls.append(name)  # noqa: E704
    def tool_result(self, name, result): pass
    def notice(self, message): self.notices.append(message)  # noqa: E704


def text_reply(text):
    return [{"choices": [{"delta": {"content": text}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}]


def tool_reply(*calls):
    """A reply that calls tools; each call's arguments arrive in two fragments."""
    chunks = []
    for i, (name, args) in enumerate(calls):
        raw = json.dumps(args)
        cut = len(raw) // 2
        chunks.append({"choices": [{"delta": {"tool_calls": [
            {"index": i, "id": f"id{i}", "type": "function", "function": {"name": name, "arguments": raw[:cut]}}]}}]})
        chunks.append({"choices": [{"delta": {"tool_calls": [
            {"index": i, "function": {"arguments": raw[cut:]}}]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    return chunks


class FakeModel:
    """Plays back one scripted reply per request and records each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def make_body(self, messages, with_tools):
        return {"messages": json.loads(json.dumps(messages)), "with_tools": with_tools}

    def stream(self, body):
        self.requests.append(body)
        reply = self.replies.pop(0) if self.replies else text_reply("(out of script)")
        yield from reply


def run(model, toolbox, ui=None, max_steps=25, prompt="go"):
    messages = [{"role": "user", "content": prompt}]
    result = agent.run_turn(messages, model.make_body, model.stream, toolbox, ui or NullUI(), max_steps)
    return result, messages


# ------------------------------------------------------------- streaming ---

def test_consume_stream_reassembles_fragmented_tool_calls():
    res = agent.consume_stream(tool_reply(("read_file", {"path": "a.py"}), ("grep", {"pattern": "x"})), NullUI())
    assert [(c["id"], c["name"], json.loads(c["arguments"])) for c in res.tool_calls] == [
        ("id0", "read_file", {"path": "a.py"}), ("id1", "grep", {"pattern": "x"})]
    assert res.finish_reason == "tool_calls"


def test_consume_stream_fills_in_missing_ids_and_keeps_stats():
    chunks = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "list_files", "arguments": "{}"}}]}}]},
              {"choices": [], "usage": {"total_tokens": 99}, "timings": {"predicted_n": 5}}]
    res = agent.consume_stream(chunks, NullUI())
    assert res.tool_calls[0]["id"] == "call_0"
    assert res.usage == {"total_tokens": 99} and res.timings == {"predicted_n": 5}


# ------------------------------------------------------------------ loop ---

def test_plain_answer_without_tools():
    model = FakeModel(text_reply("hello"))
    result, messages = run(model, None)
    assert result.text == "hello" and result.steps == 0
    assert messages[-1] == {"role": "assistant", "content": "hello"}
    assert model.requests[0]["with_tools"] is False


def test_tool_results_are_fed_back_until_the_model_answers(ws_root, make_toolbox):
    write(ws_root, "a.py", "x = 1\n")
    model = FakeModel(tool_reply(("read_file", {"path": "a.py"})), text_reply("it sets x"))
    result, messages = run(model, make_toolbox())
    assert result.text == "it sets x" and result.steps == 1
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["id"] == "id0"
    assert messages[2]["tool_call_id"] == "id0" and "x = 1" in messages[2]["content"]
    # The second request carried the tool result back to the model.
    assert model.requests[1]["messages"][-1]["role"] == "tool"


def test_edit_flow_end_to_end(ws_root, make_toolbox):
    write(ws_root, "a.py", "greeting = 'hi'\n")
    model = FakeModel(
        tool_reply(("read_file", {"path": "a.py"})),
        tool_reply(("edit_file", {"path": "a.py", "old_string": "'hi'", "new_string": "'hello'"})),
        text_reply("done"))
    result, _ = run(model, make_toolbox())
    assert (ws_root / "a.py").read_text() == "greeting = 'hello'\n" and result.steps == 2


# ------------------------------------------------------------- hard stop ---

def test_hard_stop_after_a_run(ws_root, make_toolbox):
    write(ws_root, "hello.py", "print('ran ok')\n")
    write(ws_root, "other.py", "untouched = True\n")
    ui = NullUI()
    model = FakeModel(
        tool_reply(("read_file", {"path": "other.py"})),
        tool_reply(("check_dependencies", {"path": "hello.py"})),
        # Run, then try to sneak in an edit in the same reply.
        tool_reply(("run_file", {"path": "hello.py"}),
                   ("edit_file", {"path": "other.py", "old_string": "True", "new_string": "False"})),
        # Tools are off now; even if the model emits a call it must not run.
        tool_reply(("write_file", {"path": "evil.py", "content": "x"})),
        text_reply("should never be requested"))
    result, messages = run(model, make_toolbox(), ui)

    assert result.hard_stopped
    assert ui.calls == ["read_file", "check_dependencies", "run_file"]  # edit never dispatched
    assert (ws_root / "other.py").read_text() == "untouched = True\n"
    assert not (ws_root / "evil.py").exists()
    skipped = [m for m in messages if m["role"] == "tool" and m["content"].startswith("skipped")]
    assert len(skipped) == 1
    # Exactly one request after the run, with tools switched off, then the turn ends.
    assert len(model.requests) == 4
    assert [r["with_tools"] for r in model.requests] == [True, True, True, False]
    assert messages[-1]["role"] == "assistant"


def test_declined_run_also_ends_tool_use(ws_root, make_toolbox):
    write(ws_root, "hello.py", "print('x')\n")
    tb = make_toolbox(FakeApprover(run=[Decision(False)]))
    model = FakeModel(
        tool_reply(("check_dependencies", {"path": "hello.py"})),
        tool_reply(("run_file", {"path": "hello.py"})),
        text_reply("ok, I won't run it"))
    result, _ = run(model, tb)
    assert result.hard_stopped and result.text == "ok, I won't run it"
    assert model.requests[-1]["with_tools"] is False


def test_a_failed_check_does_not_trigger_the_hard_stop(ws_root, make_toolbox):
    write(ws_root, "needs.py", "import not_installed_pkg_xyz\n")
    model = FakeModel(
        tool_reply(("check_dependencies", {"path": "needs.py"})),
        text_reply("install not_installed_pkg_xyz first"))
    result, _ = run(model, make_toolbox())
    assert not result.hard_stopped and "install" in result.text


def test_step_limit_forces_a_wrap_up(ws_root, make_toolbox):
    write(ws_root, "a.py", "x\n")
    ui = NullUI()
    model = FakeModel(*[tool_reply(("read_file", {"path": "a.py"}))] * 3, text_reply("summary"))
    result, messages = run(model, make_toolbox(), ui, max_steps=3)
    assert result.steps == 3 and result.text == "summary"
    assert model.requests[-1]["with_tools"] is False
    assert ui.notices and "3 tool steps" in ui.notices[0]
