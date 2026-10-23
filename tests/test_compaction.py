"""Context compaction: stubbing old tool output, summarising old turns."""
from __future__ import annotations

import json

import compaction
from compaction import CompactionSettings, Compactor


def call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def conversation(turns: int, result_size: int = 4000) -> list[dict]:
    """`turns` user turns, each: read a file, then answer."""
    msgs = []
    for t in range(turns):
        msgs += [
            {"role": "user", "content": f"request {t}"},
            {"role": "assistant", "content": "", "tool_calls": [call(f"r{t}", "read_file", path=f"f{t}.py")]},
            {"role": "tool", "tool_call_id": f"r{t}", "content": f"contents of f{t}\n" + "x" * result_size},
            {"role": "assistant", "content": f"answer {t}"},
        ]
    return msgs


def chars(messages):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in messages)


def make(context_tokens=10_000, summary="Goal: things.\nDone so far: stuff.", **settings):
    log = {"notices": [], "forgot": [], "transcripts": []}

    def summarize(text):
        log["transcripts"].append(text)
        if isinstance(summary, Exception):
            raise summary
        return summary

    c = Compactor(CompactionSettings(**settings), context_tokens, measure=chars, summarize=summarize,
                  notice=log["notices"].append, forget_file=log["forgot"].append)
    c.chars_per_token = 1.0  # make the arithmetic obvious: 1 char = 1 token
    return c, log


def test_below_the_threshold_nothing_happens():
    c, log = make(context_tokens=100_000)
    msgs = conversation(3)
    before = json.dumps(msgs)
    assert c.maybe_compact(msgs) is False
    assert json.dumps(msgs) == before and not log["notices"]


def test_tier_one_stubs_old_tool_output_and_forgets_those_files():
    msgs = conversation(5)              # ~20k "tokens"
    c, log = make(context_tokens=26_000, keep_tool_results=2)  # 70% = 18.2k: over; 50% = 13k
    assert c.maybe_compact(msgs) is True
    tool = [m for m in msgs if m["role"] == "tool"]
    assert [m["content"].startswith(compaction.STUB_PREFIX) for m in tool] == [True, True, True, False, False]
    assert "read_file(path='f0.py')" in tool[0]["content"]
    assert log["forgot"] == ["f0.py", "f1.py", "f2.py"]
    assert not log["transcripts"]  # tier 1 was enough: no summary
    assert "dropped 3 old tool outputs" in log["notices"][1]


def test_tier_two_summarises_old_turns_into_the_first_kept_message():
    msgs = conversation(6, result_size=200)  # small results: tier 1 can't help
    msgs[-1]["content"] = "answer 5 " + "y" * 3000
    c, log = make(context_tokens=5_000, keep_turns=2)
    assert c.maybe_compact(msgs) is True
    starts = compaction.turn_starts(msgs)
    assert len(starts) == 2  # only the kept turns remain
    first = msgs[0]
    assert first["role"] == "user" and first["content"].startswith(f"<{compaction.SUMMARY_TAG}>")
    assert "Goal: things." in first["content"] and first["content"].endswith("request 4")
    # The summariser saw the old turns, and not the kept ones.
    assert "request 0" in log["transcripts"][0] and "request 4" not in log["transcripts"][0]
    assert {"f0.py", "f3.py"} <= set(log["forgot"])
    # History stays well-formed: every tool result still follows its call.
    ids = {tc["id"] for m in msgs for tc in m.get("tool_calls") or []}
    assert all(m["tool_call_id"] in ids for m in msgs if m["role"] == "tool")


def test_changed_files_come_from_the_tool_log_not_the_summary():
    msgs = [
        {"role": "user", "content": "add logging"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("e1", "edit_file", path="app.py", old_string="a", new_string="b"),
            call("e2", "write_file", path="log.py", content="x"),
            call("e3", "edit_file", path="nope.py", old_string="a", new_string="b")]},
        {"role": "tool", "tool_call_id": "e1", "content": "updated app.py (+1 -1)"},
        {"role": "tool", "tool_call_id": "e2", "content": "created log.py (+1 -0)"},
        {"role": "tool", "tool_call_id": "e3", "content": "error: old_string was not found"},
        {"role": "assistant", "content": "done"},
    ] + conversation(2, result_size=10)
    c, _ = make(summary="Goal: logging.", keep_turns=2)
    c.maybe_compact(msgs, force=True)
    assert "Files changed earlier in this session: app.py, log.py" in msgs[0]["content"]


def test_a_failed_summary_falls_back_to_the_list_of_requests():
    msgs = conversation(4, result_size=10)
    c, log = make(summary=RuntimeError("server gone"), keep_turns=1)
    c.maybe_compact(msgs, force=True)
    assert "Earlier requests from the user, oldest first:\n- request 0\n- request 1\n- request 2" in msgs[0]["content"]
    assert any("couldn't summarise (server gone)" in n for n in log["notices"])


def test_harness_notes_are_not_turns():
    assert compaction.turn_starts(conversation(2)) == [0, 4]
    # The step-limit and pause notes are user messages, but always follow a tool result.
    note_after_tool = [{"role": "user", "content": "go"},
                       {"role": "assistant", "content": "", "tool_calls": [call("a", "grep", pattern="x")]},
                       {"role": "tool", "tool_call_id": "a", "content": "no matches"},
                       {"role": "user", "content": "(Tool use is paused…)"}]
    assert compaction.turn_starts(note_after_tool) == [0]


def test_big_write_arguments_are_slimmed_too():
    big = "line\n" * 500
    msgs = [{"role": "user", "content": "make it"},
            {"role": "assistant", "content": "", "tool_calls": [call("w", "write_file", path="big.py", content=big)]},
            {"role": "tool", "tool_call_id": "w", "content": "created big.py (+500 -0)"},
            {"role": "assistant", "content": "made it"}] + conversation(1, result_size=10)
    c, log = make(keep_tool_results=1)
    c._stub_tool_results(msgs, keep=1)
    args = json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "big.py", "content": f"{compaction.STUB_PREFIX} <500 lines>"}
    assert msgs[2]["content"] == "created big.py (+500 -0)"  # short results stay
    assert "big.py" in log["forgot"]


def test_messages_are_replaced_not_mutated():
    msgs = conversation(5)
    originals = [dict(m) for m in msgs]
    c, _ = make(context_tokens=26_000, keep_tool_results=2)
    snapshot = list(msgs)
    c.maybe_compact(msgs)
    assert [dict(m) for m in snapshot] == originals  # the caller's rollback copy is intact


def test_observe_calibrates_the_ratio():
    c, _ = make()
    c.chars_per_token = 3.5
    c._sent_chars = 20_000
    c.observe({"prompt_tokens": 10_000})  # this model: 2 chars/token
    assert c.chars_per_token == 2.75      # moves halfway
    c.observe({"prompt_tokens": 10})      # too small to trust
    assert c.chars_per_token == 2.75


def test_disabled_compaction_still_honours_force():
    msgs = conversation(4, result_size=10)
    c, _ = make(context_tokens=10, enabled=False, keep_turns=1)
    assert c.maybe_compact(msgs) is False
    assert c.maybe_compact(msgs, force=True) is True
