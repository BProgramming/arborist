"""The agent loop: send the conversation, run any tool calls, repeat.

One user message = one turn. Within a turn the model may call tools many
times (up to max_steps), with one hard rule: once code has run (or the user
refused a run), the model gets exactly one more reply, with tools switched
off, to explain the result. Then the turn ends, so nothing can be run or
edited on the strength of that output without you saying so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol

from tools import ToolResult, Toolbox

TOOL_GUIDE = """\
You are working in the directory {root} on the user's computer. Tools let you \
read, search, edit and create files there, and run a single .py, .rs, .dart, \
.js or .ts file.

Rules:
- Paths are relative to {root}; nothing outside it is reachable.
- Read a file before editing it. Make focused changes with edit_file; use write_file for new files.
- Don't re-read a file you already have unless it changed.
- To run code: call check_dependencies first. If anything is missing, tell the user what to \
install and stop. If it passes, call run_file. The user approves every run.
- After run_file, explain the output (did it work, what errors mean, what you'd change) and \
stop. Don't edit or run anything else in that turn; the user decides what happens next.
"""


class AgentUI(Protocol):
    def reasoning(self, text: str) -> None: ...
    def text(self, text: str) -> None: ...
    def end_message(self) -> None: ...
    def tool_call(self, name: str, arguments: str) -> None: ...
    def tool_result(self, name: str, result: ToolResult) -> None: ...
    def notice(self, message: str) -> None: ...


@dataclass
class StreamResult:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    text: str
    steps: int
    hard_stopped: bool
    timings: dict
    usage: dict


def consume_stream(chunks: Iterable[dict], ui: AgentUI) -> StreamResult:
    """Fold OpenAI-style streaming chunks into text plus complete tool calls.

    Tool-call arguments arrive as JSON fragments keyed by index; they're only
    parseable once the stream ends.
    """
    res = StreamResult()
    text: list[str] = []
    calls: dict[int, dict] = {}
    for chunk in chunks:
        res.timings = chunk.get("timings") or res.timings
        res.usage = chunk.get("usage") or res.usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                ui.reasoning(delta["reasoning_content"])
            if delta.get("content"):
                text.append(delta["content"])
                ui.text(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", len(calls)),
                                        {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                res.finish_reason = choice["finish_reason"]
    res.text = "".join(text)
    for i, idx in enumerate(sorted(calls)):
        c = calls[idx]
        c["id"] = c["id"] or f"call_{i}"
        res.tool_calls.append(c)
    return res


def run_turn(messages: list[dict],
             make_body: Callable[[list[dict], bool], dict],
             stream: Callable[[dict], Iterator[dict]],
             toolbox: Toolbox | None,
             ui: AgentUI,
             max_steps: int = 25) -> TurnResult:
    """Run one user turn to completion, appending to `messages` as it goes.

    `messages` should already end with the user's message. The caller keeps
    a snapshot so it can roll back if the turn is interrupted.
    """
    tools_on = toolbox is not None
    steps = 0
    hard_stopped = False
    while True:
        res = consume_stream(stream(make_body(messages, tools_on)), ui)
        ui.end_message()

        if not (tools_on and res.tool_calls):
            messages.append({"role": "assistant", "content": res.text})
            return TurnResult(res.text, steps, hard_stopped, res.timings, res.usage)

        messages.append({
            "role": "assistant",
            "content": res.text,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                           for c in res.tool_calls],
        })
        for call in res.tool_calls:
            if hard_stopped:
                # The model batched more calls after a run in the same reply.
                result = ToolResult("skipped: code already ran this turn, so no further tools "
                                    "run until the user replies. Explain the run's output instead.")
            else:
                ui.tool_call(call["name"], call["arguments"])
                result = toolbox.call(call["name"], call["arguments"])
                ui.tool_result(call["name"], result)
                hard_stopped = hard_stopped or result.hard_stop
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.content})

        steps += 1
        if hard_stopped:
            tools_on = False
        elif steps >= max_steps:
            tools_on = False
            ui.notice(f"reached {max_steps} tool steps; asking the model to wrap up")
            messages.append({"role": "user", "content":
                             f"(You've used {max_steps} tool steps. Stop using tools and summarise "
                             "what you did and what's left.)"})
