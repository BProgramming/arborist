"""Keep a long conversation inside the context window.

Runs before every request. When the request is estimated to pass
`threshold` of the window, it shrinks the conversation in tiers, cheapest
and most faithful first, stopping as soon as it's under `target`:

1. Old tool output. Tool results are most of the tokens in a coding session
   and the least worth keeping verbatim: anything outside the newest
   `keep_tool_results` becomes a one-line stub saying which call produced
   it, and big file contents in old write/edit calls are dropped too.
   Files whose contents leave the conversation this way must be read again
   before they can be edited.
2. Old turns. Everything before the last `keep_turns` user messages is
   replaced by a structured summary (goal, what's done, decisions, state,
   open issues) written by the model, plus a list of files changed, which is
   taken from the tool log rather than trusted to the summary. If the model
   can't summarise, a plain list of the earlier requests is used instead.
3. If that still isn't enough (one enormous turn), the newest tool results
   are cut down to the last two.

Token counts are estimated from characters, with the ratio recalibrated
from the server's reported prompt size after every request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

STUB_PREFIX = "[compacted]"
SUMMARY_TAG = "earlier_conversation_summary"
_BIG_ARGS = ("content", "old_string", "new_string")
_KEEP_SHORT = 300  # results shorter than this ("updated a.py (+3 -1)") are kept

SUMMARY_INSTRUCTIONS = """\
Above is the earlier part of a coding session between a user and an AI \
assistant working in their project. The assistant will continue the session \
with only your summary in place of it, so write what it needs to carry on, \
under these headings:

Goal: what the user wants overall.
Done so far: what was changed, found or learned. Use exact file paths, \
function names, commands and error messages.
Decisions: choices that were made and why, including anything the user \
approved, refused or asked for.
Current state: where the work stood at the end of the transcript.
Open issues: unfixed errors, unanswered questions, agreed next steps.

Only include what the transcript supports. Plain text, at most about 400 words.\
"""


@dataclass
class CompactionSettings:
    enabled: bool = True
    threshold: float = 0.7      # compact when a request would pass this share of the window
    target: float = 0.5         # ...and stop once it's below this
    keep_turns: int = 2         # recent user turns never summarised (the current one included)
    keep_tool_results: int = 8  # recent tool results never stubbed in tier 1

    @classmethod
    def from_config(cls, cfg: dict) -> "CompactionSettings":
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


class Compactor:
    def __init__(self, settings: CompactionSettings, context_tokens: int,
                 measure: Callable[[list[dict]], int],
                 summarize: Callable[[str], str],
                 notice: Callable[[str], None] = lambda m: None,
                 forget_file: Callable[[str], None] = lambda p: None):
        """measure(messages) -> characters in the whole request (system prompt
        and tool definitions included); summarize(transcript) -> summary text."""
        self.s = settings
        self.context_tokens = context_tokens
        self.measure = measure
        self.summarize = summarize
        self.notice = notice
        self.forget_file = forget_file
        self.chars_per_token = 3.5  # a fair start for code; corrected by observe()
        self.files_changed: dict[str, None] = {}  # every file changed in compacted turns
        self._sent_chars: int | None = None

    def reset(self) -> None:
        self.files_changed.clear()

    # -- estimates --------------------------------------------------------------

    def tokens(self, messages: list[dict]) -> int:
        return int(self.measure(messages) / self.chars_per_token)

    def observe(self, usage: dict) -> None:
        """Calibrate the chars-per-token estimate against what the server counted."""
        prompt = usage.get("prompt_tokens")
        if self._sent_chars and isinstance(prompt, int) and prompt > 200:
            ratio = self._sent_chars / prompt
            if 1.0 <= ratio <= 10.0:
                self.chars_per_token = 0.5 * self.chars_per_token + 0.5 * ratio

    def _pct(self, tokens: int) -> str:
        return f"{100 * tokens / self.context_tokens:.0f}%"

    # -- entry point --------------------------------------------------------------

    def maybe_compact(self, messages: list[dict], force: bool = False) -> bool:
        """Shrink `messages` in place if needed. Returns True if anything changed."""
        before = self.tokens(messages)
        limit = self.s.threshold * self.context_tokens
        goal = self.s.target * self.context_tokens
        if not force and (not self.s.enabled or before <= limit):
            self._sent_chars = self.measure(messages)
            return False

        self.notice(f"context ≈{self._pct(before)} full (~{before:,} of {self.context_tokens:,} "
                    "tokens): compacting")
        changed = False
        n = self._stub_tool_results(messages, self.s.keep_tool_results)
        if n:
            changed = True
            self.notice(f"  dropped {n} old tool output{'s' * (n != 1)} "
                        f"(now ≈{self._pct(self.tokens(messages))})")
        if force or self.tokens(messages) > goal:
            turns = self._summarize_old_turns(messages)
            if turns:
                changed = True
                self.notice(f"  summarised {turns} earlier turn{'s' * (turns != 1)} "
                            f"(now ≈{self._pct(self.tokens(messages))})")
        if self.tokens(messages) > goal:
            n = self._stub_tool_results(messages, 2)
            if n:
                changed = True
                self.notice(f"  dropped {n} more tool output{'s' * (n != 1)} from this turn")
        after = self.tokens(messages)
        if after > limit:
            self.notice(f"  still ≈{self._pct(after)} full; consider /save and /clear")
        self._sent_chars = self.measure(messages)
        return changed

    # -- tier 1: stub old tool results ------------------------------------------

    def _stub_tool_results(self, messages: list[dict], keep: int) -> int:
        calls = _calls_by_id(messages)
        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = tool_idx[:-keep] if keep else tool_idx
        count = 0
        for i in old:
            m = messages[i]
            content = m.get("content") or ""
            if content.startswith(STUB_PREFIX) or len(content) <= _KEEP_SHORT:
                continue
            name, args = calls.get(m.get("tool_call_id"), ("tool", {}))
            messages[i] = {**m, "content": f"{STUB_PREFIX} The output of {_describe(name, args)} "
                                           "was removed to save context. Call it again if you "
                                           "need it."}
            if name == "read_file" and isinstance(args.get("path"), str):
                self.forget_file(args["path"])
            count += 1
        # Big arguments (whole files in write_file, long edits) in old calls.
        old_ids = {messages[i].get("tool_call_id") for i in old}
        for i, m in enumerate(messages):
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            new_calls, touched = [], False
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                if tc.get("id") in old_ids:
                    slim = _slim_arguments(fn.get("arguments") or "")
                    if slim is not None:
                        tc = {**tc, "function": {**fn, "arguments": slim}}
                        touched = True
                        args = _parse(fn.get("arguments"))
                        if isinstance(args.get("path"), str):
                            self.forget_file(args["path"])
                new_calls.append(tc)
            if touched:
                messages[i] = {**m, "tool_calls": new_calls}
        return count

    # -- tier 2: summarise old turns --------------------------------------------

    def _summarize_old_turns(self, messages: list[dict]) -> int:
        starts = turn_starts(messages)
        if len(starts) <= self.s.keep_turns:
            return 0
        cut = starts[len(starts) - self.s.keep_turns]
        old, kept = messages[:cut], messages[cut:]

        self._collect_changed_files(old)
        for name, args in _calls_by_id(old).values():
            if name in ("read_file", "edit_file", "write_file") and isinstance(args.get("path"), str):
                self.forget_file(args["path"])

        # The summary request has to fit in the window too.
        budget = int(self.context_tokens * 0.45 * self.chars_per_token)
        transcript = _clip(render_transcript(old), budget)
        try:
            summary = (self.summarize(transcript) or "").strip()
        except Exception as e:  # the server may be busy, slow or broken
            self.notice(f"  couldn't summarise ({e}); keeping a plain list of earlier requests")
            summary = ""
        if not summary:
            summary = mechanical_summary(old)

        files = ""
        if self.files_changed:
            files = "\n\nFiles changed earlier in this session: " + ", ".join(self.files_changed)
        block = (f"<{SUMMARY_TAG}>\n(Earlier turns of this conversation were compressed to save "
                 "context. Read a file again before editing it.)\n\n"
                 f"{summary}{files}\n</{SUMMARY_TAG}>")
        first = kept[0]
        messages[:] = [{**first, "content": f"{block}\n\n{first.get('content') or ''}"}] + kept[1:]
        return len(starts) - self.s.keep_turns

    def _collect_changed_files(self, old: list[dict]) -> None:
        results = {m.get("tool_call_id"): m.get("content") or "" for m in old if m.get("role") == "tool"}
        for cid, (name, args) in _calls_by_id(old).items():
            if name in ("edit_file", "write_file") and isinstance(args.get("path"), str):
                r = results.get(cid, "")
                if r.startswith(("created ", "updated ")):
                    self.files_changed[args["path"]] = None


# ------------------------------------------------------------------ helpers ---

def turn_starts(messages: list[dict]) -> list[int]:
    """Indices of the user's own messages. The harness's notes to the model
    (step limit, pause) are user messages too, but they always follow a tool
    result, so they don't start a turn."""
    return [i for i, m in enumerate(messages)
            if m.get("role") == "user" and (i == 0 or messages[i - 1].get("role") != "tool")]


def _parse(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def _calls_by_id(messages: list[dict]) -> dict[str, tuple[str, dict]]:
    out = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            out[tc.get("id")] = (fn.get("name") or "tool", _parse(fn.get("arguments")))
    return out


def _describe(name: str, args: dict) -> str:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items()
                      if k not in _BIG_ARGS and not (isinstance(v, str) and len(v) > 80))
    return f"{name}({shown})"


def _slim_arguments(raw: str) -> str | None:
    """The call's JSON with big text fields replaced by a line count, or None
    if there's nothing big to drop."""
    args = _parse(raw)
    if not any(isinstance(args.get(k), str) and len(args[k]) > _KEEP_SHORT for k in _BIG_ARGS):
        return None
    slim = {k: (f"{STUB_PREFIX} <{len(v.splitlines()) or 1} lines>"
                if k in _BIG_ARGS and isinstance(v, str) and len(v) > _KEEP_SHORT else v)
            for k, v in args.items()}
    return json.dumps(slim)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return text[:head] + f"\n\n[... {len(text) - limit:,} characters of the middle omitted ...]\n\n" + text[-tail:]


def render_transcript(messages: list[dict], result_chars: int = 1500) -> str:
    """The conversation as plain text for the summariser."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            out.append(f"USER:\n{content}")
        elif role == "assistant":
            if content:
                out.append(f"ASSISTANT:\n{content}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                out.append(f"ASSISTANT CALLED: {_describe(fn.get('name') or 'tool', _parse(fn.get('arguments')))}")
        elif role == "tool":
            if len(content) > result_chars:
                content = content[:result_chars] + f"\n[... {len(content) - result_chars:,} more characters]"
            out.append(f"RESULT:\n{content}")
    return "\n\n".join(out)


def mechanical_summary(old: list[dict]) -> str:
    """A summary that needs no model: what the user asked for, in order."""
    asks = []
    for i in turn_starts(old):
        text = old[i].get("content") or ""
        # Skip an older summary block; its file list is carried separately.
        if text.startswith(f"<{SUMMARY_TAG}>") and f"</{SUMMARY_TAG}>" in text:
            text = text.split(f"</{SUMMARY_TAG}>", 1)[1]
        text = " ".join(text.split())
        if text:
            asks.append("- " + (text if len(text) <= 200 else text[:199] + "…"))
    return "Earlier requests from the user, oldest first:\n" + "\n".join(asks or ["- (none recorded)"])
