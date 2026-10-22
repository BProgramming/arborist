#!/usr/bin/env python3
"""Terminal chat client for a local llama-server running Bonsai 2 27B.

The heavy lifting (inference) happens in the C++ server; this is just I/O
plus the tools the model can call on files in your workspace (the directory
you start it from, or --workspace).

Standard library only, so there's nothing to pip install.
Type /help at the prompt for commands.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    import readline  # noqa: F401  (line editing + history for input())
except ImportError:  # pragma: no cover
    readline = None

import agent
import config as model_config
from tools import Decision, Limits, ToolResult, Toolbox, Workspace, TOOL_SCHEMAS

# ---------------------------------------------------------------- terminal ---

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def dim(t: str) -> str: return _c("2", t)
def bold(t: str) -> str: return _c("1", t)
def cyan(t: str) -> str: return _c("36", t)
def red(t: str) -> str: return _c("31", t)
def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)


# Readline needs non-printing escapes wrapped in \001..\002 or the cursor drifts.
PROMPT = "\001\033[1;32m\002› \001\033[0m\002" if USE_COLOR else "> "
CONT_PROMPT = "\001\033[2m\002… \001\033[0m\002" if USE_COLOR else "... "

HELP = """\
Commands:
  /help                 this list
  /exit, /quit          leave (Ctrl-D works too)
  /clear                forget the conversation (keeps the system prompt)
  /effort none|medium|xhigh
                        how long the model thinks before answering
  /think                toggle showing the model's thinking
  /tools                toggle file, run and web tools for this session
  /file PATH [PATH…]    attach file contents to your next message
  /system TEXT          replace the system prompt (clears the conversation)
  /retry                regenerate the last reply (file changes are not undone)
  /save [PATH]          write the conversation to a Markdown file
  /stats                context usage and last-reply speed

Input:
  \"\"\"                   on its own line starts/ends a multi-line block
  Ctrl-C                stops the current reply, tool or run (that turn is discarded)
  A line starting with / is a command only if its first word has no other /,
  so pasting a path like /etc/hosts sends it as a message. Use // to send a
  message that starts with a command word (//help me → "/help me").
"""


class Interrupted(BaseException):
    """Ctrl-C during a turn. A BaseException (like KeyboardInterrupt) so that
    the catch-all error handling in tools and turns lets it through."""


# ------------------------------------------------------------------ client ---

class Client:
    def __init__(self, base_url: str):
        u = urlparse(base_url)
        self.host, self.port = u.hostname, u.port

    def _conn(self, timeout: float | None) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def healthy(self) -> bool:
        try:
            c = self._conn(2)
            c.request("GET", "/health")
            r = c.getresponse()
            r.read()
            return r.status == 200
        except OSError:
            return False

    def stream_chat(self, body: dict):
        """Yield parsed SSE chunks from /v1/chat/completions."""
        # No read timeout: a long think can go quiet between visible tokens.
        conn = self._conn(None)
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read().decode(errors="replace")
                try:
                    detail = json.loads(detail)["error"]["message"]
                except Exception:
                    pass
                raise RuntimeError(f"server returned HTTP {resp.status}: {detail}")
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    raise RuntimeError(f"malformed data from the server: {data[:200]!r}") from None
        finally:
            # Closing mid-stream tells llama-server to stop generating.
            conn.close()


# -------------------------------------------------------------- terminal UI ---

def _short(value, n: int = 60) -> str:
    s = value if isinstance(value, str) else json.dumps(value)
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def describe_call(name: str, arguments: str) -> str:
    """One readable line for a tool call, without dumping file contents."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return f"{name}({_short(arguments)})"
    parts = []
    for k, v in args.items():
        if k in ("content", "old_string", "new_string") and isinstance(v, str):
            parts.append(f"{k}=<{len(v.splitlines()) or 1} lines>")
        else:
            parts.append(f"{k}={_short(v, 50)}")
    return f"{name}({', '.join(parts)})"


class TerminalUI:
    """Renders the stream and tool activity (agent.AgentUI)."""

    def __init__(self, show_thinking: bool):
        self.show_thinking = show_thinking
        self._reset()

    def _reset(self):
        self.started = time.monotonic()
        self.thinking_chars = 0
        self.thinking_open = False
        self.status_shown = False
        self.wrote_text = False

    def _clear_status(self):
        if self.status_shown:
            sys.stdout.write("\r\033[K" if USE_COLOR else "\n")
            self.status_shown = False

    def reasoning(self, text: str) -> None:
        self.thinking_chars += len(text)
        if self.show_thinking:
            if not self.thinking_open:
                sys.stdout.write(dim("thinking: "))
                self.thinking_open = True
            sys.stdout.write(dim(text))
        else:
            secs = time.monotonic() - self.started
            sys.stdout.write(f"\r{dim(f'thinking… {secs:4.0f}s, ~{self.thinking_chars // 4} tokens')}")
            self.status_shown = True
        sys.stdout.flush()

    def text(self, text: str) -> None:
        self._clear_status()
        if self.thinking_open:
            sys.stdout.write("\n\n")
            self.thinking_open = False
        self.wrote_text = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_message(self) -> None:
        self._clear_status()
        if self.wrote_text or self.thinking_open:
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._reset()

    def tool_call(self, name: str, arguments: str) -> None:
        print(cyan("● ") + describe_call(name, arguments))

    def tool_result(self, name: str, result: ToolResult) -> None:
        lines = result.content.splitlines() or [""]
        first = _short(lines[0], 100)
        if result.content.startswith("error:"):
            print("  " + red(first))
        elif name == "run_file":
            pass  # the output already streamed live
        else:
            extra = f"  (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
            print("  " + dim(first + extra))

    def notice(self, message: str) -> None:
        print(yellow(message))

    def run_output(self, line: str) -> None:
        print(dim("  │ ") + line)


class TerminalApprover:
    """Asks before anything changes on disk or runs (tools.Approver)."""

    MAX_DIFF_LINES = 150

    def approve_change(self, rel_path: str, diff: str, is_new: bool) -> Decision:
        print()
        print(bold(("Create " if is_new else "Edit ") + rel_path))
        lines = diff.splitlines()
        for line in lines[: self.MAX_DIFF_LINES]:
            if line.startswith(("+++", "---")):
                print(dim(line))
            elif line.startswith("+"):
                print(green(line))
            elif line.startswith("-"):
                print(red(line))
            elif line.startswith("@@"):
                print(cyan(line))
            else:
                print(line)
        if len(lines) > self.MAX_DIFF_LINES:
            print(dim(f"… {len(lines) - self.MAX_DIFF_LINES} more diff lines not shown"))
        while True:
            ans = input("Apply? [y]es / [n]o / [a]ll edits this session: ").strip().lower()
            if ans in ("y", "yes"):
                return Decision(True)
            if ans in ("a", "all"):
                return Decision(True, always=True)
            if ans in ("n", "no"):
                return Decision(False, input("Reason for the model (optional): ").strip())

    def approve_fetch(self, host: str, url: str) -> Decision:
        print()
        print(bold(yellow(f"Allow web access to {host}?")))
        print("  " + url)
        print(dim("  approving covers this site for the rest of the session"))
        ans = input("Fetch? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return Decision(True)
        return Decision(False, input("Reason for the model (optional): ").strip())

    def approve_run(self, rel_path: str, steps: list[list[str]], cwd: str, timeout: int) -> Decision:
        print()
        print(bold(yellow(f"Run {rel_path}?")))
        for step in steps:
            print("  $ " + " ".join(step))
        print(dim(f"  in {cwd}, killed after {timeout}s"))
        ans = input("Run it? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return Decision(True)
        return Decision(False, input("Reason for the model (optional): ").strip())


# -------------------------------------------------------------------- chat ---

class Chat:
    def __init__(self, cfg: dict, workspace: Path | None, tools_enabled: bool):
        self.cfg = cfg
        ch = cfg["chat"]
        self.client = Client(cfg["derived"]["base_url"])
        self.system_prompt = ch["system_prompt"]
        self.effort = ch["reasoning_effort"]
        self.ui = TerminalUI(ch["show_thinking"])
        self.messages: list[dict] = []
        self.pending_files: list[tuple[str, str]] = []
        self.last: agent.TurnResult | None = None
        self.last_wall = 0.0
        self.ctx_used = 0
        self.busy = False
        self.history_dir = Path(cfg["derived"]["history_dir"])
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.toolbox: Toolbox | None = None
        self.tools_enabled = False
        if workspace is not None:
            self.toolbox = Toolbox(
                Workspace(workspace), TerminalApprover(),
                Limits.from_config(cfg["tools"]),
                log_dir=cfg["derived"]["log_dir"],
                on_output=self.ui.run_output,
                on_notice=self.ui.notice,
            )
            self.tools_enabled = tools_enabled

    # -- request -------------------------------------------------------------

    def _system(self) -> str:
        parts = [self.system_prompt] if self.system_prompt else []
        if self.tools_enabled and self.toolbox:
            parts.append(agent.TOOL_GUIDE.format(root=self.toolbox.ws.root))
        return "\n\n".join(parts)

    def _body(self, messages: list[dict], with_tools: bool) -> dict:
        ch = self.cfg["chat"]
        system = self._system()
        body = {
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": ch["max_tokens"],
            "temperature": ch["temperature"],
            "top_p": ch["top_p"],
            "top_k": ch["top_k"],
            "min_p": ch["min_p"],
            "presence_penalty": ch["presence_penalty"],
            "reasoning_effort": self.effort,
            # PERF: reuse the KV cache for the unchanged prefix of the
            # conversation, so each request only processes the new tokens.
            # The tool definitions and system prompt sit in that prefix too.
            "cache_prompt": True,
        }
        if self.tools_enabled and self.toolbox:
            # Tool definitions stay in every request (even with tools switched
            # off for the post-run explanation) so the cached prefix holds;
            # "none" stops the model from calling them.
            body["tools"] = self.toolbox.schemas
            body["tool_choice"] = "auto" if with_tools else "none"
        return body

    def _print_stats(self, compact: bool = False) -> None:
        ctx = self.cfg["server"]["ctx_size"] // self.cfg["server"]["parallel"]
        pct = 100 * self.ctx_used / ctx if ctx else 0
        parts = []
        t = self.last.timings if self.last else {}
        if t.get("predicted_per_second"):
            parts.append(f"{t['predicted_per_second']:.1f} tok/s")
        if t.get("prompt_n") and t.get("prompt_per_second"):
            parts.append(f"prompt {t['prompt_n']} @ {t['prompt_per_second']:.0f} tok/s")
        if self.last and self.last.steps:
            parts.append(f"{self.last.steps} tool step{'s' * (self.last.steps != 1)}")
        if self.last:
            parts.append(f"{self.last_wall:.1f}s")
        parts.append(f"ctx {self.ctx_used:,}/{ctx:,} ({pct:.0f}%)")
        line = " · ".join(parts)
        print(dim(line) if compact else line)
        if pct > 85:
            print(red("context nearly full — /save then /clear soon"))

    # -- turn ----------------------------------------------------------------

    def _reply(self) -> None:
        print()
        snapshot = len(self.messages) - 1  # index of this turn's user message
        started = time.monotonic()
        self.busy = True
        try:
            result = agent.run_turn(
                self.messages, self._body, self.client.stream_chat,
                self.toolbox if self.tools_enabled else None, self.ui,
                max_steps=self.cfg["tools"]["max_steps"],
            )
        except Interrupted:
            self.ui.end_message()
            print(red("[stopped]"))
            self._rollback(snapshot)
        except (RuntimeError, OSError) as e:
            self.ui.end_message()
            print(red(f"error: {e}"))
            if not self.client.healthy():
                print(red("the server isn't responding — check the server log"))
            self._rollback(snapshot)
        except Exception as e:
            # Anything else is a bug here, not in the model. Keep the session.
            self.ui.end_message()
            log = self._log_error()
            print(red(f"internal error: {type(e).__name__}: {e}"))
            print(dim(f"traceback written to {log}; the conversation is intact, minus this turn"))
            self._rollback(snapshot)
        else:
            self.last = result
            self.last_wall = time.monotonic() - started
            if result.usage:
                # Size of the last request (prompt + reply). Slightly high for
                # the next turn, since the template drops old replies' thinking.
                self.ctx_used = result.usage.get("total_tokens", self.ctx_used)
            if result.hard_stopped:
                print(dim("(code ran; tools stay off until your next message)"))
            if self.cfg["chat"]["show_stats"]:
                self._print_stats(compact=True)
            if not result.text.strip():
                print(dim("(empty reply — if thinking ran long, raise chat.max_tokens or lower /effort)"))
        finally:
            self.busy = False
        print()

    def _log_error(self) -> Path | str:
        try:
            path = Path(self.cfg["derived"]["log_dir"]) / "chat-errors.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S}\n{traceback.format_exc()}\n")
            return path
        except OSError:
            return "(couldn't write the log)"

    def _rollback(self, snapshot: int) -> None:
        """Drop the interrupted turn so the history stays well-formed."""
        changed = any(m.get("role") == "tool" for m in self.messages[snapshot:])
        del self.messages[snapshot:]
        if changed:
            print(dim("note: any file changes made during that turn are still on disk"))

    # -- commands ------------------------------------------------------------

    def _attach(self, paths: list[str]) -> None:
        if not paths:
            print(red("usage: /file PATH [PATH…]"))
            return
        for p in paths:
            path = Path(os.path.expanduser(p))
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                print(red(f"can't read {p}: {e}"))
                continue
            self.pending_files.append((str(path), text))
            print(dim(f"attached {path} ({len(text):,} chars, ~{len(text) // 4:,} tokens)"))

    def _save(self, arg: str) -> None:
        path = Path(os.path.expanduser(arg)) if arg else \
            self.history_dir / f"chat-{datetime.now():%Y%m%d-%H%M%S}.md"
        lines = [f"# Agent chat — {datetime.now():%Y-%m-%d %H:%M}", ""]
        if self.system_prompt:
            lines += ["**System:** " + self.system_prompt, ""]
        for m in self.messages:
            role = m["role"]
            if role == "user":
                lines += ["## You", "", m["content"], ""]
            elif role == "assistant":
                if m.get("content"):
                    lines += ["## Agent", "", m["content"], ""]
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    lines += [f"→ `{describe_call(fn['name'], fn['arguments'])}`", ""]
            elif role == "tool":
                body = m["content"]
                if len(body) > 2000:
                    body = body[:2000] + "\n…"
                lines += ["```", body, "```", ""]
        try:
            path.write_text("\n".join(lines), encoding="utf-8")
            print(dim(f"saved {path}"))
        except OSError as e:
            print(red(f"save failed: {e}"))

    def _clear(self) -> None:
        self.messages.clear()
        self.pending_files.clear()
        self.ctx_used = 0
        if self.toolbox:
            # The model forgot what it read, so it has to read files again
            # before editing them.
            self.toolbox.reset()

    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False to exit."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("exit", "quit", "q"):
            return False
        if cmd == "help":
            print(HELP)
        elif cmd == "clear":
            self._clear()
            print(dim("conversation cleared"))
        elif cmd == "effort":
            if arg in model_config.EFFORTS:
                self.effort = arg
                print(dim(f"reasoning effort: {arg}"))
            else:
                print(red(f"usage: /effort {'|'.join(sorted(model_config.EFFORTS))}  (now: {self.effort})"))
        elif cmd == "think":
            self.ui.show_thinking = not self.ui.show_thinking
            print(dim(f"show thinking: {'on' if self.ui.show_thinking else 'off'}"))
        elif cmd == "tools":
            if self.toolbox is None:
                print(red("tools are unavailable in this session (started with --no-tools)"))
            else:
                self.tools_enabled = not self.tools_enabled
                print(dim(f"tools: {'on' if self.tools_enabled else 'off'}  (workspace {self.toolbox.ws.root})"))
        elif cmd == "file":
            self._attach(arg.split())
        elif cmd == "system":
            if not arg:
                print(dim(f"system prompt: {self.system_prompt}"))
            else:
                self.system_prompt = arg
                self._clear()
                print(dim("system prompt replaced; conversation cleared"))
        elif cmd == "retry":
            last_user = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=None)
            if last_user is None or last_user == len(self.messages) - 1:
                print(red("nothing to retry"))
            else:
                self._rollback(last_user + 1)
                self._reply()
        elif cmd == "save":
            self._save(arg)
        elif cmd == "stats":
            self._print_stats()
        else:
            print(red(f"unknown command /{cmd} — try /help"))
        return True

    # -- loop ----------------------------------------------------------------

    def _read_input(self) -> str | None:
        line = input(PROMPT)
        if line.strip() != '"""':
            return line
        block = []
        while True:
            nxt = input(CONT_PROMPT)
            if nxt.strip() == '"""':
                return "\n".join(block)
            block.append(nxt)

    def run(self) -> None:
        hist = self.history_dir / "input_history"
        if readline:
            try:
                readline.read_history_file(hist)
            except OSError:
                pass
            readline.set_history_length(2000)

        print(dim(f"Model: Bonsai 2 27B"))
        print(dim(f"Server: {self.cfg['derived']['base_url']}"))
        print(dim(f"Effort: {self.effort}"))
        print(dim(f"Tools: {"ON" if self.tools_enabled else "OFF"}"))

        try:
            while True:
                try:
                    line = self._read_input()
                except KeyboardInterrupt:
                    print()
                    continue
                except EOFError:
                    print()
                    break
                if line is None or not line.strip():
                    continue
                if is_command(line):
                    if not self.command(line.strip()):
                        break
                    continue
                if line.startswith("//"):
                    line = line[1:]
                content = line
                if self.pending_files:
                    parts = [f"<file path=\"{p}\">\n{t}\n</file>" for p, t in self.pending_files]
                    content = "\n\n".join(parts) + "\n\n" + line
                    self.pending_files.clear()
                self.messages.append({"role": "user", "content": content})
                self._reply()
        finally:
            if readline:
                try:
                    readline.write_history_file(hist)
                except OSError:
                    pass


def is_command(line: str) -> bool:
    """'/clear' is a command; '/etc/hosts …' and '//help' are messages."""
    if not line.startswith("/") or line.startswith("//"):
        return False
    first = line.split(maxsplit=1)[0]
    return len(first) > 1 and "/" not in first[1:]


def resolve_workspace(arg: str | None) -> Path:
    root = Path(os.path.expanduser(arg)) if arg else Path.cwd()
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(red(f"workspace is not a directory: {root}"))
    if root == Path(root.anchor):
        raise SystemExit(red("refusing to use the filesystem root as the workspace; "
                             "cd into a project or pass --workspace"))
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chat with the local LLM server.")
    ap.add_argument("--workspace", "-w", help="directory the model's tools may access (default: current directory)")
    ap.add_argument("--no-tools", action="store_true", help="plain chat: no file or run tools at all")
    args = ap.parse_args(argv)

    try:
        cfg = model_config.load()
    except model_config.ConfigError as e:
        print(red(f"config error: {e}"), file=sys.stderr)
        return 1

    workspace = None if args.no_tools else resolve_workspace(args.workspace)
    if workspace is not None and workspace == Path.home().resolve():
        print(yellow(f"warning: the workspace is your whole home directory ({workspace}); "
                     "consider starting from a project folder"))

    chat = Chat(cfg, workspace, tools_enabled=cfg["tools"]["enabled"])

    # Ctrl-C while a turn is in progress (streaming, a tool, an approval
    # prompt or a run) cancels the turn; at the prompt it just clears the line.
    def on_sigint(signum, frame):
        if chat.busy:
            raise Interrupted()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, on_sigint)

    if not chat.client.healthy():
        print(red(f"no server at {cfg['derived']['base_url']} — start it with ./bonsai.sh"), file=sys.stderr)
        return 1
    chat.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
