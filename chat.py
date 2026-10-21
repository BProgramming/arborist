#!/usr/bin/env python3
"""Terminal chat client for a local llama-server running Bonsai 2 27B.

The heavy lifting (inference) happens in the C++ server; this is just I/O.
Standard library only, so there's nothing to pip install.

Type /help at the prompt for commands.
"""
from __future__ import annotations

import http.client
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    import readline  # noqa: F401  (line editing + history for input())
except ImportError:  # pragma: no cover
    readline = None

import config as bonsai_config

# ---------------------------------------------------------------- terminal ---

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def dim(t: str) -> str: return _c("2", t)
def bold(t: str) -> str: return _c("1", t)
def cyan(t: str) -> str: return _c("36", t)
def red(t: str) -> str: return _c("31", t)
def green(t: str) -> str: return _c("32", t)


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
  /file PATH [PATH…]    attach file contents to your next message
  /system TEXT          replace the system prompt (clears the conversation)
  /retry                regenerate the last reply
  /save [PATH]          write the conversation to a Markdown file
  /stats                context usage and last-reply speed

Input:
  \"\"\"                   on its own line starts/ends a multi-line block
  Ctrl-C                stops a reply mid-stream (that turn is discarded)
"""


# ------------------------------------------------------------------ client ---

class Interrupted(Exception):
    pass


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
                yield json.loads(data)
        finally:
            # Closing mid-stream tells llama-server to stop generating.
            conn.close()


# -------------------------------------------------------------------- chat ---

class Chat:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ch = cfg["chat"]
        self.client = Client(cfg["derived"]["base_url"])
        self.system_prompt = ch["system_prompt"]
        self.effort = ch["reasoning_effort"]
        self.show_thinking = ch["show_thinking"]
        self.messages: list[dict] = []
        self.pending_files: list[tuple[str, str]] = []
        self.last_stats: dict = {}
        self.ctx_used = 0
        self.history_dir = Path(cfg["derived"]["history_dir"])
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # -- request -------------------------------------------------------------

    def _body(self) -> dict:
        ch = self.cfg["chat"]
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        msgs += self.messages
        return {
            "messages": msgs,
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
            # conversation, so each turn only processes the new tokens.
            # Without this every turn re-reads the whole history.
            "cache_prompt": True,
        }

    def _generate(self) -> str | None:
        """Stream one reply. Returns the answer text, or None if interrupted."""
        answer: list[str] = []
        thinking_chars = 0
        thinking_open = False
        status_shown = False
        t0 = time.monotonic()
        first_token_at = None
        timings, usage = {}, {}

        def clear_status():
            nonlocal status_shown
            if status_shown:
                sys.stdout.write("\r\033[K" if USE_COLOR else "\n")
                status_shown = False

        try:
            for chunk in self.client.stream_chat(self._body()):
                timings = chunk.get("timings") or timings
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    think = delta.get("reasoning_content")
                    text = delta.get("content")
                    if think:
                        thinking_chars += len(think)
                        if self.show_thinking:
                            if not thinking_open:
                                sys.stdout.write(dim("thinking: "))
                                thinking_open = True
                            sys.stdout.write(dim(think))
                        else:
                            secs = time.monotonic() - t0
                            sys.stdout.write(f"\r{dim(f'thinking… {secs:4.0f}s, ~{thinking_chars // 4} tokens')}")
                            status_shown = True
                        sys.stdout.flush()
                    if text:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                            clear_status()
                            if thinking_open:
                                sys.stdout.write("\n\n")
                                thinking_open = False
                        answer.append(text)
                        sys.stdout.write(text)
                        sys.stdout.flush()
        except Interrupted:
            clear_status()
            print(red("\n[stopped]"))
            return None
        finally:
            clear_status()

        print()
        self.last_stats = {
            "wall_s": time.monotonic() - t0,
            "ttft_s": (first_token_at - t0) if first_token_at else None,
            "prompt_n": timings.get("prompt_n"),
            "prompt_tps": timings.get("prompt_per_second"),
            "gen_n": timings.get("predicted_n"),
            "gen_tps": timings.get("predicted_per_second"),
            "cache_n": timings.get("cache_n"),
        }
        if usage:
            self.ctx_used = usage.get("total_tokens", 0)
        if self.cfg["chat"]["show_stats"]:
            self._print_stats(compact=True)
        text = "".join(answer)
        if not text.strip():
            print(dim("(empty reply — if thinking ran long, raise chat.max_tokens or lower /effort)"))
        return text

    def _print_stats(self, compact: bool = False) -> None:
        s = self.last_stats
        ctx = self.cfg["server"]["ctx_size"] // self.cfg["server"]["parallel"]
        pct = 100 * self.ctx_used / ctx if ctx else 0
        parts = []
        if s.get("gen_tps"):
            parts.append(f"{s['gen_tps']:.1f} tok/s")
        if s.get("gen_n") is not None:
            parts.append(f"{s['gen_n']} generated")
        if s.get("prompt_tps") and s.get("prompt_n"):
            parts.append(f"prompt {s['prompt_n']} @ {s['prompt_tps']:.0f} tok/s")
        if s.get("cache_n"):
            parts.append(f"{s['cache_n']} cached")
        if s.get("ttft_s") is not None:
            parts.append(f"answer after {s['ttft_s']:.1f}s")
        parts.append(f"ctx {self.ctx_used:,}/{ctx:,} ({pct:.0f}%)")
        line = " · ".join(parts)
        print(dim(line) if compact else line)
        if pct > 85:
            print(red("context nearly full — /save then /clear soon"))

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
        lines = [f"# Bonsai chat — {datetime.now():%Y-%m-%d %H:%M}", ""]
        if self.system_prompt:
            lines += ["**System:** " + self.system_prompt, ""]
        for m in self.messages:
            who = "You" if m["role"] == "user" else "Bonsai"
            lines += [f"## {who}", "", m["content"], ""]
        try:
            path.write_text("\n".join(lines), encoding="utf-8")
            print(dim(f"saved {path}"))
        except OSError as e:
            print(red(f"save failed: {e}"))

    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False to exit."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("exit", "quit", "q"):
            return False
        if cmd == "help":
            print(HELP)
        elif cmd == "clear":
            self.messages.clear()
            self.pending_files.clear()
            self.ctx_used = 0
            print(dim("conversation cleared"))
        elif cmd == "effort":
            if arg in bonsai_config.EFFORTS:
                self.effort = arg
                print(dim(f"reasoning effort: {arg}"))
            else:
                print(red(f"usage: /effort {'|'.join(sorted(bonsai_config.EFFORTS))}  (now: {self.effort})"))
        elif cmd == "think":
            self.show_thinking = not self.show_thinking
            print(dim(f"show thinking: {'on' if self.show_thinking else 'off'}"))
        elif cmd == "file":
            self._attach(arg.split())
        elif cmd == "system":
            if not arg:
                print(dim(f"system prompt: {self.system_prompt}"))
            else:
                self.system_prompt = arg
                self.messages.clear()
                self.ctx_used = 0
                print(dim("system prompt replaced; conversation cleared"))
        elif cmd == "retry":
            if self.messages and self.messages[-1]["role"] == "assistant":
                self.messages.pop()
                self._reply()
            else:
                print(red("nothing to retry"))
        elif cmd == "save":
            self._save(arg)
        elif cmd == "stats":
            self._print_stats()
        else:
            print(red(f"unknown command /{cmd} — try /help"))
        return True

    # -- loop ----------------------------------------------------------------

    def _reply(self) -> None:
        print()
        try:
            text = self._generate()
        except (RuntimeError, OSError) as e:
            print(red(f"\nerror: {e}"))
            if not self.client.healthy():
                print(red("the server isn't responding — check the server log"))
            text = None
        if text is None:
            self.messages.pop()  # drop the user turn so history stays consistent
        else:
            self.messages.append({"role": "assistant", "content": text})
        print()

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

        print(bold("Bonsai 2 27B") + dim(f"  ·  {self.cfg['derived']['base_url']}  ·  "
                                         f"effort {self.effort}  ·  /help for commands"))
        print()
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
                if line.startswith("/"):
                    if not self.command(line.strip()):
                        break
                    continue
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


def main() -> int:
    try:
        cfg = bonsai_config.load()
    except bonsai_config.ConfigError as e:
        print(red(f"config error: {e}"), file=sys.stderr)
        return 1

    # Ctrl-C during a reply raises Interrupted inside the stream loop; at the
    # prompt, input() gets a normal KeyboardInterrupt.
    def on_sigint(signum, frame):
        if frame is not None and any(
            f.f_code.co_name == "_generate" for f in _walk(frame)
        ):
            raise Interrupted()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, on_sigint)

    chat = Chat(cfg)
    if not chat.client.healthy():
        print(red(f"no server at {cfg['derived']['base_url']} — start it with ./bonsai.sh"), file=sys.stderr)
        return 1
    chat.run()
    return 0


def _walk(frame):
    while frame is not None:
        yield frame
        frame = frame.f_back


if __name__ == "__main__":
    sys.exit(main())
