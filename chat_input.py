"""The chat prompt.

With prompt_toolkit installed and a real terminal, the prompt gets a live
completion menu (commands after /, checkpoint IDs after /undo, a file name
after /save), history suggestions, bracketed paste, multi-line editing with
Alt+Enter for a new line, and a bottom toolbar. Without it, chat.py falls back
to plain input() with readline.

./arborist.sh installs prompt_toolkit (see pyproject.toml) when it's missing.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output import create_output
    HAVE_PROMPT_TOOLKIT = True
except ImportError:  # the fallback in chat.py takes over
    HAVE_PROMPT_TOOLKIT = False
    Completer = object  # type: ignore[assignment,misc]


def available(stdin=None, stdout=None) -> bool:
    """Whether the rich prompt can be used: the library is there and both ends
    are a terminal (pipes and tests get plain input())."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        return HAVE_PROMPT_TOOLKIT and stdin.isatty() and stdout.isatty()
    except (AttributeError, ValueError):
        return False


# ------------------------------------------------------------------ commands --

_HELP_ENTRY = re.compile(r"^  (/\S.*?)(?:\s{2,}(\S.*))?$")
_HELP_MORE = re.compile(r"^ {8,}(\S.*)$")


def command_help(help_text: str) -> dict[str, tuple[str, str]]:
    """Command name -> (usage, description), in the order the help lists them.
    Read from the "Commands:" section of chat.HELP, so the menu can't drift
    from /help."""
    out: dict[str, tuple[str, str]] = {}
    in_commands = False
    last: list[str] = []
    for line in help_text.splitlines():
        if line.strip() == "Commands:":
            in_commands = True
            continue
        if not in_commands:
            continue
        if not line.strip():
            break
        m = _HELP_ENTRY.match(line)
        if m:
            usage, desc = m.group(1).strip(), (m.group(2) or "").strip()
            # "/exit, /quit" lists two names for one command.
            forms = [f.strip() for f in usage.split(",")] if "," in usage else [usage]
            last = []
            for form in forms:
                name = form.split()[0][1:]
                out[name] = (form, desc)
                last.append(name)
            continue
        m = _HELP_MORE.match(line)
        if m and last:
            for name in last:
                usage, desc = out[name]
                out[name] = (usage, f"{desc} {m.group(1).strip()}".strip())
    return out


def save_suggestion(now: datetime) -> str:
    return f"arborist_conversation_{now:%Y%m%d%H%M%S}.md"


def toolbar_text(model: str, effort: str, tools_on: bool, context_pct: float) -> str:
    return (f"Model: {model}, Effort: {effort or 'n/a'}, Tools: {'ON' if tools_on else 'OFF'}, "
            f"Context: {context_pct:.0f}%")


class ChatCompleter(Completer):
    """What the menu offers for the text before the cursor:

    /par...         commands starting with "par", with their help text
    /undo ab...     checkpoint IDs starting with "ab", newest first
    /save ...       arborist_conversation_<now>.md

    Nothing for anything else, including multi-line messages, //escaped text
    and pasted paths like /etc/hosts.
    """

    def __init__(self, commands: dict[str, tuple[str, str]],
                 checkpoints: Callable[[], list[tuple[str, str]]],
                 now: Callable[[], datetime] = datetime.now):
        self.commands = commands
        self._checkpoints = checkpoints
        self._checkpoint_cache: list[tuple[str, str]] | None = None
        self.now = now

    def reset(self) -> None:
        """Forget cached checkpoints (called once per prompt: listing them runs git)."""
        self._checkpoint_cache = None

    def checkpoints(self) -> list[tuple[str, str]]:
        if self._checkpoint_cache is None:
            try:
                self._checkpoint_cache = list(self._checkpoints())
            except Exception:  # a broken store must never break typing
                self._checkpoint_cache = []
        return self._checkpoint_cache

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if "\n" in text or not text.startswith("/") or text.startswith("//"):
            return
        m = re.fullmatch(r"/(\w*)", text)
        if m:
            typed = m.group(1).lower()
            for name, (usage, desc) in self.commands.items():
                if name.startswith(typed):
                    takes_args = usage != f"/{name}"
                    yield Completion(f"/{name}" + (" " if takes_args else ""),
                                     start_position=-len(text), display=usage,
                                     display_meta=desc)
            return
        m = re.fullmatch(r"/undo\s+(\S*)", text)
        if m:
            typed = m.group(1).lower()
            for short, desc in self.checkpoints():
                if short.lower().startswith(typed):
                    yield Completion(short, start_position=-len(typed), display_meta=desc)
            return
        m = re.fullmatch(r"/save\s+(\S*)", text)
        if m:
            typed = m.group(1)
            name = save_suggestion(self.now())
            if name.startswith(typed):
                yield Completion(name, start_position=-len(typed),
                                 display_meta="save the conversation here")


# ------------------------------------------------------------------- session --

def key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        buf = event.current_buffer
        state = buf.complete_state
        if state is not None:
            # A highlighted entry is already in the text. With none
            # highlighted, a command name typed partway that only one command
            # matches is finished for you ("/che" -> "/checkpoints"); after
            # that, Enter sends what's there, so "/undo" or "/save" with no
            # argument keep their usual meaning.
            if (state.current_completion is None and len(state.completions) == 1
                    and re.fullmatch(r"/\w*", buf.document.text_before_cursor)):
                buf.apply_completion(state.completions[0])
            buf.complete_state = None
        buf.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return kb


class RichPrompt:
    """One PromptSession for the whole chat. read() returns the message, or
    raises KeyboardInterrupt (Ctrl-C) / EOFError (Ctrl-D), like input()."""

    def __init__(self, history_file: Path, completer: ChatCompleter,
                 toolbar: Callable[[], str], color: bool = True,
                 output=None, input=None):
        self.completer = completer
        green = "\x1b[1;32m" if color else ""
        dim = "\x1b[2m" if color else ""
        reset = "\x1b[0m" if color else ""
        self.message = ANSI(f"{green}› {reset}") if color else "> "
        self.cont_message = ANSI(f"{dim}… {reset}") if color else "... "
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=key_bindings(),
            multiline=True,  # Enter is rebound to send; Alt+Enter adds a line
            prompt_continuation=lambda width, _line, _wrap: ANSI(f"{dim}{'… ':>{width}}{reset}"),
            bottom_toolbar=lambda: toolbar_text_safe(toolbar),
            enable_history_search=False,
            mouse_support=False,
            output=output,
            input=input,
        )

    def read(self) -> str:
        self.completer.reset()
        text = self.session.prompt(self.message)
        if text.strip() != '"""':
            return text
        # The old block syntax still works, for muscle memory and pasted
        # instructions that use it.
        block = []
        while True:
            nxt = self.session.prompt(self.cont_message)
            if nxt.strip() == '"""':
                return "\n".join(block)
            block.append(nxt)


def terminal_output(stream):
    """prompt_toolkit output that writes to `stream` directly. Give it the real
    terminal, not chat.SafeStdout: SafeStdout would show the menu's and
    toolbar's escape sequences as ␛[… text."""
    return create_output(stdout=stream)


def toolbar_text_safe(toolbar: Callable[[], str]) -> str:
    try:
        return toolbar()
    except Exception:  # drawn on every keystroke; never let it take the prompt down
        return ""
