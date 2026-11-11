"""chat_input helpers that work with or without prompt_toolkit installed."""
from __future__ import annotations

import io
from datetime import datetime

import pytest

import chat
import chat_input

NOW = datetime(2026, 11, 11, 9, 8, 7)


def commands():
    return chat_input.command_help(chat.HELP)


# ---------------------------------------------------------------- pure helpers

def test_command_help_reads_names_usages_and_wrapped_descriptions():
    text = """\
Commands:
  /help                 this list
  /exit, /quit          leave
  /undo [ID]            put the files back
                        as they were
  /stats                speed

Input:
  /notacommand          not in the Commands section
"""
    assert chat_input.command_help(text) == {
        "help": ("/help", "this list"),
        "exit": ("/exit", "leave"),
        "quit": ("/quit", "leave"),
        "undo": ("/undo [ID]", "put the files back as they were"),
        "stats": ("/stats", "speed"),
    }


def test_command_help_lists_the_real_commands_in_help_order():
    names = list(commands())
    assert names[:3] == ["help", "exit", "quit"]
    assert {"undo", "save", "checkpoints", "resume", "diff", "config"} <= set(names)
    assert commands()["undo"][0] == "/undo [ID]"


def test_save_suggestion_uses_the_timestamp():
    assert chat_input.save_suggestion(NOW) == "arborist_conversation_20261111090807.md"


@pytest.mark.parametrize("effort,tools,pct,expected", [
    ("medium", True, 12.4, "Model: Bonsai, Effort: medium, Tools: ON, Context: 12%"),
    ("", False, 99.6, "Model: Bonsai, Effort: n/a, Tools: OFF, Context: 100%"),
])
def test_toolbar_text(effort, tools, pct, expected):
    assert chat_input.toolbar_text("Bonsai", effort, tools, pct) == expected


def test_toolbar_errors_never_reach_the_prompt():
    def broken():
        raise RuntimeError("boom")
    assert chat_input.toolbar_text_safe(broken) == ""
    assert chat_input.toolbar_text_safe(lambda: "ok") == "ok"


class _Tty(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_available_needs_the_library_and_a_terminal_on_both_ends(monkeypatch):
    monkeypatch.setattr(chat_input, "HAVE_PROMPT_TOOLKIT", True)
    assert chat_input.available(_Tty(True), _Tty(True))
    assert not chat_input.available(_Tty(False), _Tty(True))
    assert not chat_input.available(_Tty(True), _Tty(False))
    monkeypatch.setattr(chat_input, "HAVE_PROMPT_TOOLKIT", False)
    assert not chat_input.available(_Tty(True), _Tty(True))
