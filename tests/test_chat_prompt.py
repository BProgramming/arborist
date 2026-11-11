"""The prompt_toolkit side of the chat prompt: completions and key bindings.
Skipped when prompt_toolkit isn't installed (chat.py then uses input())."""
from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

pytest.importorskip("prompt_toolkit")

from prompt_toolkit.completion import CompleteEvent  # noqa: E402
from prompt_toolkit.document import Document  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

import chat  # noqa: E402
import chat_input  # noqa: E402

NOW = datetime(2026, 11, 11, 9, 8, 7)
CHECKPOINTS = [("abc1234", "11-11 09:00  fix the parser"), ("abd9999", "11-10 17:30  add tests"),
               ("f00ba47", "11-09 12:00  rename [safety]")]


def commands():
    return chat_input.command_help(chat.HELP)


# ----------------------------------------------------------------- completer



def make_completer(checkpoints=lambda: CHECKPOINTS):
    return chat_input.ChatCompleter(commands(), checkpoints, now=lambda: NOW)


def complete(text, completer=None):
    completer = completer or make_completer()
    return list(completer.get_completions(Document(text), CompleteEvent(text_inserted=True)))


def test_slash_alone_offers_every_command_with_its_help():
    got = complete("/")
    assert [c.text.strip() for c in got] == [f"/{n}" for n in commands()]
    undo = next(c for c in got if c.text.startswith("/undo"))
    assert undo.display_text == "/undo [ID]"
    assert "put the workspace's files back" in undo.display_meta_text
    assert undo.start_position == -1


def test_commands_narrow_as_you_type_and_ignore_case():
    assert [c.text for c in complete("/re")] == ["/retry", "/resume", "/recall "]
    assert [c.text for c in complete("/RE")] == ["/retry", "/resume", "/recall "]
    assert complete("/zzz") == []


def test_commands_that_take_arguments_end_with_a_space():
    texts = {c.text for c in complete("/")}
    assert {"/undo ", "/save ", "/file ", "/effort "} <= texts
    assert {"/help", "/clear", "/checkpoints", "/exit"} <= texts


@pytest.mark.parametrize("text", [
    "hello", "//he", "/etc/ho", "first line\n/he", "/help me", "/unknown arg",
])
def test_no_completions_outside_commands(text):
    assert complete(text) == []


def test_undo_offers_checkpoint_ids_and_filters_them():
    got = complete("/undo ")
    assert [c.text for c in got] == ["abc1234", "abd9999", "f00ba47"]
    assert got[0].display_meta_text == "11-11 09:00  fix the parser"
    assert [c.text for c in complete("/undo ab")] == ["abc1234", "abd9999"]
    narrowed = complete("/undo  ABD")
    assert [c.text for c in narrowed] == ["abd9999"] and narrowed[0].start_position == -3
    assert complete("/undo xyz") == []


def test_checkpoints_are_listed_once_per_prompt():
    calls = []

    def listing():
        calls.append(1)
        return CHECKPOINTS
    c = make_completer(listing)
    complete("/undo ", c)
    complete("/undo a", c)
    assert len(calls) == 1
    c.reset()
    complete("/undo ", c)
    assert len(calls) == 2


def test_a_broken_checkpoint_store_offers_nothing():
    def broken():
        raise OSError("store gone")
    assert complete("/undo ", make_completer(broken)) == []


def test_save_suggests_a_timestamped_file_name():
    got = complete("/save ")
    assert [c.text for c in got] == ["arborist_conversation_20261111090807.md"]
    assert got[0].start_position == 0
    partial = complete("/save arbor")
    assert [c.text for c in partial] == ["arborist_conversation_20261111090807.md"]
    assert partial[0].start_position == -5
    assert complete("/save notes.md") == []


def test_save_suggestion_uses_the_time_of_asking():
    times = iter([datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 1, 2, 3, 4, 6)])
    c = chat_input.ChatCompleter(commands(), lambda: [], now=lambda: next(times))
    assert complete("/save ", c)[0].text == "arborist_conversation_20260102030405.md"
    assert complete("/save ", c)[0].text == "arborist_conversation_20260102030406.md"


# ---------------------------------------------------------------- the prompt
#
# Driven through a pipe: keystrokes go in as the bytes a terminal would send.
# Each chunk is sent after a pause so completions (which run in the
# background) are ready before the next key, as when a person types.

ENTER, ALT_ENTER, TAB, CTRL_C, CTRL_D = "\r", "\x1b\r", "\t", "\x03", "\x04"


def paste(text):
    return f"\x1b[200~{text}\x1b[201~"


@pytest.fixture
def prompt_reader(tmp_path):
    """read(*chunks) -> what one call to RichPrompt.read() returns."""
    def read(*chunks, history=None, toolbar=lambda: "tb", reads=1):
        with create_pipe_input() as pipe:
            p = chat_input.RichPrompt(history or tmp_path / "prompt_history", make_completer(),
                                      toolbar, color=False, output=DummyOutput(), input=pipe)

            def feed():
                for chunk in chunks:
                    time.sleep(0.15)
                    pipe.send_text(chunk)
            threading.Thread(target=feed, daemon=True).start()
            results = [p.read() for _ in range(reads)]
            return results[0] if reads == 1 else results
    return read


def test_enter_sends_a_plain_message(prompt_reader):
    assert prompt_reader("hello there", ENTER) == "hello there"


def test_alt_enter_adds_a_line_and_enter_sends(prompt_reader):
    assert prompt_reader("first", ALT_ENTER, "second", ALT_ENTER, "third", ENTER) == \
        "first\nsecond\nthird"


def test_bracketed_paste_keeps_newlines_and_waits_for_enter(prompt_reader):
    assert prompt_reader(paste("def f():\r\n    return 1\nprint(f())"), " # ok", ENTER) == \
        "def f():\n    return 1\nprint(f()) # ok"


def test_multi_line_text_can_be_edited_in_the_middle(prompt_reader):
    # Up arrow moves into the first line; End then typing appends there.
    assert prompt_reader("one", ALT_ENTER, "two", "\x1b[A", "\x1b[F", "!", ENTER) == "one!\ntwo"


def test_enter_finishes_a_command_only_one_name_matches(prompt_reader):
    assert prompt_reader("/che", ENTER) == "/checkpoints"


def test_enter_sends_an_ambiguous_prefix_as_typed(prompt_reader):
    assert prompt_reader("/re", ENTER) == "/re"


def test_tab_picks_the_first_match(prompt_reader):
    assert prompt_reader("/he", TAB, ENTER) == "/help"
    assert prompt_reader("/re", TAB, TAB, ENTER) == "/resume"


def test_undo_tab_fills_a_checkpoint_id(prompt_reader):
    assert prompt_reader("/undo abd", TAB, ENTER) == "/undo abd9999"


def test_undo_without_picking_keeps_its_meaning(prompt_reader):
    # One checkpoint matching must not turn "/undo" into "/undo <that id>".
    assert prompt_reader("/undo f", ENTER) == "/undo f"
    assert prompt_reader("/undo ", ENTER) == "/undo "


def test_save_tab_fills_the_suggested_name(prompt_reader):
    assert prompt_reader("/save ", TAB, ENTER) == "/save arborist_conversation_20261111090807.md"


def test_triple_quotes_still_open_a_block(prompt_reader):
    assert prompt_reader('"""', ENTER, "a", ENTER, "  b", ENTER, '"""', ENTER) == "a\n  b"


def test_ctrl_c_and_ctrl_d_raise_like_input(prompt_reader):
    with pytest.raises(KeyboardInterrupt):
        prompt_reader("half a thought", CTRL_C)
    with pytest.raises(EOFError):
        prompt_reader(CTRL_D)


def test_history_is_saved_and_recalled_with_up(prompt_reader, tmp_path):
    hist = tmp_path / "h"
    assert prompt_reader("remember me", ALT_ENTER, "too", ENTER, history=hist) == "remember me\ntoo"
    assert "remember me" in hist.read_text()
    assert prompt_reader("\x1b[A", ENTER, history=hist) == "remember me\ntoo"


def test_toolbar_is_asked_for_while_the_prompt_is_up(prompt_reader):
    calls = []

    def toolbar():
        calls.append(1)
        return "Model: x"
    prompt_reader("hi", ENTER, toolbar=toolbar)
    assert calls
