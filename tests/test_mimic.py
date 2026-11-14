"""/mimic: argument parsing, which files are read and in what order, the
context budget, banned phrases, the one shortening pass, and the draft file."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import mimic


def chars(text: str) -> int:
    """A token counter for tests: one token per character."""
    return len(text)


def post(folder: Path, name: str, text: str, mtime: int) -> Path:
    p = folder / name
    p.write_text(text, encoding="utf-8")
    os.utime(p, ns=(mtime * 10**9, mtime * 10**9))
    return p


class Model:
    """Scripted `generate`: returns replies in order, records each request."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.requests.append(messages)
        return self.replies.pop(0)


@pytest.fixture
def banned_path(tmp_path) -> Path:
    return tmp_path / "data" / "banned_phrases.txt"


# --- arguments ----------------------------------------------------------------

@pytest.mark.parametrize("raw, limit, prompt", [
    ("3000 write about data contracts", 3000, "write about data contracts"),
    ("write about data contracts", None, "write about data contracts"),
    ("0 3 reasons to index", None, "3 reasons to index"),
    ("  250   short one  ", 250, "short one"),
])
def test_parse(raw, limit, prompt):
    assert mimic.parse(raw) == mimic.Request(limit, prompt)


@pytest.mark.parametrize("raw", ["", "   ", "3000", "3000   "])
def test_parse_needs_a_prompt(raw):
    with pytest.raises(ValueError, match="usage"):
        mimic.parse(raw)


# --- which files ------------------------------------------------------------

def test_candidates_are_newest_first_and_skip_drafts_hidden_links_and_folders(tmp_path):
    post(tmp_path, "old.md", "old", 100)
    post(tmp_path, "new.txt", "new", 300)
    post(tmp_path, "mid", "mid", 200)
    post(tmp_path, "made-by-mimic.draft", "draft", 400)
    post(tmp_path, ".hidden", "hidden", 500)
    (tmp_path / "sub").mkdir()
    post(tmp_path / "sub", "deep.md", "deep", 600)
    (tmp_path / "link.md").symlink_to(tmp_path / "old.md")
    assert [p.name for p in mimic.candidates(tmp_path)] == ["new.txt", "mid", "old.md"]


def test_same_mtime_orders_by_name(tmp_path):
    post(tmp_path, "a.md", "a", 100)
    post(tmp_path, "b.md", "b", 100)
    assert [p.name for p in mimic.candidates(tmp_path)] == ["b.md", "a.md"]


def test_binary_files_are_not_text(tmp_path):
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    assert mimic.read_text(tmp_path / "img.png") is None
    assert mimic.read_text(tmp_path / "latin1.txt") is None


def test_select_stops_at_the_first_file_that_does_not_fit():
    refs = [mimic.Reference("new", "x" * 50), mimic.Reference("big", "x" * 500),
            mimic.Reference("old", "x" * 10)]
    sel = mimic.select(refs, budget=200, count=chars)
    # "old" would fit, but taking it would skip "big": the set is the newest N.
    assert [r.name for r in sel.refs] == ["new"]
    assert sel.hit_limit and sel.considered == 3


def test_select_takes_everything_that_fits():
    refs = [mimic.Reference("a", "x"), mimic.Reference("b", "y")]
    sel = mimic.select(refs, budget=10_000, count=chars)
    assert [r.name for r in sel.refs] == ["a", "b"] and not sel.hit_limit


# --- banned phrases -----------------------------------------------------------

def test_banned_list_is_created_with_defaults_then_read_as_edited(banned_path):
    phrases = mimic.load_banned(banned_path)
    assert banned_path.is_file() and "let that sink in" in phrases
    assert not any(p.startswith("#") for p in phrases)
    banned_path.write_text("# mine\n\nsynergy\n  paradigm shift  \n", encoding="utf-8")
    assert mimic.load_banned(banned_path) == ["synergy", "paradigm shift"]


def test_phrase_matching_is_case_and_space_insensitive_and_whole_word():
    phrases = ["let that sink in", "foster", "thoughts?", "🚀", "here's the thing"]
    text = "LET  that\nsink in. Fosters beer. Thoughts? Launch 🚀. Here’s the thing:"
    assert mimic.found_in(text, phrases) == ["let that sink in", "thoughts?", "🚀",
                                             "here's the thing"]


def test_phrases_the_author_uses_are_not_banned():
    corpus = ["We built a robust, seamless pipeline.", "Nothing to see."]
    assert mimic.banned_for(["seamless", "delve", "robust"], corpus) == ["delve"]


# --- the whole run --------------------------------------------------------------

def test_run_writes_a_draft_and_shows_newest_last(tmp_path, banned_path):
    post(tmp_path, "first.md", "My first post. I delve into things.", 100)
    post(tmp_path, "second.md", "My second post.", 200)
    post(tmp_path, "earlier.draft", "ignore me", 300)
    model = Model("A new post about indexes.")
    result = mimic.run(mimic.parse("write about indexes"), tmp_path, banned_path,
                       context_tokens=100_000, count=chars, generate=model)

    assert result.path == tmp_path / "write-about-indexes.draft"
    assert result.path.read_text() == "A new post about indexes.\n"
    [messages] = model.requests
    user = messages[1]["content"]
    assert user.index('file="first.md"') < user.index('file="second.md"')
    assert "ignore me" not in user and "earlier.draft" not in user
    assert "write about indexes" in user and "Length:" not in user
    # "delve" is the author's own word, so it isn't banned; the rest are listed.
    assert "- delve" not in user and "- let that sink in" in user
    assert result.lines() == ["read 2/2 files (all files)",
                              "file write-about-indexes.draft generated, character count = 25"]


def test_limit_is_asked_for_then_enforced_with_one_shortening_pass(tmp_path, banned_path):
    post(tmp_path, "a.md", "one two three", 100)
    long = "word " * 100            # 500 chars, 100 words
    shorter = "word " * 50
    model = Model(long, shorter)
    result = mimic.run(mimic.parse("300 a post"), tmp_path, banned_path,
                       context_tokens=100_000, count=chars, generate=model)

    assert "approximately 300 characters (about 50 words)" in model.requests[0][1]["content"]
    assert len(model.requests) == 2
    retry = model.requests[1]
    assert retry[:2] == model.requests[0]  # same prefix, so the KV cache is reused
    assert retry[2] == {"role": "assistant", "content": long.strip()}
    # 100 words in 499 chars, aiming at 90% of 300: 100 * 0.9 * 300 / 499 = 54.
    assert "That's 499 characters; the limit is 300" in retry[3]["content"]
    assert "about 54 words" in retry[3]["content"]
    assert result.chars == len(shorter.strip()) and result.shortened and not result.over
    assert result.lines()[-1].endswith("(limit 300, shortened once)")


def test_only_one_shortening_pass_and_the_report_says_still_over(tmp_path, banned_path):
    post(tmp_path, "a.md", "text", 100)
    model = Model("x" * 500, "y" * 400)
    result = mimic.run(mimic.parse("300 a post"), tmp_path, banned_path,
                       context_tokens=100_000, count=chars, generate=model)
    assert len(model.requests) == 2
    assert result.path.read_text() == "y" * 400 + "\n"
    assert result.over
    assert result.lines()[-1] == ("file a-post.draft generated, character count = 400 "
                                  "(limit 300, still over after one shortening pass)")


def test_no_shortening_when_under_the_limit(tmp_path, banned_path):
    post(tmp_path, "a.md", "text", 100)
    model = Model("short")
    result = mimic.run(mimic.parse("300 a post"), tmp_path, banned_path,
                       context_tokens=100_000, count=chars, generate=model)
    assert len(model.requests) == 1 and not result.shortened
    assert result.lines()[-1].endswith("character count = 5 (limit 300)")


def test_reading_stops_at_half_the_context(tmp_path, banned_path):
    post(tmp_path, "old.md", "o" * 2000, 100)
    post(tmp_path, "new.md", "n" * 200, 200)
    model = Model("draft")
    # Half of 6000 is 3000 tokens: the prompt's fixed parts and new.md fit,
    # old.md doesn't.
    result = mimic.run(mimic.parse("a post"), tmp_path, banned_path,
                       context_tokens=6000, count=chars, generate=model)
    assert [r.name for r in result.selection.refs] == ["new.md"]
    assert "o" * 2000 not in model.requests[0][1]["content"]
    assert result.lines()[0] == "read 1/2 files (context limit)"


def test_newest_file_too_big_is_an_error(tmp_path, banned_path):
    post(tmp_path, "huge.md", "h" * 10_000, 100)
    with pytest.raises(mimic.MimicError, match="huge.md"):
        mimic.run(mimic.parse("a post"), tmp_path, banned_path,
                  context_tokens=4000, count=chars, generate=Model("x"))


def test_no_references_is_an_error(tmp_path, banned_path):
    post(tmp_path, "only.draft", "draft", 100)
    with pytest.raises(mimic.MimicError, match="no text files"):
        mimic.run(mimic.parse("a post"), tmp_path, banned_path,
                  context_tokens=100_000, count=chars, generate=Model("x"))


def test_empty_reply_writes_nothing(tmp_path, banned_path):
    post(tmp_path, "a.md", "text", 100)
    with pytest.raises(mimic.MimicError, match="nothing"):
        mimic.run(mimic.parse("a post"), tmp_path, banned_path,
                  context_tokens=100_000, count=chars, generate=Model("  \n"))
    assert not list(tmp_path.glob("*.draft"))


def test_banned_phrases_in_the_draft_are_reported(tmp_path, banned_path):
    post(tmp_path, "a.md", "plain words", 100)
    result = mimic.run(mimic.parse("a post"), tmp_path, banned_path, context_tokens=100_000,
                       count=chars, generate=Model("Let that sink in. 🚀"))
    assert result.banned_hits == ["let that sink in", "🚀"]
    assert result.lines()[-1] == "contains banned phrases: let that sink in, 🚀"


def test_binary_files_are_skipped_and_reported(tmp_path, banned_path):
    post(tmp_path, "a.md", "text", 100)
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\0\0")
    result = mimic.run(mimic.parse("a post"), tmp_path, banned_path,
                       context_tokens=100_000, count=chars, generate=Model("ok"))
    assert "skipped 1 non-text file: photo.jpg" in result.lines()


def test_drafts_never_replace_a_file(tmp_path):
    (tmp_path / "a-post.draft").write_text("mine")
    p = mimic.write_draft(tmp_path, "A post!", "new")
    assert p.name == "a-post-2.draft" and (tmp_path / "a-post.draft").read_text() == "mine"
    assert mimic.write_draft(tmp_path, "a post", "again").name == "a-post-3.draft"
    assert mimic.write_draft(tmp_path, "!!!", "x").name == "mimic.draft"


# --- in the chat --------------------------------------------------------------

@pytest.fixture
def chat_for(tmp_path, monkeypatch):
    import config

    def make(workspace: Path | None, scope: Path | None = None):
        import chat
        p = tmp_path / "test.toml"
        p.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n'
                     '[model]\nrepo = "a/b"\nfile = "c.gguf"\n')
        monkeypatch.setenv("ARBORIST_CONFIG", str(p))
        c = chat.Chat(config.load(), workspace, tools_enabled=True, memory_scope=scope)
        bodies: list[dict] = []

        def stream(body):
            bodies.append(body)
            yield {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
            yield {"choices": [{"delta": {"content": "Drafted "}}]}
            yield {"choices": [{"delta": {"content": "in your voice."}}]}

        def no_tokenizer(_text):
            raise OSError("no server")

        c.client.stream_chat = stream
        c.client.tokenize = no_tokenizer
        return c, bodies
    return make


def test_mimic_command_writes_into_the_workspace(chat_for, ws_root, capsys):
    post(ws_root, "post.md", "An old post of mine.", 100)
    c, bodies = chat_for(ws_root)
    assert c.command("/mimic 500 a post about tests")
    out = capsys.readouterr().out
    assert (ws_root / "a-post-about-tests.draft").read_text() == "Drafted in your voice.\n"
    assert "read 1/1 file (all files)" in out
    assert "file a-post-about-tests.draft generated, character count = 22 (limit 500)" in out
    [body] = bodies
    assert "tools" not in body and c.messages == []  # outside the conversation
    banned = Path(c.cfg["derived"]["mimic_banned_path"])
    assert banned.is_file() and banned.parent == Path(c.cfg["derived"]["memory_path"]).parent
    assert not c.busy


def test_mimic_without_tools_uses_the_starting_folder(chat_for, ws_root, capsys):
    post(ws_root, "post.md", "An old post.", 100)
    c, _ = chat_for(None, scope=ws_root)
    c.command("/mimic a post")
    assert (ws_root / "a-post.draft").is_file()


def test_mimic_errors_are_reported_not_raised(chat_for, ws_root, capsys):
    c, bodies = chat_for(ws_root)
    c.command("/mimic")
    assert "usage: /mimic" in capsys.readouterr().out
    c.command("/mimic a post")  # empty folder
    assert "no text files" in capsys.readouterr().out
    assert bodies == [] and not c.busy


def test_mimic_is_in_help_and_the_command_menu():
    import chat
    import chat_input
    assert "mimic" in chat_input.command_help(chat.HELP)


def test_token_counting_falls_back_to_an_estimate_once(chat_for, ws_root):
    c, _ = chat_for(ws_root)
    calls = []

    def flaky(text):
        calls.append(text)
        raise ValueError("no /tokenize")

    c.client.tokenize = flaky
    assert c._count_tokens("x" * 40) == 10
    assert c._count_tokens("y" * 8) == 2
    assert len(calls) == 1  # stopped asking after the first failure


def test_token_counting_uses_the_server_when_it_can(chat_for, ws_root):
    c, _ = chat_for(ws_root)
    c.client.tokenize = lambda text: 7
    assert c._count_tokens("anything") == 7
