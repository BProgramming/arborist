"""/mimic [LIMIT] PROMPT: write something in the voice of the files in a folder.

The folder is the one the chat started in. Its files are the reference set:
read newest first (by modification time) until they'd fill half the context
window, then shown to the model oldest first, so the most recent writing sits
nearest the instruction. The result is written next to them as NAME.draft.
Drafts are never read as references, so the tool doesn't learn from its own
output; rename a draft (drop .draft) once it's yours and it joins the set.

The length limit is enforced here, not trusted to the model: the model is
asked for roughly that length, and if the draft comes back longer it gets one
request to shorten it to a word count. Whatever comes back is written, with
its real character count reported.

Phrases that make writing sound machine-made live in a plain-text list next
to arborist's memory file. Any phrase found in the reference files is the
author's own and is dropped from the list; the rest are named in the prompt
and checked for in the draft.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DRAFT_SUFFIX = ".draft"
# Share of the context window the prompt (references included) may fill.
CONTEXT_SHARE = 0.5
# The shortening pass aims this far under the limit: rewrites overshoot.
SHORTEN_MARGIN = 0.9
# For turning a character limit into a word count the model can aim at.
CHARS_PER_WORD = 6

USAGE = "usage: /mimic [LIMIT] PROMPT  (LIMIT: most characters; 0 or none for no limit)"

SYSTEM = (
    "You write new pieces in the voice of one author, from examples of their writing. "
    "Match their vocabulary, sentence length, rhythm, punctuation, formatting and tone, "
    "and how they open and close a piece. Don't copy their sentences. Reply with only the "
    "piece itself: no preamble, no title unless their examples have titles, no notes after it."
)

DEFAULT_BANNED = """\
# Phrases /mimic keeps out of drafts: one per line, case doesn't matter, and
# lines starting with # are ignored. Any phrase that appears in the files you
# run /mimic on is yours, so it's dropped from the list for that folder.
here's the thing
let that sink in
game-changer
game changer
in today's fast-paced
in today's world
the real question is
delve
navigate the complexities
ever-evolving landscape
leverage
unlock the power
unleash
harness the power
at the end of the day
I'm thrilled to announce
I'm humbled
excited to share
on this journey
resonate
a testament to
tapestry
foster
seamless
elevate
cutting-edge
pivotal
moreover
furthermore
in conclusion
thoughts?
agree?
what do you think?
🚀
💡
🔥
👇
"""


# ----------------------------------------------------------------- input ---

@dataclass
class Request:
    limit: int | None   # most characters, or None
    prompt: str


def parse(raw: str) -> Request:
    """'/mimic 3000 write about X' -> Request(3000, 'write about X').
    A leading whole number is always the limit; 0 means none, so a prompt
    that starts with a number can still be written: /mimic 0 3 reasons…"""
    raw = raw.strip()
    first, _, rest = raw.partition(" ")
    if first.isdigit():
        limit: int | None = int(first) or None
        raw = rest.strip()
    else:
        limit = None
    if not raw:
        raise ValueError(USAGE)
    return Request(limit, raw)


# ------------------------------------------------------------ references ---

@dataclass
class Reference:
    name: str
    text: str


def candidates(folder: Path) -> list[Path]:
    """Files directly in `folder` that can serve as references, newest first.
    Skipped: drafts, hidden files, symlinks (they may point anywhere) and
    folders."""
    found = []
    for p in folder.iterdir():
        if p.name.startswith(".") or p.name.endswith(DRAFT_SUFFIX):
            continue
        if p.is_symlink() or not p.is_file():
            continue
        found.append((p.stat().st_mtime_ns, p.name, p))
    # Ties (same nanosecond) break on the name, so the order is stable.
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in found]


def read_text(path: Path) -> str | None:
    """The file as UTF-8 text, or None if it's binary or unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def example_block(ref: Reference) -> str:
    return f'<example file="{ref.name}">\n{ref.text.strip()}\n</example>'


@dataclass
class Selection:
    refs: list[Reference]   # newest first
    considered: int         # text files there were to read
    hit_limit: bool         # stopped because the context share was used up


def select(texts: list[Reference], budget: int, count: Callable[[str], int]) -> Selection:
    """Take references newest first until the next one would go over
    `budget` tokens. Stops at the first that doesn't fit, rather than
    skipping it for an older one, so the set is always the newest N."""
    taken: list[Reference] = []
    used = 0
    for ref in texts:
        n = count(example_block(ref))
        if used + n > budget:
            return Selection(taken, len(texts), True)
        taken.append(ref)
        used += n
    return Selection(taken, len(texts), False)


# ---------------------------------------------------------- banned phrases ---

def load_banned(path: Path) -> list[str]:
    """The phrase list, created with the defaults the first time."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_BANNED, encoding="utf-8")
    phrases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            phrases.append(line)
    return phrases


def _pattern(phrase: str) -> re.Pattern:
    """Case-insensitive, any run of spaces between words, and whole words
    at whichever end is a word character ("foster" doesn't match "Fosters")."""
    body = r"\s+".join(re.escape(w) for w in phrase.split())
    start = r"\b" if re.match(r"\w", phrase) else ""
    end = r"\b" if re.search(r"\w$", phrase) else ""
    return re.compile(start + body + end, re.IGNORECASE)


def found_in(text: str, phrases: list[str]) -> list[str]:
    """The phrases that appear in `text`, in list order."""
    # Curly apostrophes count as straight ones.
    text = text.replace("’", "'")
    return [p for p in phrases if _pattern(p.replace("’", "'")).search(text)]


def banned_for(phrases: list[str], corpus: list[str]) -> list[str]:
    """Drop every phrase the author has used in any reference file."""
    used = set()
    for text in corpus:
        used.update(found_in(text, phrases))
    return [p for p in phrases if p not in used]


# ---------------------------------------------------------------- prompt ---

def target_words(chars: int) -> int:
    return max(1, round(chars / CHARS_PER_WORD))


def instruction(req: Request, banned: list[str]) -> str:
    parts = ["The examples above are oldest first: the last ones are the most recent and "
             "the best guide to how the author writes now."]
    if banned:
        parts.append("The author never uses these phrases; don't use them:\n"
                     + "\n".join(f"- {p}" for p in banned))
    if req.limit:
        parts.append(f"Length: approximately {req.limit} characters "
                     f"(about {target_words(req.limit)} words), and no more.")
    parts.append(f"Write this, in the author's voice:\n{req.prompt}")
    return "\n\n".join(parts)


def user_message(refs_newest_first: list[Reference], instr: str) -> str:
    examples = "\n\n".join(example_block(r) for r in reversed(refs_newest_first))
    return f"<examples>\n{examples}\n</examples>\n\n{instr}" if examples else instr


def shorten_words(draft: str, limit: int) -> int:
    """Words to ask for so the rewrite lands a little under `limit` chars."""
    words = len(draft.split())
    return max(1, int(words * SHORTEN_MARGIN * limit / max(1, len(draft))))


def shorten_message(draft: str, limit: int) -> str:
    return (f"That's {len(draft)} characters; the limit is {limit}. Rewrite it in about "
            f"{shorten_words(draft, limit)} words: same voice, same main point. Reply with "
            "only the rewritten piece.")


# ----------------------------------------------------------------- output ---

def slug(prompt: str, words: int = 6, max_len: int = 50) -> str:
    parts = re.findall(r"[a-z0-9]+", prompt.lower())[:words]
    s = "-".join(parts)[:max_len].rstrip("-")
    return s or "mimic"


def write_draft(folder: Path, prompt: str, text: str) -> Path:
    """NAME.draft in `folder`, never replacing a file: NAME-2.draft and so on."""
    base = slug(prompt)
    n = 1
    while True:
        name = f"{base}{'' if n == 1 else f'-{n}'}{DRAFT_SUFFIX}"
        path = folder / name
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(text if text.endswith("\n") else text + "\n")
            return path
        except FileExistsError:
            n += 1


# ------------------------------------------------------------------- run ---

@dataclass
class Result:
    path: Path
    chars: int
    limit: int | None
    over: bool                     # still over the limit after shortening
    shortened: bool
    selection: Selection
    skipped: list[str] = field(default_factory=list)   # non-text files
    banned_hits: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        s = self.selection
        why = "context limit" if s.hit_limit else "all files"
        out = [f"read {len(s.refs)}/{s.considered} file{'s' * (s.considered != 1)} ({why})"]
        if self.skipped:
            out.append(f"skipped {len(self.skipped)} non-text file"
                       f"{'s' * (len(self.skipped) != 1)}: {', '.join(self.skipped[:5])}"
                       + (" …" if len(self.skipped) > 5 else ""))
        line = f"file {self.path.name} generated, character count = {self.chars}"
        if self.limit:
            line += f" (limit {self.limit}"
            if self.over:
                line += ", still over after one shortening pass"
            elif self.shortened:
                line += ", shortened once"
            line += ")"
        out.append(line)
        if self.banned_hits:
            out.append("contains banned phrases: " + ", ".join(self.banned_hits))
        return out


class MimicError(RuntimeError):
    pass


def run(req: Request, folder: Path, banned_path: Path, context_tokens: int,
        count: Callable[[str], int], generate: Callable[[list[dict]], str],
        notice: Callable[[str], None] = lambda s: None) -> Result:
    """Pick the references, ask for the draft, shorten it once if needed,
    and write it. `generate` takes chat messages and returns the reply text;
    `count` returns a text's length in tokens."""
    texts: list[Reference] = []
    skipped: list[str] = []
    for p in candidates(folder):
        t = read_text(p)
        if t is None or not t.strip():
            if t is None:
                skipped.append(p.name)
            continue
        texts.append(Reference(p.name, t))
    if not texts:
        raise MimicError(f"no text files to learn the style from in {folder} "
                         f"(drafts and hidden files don't count)")

    banned = banned_for(load_banned(banned_path), [r.text for r in texts])
    instr = instruction(req, banned)
    fixed = count(SYSTEM) + count(instr)
    budget = int(context_tokens * CONTEXT_SHARE) - fixed
    sel = select(texts, budget, count)
    if not sel.refs:
        raise MimicError(f"the newest file, {texts[0].name}, doesn't fit in half the context "
                         f"window ({int(context_tokens * CONTEXT_SHARE):,} tokens)")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(sel.refs, instr)}]
    notice(f"  writing in the style of {len(sel.refs)} file{'s' * (len(sel.refs) != 1)}…")
    draft = generate(messages).strip()
    if not draft:
        raise MimicError("the model returned nothing")

    shortened = False
    if req.limit and len(draft) > req.limit:
        notice(f"  {len(draft)} characters, over the {req.limit} limit; shortening once…")
        retry = generate(messages + [{"role": "assistant", "content": draft},
                                     {"role": "user", "content": shorten_message(draft, req.limit)}])
        retry = retry.strip()
        if retry:
            draft, shortened = retry, True

    path = write_draft(folder, req.prompt, draft)
    return Result(path=path, chars=len(draft), limit=req.limit,
                  over=bool(req.limit and len(draft) > req.limit), shortened=shortened,
                  selection=sel, skipped=skipped, banned_hits=found_in(draft, banned))
