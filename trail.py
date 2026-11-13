"""What a session has written to arborist's shared files, so /private can
take it back partway through a chat.

Files the session created (run logs, build output) are deleted. Text it
appended to a file other sessions also write to (the error logs, the prompt
history) is cut out exactly, so lines from other sessions stay where they are.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trail:
    created: list[Path] = field(default_factory=list)
    appended: list[tuple[Path, str]] = field(default_factory=list)

    def file(self, path: Path) -> None:
        self.created.append(Path(path))

    def append(self, path: Path, text: str) -> None:
        """Append `text` to `path`, remembering it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
        self.appended.append((path, text))

    def take_back(self) -> tuple[int, list[str]]:
        """Undo everything recorded. Returns (how many items went, problems)."""
        done, problems = 0, []
        for path in dict.fromkeys(self.created):
            try:
                path.unlink()
                done += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                problems.append(f"couldn't delete {path}: {e.strerror or e}")
        by_file: dict[Path, list[str]] = {}
        for path, text in self.appended:
            by_file.setdefault(path, []).append(text)
        for path, texts in by_file.items():
            n, problem = cut(path, texts)
            done += n
            if problem:
                problems.append(problem)
        self.created.clear()
        self.appended.clear()
        return done, problems


def cut(path: Path, texts: list[str]) -> tuple[int, str]:
    """Remove each of `texts` (the last occurrence, newest first) from a file,
    rewriting it atomically with the same permissions. Returns (how many were
    found and removed, a problem or "")."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0, ""
    except (OSError, UnicodeDecodeError) as e:
        return 0, f"couldn't read {path} to take this session's lines out: {e}"
    removed = missing = 0
    for text in reversed(texts):
        i = content.rfind(text)
        if i == -1:
            missing += 1
            continue
        content = content[:i] + content[i + len(text):]
        removed += 1
    if removed:
        tmp = path.with_name(path.name + ".tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, path.stat().st_mode & 0o777)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, path)
        except OSError as e:
            return 0, f"couldn't rewrite {path}: {e.strerror or e}"
    problem = (f"{missing} of this session's entries weren't in {path} any more "
               "(edited or trimmed since?)") if missing else ""
    return removed, problem
