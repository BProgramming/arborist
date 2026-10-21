"""Workspace confinement and the five file tools."""
from __future__ import annotations

import os
import time

import pytest

from conftest import FakeApprover, write
from tools import Decision, ToolError, Workspace


# --------------------------------------------------------------- workspace ---

class TestWorkspace:
    def test_relative_and_absolute_paths_inside_are_allowed(self, ws_root):
        ws = Workspace(ws_root)
        assert ws.resolve("src/a.py") == ws_root / "src" / "a.py"
        assert ws.resolve(str(ws_root / "b.py")) == ws_root / "b.py"
        assert ws.resolve(".") == ws_root

    @pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt", "/etc/passwd", "~/x"])
    def test_paths_outside_are_rejected(self, ws_root, path):
        with pytest.raises(ToolError, match="outside the workspace"):
            Workspace(ws_root).resolve(path)

    def test_symlink_escaping_the_workspace_is_rejected(self, ws_root, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("nope")
        os.symlink(secret, ws_root / "link.txt")
        with pytest.raises(ToolError, match="outside the workspace"):
            Workspace(ws_root).resolve("link.txt")

    def test_sibling_directory_with_shared_prefix_is_rejected(self, ws_root):
        # /tmp/x/project-evil must not pass as "inside" /tmp/x/project
        (ws_root.parent / (ws_root.name + "-evil")).mkdir()
        with pytest.raises(ToolError):
            Workspace(ws_root).resolve(f"../{ws_root.name}-evil/f")


# --------------------------------------------------------------- read_file ---

class TestReadFile:
    def test_numbers_lines(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "one\ntwo\nthree\n")
        out = make_toolbox().call("read_file", {"path": "a.py"}).content
        assert out.splitlines() == ["     1\tone", "     2\ttwo", "     3\tthree"]

    def test_pages_long_files(self, ws_root, make_toolbox):
        write(ws_root, "long.txt", "\n".join(f"line {i}" for i in range(1, 51)))
        tb = make_toolbox(max_read_lines=20)
        first = tb.call("read_file", {"path": "long.txt"}).content
        assert "line 20" in first and "line 21" not in first
        assert "continue with offset=21" in first
        last = tb.call("read_file", {"path": "long.txt", "offset": 41}).content
        assert "line 50" in last and "[lines 41-50 of 50]" in last

    def test_limit_cannot_exceed_configured_max(self, ws_root, make_toolbox):
        write(ws_root, "long.txt", "\n".join(str(i) for i in range(100)))
        out = make_toolbox(max_read_lines=10).call("read_file", {"path": "long.txt", "limit": 1000}).content
        assert "[lines 1-10 of 100" in out

    def test_errors_come_back_as_text(self, ws_root, make_toolbox):
        (ws_root / "bin.dat").write_bytes(b"\x00\x01\x02")
        (ws_root / "dir").mkdir()
        tb = make_toolbox()
        assert "binary" in tb.call("read_file", {"path": "bin.dat"}).content
        assert "directory" in tb.call("read_file", {"path": "dir"}).content
        assert tb.call("read_file", {"path": "missing.py"}).content.startswith("error: no such file")
        assert "outside the workspace" in tb.call("read_file", {"path": "../x"}).content


# -------------------------------------------------------------- list_files ---

class TestListFiles:
    def test_matches_at_any_depth_and_skips_ignored_dirs(self, ws_root, make_toolbox):
        for rel in ["a.py", "src/b.py", "src/deep/c.py", "src/d.rs",
                    "node_modules/pkg/e.py", ".git/f.py", ".venv/lib/g.py"]:
            write(ws_root, rel, "")
        out = make_toolbox().call("list_files", {"pattern": "*.py"}).content.splitlines()
        assert out == ["a.py", "src/b.py", "src/deep/c.py"]

    def test_path_narrows_the_search(self, ws_root, make_toolbox):
        write(ws_root, "a.rs", "")
        write(ws_root, "src/b.rs", "")
        assert make_toolbox().call("list_files", {"pattern": "*.rs", "path": "src"}).content == "src/b.rs"

    def test_caps_long_listings(self, ws_root, make_toolbox):
        for i in range(12):
            write(ws_root, f"f{i:02}.txt", "")
        out = make_toolbox(max_list_entries=5).call("list_files", {}).content
        assert len(out.splitlines()) == 6 and "7 more not shown" in out


# ------------------------------------------------------------------- grep ---

class TestGrep:
    def test_finds_matches_with_line_numbers(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "import os\ndef main():\n    pass\n")
        write(ws_root, "b.rs", "fn main() {}\n")
        out = make_toolbox().call("grep", {"pattern": r"def main|fn main"}).content.splitlines()
        assert out == ["a.py:2: def main():", "b.rs:1: fn main() {}"]

    def test_glob_and_case(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "TODO: x\n")
        write(ws_root, "b.js", "todo: y\n")
        tb = make_toolbox()
        assert tb.call("grep", {"pattern": "todo", "glob": "*.js"}).content == "b.js:1: todo: y"
        assert len(tb.call("grep", {"pattern": "todo", "ignore_case": True}).content.splitlines()) == 2

    def test_bad_regex_and_binary_files(self, ws_root, make_toolbox):
        (ws_root / "x.bin").write_bytes(b"needle\x00\x00")
        tb = make_toolbox()
        assert "invalid regex" in tb.call("grep", {"pattern": "(unclosed"}).content
        assert tb.call("grep", {"pattern": "needle"}).content.startswith("no matches")


# -------------------------------------------------------------- edit_file ---

class TestEditFile:
    def test_must_read_before_editing(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "x = 1\n")
        out = make_toolbox().call("edit_file", {"path": "a.py", "old_string": "1", "new_string": "2"}).content
        assert "read a.py with read_file" in out
        assert (ws_root / "a.py").read_text() == "x = 1\n"

    def test_replaces_a_unique_match_after_approval(self, ws_root, make_toolbox, approver):
        write(ws_root, "a.py", "x = 1\ny = 2\n")
        tb = make_toolbox(approver)
        tb.call("read_file", {"path": "a.py"})
        out = tb.call("edit_file", {"path": "a.py", "old_string": "y = 2", "new_string": "y = 3"}).content
        assert out == "updated a.py (+1 -1)"
        assert (ws_root / "a.py").read_text() == "x = 1\ny = 3\n"
        rel, diff, is_new = approver.change_requests[0]
        assert rel == "a.py" and not is_new and "-y = 2" in diff and "+y = 3" in diff

    def test_ambiguous_match_is_refused_unless_replace_all(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "a = 0\na = 0\n")
        tb = make_toolbox()
        tb.call("read_file", {"path": "a.py"})
        out = tb.call("edit_file", {"path": "a.py", "old_string": "a = 0", "new_string": "a = 1"}).content
        assert "appears 2 times" in out
        tb.call("edit_file", {"path": "a.py", "old_string": "a = 0", "new_string": "a = 1", "replace_all": True})
        assert (ws_root / "a.py").read_text() == "a = 1\na = 1\n"

    def test_missing_text_gives_a_correctable_error(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "    indented = True\n")
        tb = make_toolbox()
        tb.call("read_file", {"path": "a.py"})
        out = tb.call("edit_file", {"path": "a.py", "old_string": "indented =  True", "new_string": "x"}).content
        assert "was not found" in out and "indentation" in out

    def test_refuses_if_file_changed_since_read(self, ws_root, make_toolbox):
        p = write(ws_root, "a.py", "v = 1\n")
        tb = make_toolbox()
        tb.call("read_file", {"path": "a.py"})
        time.sleep(0.01)
        p.write_text("v = 1  # edited in your editor\n")
        os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1_000_000))
        out = tb.call("edit_file", {"path": "a.py", "old_string": "v = 1", "new_string": "v = 2"}).content
        assert "changed on disk" in out
        assert "edited in your editor" in p.read_text()

    def test_declined_edit_leaves_file_alone_and_reports_reason(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "keep = True\n")
        tb = make_toolbox(FakeApprover(change=[Decision(False, "wrong variable")]))
        tb.call("read_file", {"path": "a.py"})
        out = tb.call("edit_file", {"path": "a.py", "old_string": "True", "new_string": "False"}).content
        assert "declined" in out and "wrong variable" in out
        assert (ws_root / "a.py").read_text() == "keep = True\n"

    def test_always_stops_asking_for_edits(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "n = 0\n")
        approver = FakeApprover(change=[Decision(True, always=True)])
        tb = make_toolbox(approver)
        tb.call("read_file", {"path": "a.py"})
        for i in range(3):
            tb.call("edit_file", {"path": "a.py", "old_string": f"n = {i}", "new_string": f"n = {i + 1}"})
        assert (ws_root / "a.py").read_text() == "n = 3\n"
        assert len(approver.change_requests) == 1

    def test_auto_mode_never_asks(self, ws_root, make_toolbox, approver):
        write(ws_root, "a.py", "a\n")
        tb = make_toolbox(approver, approve_edits="auto")
        tb.call("read_file", {"path": "a.py"})
        tb.call("edit_file", {"path": "a.py", "old_string": "a", "new_string": "b"})
        assert approver.change_requests == []


# ------------------------------------------------------------- write_file ---

class TestWriteFile:
    def test_creates_new_file_and_parents(self, ws_root, make_toolbox, approver):
        out = make_toolbox(approver).call("write_file", {"path": "pkg/new.py", "content": "print(1)\n"}).content
        assert out == "created pkg/new.py (+1 -0)"
        assert (ws_root / "pkg" / "new.py").read_text() == "print(1)\n"
        assert approver.change_requests[0][2] is True  # is_new

    def test_overwriting_requires_a_read(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "old\n")
        tb = make_toolbox()
        assert "read a.py with read_file" in tb.call("write_file", {"path": "a.py", "content": "new\n"}).content
        tb.call("read_file", {"path": "a.py"})
        tb.call("write_file", {"path": "a.py", "content": "new\n"})
        assert (ws_root / "a.py").read_text() == "new\n"

    def test_cannot_write_outside(self, ws_root, make_toolbox):
        out = make_toolbox().call("write_file", {"path": "../escape.py", "content": "x"}).content
        assert "outside the workspace" in out
        assert not (ws_root.parent / "escape.py").exists()

    def test_reset_forgets_reads(self, ws_root, make_toolbox):
        write(ws_root, "a.py", "x\n")
        tb = make_toolbox()
        tb.call("read_file", {"path": "a.py"})
        tb.reset()
        assert "read a.py with read_file" in tb.call("write_file", {"path": "a.py", "content": "y"}).content


# --------------------------------------------------------------- dispatch ---

class TestDispatch:
    def test_unknown_tool_and_bad_arguments(self, make_toolbox):
        tb = make_toolbox()
        assert "unknown tool" in tb.call("delete_everything", "{}").content
        assert "couldn't parse" in tb.call("read_file", "{not json").content
        assert "bad arguments" in tb.call("read_file", {"path": "a", "colour": "red"}).content
        assert "couldn't parse" in tb.call("read_file", "[1, 2]").content

    def test_internal_methods_are_not_callable(self, make_toolbox):
        assert "unknown tool" in make_toolbox().call("_apply", {}).content

    def test_output_is_truncated(self, ws_root, make_toolbox):
        write(ws_root, "big.txt", "x" * 5000)
        out = make_toolbox(max_output_chars=1000).call("read_file", {"path": "big.txt"}).content
        assert len(out) < 1200 and "characters omitted" in out
