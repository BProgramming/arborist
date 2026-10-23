"""view_tree, outline_file, and .gitignore-aware walking."""
from __future__ import annotations

import subprocess

import pytest

import outline
from conftest import needs, write


def git_init(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)


# --------------------------------------------------------------- view_tree ---

def test_tree_lists_directories_first_and_skips_noise(ws_root, make_toolbox):
    write(ws_root, "main.py", "")
    write(ws_root, "src/app/core.py", "")
    write(ws_root, "src/util.py", "")
    write(ws_root, "node_modules/pkg/index.js", "")
    out = make_toolbox().call("view_tree", {}).content
    assert out.splitlines() == [
        "./ (3 files)",
        "  src/",
        "    app/",
        "      core.py",
        "    util.py",
        "  main.py",
    ]


def test_tree_collapses_below_depth_and_summarises_big_directories(ws_root, make_toolbox):
    write(ws_root, "a/b/c/deep.py", "")
    for i in range(30):
        write(ws_root, f"data/file{i:02}.json", "")
    write(ws_root, "data/readme.md", "")
    out = make_toolbox().call("view_tree", {"depth": 2}).content
    assert "  a/\n    b/ (1 files)" in out
    assert "… 16 more files (.json ×15, .md ×1)" in out


def test_tree_depth_is_capped_and_bad_paths_error(ws_root, make_toolbox):
    write(ws_root, "a/b/c/d/e/f.py", "")
    tb = make_toolbox(max_tree_depth=2)
    assert "b/ (1 files)" in tb.call("view_tree", {"depth": 10}).content
    assert tb.call("view_tree", {"path": "nope"}).content.startswith("error: no such directory")
    write(ws_root, "x.py", "")
    assert "not a directory" in tb.call("view_tree", {"path": "x.py"}).content


@needs("git")
def test_gitignored_files_are_left_out_of_tree_list_and_grep(ws_root, make_toolbox):
    git_init(ws_root)
    write(ws_root, ".gitignore", "build/\n*.log\n.env\n")
    write(ws_root, "app.py", "SECRET = load()\n")
    write(ws_root, "build/out.py", "SECRET = 'compiled'\n")
    write(ws_root, "debug.log", "SECRET leaked\n")
    write(ws_root, ".env", "SECRET=hunter2\n")
    tb = make_toolbox()
    tree = tb.call("view_tree", {}).content
    assert "app.py" in tree and "build" not in tree and "debug.log" not in tree and ".env" not in tree
    assert tb.call("list_files", {"pattern": "*.py"}).content == "app.py"
    assert tb.call("grep", {"pattern": "SECRET"}).content == "app.py:1: SECRET = load()"
    # Asking for an ignored directory by name still works…
    assert "build/out.py" in tb.call("grep", {"pattern": "SECRET", "path": "build"}).content
    # …and so does reading an ignored file directly.
    assert "hunter2" in tb.call("read_file", {"path": ".env"}).content


@needs("git")
def test_untracked_files_still_show_in_a_repo(ws_root, make_toolbox):
    git_init(ws_root)
    write(ws_root, "new_module.py", "x = 1\n")
    assert "new_module.py" in make_toolbox().call("view_tree", {}).content


# -------------------------------------------------------------- outline ---

PY = '''\
"""Module doc."""
import os
from pathlib import Path

LIMIT = 10


class Store(Base):
    """Keeps things."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def size(self) -> int:
        return 0

    async def load(self, key, *, retries=3):
        pass


def helper(a, b=2):
    """Adds."""
    return a + b


if __name__ == "__main__":
    helper(1)
'''


def test_python_outline_has_line_ranges_and_signatures():
    out = outline.outline(PY, ".py")
    lines = out.splitlines()
    assert lines[0] == "L2-3 imports: os, pathlib.Path"
    assert "L5 LIMIT = …" in lines
    assert "L8-19 class Store(Base)  — Keeps things." in lines
    assert "  L11-12 def __init__(self, root: Path)" in lines
    assert "  L14-16 @property def size(self) -> int" in lines
    assert "  L18-19 async def load(self, key, *, retries=3)" in lines
    assert "L22-24 def helper(a, b=2)  — Adds." in lines
    assert "L27-28 if __name__ == '__main__': …" in lines
    assert lines[-1] == "[28 lines]"


def test_outline_tool_points_at_read_file(ws_root, make_toolbox):
    write(ws_root, "store.py", PY)
    out = make_toolbox().call("outline_file", {"path": "store.py"}).content
    assert out.startswith("Outline of store.py") and "offset/limit" in out
    # Outlining doesn't count as reading: an edit still needs read_file first.
    res = make_toolbox().call("edit_file", {"path": "store.py", "old_string": "LIMIT = 10",
                                            "new_string": "LIMIT = 11"})
    assert "read store.py with read_file" in res.content


def test_python_syntax_error_falls_back_to_patterns():
    out = outline.outline("def ok():\n    pass\n\ndef broken(:\n", ".py")
    assert "syntax error at line 4" in out and "L1 def ok():" in out


def test_typescript_outline():
    src = '''import { x } from "y";
export interface Opts { a: number }
export class Server extends Base {
  private port = 0;
  constructor(opts: Opts) {
    super();
  }
  async start(): Promise<void> {
    if (this.port) {
    }
  }
}
export const handler = async (req: Request) => {
  return 1;
};
function local(a, b) {
}
'''
    out = outline.outline(src, ".ts")
    assert "(1 import line at the top)" in out
    assert "L2 export interface Opts { a: number }" in out
    assert "L3 export class Server extends Base" in out
    assert "  L5 constructor(opts: Opts)" in out
    assert "  L8 async start(): Promise<void>" in out
    assert "L13 export const handler = async (req: Request) =>" in out
    assert "L16 function local(a, b)" in out
    assert "if (" not in out


def test_rust_outline():
    src = '''use std::io;
pub struct Point { x: i32 }
impl Point {
    pub fn new(x: i32) -> Self { Point { x } }
    async fn fetch(&self) {}
}
pub(crate) enum Kind { A }
const MAX: usize = 3;
fn main() {}
'''
    out = outline.outline(src, ".rs")
    for expected in ("L2 pub struct Point { x: i32 }", "L3 impl Point", "  L4 pub fn new(x: i32) -> Self",
                     "  L5 async fn fetch(&self) {}", "L7 pub(crate) enum Kind { A }",
                     "L8 const MAX: usize = 3;", "L9 fn main() {}"):
        assert expected in out


def test_dart_outline():
    src = '''import 'package:flutter/material.dart';
class Counter extends StatelessWidget {
  final int count;
  const Counter({super.key, required this.count});
  @override
  Widget build(BuildContext context) {
    if (count > 0) {
    }
    return Text('$count');
  }
  Future<void> save() async {
  }
}
void main() => runApp(Counter(count: 1));
'''
    out = outline.outline(src, ".dart")
    assert "L2 class Counter extends StatelessWidget" in out
    assert "  L6 Widget build(BuildContext context)" in out
    assert "  L11 Future<void> save() async" in out
    assert "L14 void main() => runApp(Counter(count: 1));" in out
    assert "if (" not in out


def test_outline_rejects_unknown_types(ws_root, make_toolbox):
    write(ws_root, "notes.md", "# hi\n")
    assert "no outline for .md" in make_toolbox().call("outline_file", {"path": "notes.md"}).content
