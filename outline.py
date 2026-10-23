"""A file's structure without its bodies: classes, functions, methods, imports.

Python is parsed properly (ast), with line ranges, so the model can read just
the function it wants with read_file(offset, limit). Rust, Dart, JavaScript
and TypeScript get a line-based outline from regexes: approximate, but it
costs a few hundred tokens instead of the whole file.
"""
from __future__ import annotations

import ast
import re

OUTLINE_SUFFIXES = {".py", ".pyi", ".rs", ".dart", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"}

_MAX_SIG = 140  # characters of a signature before it's cut


def outline(source: str, suffix: str) -> str:
    """Outline `source` (the text of a file with this suffix)."""
    if suffix in (".py", ".pyi"):
        try:
            return _python(source)
        except SyntaxError as e:
            body = _regex(source, _PATTERNS["py"])
            return f"(syntax error at line {e.lineno}; approximate outline)\n{body}"
    lang = {".rs": "rs", ".dart": "dart"}.get(suffix, "js")
    return "(approximate, from pattern matching)\n" + _regex(source, _PATTERNS[lang])


# ----------------------------------------------------------------- python ---

def _cut(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= _MAX_SIG else s[: _MAX_SIG - 1] + "…"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return _cut(f"{prefix} {node.name}({ast.unparse(node.args)}){ret}")


def _decorators(node) -> str:
    # Only the short, meaningful ones (@property, @staticmethod, @dataclass…).
    names = []
    for d in node.decorator_list:
        text = ast.unparse(d)
        if len(text) <= 40:
            names.append("@" + text)
    return " ".join(names) + " " if names else ""


def _doc(node) -> str:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return ""
    first = doc.strip().splitlines()[0]
    return "  — " + (first if len(first) <= 80 else first[:79] + "…")


def _span(node) -> str:
    # Start at the first decorator, so read_file from here shows it too.
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = getattr(node, "end_lineno", None) or node.lineno
    return f"L{start}" if end == start else f"L{start}-{end}"


def _python(source: str) -> str:
    tree = ast.parse(source)
    out: list[str] = []

    imports, first_import, last_import = [], None, None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            first_import = first_import or node.lineno
            last_import = node.end_lineno or node.lineno
            if isinstance(node, ast.Import):
                imports += [a.name for a in node.names]
            else:
                mod = "." * node.level + (node.module or "")
                imports += [f"{mod}.{a.name}" if mod else a.name for a in node.names]
    if imports:
        shown = ", ".join(imports[:40]) + (f", … {len(imports) - 40} more" if len(imports) > 40 else "")
        out.append(f"L{first_import}-{last_import} imports: {shown}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"{_span(node)} {_decorators(node)}{_signature(node)}{_doc(node)}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            out.append(f"{_span(node)} {_decorators(node)}class {node.name}"
                       f"{f'({_cut(bases)})' if bases else ''}{_doc(node)}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"  {_span(item)} {_decorators(item)}{_signature(item)}")
                elif isinstance(item, ast.ClassDef):
                    out.append(f"  {_span(item)} class {item.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            # Module-level constants and dunders; skip ordinary variables.
            names = [n for n in names if n.isupper() or (n.startswith("__") and n.endswith("__"))]
            if names:
                out.append(f"{_span(node)} {', '.join(names)} = …")
        elif isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            out.append(f"{_span(node)} if __name__ == '__main__': …")
    total = len(source.splitlines())
    return "\n".join(out) + f"\n[{total} lines]" if out else f"(no top-level definitions; {total} lines)"


# ---------------------------------------------------------------- regexes ---

_KEYWORDS = r"(?!(?:if|for|while|switch|catch|return|else|new|await|throw|do|try|with)\b)"

_PATTERNS: dict[str, list[re.Pattern]] = {
    "py": [re.compile(r"^\s*(async\s+def|def|class)\s+\w+")],
    "rs": [
        re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(async\s+|const\s+|unsafe\s+|extern\s+\"C\"\s+)*fn\s+\w+"),
        re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(struct|enum|trait|union|type|mod)\s+\w+"),
        re.compile(r"^\s*(unsafe\s+)?impl\b"),
        re.compile(r"^(pub(\([^)]*\))?\s+)?(const|static)\s+[A-Z_][A-Z0-9_]*\s*:"),
        re.compile(r"^\s*macro_rules!\s*\w+"),
    ],
    "dart": [
        re.compile(r"^\s*(abstract\s+|sealed\s+|base\s+|final\s+|interface\s+)*(class|mixin|enum|extension|typedef)\b"),
        # Functions and methods: `Type name(args) {`, `name(args) async {`, `=> …`.
        re.compile(r"^\s*(?:@\w+\s+)*(?:(?:static|external|factory|const|final)\s+)*"
                   r"(?:[\w.]+(?:<[^()]*?>)?\??\s+)?(?:get\s+|set\s+)?"
                   + _KEYWORDS + r"[\w.]+\s*\([^;]*\)\s*(?:async\*?|sync\*)?\s*(?:\{|=>)"),
    ],
    "js": [
        re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\b"),
        re.compile(r"^\s*(export\s+)?(default\s+)?(abstract\s+)?class\b"),
        re.compile(r"^\s*(export\s+)?(declare\s+)?(interface|type|enum|namespace)\s+\w+"),
        re.compile(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*(:[^=]+)?=\s*(async\s*)?"
                   r"(function\b|\([^)]*\)\s*(:[^=]+)?=>|\w+\s*=>)"),
        # Class members: `name(args) {`, `async name(…) {`, `static get x() {`.
        re.compile(r"^\s+(?:public\s+|private\s+|protected\s+|static\s+|readonly\s+|async\s+|get\s+|set\s+|\*)*"
                   + _KEYWORDS + r"#?\w+\s*(<[^>]*>)?\s*\([^;]*\)\s*(:[^{=]+)?\{\s*$"),
    ],
}


def _regex(source: str, patterns: list[re.Pattern]) -> str:
    out = []
    imports = 0
    lines = source.splitlines()
    for n, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"(import|use|export \* from|from \S+ import)\b", stripped) and not line[:1].isspace():
            imports += 1
            continue
        if any(p.match(line) for p in patterns):
            indent = len(line) - len(line.lstrip())
            depth = min(indent // 2, 6)
            sig = stripped.rstrip("{").rstrip()
            out.append(f"{'  ' * depth}L{n} {_cut(sig)}")
    head = [f"({imports} import line{'s' * (imports != 1)} at the top)"] if imports else []
    if not out:
        return "\n".join(head + [f"no definitions found; {len(lines)} lines"])
    return "\n".join(head + out + [f"[{len(lines)} lines]"])
