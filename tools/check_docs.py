"""Check that the documentation's claims about the engine still hold.

Two rules, deliberately narrow:

1. Every symbol named in a ``**Code:**`` line exists in the file it names, and
   no bare ``::name`` reference appears in body prose, where it would silently
   inherit the file from whatever reference came before it.
2. Every constant a document cites by name carries the value the source gives
   it, in either of two shapes: a module-level constant written as
   ```NAME` = value``, or a keyword default written as ```func(kw=value)```.

Deliberately not checked: whether a formula in a document matches the
mathematics the code implements. That is not reliably automatable, and the
golden-master suite already locks the engine's numbers.

Run: ``python tools/check_docs.py``
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = sorted((ROOT / "pipeline").glob("*.py"))
DOCS = ["README.md", "CONTRIBUTING.md", "docs/MATHEMATICS.md",
        "docs/PHYSICS.md", "docs/STAGES.md", "docs/INPUT_FORMAT.md"]

REF = re.compile(r"`(pipeline/[\w/]+\.py)?::(\w+)`")
CONST = re.compile(r"`([A-Z][A-Z0-9_]+)`\s*=\s*([-\w.]+)")
KWARG = re.compile(r"`(\w+)\(\s*(\w+)\s*=\s*([-\w.]+)\s*\)`")
CODE_LINE = ("**Code:**", "`::")


def symbols(tree: ast.Module) -> set[str]:
    """Every function and class name defined anywhere in a module.

    >>> sorted(symbols(ast.parse("def a():\\n    def b(): pass")))
    ['a', 'b']
    """
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def constants(tree: ast.Module) -> dict[str, object]:
    """Module-level uppercase constants with a literal value.

    >>> constants(ast.parse("MIN_POINTS: int = 8"))
    {'MIN_POINTS': 8}
    """
    out: dict[str, object] = {}
    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else getattr(node, "targets", []))
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return out


def kwargs(tree: ast.Module) -> dict[tuple[str, str], object]:
    """Literal keyword defaults, keyed by (function name, keyword).

    >>> kwargs(ast.parse("def f(a, b=0.5): pass"))
    {('f', 'b'): 0.5}
    """
    out: dict[tuple[str, str], object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        named = args.args[len(args.args) - len(args.defaults):] if args.defaults else []
        pairs = list(zip(named, args.defaults)) + list(
            zip(args.kwonlyargs, args.kw_defaults))
        for arg, default in pairs:
            if default is None:
                continue
            try:
                out[(node.name, arg.arg)] = ast.literal_eval(default)
            except (ValueError, TypeError, SyntaxError):
                pass
    return out


def same(cited: str, actual: object) -> bool:
    """Compare a cited literal with a source value, tolerating float spelling.

    >>> same("0.50", 0.5)
    True
    >>> same("8", 8)
    True
    >>> same("9", 8)
    False
    """
    try:
        return float(cited) == float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return cited.strip("'\"") == str(actual)


def main() -> int:
    """Run both rules over every document and report every failure."""
    try:
        trees = {p.relative_to(ROOT).as_posix(): ast.parse(p.read_text(encoding="utf-8"))
                 for p in ENGINE}
    except (OSError, SyntaxError) as exc:
        print(f"check_docs: cannot parse the engine: {exc}", file=sys.stderr)
        return 1

    defs = {name: symbols(tree) for name, tree in trees.items()}
    consts: dict[str, object] = {}
    kws: dict[tuple[str, str], object] = {}
    for tree in trees.values():
        consts.update(constants(tree))
        kws.update(kwargs(tree))

    problems: list[str] = []
    for doc in DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            problems.append(f"{doc}: cannot read ({exc})")
            continue
        for no, line in enumerate(lines, 1):
            in_code_line = line.lstrip().startswith(CODE_LINE)
            for file_ref, name in REF.findall(line):
                if file_ref and name not in defs.get(file_ref, set()):
                    problems.append(f"{doc}:{no}: {file_ref}::{name} does not exist")
                elif not file_ref and not in_code_line:
                    problems.append(f"{doc}:{no}: bare ::{name} in prose, name the file")
            for name, cited in CONST.findall(line):
                if name in consts and not same(cited, consts[name]):
                    problems.append(
                        f"{doc}:{no}: {name} cited as {cited}, source says {consts[name]}")
            for func, kw, cited in KWARG.findall(line):
                actual = kws.get((func, kw))
                if actual is not None and not same(cited, actual):
                    problems.append(
                        f"{doc}:{no}: {func}({kw}=) cited as {cited}, source says {actual}")

    for problem in problems:
        print(problem)
    print(f"check_docs: {len(problems)} problem(s) in {len(DOCS)} document(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
