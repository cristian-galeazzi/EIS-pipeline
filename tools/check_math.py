#!/usr/bin/env python3
"""Check that every equation in the documentation renders on GitHub.

GitHub runs its Markdown parser before the math extension, so a bare ``$...$``
span loses the underscores and backslashes Markdown claims for itself and
KaTeX is handed a mangled string. Wrapping inline math in backticks,
``$`...`$``, makes it a code span that Markdown leaves alone.

This checker enforces that form and the other constraints GitHub's renderer
imposes. It says nothing about whether a formula is correct: that is what
review is for.

Notebooks are deliberately not checked. The backtick form is a GitHub
extension that Jupyter does not implement, so ``.ipynb`` markdown keeps bare
``$...$`` and is rendered by a different viewer.

Exit code 0 when clean, 1 with a report otherwise.

>>> bool(SAFE_MACROS)
True
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_PREFIX = "docs/superpowers/"

#: Macros GitHub's KaTeX build accepts. A macro outside this set is reported
#: rather than rejected outright: the set grows as the documents need it.
SAFE_MACROS = frozenset("""
text mathrm mathbf mathsf mathbb boldsymbol frac sqrt sum int prod
left right lVert rVert lvert rvert langle rangle begin end vdots cdots ldots dots
alpha beta gamma delta Delta epsilon varepsilon theta lambda mu nu pi rho sigma
Sigma tau phi varphi omega Omega infty partial nabla propto approx neq leq geq ll gg
pm mp times cdot in notin subset cup cap exp ln log min max arg dim det
quad qquad colon to mapsto hat bar tilde vec dot ddot
mathcal mathfrak overline underline binom
""".split())

INLINE = re.compile(r"\$`([^`\n]+)`\$")
BARE = re.compile(r"(?<!`)\$(?!`)[^$\n]*\$")
DISPLAY = re.compile(r"^```math\n(.*?)\n```$", re.S | re.M)


def tracked_markdown() -> list[pathlib.Path]:
    """Every tracked .md file the documentation rules apply to.

    >>> all(p.suffix == ".md" for p in tracked_markdown())
    True
    """
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split()
    return [ROOT / f for f in out if not f.startswith(SKIP_PREFIX)]


def prose_lines(text: str):
    """Yield (line number, line) for lines outside fenced blocks.

    >>> list(prose_lines("a\\n```\\nb\\n```\\nc"))
    [(1, 'a'), (5, 'c')]
    """
    fenced = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            yield i, line


def check_span(body: str) -> list[str]:
    """Structural problems in one math span, empty when it is sound.

    >>> check_span(r"\\frac{a}{b}")
    []
    >>> check_span(r"\\operatorname{x}")
    ['\\\\operatorname is rejected by GitHub']
    """
    problems: list[str] = []
    if body.count("{") != body.count("}"):
        problems.append("unbalanced braces")
    if body.count(r"\left") != body.count(r"\right"):
        problems.append(r"unbalanced \left/\right")
    if body.count(r"\begin") != body.count(r"\end"):
        problems.append(r"unbalanced \begin/\end")
    if r"\operatorname" in body:
        problems.append(r"\operatorname is rejected by GitHub")
    if r"\!" in body:
        problems.append(r"\! is ignored by GitHub and prints literally")
    # "operatorname" already has its own message above; do not report it twice.
    unknown = sorted({m for m in re.findall(r"\\([a-zA-Z]+)", body)
                      if m not in SAFE_MACROS and m != "operatorname"})
    if unknown:
        problems.append("macro outside the accepted set: " + ", ".join(unknown))
    return problems


def check_file(path: pathlib.Path) -> list[str]:
    """Every rendering problem in one document."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable ({exc})"]

    rel = path.relative_to(ROOT)
    report: list[str] = []

    for i, line in prose_lines(text):
        outside = INLINE.sub("", line)
        if BARE.search(outside):
            report.append(f"{rel}:{i}: inline math not wrapped in backticks; "
                          "Markdown will eat its underscores")
        if outside.count("$") % 2:
            report.append(f"{rel}:{i}: math span crosses a line break; "
                          "GitHub cannot render it")
        if line.startswith("#") and ("$" in line or "\\" in line):
            report.append(f"{rel}:{i}: LaTeX in a heading")
        for m in INLINE.finditer(line):
            if m.group(1) != m.group(1).strip():
                report.append(f"{rel}:{i}: whitespace against a math delimiter")

    spans = [(m.group(1), m.start()) for m in DISPLAY.finditer(text)]
    spans += [(m.group(1), m.start()) for m in INLINE.finditer(DISPLAY.sub("", text))]
    for body, pos in spans:
        line = text.count("\n", 0, pos) + 1
        for problem in check_span(body):
            report.append(f"{rel}:{line}: {problem}")

    return report


def main() -> int:
    """Report every documentation rendering problem; 0 when clean."""
    files = tracked_markdown()
    report: list[str] = []
    for path in files:
        report.extend(check_file(path))

    if report:
        print("check_math: documentation will not render correctly on GitHub\n")
        for line in report:
            print(f"  {line}")
        print(f"\n{len(report)} problem(s) in {len(files)} document(s)")
        return 1

    print(f"check_math: 0 problem(s) in {len(files)} document(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
