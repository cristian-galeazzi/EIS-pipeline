"""One sweep for the rule no single test could hold: a value and its unit.

ISO 80000-1 and the IUPAC Green Book put a space between a numerical value and
its unit symbol. The defect is mechanical and spreads by copy: every stage grew
its own "600°C". A per-file test would have to be written six times and would
still miss the seventh file, so this walks the whole project instead.

What it checks is the end of an interpolation closed against a unit, the only
shape this defect takes in an f-string. A literal that already reads "600 °C"
is invisible to it, as is a plural like "{n} files".

The three engine files are swept like everything else and are clean, so the
guard needs no exception for them.
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A unit closed against the brace that ends an interpolation. Multi-letter and
# non-letter units are unambiguous. A single letter takes the lookahead, so a
# plural ("{n} files") is not read as a second.
GLUED_UNIT = re.compile(r"\}(°C|%|Ω|bar|eV|Hz)|\}[sFKVA](?![A-Za-z0-9])")
# Jupyter line magics are not Python; ast cannot parse a cell that opens with one
MAGIC = re.compile(r"^\s*[%!]", re.M)

SWEPT_PY = sorted(ROOT.glob("pipeline/*.py")) + sorted(ROOT.glob("tools/*.py"))
SWEPT_NB = sorted(ROOT.glob("stage*.ipynb"))


def _string_literals(src: str) -> list[str]:
    """Every string literal in a source, f-strings as written.

    An f-string is returned as its source segment rather than as its parts: the
    defect lives exactly at the seam between an interpolation and the text after
    it, which ``ast`` would otherwise hand over already split. Its own fragments
    are therefore dropped, or the seam would be reported twice and the second
    report would name a literal nobody wrote.

    >>> _string_literals('x = f"{T}°C"')
    ['f"{T}°C"']
    >>> _string_literals('x = "plain"')
    ['plain']
    """
    tree = ast.parse(MAGIC.sub("pass  #", src))
    fragments = {id(part)
                 for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                 for part in ast.walk(node) if isinstance(part, ast.Constant)}
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            seg = ast.get_source_segment(src, node)
            if seg:
                out.append(seg)
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in fragments):
            out.append(node.value)
    return out


def _offenders() -> list[str]:
    """Every ``where: literal`` in the project that closes a unit against a value.

    >>> isinstance(_offenders(), list)
    True
    """
    found: list[str] = []
    for py in SWEPT_PY:
        for lit in _string_literals(py.read_text(encoding="utf-8")):
            if GLUED_UNIT.search(lit):
                found.append(f"{py.relative_to(ROOT)}: {lit!r}")
    for nb in SWEPT_NB:
        doc = json.loads(nb.read_text(encoding="utf-8"))
        for i, cell in enumerate(doc["cells"]):
            if cell["cell_type"] != "code":
                continue
            for lit in _string_literals("".join(cell["source"])):
                if GLUED_UNIT.search(lit):
                    found.append(f"{nb.name} cell {i}: {lit!r}")
    return found


def test_no_value_is_closed_against_its_unit() -> None:
    assert _offenders() == []


def test_the_sweep_reaches_every_stage_and_every_module() -> None:
    # the guard is only as good as its reach: a renamed notebook or a new
    # pipeline module must not drop out of it silently
    assert len(SWEPT_NB) == 6
    assert {p.name for p in SWEPT_PY} >= {
        "matching.py", "plots.py", "utils.py", "interactive.py",
        "quality.py", "drt.py", "fitting.py",
    }


def test_the_sweep_can_see_the_defect_it_guards_against() -> None:
    # a regex that matches nothing would pass the test above for ever
    assert GLUED_UNIT.search('f"{T}°C"')
    assert GLUED_UNIT.search('f"{t:.0f}s"')
    assert not GLUED_UNIT.search('f"{T} °C"')
    assert not GLUED_UNIT.search('f"{n} files"')
