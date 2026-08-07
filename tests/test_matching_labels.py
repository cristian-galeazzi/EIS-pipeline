"""The furnace figures are the only ones drawn outside pipeline/plots.py.

They were written before the module-wide typography rule and set their axis
labels as plain text, so the quantity symbol T came out upright. IUPAC Green
Book and ISO 80000-1: a quantity symbol is italic. This test reaches the label
strings through the source, the way tests/test_plots.py does, because drawing
either figure needs a parsed furnace log.
"""

import ast
import re
from pathlib import Path

from matplotlib import mathtext

MATCHING_SRC = Path(__file__).resolve().parents[1] / "pipeline" / "matching.py"
# what a label may never set outside a math span: the solidus that separates a
# quantity from its unit, and the degree sign. Checking the leftovers rather
# than the label catches "Time / s" as well as a bare "T / °C"
OUTSIDE_MATH = re.compile(r"[/°]")


def _outside_math(label: str) -> str:
    """The parts of a label matplotlib sets in the text font.

    >>> _outside_math(r"$t$  (D:HH:MM:SS)")
    '  (D:HH:MM:SS)'
    >>> _outside_math("T / °C")
    'T / °C'
    """
    return "".join(label.split("$")[::2])


def _label_literals() -> list[str]:
    """Every string literal handed to set_xlabel or set_ylabel in matching.py.

    >>> len(_label_literals()) >= 4
    True
    """
    out = []
    for node in ast.walk(ast.parse(MATCHING_SRC.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("set_xlabel", "set_ylabel")
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            out.append(node.args[0].value)
    return out


def test_no_axis_label_states_a_quantity_in_plain_text() -> None:
    offenders = [s for s in _label_literals()
                 if OUTSIDE_MATH.search(_outside_math(s))]
    assert offenders == []


def test_every_axis_label_parses_as_mathtext() -> None:
    # an unbalanced $ prints the label's own markup on the figure
    parser = mathtext.MathTextParser("path")
    for label in _label_literals():
        parts = label.split("$")
        assert len(parts) % 2, f"unbalanced $ in {label!r}"
        for span in parts[1::2]:
            parser.parse(f"${span}$")
