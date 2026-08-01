# Contributing

Thanks for your interest in the EIS Analysis Pipeline.

## Questions and bug reports

Open a GitHub issue. For bugs, include the stage notebook, the cell that failed,
the full traceback, and (if possible) a minimal CSV spectrum that reproduces the
problem; never attach raw measurement data you cannot share.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Set up the environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install nbstripout && nbstripout --install
   ```

   `nbstripout --install` is required: it strips notebook outputs on commit, so
   no measurement data or sample names ever land in the history. The filter is
   declared in `.gitattributes` but each clone must install the tool once.
3. Make your changes and run the regression suite:

   ```bash
   pytest tests/test_engine_golden.py
   ```

   All tests must pass.

## The byte-identical engine constraint

The calculation engine (`pipeline/quality.py`, `pipeline/drt.py`,
`pipeline/fitting.py`) and every non-Metadata `.xlsx` sheet must produce
byte-identical numeric outputs across refactors. Changes to these modules are
limited to guards on degenerate inputs that currently crash; anything that
alters the numbers needs a dedicated discussion in an issue first.

Plot styling (`pipeline/plots.py`), notebook UX (`pipeline/interactive.py`,
the notebooks themselves) and documentation are free to evolve.

## Documentation and the engine

`tools/check_docs.py` runs in CI and enforces two things about the documents:

- every symbol named in a `**Code:**` line exists in the file it names, and a
  reference in ordinary prose spells out its file rather than writing a bare
  `::name`, which would silently inherit the file from an earlier reference;
- every constant a document cites by name, as `` `NAME` = value `` or as
  `` `func(kw=value)` ``, matches the value in `pipeline/`.

It does not check that a formula matches the mathematics the code implements.
Nothing can, reliably. The golden-master suite locks the engine's numbers, and
derivations are reviewed by a person.

When the checker reports a disagreement, the document is what changes: the code
is the specification. If the engine itself is wrong, fix it in its own commit,
never inside a `docs:` change.
