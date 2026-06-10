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
