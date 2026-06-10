"""
Shared session.json persistence for all stage notebooks.

session.json is a JSON array of sample objects keyed by ``sample_id``.
Every notebook previously carried its own copy of the load/save logic;
this module replaces those copies with a single implementation that is

* atomic: written to a temp file then ``os.replace``d, so an interrupted
             save can never truncate session.json;
* backed up: the previous version is kept as ``session.json.bak``;
* merge-safe: per-condition dictionaries (``condition_params``,
             ``kk_overrides``, ``overrides``) are merged key-by-key, so
             re-running a notebook that only touched one condition cannot
             wipe the calibrated parameters of the others.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

SESSION_FILE = Path("session.json")

# Dict-of-dict keys holding per-condition calibrations: merged, never replaced
# wholesale, because a notebook session may legitimately hold only a subset
# of the conditions that were calibrated in earlier sessions.
MERGE_KEYS = frozenset({"condition_params", "kk_overrides", "overrides"})


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and nested containers) to plain Python."""
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def load_session(path: Path | str = SESSION_FILE) -> list[dict]:
    """
    Load session.json as a list of sample entries (legacy single-dict files
    are wrapped). Returns [] if the file does not exist.

    Raises ValueError on corrupted JSON instead of silently starting from
    scratch, since overwriting a corrupted file would destroy recoverable data.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{path} is unreadable or corrupted ({exc}). "
            f"A previous version may exist at {path}.bak"
        ) from exc
    if not isinstance(data, list):
        data = [data] if data else []
    return data


def load_sample(sample_id: str, path: Path | str = SESSION_FILE) -> dict:
    """Return the entry for ``sample_id`` ({} if absent)."""
    data = load_session(path)
    return next((c for c in data if c.get("sample_id") == sample_id), {})


def _atomic_write(path: Path, data: list[dict]) -> None:
    if path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass  # a failed backup must not block the save itself
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent or "."),
                                    prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def update_sample(
    sample_id: str,
    /,
    path: Path | str = SESSION_FILE,
    replace: bool = False,
    **fields: Any,
) -> None:
    """
    Read-modify-write the entry for ``sample_id`` (created if missing).

    Keys in MERGE_KEYS are merged per condition (and per temperature for
    nested dicts) so partial sessions cannot erase earlier calibrations.
    Pass ``replace=True`` to overwrite those keys wholesale, e.g. to
    intentionally delete a condition's stored parameters.
    """
    path = Path(path)
    data = load_session(path)
    entry = next((c for c in data if c.get("sample_id") == sample_id), None)
    if entry is None:
        entry = {"sample_id": sample_id}
        data.append(entry)

    for key, value in fields.items():
        value = _to_jsonable(value)
        if (not replace and key in MERGE_KEYS
                and isinstance(value, dict)
                and isinstance(entry.get(key), dict)):
            merged = entry[key]
            for cond, sub in value.items():
                if isinstance(sub, dict) and isinstance(merged.get(cond), dict):
                    merged[cond].update(sub)
                else:
                    merged[cond] = sub
        else:
            entry[key] = value

    _atomic_write(path, data)
