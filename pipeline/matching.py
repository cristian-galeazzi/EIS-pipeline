"""
pipeline/matching.py
====================
Parse furnace log files and match ISM measurements to furnace conditions.

Furnace log format (Eurotherm .txt export)
------------------------------------------
===Sample===
Sample_ID_Ar_SCCM_O2_SCCM_Tmax_Tmin_delta
===Start date===
DD.MM.YYYY_HH:MM:SS
===Data===
Time/s  Tsample (C)  dT/dt sample  Toven (C)  dT/dt oven  Tlambda (C)  dT/dt lambda  pO2 (bar)  ...
t         xE+n           NaN       xE+n           NaN       ...

Column indices (0-based)
------------------------
0 : Time/s     (elapsed seconds since measurement start)
1 : Tsample    (sample thermocouple temperature, °C)
3 : Toven      (oven thermocouple temperature, °C)
7 : pO2        (oxygen partial pressure from lambda probe, bar)

Key conventions
---------------
- Decimal separator: comma  -> replaced with dot before parsing
- Start date format: DD.MM.YYYY_HH:MM:SS
- Absolute datetime = start_dt + timedelta(seconds=elapsed_s)
- pO2 assigned to each ISM window = (max + min) / 2  (not arithmetic mean)
- SCCM labels in folder/file names are gas flow settings,
  NOT the actual pO2 -> always read pO2 from the .txt file (bar)
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from io import StringIO # treat modified log text as file-like object for pd.read_csv
from pathlib import Path
from typing import Optional
from .ingest import IsmRecord, extract_T_from_filename, extract_replica_from_filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def _round_sigfigs(x: float, n: int = 6) -> float:
    """
    Round x to n significant figures.

    Works correctly at any order of magnitude: safe for pO2 values
    ranging from ~1 bar (pure O2) down to 10^-20 bar (very reducing).

    Examples
    --------
    _round_sigfigs(0.001923456789, 6)  ->  0.00192346
    _round_sigfigs(1.923456789e-18, 6) ->  1.92346e-18
    _round_sigfigs(0.0, 6)             ->  0.0
    """
    if x == 0.0 or not math.isfinite(x):
        return x
    magnitude = math.floor(math.log10(abs(x)))
    return round(x, -int(magnitude) + (n - 1))




# ---------------------------------------------------------------------------
# Furnace log parsing
# ---------------------------------------------------------------------------

def find_furnace_log(sample_dir: Path, condition_folder: str) -> Path:
    """
    Automatically locate the furnace .txt log that matches a condition folder.

    The condition key is extracted from the folder name by stripping the
    sample prefix (e.g. 'SampleID') and matched against files in
    the 'Raw oven*' directory (handles trailing space in folder name).

    Parameters
    ----------
    sample_dir       : root sample folder, e.g. EIS program/SampleID/
    condition_folder : condition subfolder name,
                       e.g. 'SampleID_Ar-SCCM_O2-SCCM_Tmax_Tmin_delta'

    Returns
    -------
    Path to the matching .txt furnace log file.

    Raises
    ------
    FileNotFoundError if no match is found.
    """
    # Locate the "Raw oven" folder (may have trailing space)
    raw_oven_candidates = list(sample_dir.glob("Raw oven*"))
    if not raw_oven_candidates:
        raise FileNotFoundError(f"No 'Raw oven' folder found in {sample_dir}")
    raw_oven_dir = raw_oven_candidates[0]

    # Extract condition key by stripping the sample prefix
    # e.g. 'SampleID_Ar-SCCM_O2-SCCM_Tmax_Tmin_delta' -> 'Ar-SCCM_O2-SCCM_Tmax_Tmin_delta'
    parts = condition_folder.split("_")
    gas_tokens = []
    collecting = False
    for p in parts:    
        # p is each token from the folder name split by '_'
        if not collecting:
            # Start collecting from the first recognised gas token (Ar, O2, N2, H2, H2O, CO2, CO, He)
            if p and p[0].isalpha() and not p.isdigit():
                if p.upper() in ("AR", "O2", "N2", "H2", "CO2", "CO", "HE", "H2O") or re.match(r"^(Ar|O2|N2|H2O|H2|CO2|CO|He)", p, re.I):
                    collecting = True
                    gas_tokens.append(p)
        else:
            gas_tokens.append(p)
    condition_key = "_".join(gas_tokens)  # e.g. 'Ar-SCCM_O2-SCCM_Tmax_Tmin_delta'

    # Search for a .txt file containing the condition key
    txt_files = list(raw_oven_dir.glob("*.txt"))
    for f in txt_files:
        if condition_key in f.stem:
            return f

    raise FileNotFoundError(
        f"No furnace log found for condition key '{condition_key}' "
        f"in {raw_oven_dir}.\nAvailable files: {[f.name for f in txt_files]}"
    )


def parse_oven_file(filepath: Path) -> dict:
    """
    Parse one furnace .txt file and return a dictionary with raw data.

    Returns
    -------
    dictionary with keys:
        sample_name   : str
        start_date_raw: str  (raw string from ===Start date===)
        start_dt      : datetime (full absolute start datetime)
        start_seconds : float (seconds since midnight, kept for compatibility)
        filepath      : Path
        df            : DataFrame with columns:
                        abs_datetime, Time_s, Tsample, Toven, pO2
    """
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    sample_name    = ""
    start_date_raw = ""
    start_dt       = None
    start_seconds  = 0.0
    data_start     = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "===Sample===":
            sample_name = lines[i + 1].strip()
        elif stripped == "===Start date===":
            start_date_raw = lines[i + 1].strip()
            # Full datetime: DD.MM.YYYY_HH:MM:SS
            try:
                start_dt = datetime.strptime(start_date_raw, "%d.%m.%Y_%H:%M:%S")
            except ValueError:
                pass
            # Fallback: extract HH:MM:SS from end of string
            m = re.search(r"(\d{2}):(\d{2}):(\d{2})$", start_date_raw)
            if m:
                h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                start_seconds = h * 3600 + mn * 60 + s
        elif stripped == "===Data===":
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find ===Data=== section in {filepath.name}")

    # Replace commas with dots (European decimal separator)
    data_text = "\n".join(lines[data_start:]).replace(",", ".")

    raw_df = pd.read_csv(
        StringIO(data_text),
        sep="\t",
        decimal=".",
        engine="python",
        skip_blank_lines=True,
    )
    raw_df.dropna(how="all", inplace=True)

    # Build clean working DataFrame with named columns
    # Column indices match Eurotherm .txt format (0-based): 0=Time_s, 1=Tsample, 3=Toven, 7=pO2
    if raw_df.shape[1] < 8:
        raise ValueError(
            f"{filepath.name}: furnace log has {raw_df.shape[1]} columns, "
            f"expected >= 8 (Eurotherm format with pO2 at column 8)"
        )
    df = pd.DataFrame()
    df["Time_s"]  = pd.to_numeric(raw_df.iloc[:, 0], errors="coerce")
    df["Tsample"] = pd.to_numeric(raw_df.iloc[:, 1], errors="coerce")
    df["Toven"]   = pd.to_numeric(raw_df.iloc[:, 3], errors="coerce")
    df["pO2"]     = pd.to_numeric(raw_df.iloc[:, 7], errors="coerce")
    df.dropna(subset=["Time_s", "Tsample", "pO2"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add absolute datetime column (critical for ISM timestamp matching).
    # A None column here would make every window comparison False and mark
    # all ISM files OUTSIDE_RANGE with no hint of the real cause, so an
    # unparseable start date must fail loudly.
    if start_dt is None:
        raise ValueError(
            f"{filepath.name}: could not parse start date '{start_date_raw}' "
            f"(expected DD.MM.YYYY_HH:MM:SS). Without it, ISM timestamp "
            f"matching is impossible."
        )
    df["abs_datetime"] = start_dt + pd.to_timedelta(df["Time_s"], unit="s")

    return {
        "sample_name":    sample_name,
        "start_date_raw": start_date_raw,
        "start_dt":       start_dt,
        "start_seconds":  start_seconds,
        "filepath":       filepath,
        "df":             df,
    }


# ---------------------------------------------------------------------------
# ISM <-> furnace matching
# ---------------------------------------------------------------------------

def match_ism_to_furnace(
    records: list[IsmRecord],
    furnace_df: pd.DataFrame,
    T_stability_std: float = 1.5,
    T_round_step: int = 25,
    T_plateau_range: tuple[float, float] = (390.0, 610.0),
    pre_margin_min:  float = 20.0,
    post_margin_min: float = 5.0,
) -> list[IsmRecord]:
    """
    Assign furnace conditions (T_nominal, pO2_mean, status) to each ISM record.

    For each ISM record the furnace log rows that fall within the measurement
    time window [t_start, t_end] are extracted. Then:
      - If no rows match                   -> status = OUTSIDE_RANGE
      - If T_std > threshold               -> status = UNSTABLE  (ramp or transition)
      - If T_mean out of range             -> status = OUT_OF_RANGE
      - If extended window fails T_std     -> status = NEAR_TRANSITION
      - Otherwise                          -> status = VALID

    Asymmetric margins - pre vs post:
      pre_margin_min  : minutes checked BEFORE t_start.
                        Guards against files taken right as a new plateau starts
                        (furnace not yet stabilized). Keep large (15-25 min).
      post_margin_min : minutes checked AFTER t_end.
                        Guards against files whose window ends just as the
                        furnace begins the next ramp down. Can be small (0-10 min):
                        if the measurement itself was stable, the furnace ramping
                        down afterwards does not invalidate it.

    After classification, VALID records are grouped by T_nominal and assigned
    replica indices (1-based, in chronological order).

    Parameters
    ----------
    records          : list of IsmRecord (from scan_condition_dir)
    furnace_df       : DataFrame with columns abs_datetime, Tsample, pO2
    T_stability_std  : max allowed std(Tsample) [°C] inside window and margins
    T_round_step     : nominal temperature rounding step [°C]
    T_plateau_range  : (T_min, T_max) valid plateau range [°C]
    pre_margin_min   : safety margin before t_start [minutes] (default 20)
    post_margin_min  : safety margin after t_end [minutes] (default 5)

    Returns
    -------
    Same list with T_nominal, T_mean, T_std, pO2_mean, replica, status filled in.
    """
    if "abs_datetime" not in furnace_df.columns or furnace_df["abs_datetime"].isna().all():
        raise ValueError(
            "furnace_df has no valid abs_datetime column. "
            "Check that the furnace log start date was parsed correctly."
        )

    pre_margin  = pd.Timedelta(minutes=pre_margin_min)
    post_margin = pd.Timedelta(minutes=post_margin_min)

    for rec in records:
        if rec.t_start is None or rec.t_end is None:
            rec.status = "OUTSIDE_RANGE"
            continue

        # Make t_start/t_end timezone-naive for comparison
        t0 = rec.t_start.replace(tzinfo=None)
        t1 = rec.t_end.replace(tzinfo=None)

        mask = (furnace_df["abs_datetime"] >= t0) & (furnace_df["abs_datetime"] <= t1)
        window = furnace_df.loc[mask]

        if len(window) == 0:
            rec.status = "OUTSIDE_RANGE"
            continue

        if len(window) < 2:
            # Single furnace log point: std is undefined - treat as insufficient data.
            rec.T_mean = round(float(window["Tsample"].mean()), 2)
            rec.T_std  = None
            rec.status = "OUTSIDE_RANGE"
            continue

        # Round to physically meaningful precision:
        # T_mean: 0.01 °C (well beyond furnace stability limit of ~0.5–2 °C)
        # T_std:  0.001 °C (enough to distinguish stable vs unstable)
        # pO2:    6 significant figures (lambda probe accuracy ~3–4 sig. figs.)
        T_mean = round(float(window["Tsample"].mean()), 2)
        T_std  = round(float(window["Tsample"].std(ddof=0)), 3)  # ddof=0: population std
        pO2_max = float(window["pO2"].max())
        pO2_min = float(window["pO2"].min())
        pO2_mean = _round_sigfigs((pO2_max + pO2_min) / 2.0, n=6)

        rec.T_mean   = T_mean
        rec.T_std    = T_std
        rec.pO2_mean = pO2_mean

        if T_std > T_stability_std:
            rec.status = "UNSTABLE"
        elif not (T_plateau_range[0] <= T_mean <= T_plateau_range[1]):
            rec.status = "OUT_OF_RANGE"
        else:
            # Extended window check: pre_margin before t_start, post_margin after t_end.
            # A file taken just before the descending ramp uses a small post_margin
            # so it is not penalized for what happens after the measurement ends.
            if pre_margin_min > 0 or post_margin_min > 0:
                ext_mask = (
                    (furnace_df["abs_datetime"] >= t0 - pre_margin) &
                    (furnace_df["abs_datetime"] <= t1 + post_margin)
                )
                ext_window = furnace_df.loc[ext_mask]
                if len(ext_window) >= 2:
                    T_std_ext = round(float(ext_window["Tsample"].std(ddof=0)), 3)
                    if T_std_ext > T_stability_std:
                        rec.status = "NEAR_TRANSITION"
                        continue

            rec.T_nominal = float(round(T_mean / T_round_step) * T_round_step)
            rec.status    = "VALID"

    # Assign replica indices within each T_nominal group (chronological order)
    from collections import defaultdict
    group_counter: dict[float, int] = defaultdict(int)

    for rec in records:
        if rec.status == "VALID" and rec.T_nominal is not None:
            group_counter[rec.T_nominal] += 1
            rec.replica = group_counter[rec.T_nominal]

    return records


def validate_against_filename_labels(records: list[IsmRecord]) -> pd.DataFrame:
    """
    For already-labeled files, compare T_nominal from furnace matching
    against the temperature encoded in the filename.

    Returns a DataFrame with columns:
      file, T_nominal, T_file_label, label_match, status

    label_match:
      True  -> furnace T_nominal agrees with filename label
      False -> mismatch (potential issue)
      None  -> filename has no temperature label (sequential / unlabeled file)
    """
    rows = []
    for r in records:
        T_file = extract_T_from_filename(r.path.name)
        if r.T_nominal is not None and T_file is not None:
            match = (int(r.T_nominal) == T_file)
        else:
            match = None
        rows.append({
            "file":         r.path.name,
            "T_nominal":    r.T_nominal,
            "T_file_label": T_file,
            "label_match":  match,
            "status":       r.status,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auto-labeling  (used for new data with no manual temperature labels)
# ---------------------------------------------------------------------------

_LABELED_RE  = re.compile(r"_\d{2,4}[Cc](?:_\d+)?\.ism$",  re.IGNORECASE)
# Strip only the trailing run counter (e.g. _01.ism) - NOT the T-range prefix (_600_400_25).
# The old 4-group pattern ate the T-range for pure-gas conditions (Ar, O2) where
# the condition name has no hyphens and the T-range tokens look numeric.
_SEQ_TAIL_RE = re.compile(r"_\d{1,3}\.ism$", re.IGNORECASE)


def get_label_prefix(records: list[IsmRecord],
                     condition_folder: str = "") -> str:
    """
    Derive the filename prefix used for auto-labeling.

    Strategy:
    1. Collect all labeled files (matching _LABELED_RE), strip the T-label
       suffix from each, and return the most frequently occurring prefix.
       Ties are broken by length (longer = more specific wins).
       Using the majority prefix rather than the first match avoids single
       oddly-named files biasing the result.
    2. Fall through to sequential files (_SEQ_TAIL_RE strips only the
       trailing run counter, e.g. _01.ism).
    3. Fallback: lowercased condition_folder minus the T-range suffix.

    Parameters
    ----------
    records          : list of IsmRecord for this condition
    condition_folder : condition folder name (used as fallback)

    Returns
    -------
    str : prefix without trailing underscore
    """
    from collections import Counter

    # Collect prefixes from all labeled files
    label_prefixes = []
    for rec in records:
        m = _LABELED_RE.search(rec.path.name)
        if m:
            label_prefixes.append(rec.path.name[: m.start()])

    if label_prefixes:
        counts = Counter(label_prefixes)
        max_count = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_count]
        return max(candidates, key=len)   # longest wins on tie

    # Try sequential files
    for rec in records:
        m = _SEQ_TAIL_RE.search(rec.path.name)
        if m:
            return rec.path.name[: m.start()]

    # Fallback: condition folder → lowercase, strip T-range suffix _NNN_NNN_NN
    name = condition_folder.lower()
    name = re.sub(r"_\d{2,3}_\d{2,3}_\d{2,3}$", "", name)
    return name or "sample"


def generate_auto_label(prefix: str, T_nominal: int, replica: int) -> str:
    """
    Generate a temperature-labeled filename for a VALID ISM file.

    Convention (matches user's manual labeling for SampleID):
        replica 1  ->  {prefix}_{T}C.ism
        replica 2  ->  {prefix}_{T}C_1.ism
        replica k  ->  {prefix}_{T}C_{k-1}.ism

    Parameters
    ----------
    prefix    : filename prefix  (e.g. 'SampleID_ar-SCCM_o2-SCCM')
    T_nominal : nominal temperature [°C]  (e.g. 400)
    replica   : 1-based replica index within this T group

    Returns
    -------
    str : auto-generated filename  (e.g. 'SampleID_ar-SCCM_o2-SCCM_400C.ism')
    """
    if replica == 1:
        return f"{prefix}_{T_nominal}C.ism"
    return f"{prefix}_{T_nominal}C_{replica - 1}.ism"


def build_auto_labels(records: list[IsmRecord],
                      prefix: str) -> list[IsmRecord]:
    """
    Assign auto_label to every VALID IsmRecord.

    Two separate rules:
      1. Already-labeled files (filename matches _LABELED_RE):
         auto_label = original filename  - no rename, user has already named them.
      2. Sequential files (no T label in the filename):
         auto_label = {prefix}_{T}C[_k].ism, continuing from the highest
         replica number the user already assigned at that T.

    Why separate rules?
    -------------------
    If a sequential measurement was taken just before a user-labeled one at the
    same T (chronologically earlier), a flat replica counter would offset all
    labeled files by 1, causing apparent mismatches in the validation table.
    Keeping labeled files at their original name and starting sequential counter
    after the user's highest replica avoids any collision.

    Parameters
    ----------
    records : list of IsmRecord (after match_ism_to_furnace)
    prefix  : filename prefix from get_label_prefix()

    Returns
    -------
    Same list with auto_label filled in.
    """
    from collections import defaultdict
    from .ingest import extract_replica_from_filename

    _lab_re = re.compile(r"_\d{2,4}[Cc](?:_\d+)?\.ism$", re.IGNORECASE)

    # Step 1: labeled VALID files keep their original filename as auto_label.
    for rec in records:
        if rec.status == "VALID" and rec.T_nominal is not None:
            if _lab_re.search(rec.path.name):
                rec.auto_label = rec.path.name
            else:
                rec.auto_label = None   # filled in step 3
        else:
            rec.auto_label = None

    # Step 2: per T_nominal, find the highest replica number the user assigned.
    max_labeled: dict[float, int] = defaultdict(int)
    for rec in records:
        if rec.status == "VALID" and rec.T_nominal is not None:
            if _lab_re.search(rec.path.name):
                rep = extract_replica_from_filename(rec.path.name)
                if rep is not None:
                    max_labeled[rec.T_nominal] = max(max_labeled[rec.T_nominal], rep)

    # Step 3: sequential files get names starting from max_labeled[T] + 1.
    seq_counter: dict[float, int] = {}
    for rec in sorted(records, key=lambda r: (r.t_start or datetime.min)):
        if rec.status != "VALID" or rec.T_nominal is None:
            continue
        if _lab_re.search(rec.path.name):
            continue   # already handled
        T = rec.T_nominal
        if T not in seq_counter:
            seq_counter[T] = max_labeled.get(T, 0) + 1
        rec.auto_label = generate_auto_label(prefix, int(T), seq_counter[T])
        seq_counter[T] += 1

    return records


def validate_auto_labels(records: list[IsmRecord]) -> pd.DataFrame:
    """
    Validate the auto-labeling algorithm against manually labeled files.

    For each VALID file this function checks:

    1. T match  - furnace T_nominal == T encoded in filename
       (only for already-labeled files; '-' for sequential files)

    2. Order match - within this T group, the labeled files appear in
       chronological order consistent with their suffix numbers.
       e.g.  if _575C.ism comes before _575C_1.ism in time → ✓
             if _575C_1.ism comes *before* _575C.ism in time → ✗ ORDER ERROR

    3. Auto-label shown - what Stage 1 would name this file for NEW data.
       Note: auto_label ≠ filename is expected and is NOT an error when
       sequential files exist within a T group: they shift replica numbering.

    Returns
    -------
    pd.DataFrame with columns:
        file, is_labeled, T_nominal, replica_seq, auto_label,
        T_file, replica_file, T_match, order_ok
    """
    _lab_re = re.compile(r"_\d{2,4}[Cc](?:_\d+)?\.ism$", re.IGNORECASE)

    # Group VALID labeled records by T_nominal for order check
    from collections import defaultdict
    labeled_by_T: dict[float, list] = defaultdict(list)
    for rec in records:
        if rec.status == "VALID" and _lab_re.search(rec.path.name):
            labeled_by_T[rec.T_nominal].append(rec)

    # Sort each group by timestamp - check if replica_file is monotonically increasing
    order_ok_map: dict[str, Optional[bool]] = {}
    for T_nom, grp in labeled_by_T.items():
        grp_sorted = sorted(grp, key=lambda r: r.t_start or datetime.min)
        replica_seq = [extract_replica_from_filename(r.path.name) for r in grp_sorted]
        # Check monotonic increase
        ok = all(
            a < b
            for a, b in zip(replica_seq, replica_seq[1:])
            if a is not None and b is not None
        )
        for r in grp_sorted:
            order_ok_map[r.path.name] = ok

    rows = []
    for rec in records:
        if rec.status != "VALID":
            continue
        is_labeled = bool(_lab_re.search(rec.path.name))
        T_file      = extract_T_from_filename(rec.path.name)   if is_labeled else None
        rep_file    = extract_replica_from_filename(rec.path.name) if is_labeled else None
        T_match     = (int(rec.T_nominal) == T_file) if (is_labeled and T_file is not None) else None
        order_ok    = order_ok_map.get(rec.path.name)

        rows.append({
            "file":        rec.path.name,
            "is_labeled":  is_labeled,
            "T_nominal":   rec.T_nominal,
            "replica_seq": rec.replica,      # chronological position among ALL valid files
            "auto_label":  rec.auto_label,
            "T_file":      T_file,
            "replica_file": rep_file,        # replica extracted from the existing filename
            "T_match":     T_match,
            "order_ok":    order_ok,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting (ported from oven_plots.ipynb)
# ---------------------------------------------------------------------------

def _format_time(total_seconds: float) -> str:
    """Format absolute seconds as D:HH:MM:SS (D = days elapsed from start day)."""
    s   = int(total_seconds)
    day = s // 86400
    rem = s % 86400
    hh  = rem // 3600
    mm  = (rem % 3600) // 60
    ss  = rem % 60
    return f"{day}:{hh:02d}:{mm:02d}:{ss:02d}"


def plot_oven(parsed: dict, save_path: Optional[Path] = None, show: bool = True) -> None:
    """
    Plot furnace temperature vs time (ported from oven_plots.ipynb).

    X axis : D:HH:MM:SS, hourly ticks, trimmed to Tsample in [350, 650] °C
    Y axis : T/°C, range 350-650, gridlines every 25 °C
    """
    Y_MIN, Y_MAX, Y_STEP = 350, 650, 25

    df            = parsed["df"]
    start_seconds = parsed["start_seconds"]
    filepath      = parsed["filepath"]

    # Absolute seconds from midnight (kept for x-axis labelling)
    x       = df["Time_s"] + start_seconds
    tsample = df["Tsample"]
    label   = filepath.stem

    # Trim x range to valid temperature window
    in_range = (tsample >= Y_MIN) & (tsample <= Y_MAX)
    if in_range.any():
        x_min = x[in_range].min()
        x_max = x[in_range].max()
    else:
        x_min, x_max = x.min(), x.max()

    # Hourly tick positions
    first_tick = math.ceil(x_min / 3600) * 3600
    last_tick  = math.floor(x_max / 3600) * 3600
    ticks      = np.arange(first_tick, last_tick + 1, 3600)
    tick_labels = [_format_time(t) for t in ticks]

    n_hours = len(ticks)
    fig_w   = max(14, n_hours * 0.22)

    with plt.rc_context({
        "font.family": "sans-serif", "font.size": 11,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,
        "axes.grid": True, "grid.linestyle": "-",
        "grid.color": "#cccccc", "grid.alpha": 1.0,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "figure.dpi": 150,
    }):

        fig, ax = plt.subplots(figsize=(fig_w, 6))
        ax.plot(x, tsample, color="black", linewidth=1.0, label=label, zorder=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_STEP))
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("T / °C", fontsize=12)
        ax.set_title("")
        ax.legend(loc="upper right", framealpha=1.0, edgecolor="black",
                  fontsize=11, handlelength=1.5)
        ax.tick_params(which="major", direction="in", top=False, right=False, length=4)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
            print(f"  Saved -> {save_path.name}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_ism_selection(
    parsed: dict,
    records: list,
    condition_folder: str,
    lead_hours_at_600: float = 4.0,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot furnace temperature vs time with ISM measurement windows overlaid.

    The x-axis is trimmed to show only the last `lead_hours_at_600` hours of the
    600 °C plateau plus the full descent - the long stable period at max temperature
    before the first ramp is not informative and would squash the detail.

    Color + hatch coding:
      green solid  : VALID
      blue  hatch  : NEAR_TRANSITION (too close to a ramp edge)
      red   hatch  : UNSTABLE (T_std exceeds threshold during measurement)
      grey         : OUTSIDE_RANGE / OUT_OF_RANGE
    """
    from matplotlib.patches import Patch

    Y_MIN, Y_MAX, Y_STEP = 350, 650, 25

    C_VALID  = "#90EE90"   # light green
    C_NEAR   = "#0072B2"   # blue
    C_UNSTAB = "#D62728"   # red
    C_OOR    = "#BBBBBB"   # light grey

    _STYLE = {
        "VALID":           (C_VALID,  0.50, None),    # (color, alpha, hatch)
        "NEAR_TRANSITION": (C_NEAR,   0.45, "///"),
        "UNSTABLE":        (C_UNSTAB, 0.45, "xxx"),
        "OUT_OF_RANGE":    (C_OOR,    0.25, None),
        "OUTSIDE_RANGE":   (C_OOR,    0.25, None),
    }

    df            = parsed["df"]
    start_seconds = parsed["start_seconds"]
    start_dt      = parsed["start_dt"]

    x       = df["Time_s"] + start_seconds
    tsample = df["Tsample"]

    # --- trim x range ---
    # x_max: last point in the valid T window
    in_range = (tsample >= Y_MIN) & (tsample <= Y_MAX)
    x_max = float(x[in_range].max()) if in_range.any() else float(x.max())

    # Find the last time T >= 585 °C (top of 600°C plateau) before the first descent.
    # Walk backwards from x_max to find when T dropped below 585 for good.
    T_TOP = 585.0
    at_top = (tsample >= T_TOP) & (x <= x_max)
    if at_top.any():
        last_at_top = float(x[at_top].max())
    else:
        last_at_top = x_max

    x_min = max(last_at_top - lead_hours_at_600 * 3600,
                float(x[in_range].min()) if in_range.any() else float(x.min()))

    # --- ticks every hour ---
    first_tick = math.ceil(x_min / 3600) * 3600
    last_tick  = math.floor(x_max / 3600) * 3600
    ticks      = np.arange(first_tick, last_tick + 1, 3600)
    tick_labels = [_format_time(t) for t in ticks]

    n_hours = len(ticks)
    fig_w   = max(12, n_hours * 0.28)

    with plt.rc_context({
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.linewidth":   1.0,
        "lines.linewidth":  1.5,
        "axes.grid":        True,
        "grid.linestyle":   "-",
        "grid.color":       "#dddddd",
        "grid.alpha":       1.0,
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "figure.dpi":       140,
    }):

        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.plot(x, tsample, color="black", linewidth=1.5, zorder=5)

        for rec in records:
            if rec.t_start is None or rec.t_end is None or start_dt is None:
                continue
            t0 = rec.t_start.replace(tzinfo=None)
            t1 = rec.t_end.replace(tzinfo=None)
            x0 = (t0 - start_dt).total_seconds() + start_seconds
            x1 = (t1 - start_dt).total_seconds() + start_seconds
            if x1 < x_min or x0 > x_max:
                continue

            color, alpha, hatch = _STYLE.get(rec.status, (C_OOR, 0.20, None))
            ax.axvspan(x0, x1, facecolor=color, alpha=alpha,
                       hatch=hatch, edgecolor=color if hatch else "none",
                       linewidth=0, zorder=2)
            # Strong border line to make each window clearly visible
            ax.axvline(x0, color=color, linewidth=1.2, alpha=0.8, zorder=3)
            ax.axvline(x1, color=color, linewidth=1.2, alpha=0.8, zorder=3)

            # T_nominal label inside VALID windows (e.g. "400°C", "400°C_1")
            if rec.status == "VALID" and rec.T_nominal is not None:
                xmid = (x0 + x1) / 2
                T = int(rec.T_nominal)
                rep = rec.replica if rec.replica is not None else 1
                label = f"{T}°C" if rep == 1 else f"{T}°C_{rep-1}"
                ax.text(
                    xmid, Y_MIN + 10, label,
                    ha="center", va="bottom", fontsize=7, color="black",
                    fontweight="bold", rotation=90, zorder=6,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=7)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_STEP))
        ax.set_xlabel("Time  (D:HH:MM:SS)", fontsize=11)
        ax.set_ylabel("T / °C", fontsize=11)
        ax.tick_params(which="major", direction="in", top=False, right=False, length=4)

        legend_items = [
            Patch(facecolor=C_VALID,  alpha=0.7, label="VALID"),
            Patch(facecolor=C_NEAR,   alpha=0.6, hatch="///",
                  edgecolor=C_NEAR,   label="NEAR_TRANSITION"),
            Patch(facecolor=C_UNSTAB, alpha=0.6, hatch="xxx",
                  edgecolor=C_UNSTAB, label="UNSTABLE"),
            Patch(facecolor=C_OOR,    alpha=0.4, label="OUT_OF_RANGE"),
        ]
        ax.legend(handles=legend_items, loc="upper right",
                  framealpha=1.0, edgecolor="#888888", fontsize=9)

        # Title: condition + per-T counts
        per_T: dict[float, int] = {}
        for r in records:
            if r.status == "VALID" and r.T_nominal is not None:
                per_T[r.T_nominal] = per_T.get(r.T_nominal, 0) + 1
        summary = "  ".join(
            f"{int(T)}°C×{n}" for T, n in sorted(per_T.items(), reverse=True)
        )
        n_nt  = sum(1 for r in records if r.status == "NEAR_TRANSITION")
        n_un  = sum(1 for r in records if r.status == "UNSTABLE")
        n_val = sum(1 for r in records if r.status == "VALID")
        ax.set_title(
            f"{condition_folder}\n"
            f"VALID: {n_val}   NEAR_TRANSITION: {n_nt}   UNSTABLE: {n_un}\n"
            f"{summary}",
            fontsize=9, loc="left",
        )

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"  Saved -> {save_path.name}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def extract_plateau_table(parsed: dict, interval_s: int = 300) -> pd.DataFrame:
    """
    Extract a summary table sampled every `interval_s` seconds.

    Columns: Time (D:HH:MM:SS), Tsample (°C), Toven (°C), pO2 (bar)

    Useful for the operator to visually inspect temperature plateaus
    before running Stage 1 ISM matching.
    """
    df            = parsed["df"]
    start_seconds = parsed["start_seconds"]

    x       = df["Time_s"] + start_seconds
    tsample = df["Tsample"]
    toven   = df["Toven"]
    pO2     = df["pO2"]

    bins = np.arange(0, x.max() + interval_s, interval_s)
    rows = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.any():
            idx = mask[mask].index[0]
            rows.append({
                "Time":          _format_time(float(x.iloc[idx])),
                "Tsample (°C)":  round(float(tsample.iloc[idx]), 1),
                "Toven (°C)":    round(float(toven.iloc[idx]), 1),
                "pO2 (bar)":     float(pO2.iloc[idx]),
            })

    return pd.DataFrame(rows)
