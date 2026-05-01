"""Time-resolved ISC analysis implementation."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SUBJECT_RE = re.compile(r"sub-[A-Za-z0-9]+")
logger = logging.getLogger(__name__)


@dataclass
class TimeResolvedISCConfig:
    """Typed config for time-resolved ISC."""

    window_size_trs: int
    min_samples: float | int = 0.5
    approach: str = "loso"
    run_boundaries: list[list[int]] | None = None
    group_column: str | None = None
    roi_name: str | None = None
    timecourse_glob: str | None = None
    timecourse_files: dict[str, str] | None = None
    make_figures: bool = True
    overwrite_figures: bool = False
    activation_shared_zero: bool = False
    event_seconds: list[float] | None = None
    tr_seconds: float | None = None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value)


def _required_samples(min_samples: float | int, window_size_trs: int) -> int:
    if isinstance(min_samples, float) and min_samples < 1:
        if min_samples <= 0:
            raise ValueError("min_samples proportion must be > 0.")
        return int(math.ceil(min_samples * window_size_trs))

    required = int(min_samples)
    if required <= 0:
        raise ValueError("min_samples must be positive.")
    if required > window_size_trs:
        raise ValueError("min_samples cannot exceed window_size_trs.")
    return required


def _window_bounds(center_tr: int, window_size_trs: int) -> tuple[int, int]:
    left = (window_size_trs - 1) // 2
    right = window_size_trs - left - 1
    start = center_tr - left
    end = center_tr + right + 1
    return start, end


def _window_within_run(start: int, end: int, run_boundaries: list[tuple[int, int]]) -> bool:
    for run_start, run_end in run_boundaries:
        if start >= run_start and end <= run_end:
            return True
    return False


def _fisher_z(corr: float) -> float:
    clipped = float(np.clip(corr, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _split_fields(raw_line: str) -> list[str]:
    if "\t" in raw_line:
        return raw_line.split("\t")
    if "," in raw_line:
        return raw_line.split(",")
    return raw_line.strip().split()


def _is_numeric_or_na_token(token: str) -> bool:
    stripped = token.strip()
    if stripped == "":
        return True
    if stripped.lower() in {"na", "nan", "null", "none", "."}:
        return True
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def _token_to_float(token: str, path: Path, raw_line: str) -> float:
    stripped = token.strip()
    if stripped == "" or stripped.lower() in {"na", "nan", "null", "none", "."}:
        return float("nan")
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"Unable to parse numeric value in {path}: {raw_line}") from exc


def _read_timecourse_column(path: Path, roi_name: str | None = None) -> np.ndarray:
    values: list[float] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    first_non_empty_idx = next((idx for idx, line in enumerate(lines) if line.strip() != ""), None)
    if first_non_empty_idx is None:
        return np.asarray(values, dtype=float)

    first_fields = _split_fields(lines[first_non_empty_idx])
    has_header = any(not _is_numeric_or_na_token(field) for field in first_fields)

    column_index = 0
    data_start_idx = first_non_empty_idx
    if has_header:
        header_fields = [field.strip() for field in first_fields]
        if roi_name is None:
            if len(header_fields) == 1:
                column_index = 0
            else:
                raise ValueError(
                    f"Header detected in {path} with multiple columns. "
                    "Set time_resolved_isc config 'roi_name' to select a column."
                )
        else:
            if roi_name not in header_fields:
                raise ValueError(
                    f"roi_name '{roi_name}' not found in header columns for {path}: {header_fields}"
                )
            column_index = header_fields.index(roi_name)

        data_start_idx = first_non_empty_idx + 1

    for raw_line in lines[data_start_idx:]:
        if raw_line.strip() == "":
            values.append(float("nan"))
            continue

        fields = _split_fields(raw_line)
        token = fields[column_index] if column_index < len(fields) else ""
        values.append(_token_to_float(token=token, path=path, raw_line=raw_line))

    return np.asarray(values, dtype=float)


def _infer_subject_id(path: Path) -> str | None:
    for part in path.parts:
        match = _SUBJECT_RE.search(part)
        if match:
            return match.group(0)
    match = _SUBJECT_RE.search(path.name)
    if match:
        return match.group(0)
    return None


def _find_timecourse_files(bids_root: Path, config: TimeResolvedISCConfig) -> dict[str, Path]:
    if config.timecourse_files:
        resolved: dict[str, Path] = {}
        for subject, path in config.timecourse_files.items():
            parsed = Path(path).expanduser()
            resolved[subject] = parsed if parsed.is_absolute() else (bids_root / parsed)
        return resolved

    pattern = config.timecourse_glob or "**/sub-*_timecourse.tsv"
    files = sorted(bids_root.glob(pattern))
    subject_to_file: dict[str, Path] = {}

    for path in files:
        if not path.is_file():
            continue
        subject = _infer_subject_id(path)
        if subject is None:
            continue
        subject_to_file[subject] = path

    if not subject_to_file:
        raise FileNotFoundError(
            "No timecourse files found. Set analysis config timecourse_glob or timecourse_files."
        )

    return subject_to_file


def _find_censor_file(timecourse_path: Path) -> Path | None:
    stem = timecourse_path.stem
    parent = timecourse_path.parent

    candidates = [
        parent / f"{stem}_censor.tsv",
        parent / f"{stem}_censor.csv",
        parent / f"{stem}_censor.txt",
        parent / f"{stem}_censoring.tsv",
        parent / f"{stem}_censoring.csv",
        parent / f"{stem}_censoring.txt",
    ]

    if "timecourse" in stem:
        alt_stem = stem.replace("timecourse", "censor")
        candidates.extend(
            [
                parent / f"{alt_stem}.tsv",
                parent / f"{alt_stem}.csv",
                parent / f"{alt_stem}.txt",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    loose = sorted(parent.glob(f"*{_infer_subject_id(timecourse_path) or ''}*censor*.*"))
    for candidate in loose:
        if candidate.suffix.lower() in {".tsv", ".csv", ".txt"}:
            return candidate
    return None


def _censor_to_valid_mask(censor_values: np.ndarray) -> np.ndarray:
    valid = np.zeros(censor_values.shape[0], dtype=bool)
    finite = np.isfinite(censor_values)

    if np.any(finite):
        finite_vals = censor_values[finite]
        if np.all((finite_vals >= 0) & (finite_vals <= 1)):
            valid[finite] = finite_vals >= 0.5
        else:
            valid[finite] = finite_vals != 0

    return valid


def _load_groups(participants_tsv_path: Path | None, group_column: str | None) -> dict[str, str]:
    if group_column is None:
        return {}
    if participants_tsv_path is None or not participants_tsv_path.exists():
        return {}

    participants = pd.read_csv(participants_tsv_path, sep="\t")
    if "participant_id" not in participants.columns:
        return {}
    if group_column not in participants.columns:
        return {}

    rows = participants[["participant_id", group_column]].dropna()
    return {str(row["participant_id"]): str(row[group_column]) for _, row in rows.iterrows()}


def _build_comparison_sets(subjects: list[str], groups: dict[str, str]) -> dict[str, list[str]]:
    comparison_sets: dict[str, list[str]] = {"full": list(subjects)}
    unique_groups = sorted({groups[s] for s in subjects if s in groups})

    for group_name in unique_groups:
        group_subjects = [s for s in subjects if groups.get(s) == group_name]
        comparison_sets[_safe_name(group_name)] = group_subjects

    return comparison_sets


def _get_comparison_type(subject: str, comparison_set_name: str, groups: dict[str, str]) -> str:
    """Determine if comparison is within-group, between-group, or full."""
    if comparison_set_name == "full":
        return "full"
    subject_group = groups.get(subject, "ungrouped")
    return "within" if _safe_name(subject_group) == comparison_set_name else "between"


def _results_to_long_format(
    subject_series_results: dict[str, dict[str, np.ndarray]],
    subjects: list[str],
    n_timepoints: int,
    groups: dict[str, str],
    roi_label: str,
) -> pd.DataFrame:
    """Convert wide-format results (subject x TR) to long format."""
    rows: list[dict[str, Any]] = []

    for subject in subjects:
        for set_name, timeseries in subject_series_results.items():
            timeseries_subject = timeseries[subject]
            for tr in range(n_timepoints):
                value = timeseries_subject[tr]
                if not np.isnan(value):
                    comparison_type = _get_comparison_type(subject, set_name, groups)
                    rows.append(
                        {
                            "subject": subject,
                            "tr": tr,
                            "roi": roi_label,
                            "comparison_group": "all" if set_name == "full" else set_name,
                            "comparison_type": comparison_type,
                            "value": value,
                        }
                    )

    return pd.DataFrame(rows)


def _compute_group_averages(
    long_df: pd.DataFrame,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute mean and SEM for each comparison_group/comparison_type combination.
    
    Returns dict mapping (comparison_group, comparison_type) to
    (mean, sem, n) arrays, where n is contributing subject count per TR.
    """
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    if long_df.empty:
        return averages

    n_trs = long_df["tr"].max() + 1

    # Group by comparison_group and comparison_type
    group_combos = long_df.groupby(["comparison_group", "comparison_type"])

    for (cg, ct), group_data in group_combos:
        values_by_tr = [[] for _ in range(n_trs)]

        for _, row in group_data.iterrows():
            values_by_tr[int(row["tr"])].append(row["value"])

        means = np.zeros(n_trs)
        sems = np.zeros(n_trs)
        counts = np.zeros(n_trs)

        for tr_idx, values in enumerate(values_by_tr):
            n_vals = len(values)
            if n_vals:
                means[tr_idx] = np.mean(values)
                counts[tr_idx] = n_vals

                # Sample SEM uses ddof=1; undefined for n < 2.
                if n_vals >= 2:
                    sems[tr_idx] = np.std(values, ddof=1) / np.sqrt(n_vals)
                else:
                    sems[tr_idx] = np.nan
            else:
                means[tr_idx] = np.nan
                sems[tr_idx] = np.nan
                counts[tr_idx] = 0

        if cg not in averages:
            averages[cg] = {}
        averages[cg][ct] = (means, sems, counts)

    return averages


def _save_group_averages_tsv(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    analysis_dir: Path,
    roi_label: str,
    approach: str,
) -> Path:
    """Save group averages to a TSV file."""
    rows: list[dict[str, Any]] = []

    for comparison_group, type_dict in sorted(averages.items()):
        for comparison_type, (means, sems, counts) in sorted(type_dict.items()):
            for tr_idx, (mean_val, sem_val, n_val) in enumerate(zip(means, sems, counts)):
                if not np.isnan(mean_val):
                    rows.append(
                        {
                            "comparison_group": comparison_group,
                            "comparison_type": comparison_type,
                            "tr": tr_idx,
                            "mean": mean_val,
                            "sem": sem_val,
                            "n": int(n_val),
                        }
                    )

    df = pd.DataFrame(rows)
    out_path = analysis_dir / f"roi-{roi_label}_approach-{approach}_group_averages.tsv"
    df.to_csv(out_path, sep="\t", index=False, na_rep="")
    return out_path


def _load_group_averages_tsv(
    path: Path,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Reconstruct the nested averages dict from a saved group_averages TSV."""
    df = pd.read_csv(path, sep="\t")
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    count_col = "n" if "n" in df.columns else ("count" if "count" in df.columns else None)

    for comparison_group, cg_df in df.groupby("comparison_group"):
        averages[str(comparison_group)] = {}
        for comparison_type, ct_df in cg_df.groupby("comparison_type"):
            ct_df = ct_df.sort_values("tr")
            n_trs = int(ct_df["tr"].max()) + 1
            means = np.full(n_trs, np.nan)
            sems = np.full(n_trs, np.nan)
            counts = np.zeros(n_trs)
            for _, row in ct_df.iterrows():
                tr_idx = int(row["tr"])
                means[tr_idx] = row["mean"]
                sems[tr_idx] = row["sem"]
                if count_col is not None:
                    counts[tr_idx] = row[count_col]
            averages[str(comparison_group)][str(comparison_type)] = (means, sems, counts)

    return averages


def _compute_group_activation_averages(
    subject_series: dict[str, np.ndarray],
    subject_valid_mask: dict[str, np.ndarray],
    comparison_sets: dict[str, list[str]],
    n_timepoints: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute per-TR activation averages for each comparison group."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for set_name, set_subjects in comparison_sets.items():
        if not set_subjects:
            continue
        comparison_group = "all" if set_name == "full" else set_name
        matrix = np.vstack([subject_series[s] for s in set_subjects]).astype(float)
        valid = np.vstack([subject_valid_mask[s] for s in set_subjects])
        matrix = np.where(valid, matrix, np.nan)

        means = np.full(n_timepoints, np.nan)
        sems = np.full(n_timepoints, np.nan)
        counts = np.zeros(n_timepoints)

        for tr_idx in range(n_timepoints):
            vals = matrix[:, tr_idx]
            finite = np.isfinite(vals)
            n = int(np.sum(finite))
            counts[tr_idx] = n
            if n > 0:
                means[tr_idx] = float(np.mean(vals[finite]))
            if n >= 2:
                sems[tr_idx] = float(np.std(vals[finite], ddof=1) / np.sqrt(n))

        out[comparison_group] = (means, sems, counts)

    return out


def _save_group_activation_averages_tsv(
    activation_averages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    analysis_dir: Path,
    roi_label: str,
    approach: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for comparison_group, (means, sems, counts) in sorted(activation_averages.items()):
        for tr_idx, (m, s, n) in enumerate(zip(means, sems, counts)):
            if np.isnan(m):
                continue
            rows.append(
                {
                    "comparison_group": comparison_group,
                    "tr": tr_idx,
                    "activation_mean": float(m),
                    "activation_sem": float(s) if not np.isnan(s) else "",
                    "n": int(n),
                }
            )

    out_path = analysis_dir / f"roi-{roi_label}_approach-{approach}_activation_averages.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, na_rep="")
    return out_path


def _load_group_activation_averages_tsv(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.read_csv(path, sep="\t")
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    count_col = "n" if "n" in df.columns else ("count" if "count" in df.columns else None)

    for comparison_group, group_df in df.groupby("comparison_group"):
        group_df = group_df.sort_values("tr")
        n_trs = int(group_df["tr"].max()) + 1
        means = np.full(n_trs, np.nan)
        sems = np.full(n_trs, np.nan)
        counts = np.zeros(n_trs)
        for _, row in group_df.iterrows():
            tr_idx = int(row["tr"])
            means[tr_idx] = row["activation_mean"]
            sems[tr_idx] = row["activation_sem"]
            if count_col is not None:
                counts[tr_idx] = row[count_col]
        out[str(comparison_group)] = (means, sems, counts)

    return out


def _align_twin_zero(ax: Any, ax2: Any) -> None:
    """Shift ax2 limits so y=0 falls at the same relative height as in ax."""
    lo1, hi1 = ax.get_ylim()
    lo2, hi2 = ax2.get_ylim()
    span1, span2 = hi1 - lo1, hi2 - lo2
    if span1 == 0 or span2 == 0:
        return
    f = max(0.0, min(1.0, (0.0 - lo1) / span1))
    candidates: list[float] = [span2]
    if f > 0 and lo2 < 0:
        candidates.append(-lo2 / f)
    if f < 1 and hi2 > 0:
        candidates.append(hi2 / (1.0 - f))
    new_span = max(candidates)
    ax2.set_ylim(-f * new_span, (1.0 - f) * new_span)


def _create_group_figures_with_activation(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    activation_averages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    analysis_dir: Path,
    roi_label: str,
    approach: str,
    event_seconds: list[float] | None = None,
    tr_seconds: float | None = None,
    shared_zero: bool = False,
) -> list[Path]:
    figure_paths: list[Path] = []

    n_trs_all = max(
        (len(means) for type_dict in averages.values() for means, _, _ in type_dict.values()),
        default=0,
    )
    fig_width = max(12.0, n_trs_all * 0.075)

    colors = {"within": "#1f77b4", "between": "#ff7f0e", "full": "#2ca02c"}

    for comparison_group in sorted(averages.keys()):
        type_dict = averages[comparison_group]
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        ax2 = ax.twinx()

        for comparison_type in sorted(type_dict.keys()):
            means, sems, _counts = type_dict[comparison_type]
            trs = np.arange(len(means))
            color = colors.get(comparison_type, "#999999")
            ax.plot(trs, means, label=comparison_type, color=color, linewidth=2)
            valid_sem = ~np.isnan(sems)
            if np.any(valid_sem):
                ax.fill_between(
                    trs,
                    np.where(valid_sem, means - sems, np.nan),
                    np.where(valid_sem, means + sems, np.nan),
                    alpha=0.2,
                    color=color,
                )

        activation_tuple = activation_averages.get(comparison_group)
        if activation_tuple is not None:
            activation_means, _activation_sems, _activation_counts = activation_tuple
            trs = np.arange(len(activation_means))
            ax2.plot(
                trs,
                activation_means,
                label="activation",
                color="#111111",
                linewidth=1.25,
                alpha=0.5,
            )

        if event_seconds is not None and tr_seconds is not None and tr_seconds > 0:
            for event_sec in event_seconds:
                ax.axvline(event_sec / tr_seconds, color="gray", linestyle="--", alpha=0.3, linewidth=1)
                ax.axvline((event_sec + 6.0) / tr_seconds, color="gray", linestyle="--", alpha=0.6, linewidth=1)

        if shared_zero:
            _align_twin_zero(ax, ax2)

        ax.set_xlabel("TR")
        ax.set_ylabel("Correlation (LOSO) or Mean Fisher-z (Pairwise)")
        ax2.set_ylabel("Activation")
        ax.set_title(
            f"Time-Resolved ISC + Activation: {comparison_group} group "
            f"(ROI: {roi_label}, Approach: {approach})"
        )
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)
        ax.grid(True, alpha=0.3)

        fig_path = analysis_dir / (
            f"roi-{roi_label}_group-{comparison_group}_approach-{approach}_figure_with_activation.png"
        )
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)

    return figure_paths


def _create_group_figures(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    analysis_dir: Path,
    roi_label: str,
    approach: str,
    event_seconds: list[float] | None = None,
    tr_seconds: float | None = None,
    shared_zero: bool = False,
) -> list[Path]:
    """Create PNG figures for each comparison_group showing all comparison_types.
    
    Optionally overlays event markers (faint) and hemodynamic lag markers (darker).
    
    Returns list of created figure paths.

    Notes:
        shared_zero is accepted for cross-task API consistency. This plot currently
        uses a single y-axis, so the option is a no-op here.
    """
    figure_paths: list[Path] = []

    # Scale figure width with dataset length: ~0.075 in/TR, minimum 12 in.
    n_trs_all = max(
        (len(means) for type_dict in averages.values() for means, _, _ in type_dict.values()),
        default=0,
    )
    fig_width = max(12.0, n_trs_all * 0.075)

    for comparison_group in sorted(averages.keys()):
        type_dict = averages[comparison_group]

        fig, ax = plt.subplots(figsize=(fig_width, 6))

        colors = {"within": "#1f77b4", "between": "#ff7f0e", "full": "#2ca02c"}

        for comparison_type in sorted(type_dict.keys()):
            means, sems, _counts = type_dict[comparison_type]
            trs = np.arange(len(means))

            color = colors.get(comparison_type, "#999999")

            ax.plot(trs, means, label=comparison_type, color=color, linewidth=2)
            ax.fill_between(
                trs,
                means - sems,
                means + sems,
                alpha=0.2,
                color=color,
            )

        # Add event markers if provided
        if event_seconds is not None and tr_seconds is not None and tr_seconds > 0:
            for event_sec in event_seconds:
                event_tr = event_sec / tr_seconds

                # Event marker (faint)
                ax.axvline(event_tr, color="gray", linestyle="--", alpha=0.3, linewidth=1)

                # Hemodynamic lag marker: event + 6 seconds (darker)
                lag_tr = (event_sec + 6.0) / tr_seconds
                ax.axvline(lag_tr, color="gray", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_xlabel("TR")
        ax.set_ylabel("Correlation (LOSO) or Mean Fisher-z (Pairwise)")
        ax.set_title(
            f"Time-Resolved ISC: {comparison_group} group (ROI: {roi_label}, Approach: {approach})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = analysis_dir / f"roi-{roi_label}_group-{comparison_group}_approach-{approach}_figure.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)

    return figure_paths


def _series_to_wide_df(series_map: dict[str, np.ndarray]) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame()

    any_series = next(iter(series_map.values()))
    columns = [f"tr_{idx:05d}" for idx in range(any_series.shape[0])]
    data = {"subject": list(series_map.keys())}
    for idx, col in enumerate(columns):
        data[col] = [series_map[subject][idx] for subject in series_map]
    return pd.DataFrame(data)


def _load_cached_subject_summary(
    subject_path: Path,
    comparison_sets: list[str],
    n_timepoints: int,
) -> dict[str, np.ndarray] | None:
    """Load a cached subject long-format TSV into set_name -> per-TR summary arrays."""
    if not subject_path.exists():
        return None

    df = pd.read_csv(subject_path, sep="\t")
    required_cols = {"comparison_group", "tr", "value"}
    if not required_cols.issubset(df.columns):
        return None

    out: dict[str, np.ndarray] = {
        set_name: np.full(n_timepoints, np.nan, dtype=float) for set_name in comparison_sets
    }

    for set_name in comparison_sets:
        subset = df[df["comparison_group"] == set_name]
        if subset.empty:
            continue

        trs = subset["tr"].to_numpy(dtype=int)
        vals = subset["value"].to_numpy(dtype=float)
        valid = (trs >= 0) & (trs < n_timepoints) & np.isfinite(vals)
        out[set_name][trs[valid]] = vals[valid]

    return out


def run_time_resolved_isc_analysis(
    bids_root: Path,
    output_root: Path,
    discovered_inputs: dict[str, Any],
    config_dict: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run centered-window time-resolved ISC for one scalar timeseries per subject."""
    config = TimeResolvedISCConfig(
        window_size_trs=int(config_dict["window_size_trs"]),
        min_samples=config_dict.get("min_samples", 0.5),
        approach=str(config_dict.get("approach", "loso")).lower(),
        run_boundaries=config_dict.get("run_boundaries"),
        group_column=config_dict.get("group_column"),
        roi_name=config_dict.get("roi_name"),
        timecourse_glob=config_dict.get("timecourse_glob"),
        timecourse_files=config_dict.get("timecourse_files"),
        make_figures=bool(config_dict.get("make_figures", True)),
        overwrite_figures=bool(config_dict.get("overwrite_figures", False)),
        activation_shared_zero=bool(config_dict.get("activation_shared_zero", False)),
        event_seconds=config_dict.get("event_seconds"),
        tr_seconds=config_dict.get("tr_seconds"),
    )

    if config.approach not in {"loso", "pairwise"}:
        raise ValueError("time_resolved_isc approach must be 'loso' or 'pairwise'.")

    analysis_dir = output_root / "time_resolved_isc"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    roi_label = _safe_name(config.roi_name) if config.roi_name else "all"
    group_dir = analysis_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    # Default behavior: reuse cache unless overwrite is explicitly enabled.
    # A cache hit requires metadata, group outputs, and all per-subject outputs.
    subject_to_file = _find_timecourse_files(bids_root=bids_root, config=config)
    subjects = sorted(subject_to_file)

    metadata_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_timeseries_metadata.json"
    group_timeseries_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_timeseries.tsv"
    group_averages_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_group_averages.tsv"
    activation_averages_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_activation_averages.tsv"
    subject_timeseries_paths = [
        analysis_dir / subject / f"{subject}_roi-{roi_label}_approach-{config.approach}_timeseries.tsv"
        for subject in subjects
    ]

    # ── Phase 1: decide whether the heavy ISC computation needs to run ──────────
    core_cache_paths: list[Path] = [metadata_path, group_timeseries_path, *subject_timeseries_paths]
    missing_core_paths = [p for p in core_cache_paths if not p.exists()]
    needs_compute = overwrite or bool(missing_core_paths)

    if not needs_compute:
        # ── Phase 2 (cache hit): post-processing only ────────────────────────
        logger.info(
            "Cache hit: skipping ISC computation for roi=%s, approach=%s (%d subjects)",
            config.roi_name,
            config.approach,
            len(subjects),
        )

        cached_outputs: dict[str, Any] = {
            "analysis": "time_resolved_isc",
            "status": "skipped_cache",
            "roi_name": config.roi_name,
            "roi_label": roi_label,
            "approach": config.approach,
            "window_size_trs": config.window_size_trs,
            "n_subjects": len(subjects),
            "group_dir": str(group_dir),
            "files": {
                f"{config.approach}_group_timeseries": str(group_timeseries_path),
                "metadata": str(metadata_path),
            },
            "subject_dirs": {subject: str(analysis_dir / subject) for subject in subjects},
            "message": f"Using cached outputs from {analysis_dir}",
        }

        if config.approach == "pairwise":
            pairwise_details_path = group_dir / f"desc-pairwise_roi-{roi_label}_details.tsv"
            if pairwise_details_path.exists():
                cached_outputs["files"]["pairwise_details"] = str(pairwise_details_path)

        # Figures (and future: inferential stats) — handled independently of computation cache
        if config.make_figures and group_averages_path.exists():
            existing_figs = sorted(group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_figure.png"))
            existing_overlay_figs = sorted(
                group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_figure_with_activation.png")
            )

            needs_base = config.overwrite_figures or not existing_figs
            needs_overlay = config.overwrite_figures or not existing_overlay_figs
            if needs_base or needs_overlay:
                logger.info("Generating figures from cached averages for roi=%s", config.roi_name)
                averages = _load_group_averages_tsv(group_averages_path)

                if needs_base:
                    fig_paths = _create_group_figures(
                        averages,
                        group_dir,
                        roi_label,
                        config.approach,
                        event_seconds=config.event_seconds,
                        tr_seconds=config.tr_seconds,
                        shared_zero=config.activation_shared_zero,
                    )
                else:
                    fig_paths = existing_figs

                if activation_averages_path.exists() and needs_overlay:
                    activation_averages = _load_group_activation_averages_tsv(activation_averages_path)
                    overlay_fig_paths = _create_group_figures_with_activation(
                        averages,
                        activation_averages,
                        group_dir,
                        roi_label,
                        config.approach,
                        event_seconds=config.event_seconds,
                        tr_seconds=config.tr_seconds,
                        shared_zero=config.activation_shared_zero,
                    )
                else:
                    overlay_fig_paths = existing_overlay_figs

                cached_outputs["status"] = "refreshed_figures"
                logger.info(
                    "Generated %d base figures and %d activation-overlay figures for roi=%s",
                    len(fig_paths),
                    len(overlay_fig_paths),
                    config.roi_name,
                )
            else:
                fig_paths = existing_figs
                overlay_fig_paths = existing_overlay_figs

            cached_outputs["files"][f"{config.approach}_group_averages"] = str(group_averages_path)
            if activation_averages_path.exists():
                cached_outputs["files"][f"{config.approach}_activation_averages"] = str(activation_averages_path)
            cached_outputs["files"][f"{config.approach}_figures"] = [str(p) for p in fig_paths]
            cached_outputs["files"][f"{config.approach}_figures_with_activation"] = [str(p) for p in overlay_fig_paths]
        elif group_averages_path.exists():
            cached_outputs["files"][f"{config.approach}_group_averages"] = str(group_averages_path)
            if activation_averages_path.exists():
                cached_outputs["files"][f"{config.approach}_activation_averages"] = str(activation_averages_path)
            existing_figs = sorted(group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_figure.png"))
            existing_overlay_figs = sorted(
                group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_figure_with_activation.png")
            )
            if existing_figs:
                cached_outputs["files"][f"{config.approach}_figures"] = [str(p) for p in existing_figs]
            if existing_overlay_figs:
                cached_outputs["files"][f"{config.approach}_figures_with_activation"] = [
                    str(p) for p in existing_overlay_figs
                ]

        return cached_outputs

    # ── Phase 1 continued: run the full ISC computation ──────────────────────
    required_samples = _required_samples(config.min_samples, config.window_size_trs)
    logger.info(
        "time_resolved_isc starting with approach=%s, window_size_trs=%d, required_samples=%d",
        config.approach,
        config.window_size_trs,
        required_samples,
    )

    logger.info("Found %d participant timeseries files", len(subjects))
    subject_series: dict[str, np.ndarray] = {}
    subject_valid_mask: dict[str, np.ndarray] = {}

    for subject_index, subject in enumerate(subjects, start=1):
        logger.info("Loading participant %d/%d: %s", subject_index, len(subjects), subject)
        series = _read_timecourse_column(subject_to_file[subject], roi_name=config.roi_name)
        subject_series[subject] = series

    series_lengths = {subject: ts.shape[0] for subject, ts in subject_series.items()}
    unique_lengths = sorted(set(series_lengths.values()))
    if len(unique_lengths) != 1:
        raise ValueError(f"All subject timecourses must have equal length. Found lengths: {series_lengths}")

    n_timepoints = unique_lengths[0]

    for subject_index, subject in enumerate(subjects, start=1):
        logger.info("Preparing censor mask %d/%d: %s", subject_index, len(subjects), subject)
        series = subject_series[subject]
        censor_file = _find_censor_file(subject_to_file[subject])
        if censor_file is not None:
            censor_values = _read_timecourse_column(censor_file)
            if censor_values.shape[0] != n_timepoints:
                raise ValueError(
                    f"Censor length mismatch for {subject}: {censor_file} has {censor_values.shape[0]} rows, "
                    f"expected {n_timepoints}."
                )
            censor_valid = _censor_to_valid_mask(censor_values)
        else:
            censor_valid = np.ones(n_timepoints, dtype=bool)

        subject_valid_mask[subject] = np.isfinite(series) & censor_valid

    if config.run_boundaries:
        run_boundaries = [(int(start), int(end)) for start, end in config.run_boundaries]
    else:
        run_boundaries = [(0, n_timepoints)]

    participants_path_raw = discovered_inputs.get("participants_tsv_path")
    participants_path = Path(participants_path_raw) if participants_path_raw else None
    groups = _load_groups(
        participants_tsv_path=participants_path,
        group_column=config.group_column,
    )
    comparison_sets = _build_comparison_sets(subjects=subjects, groups=groups)
    logger.info("Using %d comparison sets: %s", len(comparison_sets), sorted(comparison_sets.keys()))

    loso_summary: dict[str, dict[str, np.ndarray]] = {
        name: {subject: np.full(n_timepoints, np.nan, dtype=float) for subject in subjects}
        for name in comparison_sets
    }
    pairwise_summary: dict[str, dict[str, np.ndarray]] = {
        name: {subject: np.full(n_timepoints, np.nan, dtype=float) for subject in subjects}
        for name in comparison_sets
    }
    pairwise_rows: list[dict[str, Any]] = []

    cached_subjects: set[str] = set()
    if not overwrite:
        for subject in subjects:
            subject_cache_path = analysis_dir / subject / (
                f"{subject}_roi-{roi_label}_approach-{config.approach}_timeseries.tsv"
            )
            cached_map = _load_cached_subject_summary(
                subject_cache_path,
                list(comparison_sets.keys()),
                n_timepoints,
            )
            if cached_map is None:
                continue

            if config.approach == "loso":
                for set_name in comparison_sets:
                    loso_summary[set_name][subject] = cached_map[set_name]
            else:
                for set_name in comparison_sets:
                    pairwise_summary[set_name][subject] = cached_map[set_name]

            cached_subjects.add(subject)

    if cached_subjects:
        logger.info(
            "Loaded cached %s summaries for %d/%d participants",
            config.approach,
            len(cached_subjects),
            len(subjects),
        )

    for subject_index, subject in enumerate(subjects, start=1):
        if subject in cached_subjects:
            logger.info(
                "Using cached ISC for participant %d/%d: %s",
                subject_index,
                len(subjects),
                subject,
            )
            continue

        logger.info("Computing ISC for participant %d/%d: %s", subject_index, len(subjects), subject)
        for tr in range(n_timepoints):
            start, end = _window_bounds(center_tr=tr, window_size_trs=config.window_size_trs)
            if start < 0 or end > n_timepoints:
                continue
            if not _window_within_run(start=start, end=end, run_boundaries=run_boundaries):
                continue

            target_window = subject_series[subject][start:end]
            target_valid = subject_valid_mask[subject][start:end]
            if int(np.sum(target_valid)) < required_samples:
                continue

            for set_name, set_subjects in comparison_sets.items():
                comparison_subjects = [s for s in set_subjects if s != subject]
                eligible_subjects: list[str] = []
                for comparison_subject in comparison_subjects:
                    comparison_valid = subject_valid_mask[comparison_subject][start:end]
                    if int(np.sum(comparison_valid)) >= required_samples:
                        eligible_subjects.append(comparison_subject)

                if not eligible_subjects:
                    continue

                if config.approach == "loso":
                    comparison_stack = np.vstack([subject_series[s][start:end] for s in eligible_subjects])
                    comparison_valid_stack = np.vstack([subject_valid_mask[s][start:end] for s in eligible_subjects])
                    comparison_stack = np.where(comparison_valid_stack, comparison_stack, np.nan)
                    group_mean = np.nanmean(comparison_stack, axis=0)
                    overlap = np.isfinite(target_window) & target_valid & np.isfinite(group_mean)
                    if int(np.sum(overlap)) < required_samples:
                        continue
                    corr = _pearson_corr(target_window[overlap], group_mean[overlap])
                    loso_summary[set_name][subject][tr] = corr

                elif config.approach == "pairwise":
                    pairwise_z_values: list[float] = []
                    for comparison_subject in eligible_subjects:
                        comparison_window = subject_series[comparison_subject][start:end]
                        comparison_valid = subject_valid_mask[comparison_subject][start:end]
                        overlap = np.isfinite(target_window) & target_valid & np.isfinite(comparison_window) & comparison_valid

                        if int(np.sum(overlap)) < required_samples:
                            fisher_z = float("nan")
                        else:
                            corr = _pearson_corr(target_window[overlap], comparison_window[overlap])
                            fisher_z = _fisher_z(corr) if np.isfinite(corr) else float("nan")

                        pairwise_rows.append(
                            {
                                "comparison_set": set_name,
                                "target_subject": subject,
                                "timepoint": tr,
                                "comparison_subject": comparison_subject,
                                "fisher_z": fisher_z,
                            }
                        )

                        if np.isfinite(fisher_z):
                            pairwise_z_values.append(fisher_z)

                    if pairwise_z_values:
                        pairwise_summary[set_name][subject][tr] = float(np.mean(pairwise_z_values))

        # Persist each newly computed participant immediately for resumable runs.
        if config.approach == "loso":
            subject_summary = {
                set_name: {subject: loso_summary[set_name][subject]} for set_name in comparison_sets
            }
        else:
            subject_summary = {
                set_name: {subject: pairwise_summary[set_name][subject]} for set_name in comparison_sets
            }

        subject_long_df = _results_to_long_format(
            subject_summary,
            [subject],
            n_timepoints,
            groups,
            roi_label,
        )
        subject_dir = analysis_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_path = subject_dir / (
            f"{subject}_roi-{roi_label}_approach-{config.approach}_timeseries.tsv"
        )
        subject_long_df.drop(columns=["subject"]).to_csv(
            subject_path,
            sep="\t",
            index=False,
            na_rep="",
        )
        logger.info("Wrote participant cache file: %s", subject_path)

    outputs: dict[str, Any] = {
        "analysis": "time_resolved_isc",
        "roi_name": config.roi_name,
        "roi_label": roi_label,
        "approach": config.approach,
        "window_size_trs": config.window_size_trs,
        "required_samples": required_samples,
        "n_subjects": len(subjects),
        "n_timepoints": n_timepoints,
        "comparison_sets": sorted(comparison_sets.keys()),
        "group_dir": str(group_dir),
        "files": {},
        "subject_dirs": {subject: str(analysis_dir / subject) for subject in subjects},
    }

    if config.approach == "loso":
        long_df = _results_to_long_format(
            loso_summary, subjects, n_timepoints, groups, roi_label
        )

        group_path = group_dir / f"roi-{roi_label}_approach-loso_timeseries.tsv"
        long_df.to_csv(group_path, sep="\t", index=False, na_rep="")
        outputs["files"]["loso_group_timeseries"] = str(group_path)

        if config.make_figures:
            logger.info("Computing group averages and generating figures for LOSO")
            averages = _compute_group_averages(long_df)
            avg_path = _save_group_averages_tsv(averages, group_dir, roi_label, "loso")
            outputs["files"]["loso_group_averages"] = str(avg_path)

            activation_averages = _compute_group_activation_averages(
                subject_series,
                subject_valid_mask,
                comparison_sets,
                n_timepoints,
            )
            activation_avg_path = _save_group_activation_averages_tsv(
                activation_averages,
                group_dir,
                roi_label,
                "loso",
            )
            outputs["files"]["loso_activation_averages"] = str(activation_avg_path)

            existing_loso_figs = sorted(group_dir.glob(f"roi-{roi_label}_group-*_approach-loso_figure.png"))
            if config.overwrite_figures or not existing_loso_figs:
                fig_paths = _create_group_figures(
                    averages,
                    group_dir,
                    roi_label,
                    "loso",
                    event_seconds=config.event_seconds,
                    tr_seconds=config.tr_seconds,
                    shared_zero=config.activation_shared_zero,
                )
                logger.info("Generated %d LOSO figures", len(fig_paths))
            else:
                fig_paths = existing_loso_figs
                logger.info("Using cached LOSO figures (%d)", len(fig_paths))
            outputs["files"]["loso_figures"] = [str(p) for p in fig_paths]

            existing_loso_overlay_figs = sorted(
                group_dir.glob(f"roi-{roi_label}_group-*_approach-loso_figure_with_activation.png")
            )
            if config.overwrite_figures or not existing_loso_overlay_figs:
                overlay_fig_paths = _create_group_figures_with_activation(
                    averages,
                    activation_averages,
                    group_dir,
                    roi_label,
                    "loso",
                    event_seconds=config.event_seconds,
                    tr_seconds=config.tr_seconds,
                    shared_zero=config.activation_shared_zero,
                )
                logger.info("Generated %d LOSO activation-overlay figures", len(overlay_fig_paths))
            else:
                overlay_fig_paths = existing_loso_overlay_figs
                logger.info("Using cached LOSO activation-overlay figures (%d)", len(overlay_fig_paths))
            outputs["files"]["loso_figures_with_activation"] = [str(p) for p in overlay_fig_paths]

    if config.approach == "pairwise":
        long_df = _results_to_long_format(
            pairwise_summary, subjects, n_timepoints, groups, roi_label
        )

        group_path = group_dir / f"roi-{roi_label}_approach-pairwise_timeseries.tsv"
        long_df.to_csv(group_path, sep="\t", index=False, na_rep="")
        outputs["files"]["pairwise_group_timeseries"] = str(group_path)

        if config.make_figures:
            logger.info("Computing group averages and generating figures for pairwise")
            averages = _compute_group_averages(long_df)
            avg_path = _save_group_averages_tsv(averages, group_dir, roi_label, "pairwise")
            outputs["files"]["pairwise_group_averages"] = str(avg_path)

            activation_averages = _compute_group_activation_averages(
                subject_series,
                subject_valid_mask,
                comparison_sets,
                n_timepoints,
            )
            activation_avg_path = _save_group_activation_averages_tsv(
                activation_averages,
                group_dir,
                roi_label,
                "pairwise",
            )
            outputs["files"]["pairwise_activation_averages"] = str(activation_avg_path)

            existing_pw_figs = sorted(group_dir.glob(f"roi-{roi_label}_group-*_approach-pairwise_figure.png"))
            if config.overwrite_figures or not existing_pw_figs:
                fig_paths = _create_group_figures(
                    averages,
                    group_dir,
                    roi_label,
                    "pairwise",
                    event_seconds=config.event_seconds,
                    tr_seconds=config.tr_seconds,
                    shared_zero=config.activation_shared_zero,
                )
                logger.info("Generated %d pairwise figures", len(fig_paths))
            else:
                fig_paths = existing_pw_figs
                logger.info("Using cached pairwise figures (%d)", len(fig_paths))
            outputs["files"]["pairwise_figures"] = [str(p) for p in fig_paths]

            existing_pw_overlay_figs = sorted(
                group_dir.glob(f"roi-{roi_label}_group-*_approach-pairwise_figure_with_activation.png")
            )
            if config.overwrite_figures or not existing_pw_overlay_figs:
                overlay_fig_paths = _create_group_figures_with_activation(
                    averages,
                    activation_averages,
                    group_dir,
                    roi_label,
                    "pairwise",
                    event_seconds=config.event_seconds,
                    tr_seconds=config.tr_seconds,
                    shared_zero=config.activation_shared_zero,
                )
                logger.info("Generated %d pairwise activation-overlay figures", len(overlay_fig_paths))
            else:
                overlay_fig_paths = existing_pw_overlay_figs
                logger.info("Using cached pairwise activation-overlay figures (%d)", len(overlay_fig_paths))
            outputs["files"]["pairwise_figures_with_activation"] = [str(p) for p in overlay_fig_paths]

        pairwise_path = group_dir / f"desc-pairwise_roi-{roi_label}_details.tsv"
        pairwise_df = pd.DataFrame(pairwise_rows)
        if not pairwise_df.empty:
            pairwise_df["roi_name"] = config.roi_name if config.roi_name is not None else "all"
        pairwise_df.to_csv(pairwise_path, sep="\t", index=False, na_rep="")
        outputs["files"]["pairwise_details"] = str(pairwise_path)

    metadata_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_timeseries_metadata.json"
    metadata = {
        "analysis": "time_resolved_isc",
        "config": {
            "window_size_trs": config.window_size_trs,
            "min_samples": config.min_samples,
            "required_samples": required_samples,
            "approach": config.approach,
            "run_boundaries": config.run_boundaries,
            "group_column": config.group_column,
            "roi_name": config.roi_name,
            "timecourse_glob": config.timecourse_glob,
            "timecourse_files": config.timecourse_files,
            "make_figures": config.make_figures,
            "activation_shared_zero": config.activation_shared_zero,
            "event_seconds": config.event_seconds,
            "tr_seconds": config.tr_seconds,
        },
        "n_subjects": len(subjects),
        "n_timepoints": n_timepoints,
        "comparison_sets": sorted(comparison_sets.keys()),
        "files": outputs["files"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["files"]["metadata"] = str(metadata_path)

    logger.info("time_resolved_isc finished. Outputs written to %s", analysis_dir)

    return outputs
