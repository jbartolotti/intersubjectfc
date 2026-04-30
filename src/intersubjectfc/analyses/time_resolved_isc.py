"""Time-resolved ISC analysis implementation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SUBJECT_RE = re.compile(r"sub-[A-Za-z0-9]+")


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
        not_group_subjects = [s for s in subjects if groups.get(s) != group_name]
        comparison_sets[f"group-{_safe_name(group_name)}"] = group_subjects
        comparison_sets[f"not-group-{_safe_name(group_name)}"] = not_group_subjects

    return comparison_sets


def _series_to_wide_df(series_map: dict[str, np.ndarray]) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame()

    any_series = next(iter(series_map.values()))
    columns = [f"tr_{idx:05d}" for idx in range(any_series.shape[0])]
    data = {"subject": list(series_map.keys())}
    for idx, col in enumerate(columns):
        data[col] = [series_map[subject][idx] for subject in series_map]
    return pd.DataFrame(data)


def run_time_resolved_isc_analysis(
    bids_root: Path,
    output_root: Path,
    discovered_inputs: dict[str, Any],
    config_dict: dict[str, Any],
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
    )

    if config.approach not in {"loso", "pairwise"}:
        raise ValueError("time_resolved_isc approach must be 'loso' or 'pairwise'.")

    required_samples = _required_samples(config.min_samples, config.window_size_trs)
    subject_to_file = _find_timecourse_files(bids_root=bids_root, config=config)

    subjects = sorted(subject_to_file)
    subject_series: dict[str, np.ndarray] = {}
    subject_valid_mask: dict[str, np.ndarray] = {}

    for subject in subjects:
        series = _read_timecourse_column(subject_to_file[subject], roi_name=config.roi_name)
        subject_series[subject] = series

    series_lengths = {subject: ts.shape[0] for subject, ts in subject_series.items()}
    unique_lengths = sorted(set(series_lengths.values()))
    if len(unique_lengths) != 1:
        raise ValueError(f"All subject timecourses must have equal length. Found lengths: {series_lengths}")

    n_timepoints = unique_lengths[0]

    for subject in subjects:
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

    loso_summary: dict[str, dict[str, np.ndarray]] = {
        name: {subject: np.full(n_timepoints, np.nan, dtype=float) for subject in subjects}
        for name in comparison_sets
    }
    pairwise_summary: dict[str, dict[str, np.ndarray]] = {
        name: {subject: np.full(n_timepoints, np.nan, dtype=float) for subject in subjects}
        for name in comparison_sets
    }
    pairwise_rows: list[dict[str, Any]] = []

    for tr in range(n_timepoints):
        start, end = _window_bounds(center_tr=tr, window_size_trs=config.window_size_trs)
        if start < 0 or end > n_timepoints:
            continue
        if not _window_within_run(start=start, end=end, run_boundaries=run_boundaries):
            continue

        for subject in subjects:
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

    analysis_dir = output_root / "time_resolved_isc"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    roi_label = _safe_name(config.roi_name) if config.roi_name else "all"

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
        "files": {},
    }

    if config.approach == "loso":
        for set_name, set_series in loso_summary.items():
            out_name = f"desc-{_safe_name(set_name)}_roi-{roi_label}_approach-loso_timeseries.tsv"
            out_path = analysis_dir / out_name
            _series_to_wide_df(set_series).to_csv(out_path, sep="\t", index=False, na_rep="")
            outputs["files"][f"loso_{set_name}"] = str(out_path)

    if config.approach == "pairwise":
        for set_name, set_series in pairwise_summary.items():
            out_name = f"desc-{_safe_name(set_name)}_roi-{roi_label}_approach-pairwise_timeseries.tsv"
            out_path = analysis_dir / out_name
            _series_to_wide_df(set_series).to_csv(out_path, sep="\t", index=False, na_rep="")
            outputs["files"][f"pairwise_mean_{set_name}"] = str(out_path)

        pairwise_path = analysis_dir / f"desc-pairwise_roi-{roi_label}_details.tsv"
        pairwise_df = pd.DataFrame(pairwise_rows)
        if not pairwise_df.empty:
            pairwise_df["roi_name"] = config.roi_name if config.roi_name is not None else "all"
        pairwise_df.to_csv(pairwise_path, sep="\t", index=False, na_rep="")
        outputs["files"]["pairwise_details"] = str(pairwise_path)

    metadata_path = analysis_dir / f"time_resolved_isc_roi-{roi_label}_metadata.json"
    metadata = {
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
        },
        "subjects": subjects,
        "subject_timecourse_files": {subject: str(path) for subject, path in subject_to_file.items()},
        "comparison_sets": comparison_sets,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["files"]["metadata"] = str(metadata_path)

    return outputs
