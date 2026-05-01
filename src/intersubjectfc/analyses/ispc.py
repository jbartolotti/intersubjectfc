"""Inter-Subject Pattern Correlation (ISPC) analysis.

For each TR, extracts the spatial pattern of BOLD signal within an ROI mask
for every subject, then computes the correlation between each subject's
pattern and the LOSO group mean pattern (or pairwise patterns).

Input:
    - 4D AFNI BRIK files (denoised residuals), one per subject.
    - A single ROI mask (NIfTI or AFNI) defining the voxels to include.
    - Optional 1D censor files (1 = good TR, 0 = exclude).

Output:
    - Per-subject long-format TSV: tr, comparison_group, comparison_type, value
    - Group-combined TSV
    - Group averages TSV (mean, SEM, n per TR)
    - PNG figures with optional event markers
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ISPCConfig:
    roi_mask: str                          # Path to ROI mask image
    roi_name: str                          # Label used in output filenames
    approach: str = "loso"                 # "loso" or "pairwise"
    brain_data_root: str | None = None     # Base directory for AFNI subject folders
    brain_glob: str | None = None          # Glob relative to brain_data_root for BRIK files
    brain_files: dict[str, str] | None = None  # Explicit subject_id -> file path override
    group_column: str | None = None        # participants.tsv column for group comparisons
    make_figures: bool = True
    overwrite_figures: bool = False
    event_seconds: list[float] | None = None
    tr_seconds: float | None = None


# ---------------------------------------------------------------------------
# Helpers: file discovery
# ---------------------------------------------------------------------------

def _find_brain_files(
    bids_root: Path,
    config: ISPCConfig,
) -> dict[str, Path]:
    """Return subject_id -> 4D brain file path mapping.

    Priority:
    1. config.brain_files (explicit mapping)
    2. config.brain_data_root + config.brain_glob (glob discovery)
    """
    if config.brain_files:
        return {sid: Path(p) for sid, p in config.brain_files.items()}

    if not config.brain_data_root or not config.brain_glob:
        raise ValueError(
            "ISPC requires either 'brain_files' or both "
            "'brain_data_root' and 'brain_glob' in config."
        )

    base = Path(config.brain_data_root).expanduser()
    matches: dict[str, Path] = {}
    for path in sorted(base.glob(config.brain_glob)):
        # Infer subject ID from the first path component after base.
        try:
            rel = path.relative_to(base)
            subject_id = rel.parts[0]
            matches[subject_id] = path
        except (ValueError, IndexError):
            logger.warning("Could not infer subject ID from path: %s", path)

    return matches


def _find_censor_file(brain_file: Path, subject_id: str) -> Path | None:
    """Look for censor_<subject_id>*2.1D in the same directory."""
    parent = brain_file.parent
    candidates = sorted(parent.glob(f"censor_{subject_id}*2.1D"))
    if candidates:
        if len(candidates) > 1:
            logger.warning(
                "Multiple censor files found for %s; using first: %s",
                subject_id,
                candidates[0],
            )
        return candidates[0]
    return None


def _load_censor_mask(censor_file: Path, n_timepoints: int) -> np.ndarray:
    """Load a 1D censor file as a boolean array (True = good TR)."""
    values = np.loadtxt(censor_file).astype(float).ravel()
    if values.shape[0] != n_timepoints:
        raise ValueError(
            f"Censor file {censor_file} has {values.shape[0]} rows; "
            f"expected {n_timepoints}."
        )
    return values > 0.5


# ---------------------------------------------------------------------------
# Helpers: mask and data loading
# ---------------------------------------------------------------------------

def _resolve_brain_load_path(brain_file: Path) -> Path:
    """Return the loadable path for a brain image (prefer AFNI .HEAD over .BRIK)."""
    brain_path = brain_file
    if brain_path.suffix.upper() == ".BRIK":
        head_path = brain_path.with_suffix(".HEAD")
        if head_path.exists():
            return head_path
    return brain_path


def _load_roi_mask_for_reference(mask_path: str | Path, reference_brain_file: Path) -> np.ndarray:
    """Load ROI mask and align it to a reference brain grid.

    If shape/affine differ, nearest-neighbor resampling is applied so that the
    mask can be indexed against the flattened 4D brain data.
    """
    mask_img = nib.load(str(mask_path))
    ref_img = nib.load(str(_resolve_brain_load_path(reference_brain_file)))

    ref_3d = nib.Nifti1Image(
        np.asanyarray(ref_img.dataobj)[..., 0],
        ref_img.affine,
        ref_img.header,
    )

    same_shape = tuple(mask_img.shape[:3]) == tuple(ref_3d.shape[:3])
    same_affine = np.allclose(mask_img.affine, ref_3d.affine, atol=1e-3)

    if not (same_shape and same_affine):
        logger.info(
            "Resampling ROI mask to reference brain grid: mask_shape=%s, ref_shape=%s",
            mask_img.shape,
            ref_3d.shape,
        )
        mask_img = resample_to_img(mask_img, ref_3d, interpolation="nearest", force_resample=True)

    mask_data = np.asanyarray(mask_img.dataobj)
    mask_bool = mask_data.ravel().astype(bool)
    if not np.any(mask_bool):
        raise ValueError(
            f"ROI mask has zero in-mask voxels after alignment: {mask_path}"
        )
    return mask_bool


def _load_4d_brain(
    brain_file: Path,
    mask_flat: np.ndarray,
) -> np.ndarray:
    """Load 4D brain and extract in-mask voxels.

    Returns array of shape (n_timepoints, n_voxels).
    Loads .HEAD/.BRIK via nibabel; also handles NIfTI.
    """
    brik_path = _resolve_brain_load_path(brain_file)

    img = nib.load(str(brik_path))
    data = np.asanyarray(img.dataobj).astype(np.float32)   # (x, y, z, t)

    # Flatten to (n_voxels, n_timepoints) then transpose.
    flat = data.reshape(-1, data.shape[-1])                 # (n_voxels_total, t)
    if flat.shape[0] != mask_flat.shape[0]:
        raise ValueError(
            "ROI mask and brain volume size mismatch after alignment: "
            f"brain_voxels={flat.shape[0]}, mask_voxels={mask_flat.shape[0]}, file={brain_file}"
        )
    masked = flat[mask_flat, :]                             # (n_mask_voxels, t)
    return masked.T                                         # (t, n_mask_voxels)


def _exclude_zero_variance_voxels(
    subject_data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    """Remove voxel columns where *any* subject has zero variance.

    Returns cleaned data dict and the number of voxels retained.
    """
    subjects = list(subject_data.keys())
    n_voxels = next(iter(subject_data.values())).shape[1]

    keep = np.ones(n_voxels, dtype=bool)
    for subject in subjects:
        variance = np.nanvar(subject_data[subject], axis=0)
        keep &= variance > 0.0

    n_kept = int(np.sum(keep))
    if n_kept < n_voxels:
        logger.info("Excluding %d zero-variance voxels (%d retained)", n_voxels - n_kept, n_kept)

    cleaned = {subject: subject_data[subject][:, keep] for subject in subjects}
    return cleaned, n_kept


# ---------------------------------------------------------------------------
# Helpers: group / comparison sets (reuse logic from time_resolved_isc)
# ---------------------------------------------------------------------------

def _load_groups(
    participants_tsv_path: Path | None,
    group_column: str | None,
) -> dict[str, str]:
    if not group_column or not participants_tsv_path or not participants_tsv_path.exists():
        return {}
    df = pd.read_csv(participants_tsv_path, sep="\t", dtype=str)
    if "participant_id" not in df.columns or group_column not in df.columns:
        return {}
    result: dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["participant_id"]).strip()
        label = str(row[group_column]).strip()
        result[pid] = label
        # Keep both sub-XXX and plain XXX keys for flexible matching.
        if pid.startswith("sub-"):
            result[pid[4:]] = label
        else:
            result[f"sub-{pid}"] = label
    return result


def _build_comparison_sets(
    subjects: list[str],
    groups: dict[str, str],
) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {"full": subjects}
    group_labels = {groups.get(s) for s in subjects if groups.get(s)}
    for label in sorted(group_labels):
        sets[label] = [s for s in subjects if groups.get(s) == label]
    return sets


def _get_comparison_type(
    subject: str,
    comparison_group: str,
    groups: dict[str, str],
) -> str:
    if comparison_group == "full":
        return "full"
    subject_group = groups.get(subject)
    if subject_group is None:
        return "full"
    return "within" if subject_group == comparison_group else "between"


# ---------------------------------------------------------------------------
# Helpers: correlation
# ---------------------------------------------------------------------------

def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays."""
    if a.size < 2:
        return float("nan")
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a_c, b_c) / denom)


def _fisher_z(r: float) -> float:
    r_clipped = float(np.clip(r, -0.9999999, 0.9999999))
    return float(np.arctanh(r_clipped))


# ---------------------------------------------------------------------------
# Helpers: group averages (identical schema to time_resolved_isc)
# ---------------------------------------------------------------------------

def _compute_group_averages(
    long_df: pd.DataFrame,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    if long_df.empty:
        return averages

    n_trs = int(long_df["tr"].max()) + 1
    for (cg, ct), group_data in long_df.groupby(["comparison_group", "comparison_type"]):
        values_by_tr: list[list[float]] = [[] for _ in range(n_trs)]
        for _, row in group_data.iterrows():
            v = row["value"]
            if pd.notna(v):
                values_by_tr[int(row["tr"])].append(float(v))

        means = np.full(n_trs, np.nan)
        sems = np.full(n_trs, np.nan)
        counts = np.zeros(n_trs)
        for tr_idx, vals in enumerate(values_by_tr):
            n = len(vals)
            counts[tr_idx] = n
            if n > 0:
                means[tr_idx] = np.mean(vals)
            if n >= 2:
                sems[tr_idx] = np.std(vals, ddof=1) / np.sqrt(n)

        if cg not in averages:
            averages[cg] = {}
        averages[cg][ct] = (means, sems, counts)

    return averages


def _save_group_averages_tsv(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    out_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for cg, type_dict in sorted(averages.items()):
        for ct, (means, sems, counts) in sorted(type_dict.items()):
            for tr_idx, (m, s, n) in enumerate(zip(means, sems, counts)):
                if not np.isnan(m):
                    rows.append({
                        "comparison_group": cg,
                        "comparison_type": ct,
                        "tr": tr_idx,
                        "mean": m,
                        "sem": float(s) if not np.isnan(s) else "",
                        "n": int(n),
                    })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, na_rep="")


def _create_group_figures(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    group_dir: Path,
    roi_label: str,
    approach: str,
    event_seconds: list[float] | None = None,
    tr_seconds: float | None = None,
) -> list[Path]:
    figure_paths: list[Path] = []
    colors = {"within": "#1f77b4", "between": "#ff7f0e", "full": "#2ca02c"}

    for comparison_group in sorted(averages.keys()):
        type_dict = averages[comparison_group]
        fig, ax = plt.subplots(figsize=(12, 6))

        for comparison_type in sorted(type_dict.keys()):
            means, sems, _counts = type_dict[comparison_type]
            trs = np.arange(len(means))
            color = colors.get(comparison_type, "#999999")
            ax.plot(trs, means, label=comparison_type, color=color, linewidth=2)
            # Only shade where SEM is valid.
            valid_sem = ~np.isnan(sems)
            if np.any(valid_sem):
                ax.fill_between(
                    trs,
                    np.where(valid_sem, means - sems, np.nan),
                    np.where(valid_sem, means + sems, np.nan),
                    alpha=0.2, color=color,
                )

        if event_seconds is not None and tr_seconds is not None and tr_seconds > 0:
            for event_sec in event_seconds:
                ax.axvline(event_sec / tr_seconds, color="gray", linestyle="--", alpha=0.3, linewidth=1)
                ax.axvline((event_sec + 6.0) / tr_seconds, color="gray", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_xlabel("TR")
        ax.set_ylabel("Pattern Correlation (ISPC)")
        ax.set_title(
            f"Inter-Subject Pattern Correlation: {comparison_group} group "
            f"(ROI: {roi_label}, Approach: {approach})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = group_dir / f"roi-{roi_label}_group-{comparison_group}_approach-{approach}_ispc_figure.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)

    return figure_paths


def _compute_group_activation_averages(
    subject_roi_paths: dict[str, Path],
    kept_voxel_indices: np.ndarray,
    subject_censor: dict[str, np.ndarray],
    comparison_sets: dict[str, list[str]],
    n_timepoints: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute per-TR group-average activation from subject ROI timeseries.

    Activation is the subject-wise mean across kept ROI voxels at each TR.
    Censored TRs are treated as missing.
    """
    per_subject_activation: dict[str, np.ndarray] = {}
    for subject, roi_path in subject_roi_paths.items():
        roi_data = np.load(roi_path, mmap_mode="r")
        roi_kept = np.asarray(roi_data[:, kept_voxel_indices], dtype=np.float32)
        activation = np.mean(roi_kept, axis=1).astype(np.float32)
        activation[~subject_censor[subject]] = np.nan
        per_subject_activation[subject] = activation

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for set_name, set_subjects in comparison_sets.items():
        if not set_subjects:
            continue
        label = "all" if set_name == "full" else set_name
        matrix = np.vstack([per_subject_activation[s] for s in set_subjects])

        means = np.full(n_timepoints, np.nan)
        sems = np.full(n_timepoints, np.nan)
        counts = np.zeros(n_timepoints)

        for tr in range(n_timepoints):
            vals = matrix[:, tr]
            valid = np.isfinite(vals)
            n = int(np.sum(valid))
            counts[tr] = n
            if n > 0:
                means[tr] = float(np.mean(vals[valid]))
            if n >= 2:
                sems[tr] = float(np.std(vals[valid], ddof=1) / np.sqrt(n))

        out[label] = (means, sems, counts)

    return out


def _save_group_activation_averages_tsv(
    activation_averages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for comparison_group, (means, sems, counts) in sorted(activation_averages.items()):
        for tr_idx, (m, s, n) in enumerate(zip(means, sems, counts)):
            if not np.isnan(m):
                rows.append(
                    {
                        "comparison_group": comparison_group,
                        "tr": tr_idx,
                        "activation_mean": float(m),
                        "activation_sem": float(s) if not np.isnan(s) else "",
                        "n": int(n),
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, na_rep="")


def _create_group_figures_with_activation(
    averages: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    activation_averages: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_dir: Path,
    roi_label: str,
    approach: str,
    event_seconds: list[float] | None = None,
    tr_seconds: float | None = None,
) -> list[Path]:
    figure_paths: list[Path] = []
    colors = {"within": "#1f77b4", "between": "#ff7f0e", "full": "#2ca02c"}

    for comparison_group in sorted(averages.keys()):
        type_dict = averages[comparison_group]
        fig, ax = plt.subplots(figsize=(12, 6))
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
                linewidth=2,
                alpha=0.9,
            )

        if event_seconds is not None and tr_seconds is not None and tr_seconds > 0:
            for event_sec in event_seconds:
                ax.axvline(event_sec / tr_seconds, color="gray", linestyle="--", alpha=0.3, linewidth=1)
                ax.axvline((event_sec + 6.0) / tr_seconds, color="gray", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_xlabel("TR")
        ax.set_ylabel("Pattern Correlation (ISPC)")
        ax2.set_ylabel("Activation")
        ax.set_title(
            f"Inter-Subject Pattern Correlation + Activation: {comparison_group} group "
            f"(ROI: {roi_label}, Approach: {approach})"
        )
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)
        ax.grid(True, alpha=0.3)

        fig_path = group_dir / (
            f"roi-{roi_label}_group-{comparison_group}_approach-{approach}_ispc_figure_with_activation.png"
        )
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)

    return figure_paths


# ---------------------------------------------------------------------------
# Helpers: long-format conversion
# ---------------------------------------------------------------------------

def _results_to_long_format(
    summary: dict[str, dict[str, np.ndarray]],
    subjects: list[str],
    n_timepoints: int,
    groups: dict[str, str],
    roi_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for set_name, subject_map in summary.items():
        comparison_group = "all" if set_name == "full" else set_name
        for subject in subjects:
            series = subject_map.get(subject, np.full(n_timepoints, np.nan))
            comparison_type = _get_comparison_type(subject, set_name, groups)
            for tr, value in enumerate(series):
                if np.isfinite(value):
                    rows.append({
                        "subject": subject,
                        "tr": tr,
                        "roi": roi_label,
                        "comparison_group": comparison_group,
                        "comparison_type": comparison_type,
                        "value": value,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _safe_name(s: Any) -> str:
    if not s:
        return "unknown"
    s_str = str(s)
    return s_str.replace(" ", "_").replace("/", "-").replace("\\", "-")


def _load_cached_subject_summary(
    subject_path: Path,
    comparison_sets: list[str],
    n_timepoints: int,
) -> dict[str, np.ndarray] | None:
    if not subject_path.exists():
        return None
    df = pd.read_csv(subject_path, sep="\t")
    if not {"comparison_group", "tr", "value"}.issubset(df.columns):
        return None

    out: dict[str, np.ndarray] = {
        s: np.full(n_timepoints, np.nan) for s in comparison_sets
    }
    for set_name in comparison_sets:
        label = "all" if set_name == "full" else set_name
        subset = df[df["comparison_group"] == label]
        if subset.empty:
            continue
        trs = subset["tr"].to_numpy(dtype=int)
        vals = subset["value"].to_numpy(dtype=float)
        valid = (trs >= 0) & (trs < n_timepoints) & np.isfinite(vals)
        out[set_name][trs[valid]] = vals[valid]
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ispc_analysis(
    bids_root: Path,
    output_root: Path,
    discovered_inputs: dict[str, Any],
    config_dict: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Inter-Subject Pattern Correlation (ISPC) for a single ROI mask."""

    roi_name = config_dict.get("roi_name")
    if roi_name is None:
        roi_name = _safe_name(config_dict.get("roi_mask", "roi"))

    config = ISPCConfig(
        roi_mask=config_dict["roi_mask"],
        roi_name=str(roi_name),
        approach=str(config_dict.get("approach", "loso")).lower(),
        brain_data_root=config_dict.get("brain_data_root"),
        brain_glob=config_dict.get("brain_glob"),
        brain_files=config_dict.get("brain_files"),
        group_column=config_dict.get("group_column"),
        make_figures=bool(config_dict.get("make_figures", True)),
        overwrite_figures=bool(config_dict.get("overwrite_figures", False)),
        event_seconds=config_dict.get("event_seconds"),
        tr_seconds=config_dict.get("tr_seconds"),
    )

    if config.approach not in {"loso", "pairwise"}:
        raise ValueError("ISPC approach must be 'loso' or 'pairwise'.")

    roi_label = _safe_name(config.roi_name)
    analysis_dir = output_root / "ispc"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    group_dir = analysis_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    # Discover brain files first (needed for cache key list).
    subject_to_file = _find_brain_files(bids_root=bids_root, config=config)
    subjects = sorted(subject_to_file)

    if not subjects:
        raise ValueError("No subject brain files found for ISPC. Check brain_data_root and brain_glob.")

    logger.info("ISPC: found %d subject brain files", len(subjects))

    # Full cache short-circuit.
    metadata_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_ispc_metadata.json"
    group_ts_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_ispc_timeseries.tsv"
    group_avg_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_ispc_group_averages.tsv"
    activation_avg_path = group_dir / f"roi-{roi_label}_approach-{config.approach}_ispc_group_activation_averages.tsv"
    subject_ts_paths = [
        analysis_dir / s / f"{s}_roi-{roi_label}_approach-{config.approach}_ispc_timeseries.tsv"
        for s in subjects
    ]

    core_required = [metadata_path, group_ts_path, *subject_ts_paths]
    existing_base_figures = sorted(
        group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_ispc_figure.png")
    )
    existing_activation_figures = sorted(
        group_dir.glob(
            f"roi-{roi_label}_group-*_approach-{config.approach}_ispc_figure_with_activation.png"
        )
    )
    figure_outputs_ready = (
        group_avg_path.exists()
        and activation_avg_path.exists()
        and len(existing_base_figures) > 0
        and len(existing_activation_figures) > 0
    )

    if not overwrite and all(p.exists() for p in core_required) and (
        (not config.make_figures)
        or (not config.overwrite_figures and figure_outputs_ready)
    ):
        logger.info(
            "Cache hit: skipping ISPC for roi=%s, approach=%s (%d subjects)",
            config.roi_name, config.approach, len(subjects),
        )
        figure_paths = sorted(existing_base_figures + existing_activation_figures)
        return {
            "analysis": "ispc",
            "status": "skipped_cache",
            "roi_name": config.roi_name,
            "roi_label": roi_label,
            "approach": config.approach,
            "n_subjects": len(subjects),
            "group_dir": str(group_dir),
            "files": {
                "group_timeseries": str(group_ts_path),
                "group_averages": str(group_avg_path),
                "group_activation_averages": str(activation_avg_path),
                "metadata": str(metadata_path),
                "figures": [str(p) for p in figure_paths],
            },
            "subject_dirs": {s: str(analysis_dir / s) for s in subjects},
        }

    # Load ROI mask aligned to the first subject's brain grid.
    logger.info("Loading ROI mask: %s", config.roi_mask)
    mask_flat = _load_roi_mask_for_reference(config.roi_mask, subject_to_file[subjects[0]])
    n_mask_voxels = int(np.sum(mask_flat))
    logger.info("Mask contains %d voxels", n_mask_voxels)

    # Extract each subject ROI matrix to disk so we do not keep full 4D data in RAM.
    roi_cache_dir = analysis_dir / "_roi_cache" / f"roi-{roi_label}"
    roi_cache_dir.mkdir(parents=True, exist_ok=True)

    subject_roi_paths: dict[str, Path] = {}
    n_timepoints_per_subject: dict[str, int] = {}
    voxel_keep_mask: np.ndarray | None = None

    for idx, subject in enumerate(subjects, start=1):
        brain_file = subject_to_file[subject]
        roi_path = roi_cache_dir / f"{subject}_roi-{roi_label}_matrix.npy"
        var_path = roi_cache_dir / f"{subject}_roi-{roi_label}_varmask.npy"

        subject_roi_paths[subject] = roi_path
        use_cached_roi = (not overwrite) and roi_path.exists() and var_path.exists()

        if use_cached_roi:
            roi_data = np.load(roi_path, mmap_mode="r")
            var_mask = np.load(var_path)
            if roi_data.ndim != 2 or roi_data.shape[1] != n_mask_voxels or var_mask.shape[0] != n_mask_voxels:
                logger.warning("Cached ROI matrix shape mismatch for %s; rebuilding cache", subject)
                use_cached_roi = False
            else:
                n_timepoints_per_subject[subject] = int(roi_data.shape[0])

        if not use_cached_roi:
            logger.info("Extracting ROI matrix %d/%d: %s from %s", idx, len(subjects), subject, brain_file)
            roi_data = _load_4d_brain(brain_file, mask_flat).astype(np.float32, copy=False)
            np.save(roi_path, roi_data)
            var_mask = np.nanvar(roi_data, axis=0) > 0.0
            np.save(var_path, var_mask.astype(np.bool_))
            n_timepoints_per_subject[subject] = int(roi_data.shape[0])
            del roi_data

        if voxel_keep_mask is None:
            voxel_keep_mask = np.asarray(var_mask, dtype=bool)
        else:
            voxel_keep_mask &= np.asarray(var_mask, dtype=bool)

    unique_lengths = sorted(set(n_timepoints_per_subject.values()))
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Subject 4D files have different numbers of TRs: {n_timepoints_per_subject}"
        )
    n_timepoints = unique_lengths[0]
    logger.info("All subjects have %d TRs", n_timepoints)

    if voxel_keep_mask is None:
        raise ValueError("Failed to compute voxel keep mask for ISPC.")

    kept_voxel_indices = np.where(voxel_keep_mask)[0]
    n_voxels_kept = int(kept_voxel_indices.shape[0])
    if n_voxels_kept == 0:
        raise ValueError("No non-zero-variance voxels remained after subject intersection.")
    if n_voxels_kept < n_mask_voxels:
        logger.info(
            "Excluding %d zero-variance voxels (%d retained)",
            n_mask_voxels - n_voxels_kept,
            n_voxels_kept,
        )

    # Load censor masks.
    subject_censor: dict[str, np.ndarray] = {}
    for subject in subjects:
        brain_file = subject_to_file[subject]
        censor_file = _find_censor_file(brain_file, subject)
        if censor_file is not None:
            logger.info("Censor file found for %s: %s", subject, censor_file)
            subject_censor[subject] = _load_censor_mask(censor_file, n_timepoints)
        else:
            logger.info("No censor file found for %s; all TRs treated as valid", subject)
            subject_censor[subject] = np.ones(n_timepoints, dtype=bool)

    # Group / comparison set setup.
    participants_path_raw = discovered_inputs.get("participants_tsv_path")
    participants_path = Path(participants_path_raw) if participants_path_raw else None
    groups = _load_groups(participants_tsv_path=participants_path, group_column=config.group_column)
    comparison_sets = _build_comparison_sets(subjects=subjects, groups=groups)
    logger.info("Comparison sets: %s", sorted(comparison_sets.keys()))

    # Initialise summary arrays.
    loso_summary: dict[str, dict[str, np.ndarray]] = {
        s: {sub: np.full(n_timepoints, np.nan) for sub in subjects}
        for s in comparison_sets
    }
    pairwise_summary: dict[str, dict[str, np.ndarray]] = {
        s: {sub: np.full(n_timepoints, np.nan) for sub in subjects}
        for s in comparison_sets
    }
    pairwise_rows: list[dict[str, Any]] = []

    # Per-subject cache loading.
    cached_subjects: set[str] = set()
    if not overwrite:
        for subject in subjects:
            cache_path = (
                analysis_dir / subject
                / f"{subject}_roi-{roi_label}_approach-{config.approach}_ispc_timeseries.tsv"
            )
            cached_map = _load_cached_subject_summary(
                cache_path, list(comparison_sets.keys()), n_timepoints
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
            "Loaded cached ISPC summaries for %d/%d subjects",
            len(cached_subjects), len(subjects),
        )

    needs_compute_subjects = overwrite or (len(cached_subjects) < len(subjects))

    # For LOSO, precompute per-set sum patterns only if we need new subject computations.
    loso_set_sums: dict[str, np.ndarray] = {}
    loso_set_counts: dict[str, np.ndarray] = {}
    if needs_compute_subjects and config.approach == "loso":
        for set_name, set_subjects in comparison_sets.items():
            set_sums = np.zeros((n_timepoints, n_voxels_kept), dtype=np.float32)
            set_counts = np.zeros(n_timepoints, dtype=np.int32)

            for set_subject in set_subjects:
                roi_data = np.load(subject_roi_paths[set_subject], mmap_mode="r")
                roi_kept = np.asarray(roi_data[:, kept_voxel_indices], dtype=np.float32)
                valid = subject_censor[set_subject]
                if np.any(valid):
                    set_sums[valid] += roi_kept[valid]
                    set_counts[valid] += 1

            loso_set_sums[set_name] = set_sums
            loso_set_counts[set_name] = set_counts

    # Per-subject ISPC computation.
    if needs_compute_subjects:
        for subject_index, subject in enumerate(subjects, start=1):
            if subject in cached_subjects:
                logger.info(
                    "Using cached ISPC for subject %d/%d: %s",
                    subject_index, len(subjects), subject,
                )
                continue

            logger.info("Computing ISPC for subject %d/%d: %s", subject_index, len(subjects), subject)
            target_roi_data = np.load(subject_roi_paths[subject], mmap_mode="r")
            target_data = np.asarray(target_roi_data[:, kept_voxel_indices], dtype=np.float32)
            target_censor = subject_censor[subject]

            for set_name, set_subjects in comparison_sets.items():
                comparison_subjects = [s for s in set_subjects if s != subject]
                if not comparison_subjects:
                    continue

                if config.approach == "loso":
                    set_sums = loso_set_sums[set_name]
                    set_counts = loso_set_counts[set_name]
                    subject_in_set = subject in set_subjects

                    for tr in range(n_timepoints):
                        if not target_censor[tr]:
                            continue

                        subtract_self = subject_in_set and bool(target_censor[tr])
                        n_other = int(set_counts[tr]) - (1 if subtract_self else 0)
                        if n_other <= 0:
                            continue

                        if subtract_self:
                            group_pattern = (set_sums[tr] - target_data[tr]) / float(n_other)
                        else:
                            group_pattern = set_sums[tr] / float(n_other)

                        corr = _pearson_corr(target_data[tr], group_pattern)
                        loso_summary[set_name][subject][tr] = corr

                elif config.approach == "pairwise":
                    z_sums = np.zeros(n_timepoints, dtype=np.float64)
                    z_counts = np.zeros(n_timepoints, dtype=np.int32)

                    for cs in comparison_subjects:
                        cs_roi_data = np.load(subject_roi_paths[cs], mmap_mode="r")
                        cs_data = np.asarray(cs_roi_data[:, kept_voxel_indices], dtype=np.float32)
                        cs_censor = subject_censor[cs]

                        for tr in range(n_timepoints):
                            if not target_censor[tr]:
                                continue

                            if not cs_censor[tr]:
                                fisher_z = float("nan")
                            else:
                                corr = _pearson_corr(target_data[tr], cs_data[tr])
                                fisher_z = _fisher_z(corr) if np.isfinite(corr) else float("nan")

                            pairwise_rows.append({
                                "comparison_set": set_name,
                                "target_subject": subject,
                                "timepoint": tr,
                                "comparison_subject": cs,
                                "fisher_z": fisher_z,
                            })

                            if np.isfinite(fisher_z):
                                z_sums[tr] += fisher_z
                                z_counts[tr] += 1

                    valid_mean = z_counts > 0
                    pairwise_summary[set_name][subject][valid_mean] = (
                        z_sums[valid_mean] / z_counts[valid_mean]
                    )

            # Write this subject's output immediately.
            summary = loso_summary if config.approach == "loso" else pairwise_summary
            subject_single = {
                set_name: {subject: summary[set_name][subject]}
                for set_name in comparison_sets
            }
            subject_long_df = _results_to_long_format(
                subject_single, [subject], n_timepoints, groups, roi_label,
            )
            subject_dir = analysis_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject_path = subject_dir / (
                f"{subject}_roi-{roi_label}_approach-{config.approach}_ispc_timeseries.tsv"
            )
            subject_long_df.drop(columns=["subject"]).to_csv(
                subject_path, sep="\t", index=False, na_rep=""
            )
            logger.info("Wrote subject ISPC file: %s", subject_path)

    # Group-level outputs.
    if needs_compute_subjects or not group_ts_path.exists():
        active_summary = loso_summary if config.approach == "loso" else pairwise_summary
        long_df = _results_to_long_format(
            active_summary, subjects, n_timepoints, groups, roi_label
        )
        long_df.to_csv(group_ts_path, sep="\t", index=False, na_rep="")
        logger.info("Wrote group ISPC timeseries: %s", group_ts_path)
    else:
        long_df = pd.read_csv(group_ts_path, sep="\t")
        logger.info("Using cached group ISPC timeseries: %s", group_ts_path)

    outputs: dict[str, Any] = {
        "analysis": "ispc",
        "roi_name": config.roi_name,
        "roi_label": roi_label,
        "approach": config.approach,
        "status": "completed" if needs_compute_subjects else "reused_cached_correlations",
        "n_subjects": len(subjects),
        "n_timepoints": n_timepoints,
        "n_voxels": n_voxels_kept,
        "comparison_sets": sorted(comparison_sets.keys()),
        "group_dir": str(group_dir),
        "files": {
            "group_timeseries": str(group_ts_path),
        },
        "subject_dirs": {s: str(analysis_dir / s) for s in subjects},
    }

    if config.make_figures:
        activation_averages = _compute_group_activation_averages(
            subject_roi_paths=subject_roi_paths,
            kept_voxel_indices=kept_voxel_indices,
            subject_censor=subject_censor,
            comparison_sets=comparison_sets,
            n_timepoints=n_timepoints,
        )
        _save_group_activation_averages_tsv(activation_averages, activation_avg_path)
        outputs["files"]["group_activation_averages"] = str(activation_avg_path)

        averages = _compute_group_averages(long_df)
        _save_group_averages_tsv(averages, group_avg_path)
        outputs["files"]["group_averages"] = str(group_avg_path)

        existing_base_figures = sorted(
            group_dir.glob(f"roi-{roi_label}_group-*_approach-{config.approach}_ispc_figure.png")
        )
        existing_activation_figures = sorted(
            group_dir.glob(
                f"roi-{roi_label}_group-*_approach-{config.approach}_ispc_figure_with_activation.png"
            )
        )
        should_render_figures = (
            needs_compute_subjects
            or config.overwrite_figures
            or len(existing_base_figures) == 0
            or len(existing_activation_figures) == 0
        )

        if should_render_figures:
            fig_paths = _create_group_figures(
                averages,
                group_dir,
                roi_label,
                config.approach,
                event_seconds=config.event_seconds,
                tr_seconds=config.tr_seconds,
            )
            fig_paths_with_activation = _create_group_figures_with_activation(
                averages,
                activation_averages,
                group_dir,
                roi_label,
                config.approach,
                event_seconds=config.event_seconds,
                tr_seconds=config.tr_seconds,
            )
            outputs["files"]["figures"] = [str(p) for p in sorted(fig_paths + fig_paths_with_activation)]
            logger.info(
                "Generated %d ISPC figures (%d with activation overlay)",
                len(fig_paths) + len(fig_paths_with_activation),
                len(fig_paths_with_activation),
            )
            if not needs_compute_subjects:
                outputs["status"] = "refreshed_figures"
        else:
            outputs["files"]["figures"] = [
                str(p)
                for p in sorted(existing_base_figures + existing_activation_figures)
            ]
            logger.info("Using cached ISPC figures")

    if config.approach == "pairwise" and pairwise_rows:
        pw_path = group_dir / f"desc-pairwise_roi-{roi_label}_ispc_details.tsv"
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df["roi_name"] = config.roi_name
        pairwise_df.to_csv(pw_path, sep="\t", index=False, na_rep="")
        outputs["files"]["pairwise_details"] = str(pw_path)

    # Metadata.
    metadata = {
        "analysis": "ispc",
        "config": {
            "roi_mask": config.roi_mask,
            "roi_name": config.roi_name,
            "approach": config.approach,
            "brain_data_root": config.brain_data_root,
            "brain_glob": config.brain_glob,
            "group_column": config.group_column,
            "make_figures": config.make_figures,
            "overwrite_figures": config.overwrite_figures,
            "event_seconds": config.event_seconds,
            "tr_seconds": config.tr_seconds,
        },
        "n_subjects": len(subjects),
        "n_timepoints": n_timepoints,
        "n_voxels": n_voxels_kept,
        "comparison_sets": sorted(comparison_sets.keys()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["files"]["metadata"] = str(metadata_path)

    logger.info("ISPC finished. Outputs written to %s", analysis_dir)
    return outputs
