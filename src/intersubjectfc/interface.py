"""High-level BIDS-aware interface for intersubjectfc analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyses import run_time_resolved_isc_analysis


def _ensure_derivative_layout(bids_root: Path, derivative_name: str) -> Path:
    """Create and return the derivatives path for this package."""
    derivatives_root = bids_root / "derivatives"
    output_root = derivatives_root / derivative_name
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _write_dataset_description(output_root: Path) -> Path:
    """Write a minimal BIDS derivative dataset_description.json file."""
    dataset_description = {
        "Name": "intersubjectfc outputs",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "intersubjectfc",
                "Version": "0.1.0",
                "Description": "Inter-subject functional connectivity analyses",
            }
        ],
        "HowToAcknowledge": "Please cite intersubjectfc and related methods publications.",
    }

    description_path = output_root / "dataset_description.json"
    description_path.write_text(json.dumps(dataset_description, indent=2), encoding="utf-8")
    return description_path


def _discover_bids_inputs(bids_root: Path) -> dict[str, Any]:
    """Collect a small set of core BIDS inputs for downstream analyses."""
    dataset_description_path = bids_root / "dataset_description.json"
    participants_tsv_path = bids_root / "participants.tsv"
    participant_dirs = sorted(
        [path.name for path in bids_root.glob("sub-*") if path.is_dir()]
    )

    dataset_description = None
    if dataset_description_path.exists():
        dataset_description = json.loads(dataset_description_path.read_text(encoding="utf-8"))

    return {
        "dataset_description_path": str(dataset_description_path) if dataset_description_path.exists() else None,
        "participants_tsv_path": str(participants_tsv_path) if participants_tsv_path.exists() else None,
        "n_subject_directories": len(participant_dirs),
        "subject_directories": participant_dirs,
        "input_bids_version": (dataset_description or {}).get("BIDSVersion"),
    }


def run_intersubject_fc(
    bids_root: Path | str,
    config: dict[str, Any] | None = None,
    derivative_name: str = "intersubjectfc",
) -> dict[str, Any]:
    """Run a placeholder BIDS-aware ISFC workflow.

    Parameters
    ----------
    bids_root:
        Path to the input BIDS dataset root.
    config:
        Optional analysis configuration dictionary.
    derivative_name:
        Name of the derivatives subdirectory created inside the BIDS dataset.

    Returns
    -------
    dict
        Metadata describing discovered paths and output locations.
    """
    bids_root = Path(bids_root).expanduser().resolve()
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {bids_root}")

    discovered_inputs = _discover_bids_inputs(bids_root=bids_root)
    output_root = _ensure_derivative_layout(bids_root=bids_root, derivative_name=derivative_name)
    description_path = _write_dataset_description(output_root=output_root)

    analysis_outputs: list[dict[str, Any]] = []
    analyses = (config or {}).get("analyses", [])
    for analysis in analyses:
        name = analysis.get("name")
        analysis_config = analysis.get("config", {})

        if name == "time_resolved_isc":
            result = run_time_resolved_isc_analysis(
                bids_root=bids_root,
                output_root=output_root,
                discovered_inputs=discovered_inputs,
                config_dict=analysis_config,
            )
            analysis_outputs.append(result)
        else:
            analysis_outputs.append(
                {
                    "analysis": name,
                    "status": "skipped",
                    "reason": "Analysis is not yet implemented.",
                }
            )

    run_info = {
        "package": "intersubjectfc",
        "status": "initialized",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bids_root": str(bids_root),
        "discovered_inputs": discovered_inputs,
        "derivative_root": str(output_root),
        "dataset_description": str(description_path),
        "config": config or {},
        "analysis_outputs": analysis_outputs,
    }

    (output_root / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return run_info
