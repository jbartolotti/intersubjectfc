from pathlib import Path

import pandas as pd

from intersubjectfc import run_intersubject_fc


def _write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_time_resolved_isc_loso_with_groups(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "test", "BIDSVersion": "1.9.0"}', encoding="utf-8"
    )
    (bids_root / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-01\tA\nsub-02\tB\nsub-03\tA\n", encoding="utf-8"
    )

    data_dir = bids_root / "derivatives" / "timeseries"
    data_dir.mkdir(parents=True)

    sub1 = data_dir / "sub-01_timecourse.tsv"
    sub2 = data_dir / "sub-02_timecourse.tsv"
    sub3 = data_dir / "sub-03_timecourse.tsv"

    _write_rows(sub1, ["0", "1", "2", "3", "4", "5", "6", "7"])
    _write_rows(sub2, ["0", "1", "2", "2.5", "3.5", "5", "6", "7"])
    _write_rows(sub3, ["0", "1", "2", "3", "4", "5", "6", "8"])

    result = run_intersubject_fc(
        bids_root=bids_root,
        config={
            "analyses": [
                {
                    "name": "time_resolved_isc",
                    "config": {
                        "approach": "loso",
                        "window_size_trs": 3,
                        "min_samples": 0.5,
                        "group_column": "group",
                        "timecourse_files": {
                            "sub-01": str(sub1),
                            "sub-02": str(sub2),
                            "sub-03": str(sub3),
                        },
                    },
                }
            ]
        },
    )

    outputs = result["analysis_outputs"][0]["files"]
    assert "loso_full" in outputs
    assert "loso_group-A" in outputs
    assert "loso_not-group-A" in outputs

    full_df = pd.read_csv(outputs["loso_full"], sep="\t")
    assert "subject" in full_df.columns
    assert full_df.shape[0] == 3


def test_time_resolved_isc_pairwise_writes_details(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "test", "BIDSVersion": "1.9.0"}', encoding="utf-8"
    )

    data_dir = bids_root / "derivatives" / "timeseries"
    data_dir.mkdir(parents=True)

    sub1 = data_dir / "sub-01_timecourse.tsv"
    sub2 = data_dir / "sub-02_timecourse.tsv"
    sub3 = data_dir / "sub-03_timecourse.tsv"

    _write_rows(sub1, ["0", "1", "", "3", "4", "5", "6", "7"])
    _write_rows(sub2, ["0", "1", "2", "3", "4", "5", "6", "7"])
    _write_rows(sub3, ["0", "2", "2", "3", "4", "6", "6", "7"])

    _write_rows(data_dir / "sub-02_timecourse_censor.tsv", ["1", "1", "0", "1", "1", "1", "1", "1"])

    result = run_intersubject_fc(
        bids_root=bids_root,
        config={
            "analyses": [
                {
                    "name": "time_resolved_isc",
                    "config": {
                        "approach": "pairwise",
                        "window_size_trs": 3,
                        "min_samples": 2,
                        "run_boundaries": [[0, 4], [4, 8]],
                        "timecourse_files": {
                            "sub-01": str(sub1),
                            "sub-02": str(sub2),
                            "sub-03": str(sub3),
                        },
                    },
                }
            ]
        },
    )

    outputs = result["analysis_outputs"][0]["files"]
    assert "pairwise_mean_full" in outputs
    assert "pairwise_details" in outputs

    details_df = pd.read_csv(outputs["pairwise_details"], sep="\t")
    assert set(["comparison_set", "target_subject", "timepoint", "comparison_subject", "fisher_z"]).issubset(
        details_df.columns
    )
    assert (details_df["comparison_set"] == "full").any()
