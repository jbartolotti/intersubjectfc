from pathlib import Path

from intersubjectfc import run_intersubject_fc


def test_run_intersubject_fc_creates_derivative_metadata(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()

    result = run_intersubject_fc(bids_root=bids_root)

    derivative_root = bids_root / "derivatives" / "intersubjectfc"
    assert derivative_root.exists()
    assert (derivative_root / "dataset_description.json").exists()
    assert (derivative_root / "run_info.json").exists()
    assert result["status"] == "initialized"
    assert "discovered_inputs" in result
    assert result["discovered_inputs"]["n_subject_directories"] == 0
