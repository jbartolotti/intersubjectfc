# intersubjectfc

`intersubjectfc` is a Python package for BIDS-aware inter-subject functional connectivity (ISFC) analyses in fMRI.

## Current status

This first scaffold release focuses on package structure and an initial high-level interface.
Core analysis methods will be added next.

## Design goals

- Accept a single BIDS dataset root as the main input
- Discover relevant files and metadata through BIDS conventions
- Write outputs to a BIDS-compliant derivatives directory
- Ensure derivative metadata is documented in `dataset_description.json`

## Quick start

```bash
pip install -e .
```

```python
from pathlib import Path
from intersubjectfc import run_intersubject_fc

result = run_intersubject_fc(
    bids_root=Path("/path/to/bids_dataset"),
    config={"analysis": "placeholder"}
)

print(result)
```

## Planned functionality

- Multiple ISFC variants (seed-based, ROI-to-ROI, voxel-wise)
- Group-level summaries and inferential utilities
- Optional confound handling and preprocessing hooks
- Rich logging and provenance capture for reproducibility
