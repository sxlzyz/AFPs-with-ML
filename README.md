# TMHF peptide screening

Core code and prepared training inputs for reproducing the locked TMHF
antifungal-peptide candidate selection workflow.

## Scope

The workflow prepares inputs, trains three models, screens candidate sequences,
and selects a final computational panel of 21 peptides (seven clusters, three
representatives per cluster). The panel has not been experimentally validated.

```text
20^7 seven-residue seeds
  -> mirror/repeat expansion and sequence filtering: 2,450,854 candidates
  -> LightGBM potency classification: 21,000 candidates
  -> XGBoost pairwise ranking: 1,000 candidates
  -> LightGBM MIC regression: regression-ranked candidates
  -> K-Means on the best 400: 21 representatives
```

## Contents

| Path | Purpose |
| --- | --- |
| `code/01_prepare.py` | Optional migration of training inputs from the original source workspace |
| `code/02_train.py` | Train and verify the locked classifier, ranker, and regressor |
| `code/03_screen.py` | Enumerate, filter, classify, rank, and score candidates |
| `code/04_cluster_top400.py` | Select and verify the final 21-peptide panel |
| `code/common.py`, `code/tmhf_repro/` | Features and utilities required by the core pipeline |
| `data/prepared/` | Small prepared training and ranking-anchor archives |
| `data/reference/` | Expected ranking and saved regression scores for selection verification |
| `data/final_selected_21_peptides.csv` | Archived final panel |
| `data/final_clustering_method_specification.json` | Locked clustering parameters and audit metadata |
| `data/top400_cluster_assignments_final.csv`, `data/cluster_summary_final.csv` | Reference cluster assignments and summary |
| `tests/` | Core selection and data-integrity tests |
| `outputs/` | Generated files, ignored by Git |

## Installation

Use Python 3.12 and run all commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Small prepared archives are ordinary Git files; Git LFS is not required.
Historical feature-schema identifiers retain their original spelling for
compatibility. Documentation and generated sheet descriptions are English.

## Verify final selection with included data

The saved regression Top1000 allows the final selection step to be checked
without retraining or obtaining the large candidate matrix:

```bash
python code/04_cluster_top400.py --input data/reference/regression_top1000.csv
python -m pytest -q
```

The script writes `outputs/clustering/top400_clustering.xlsx` with `panel_21`,
`cluster_parameters`, and `readme` sheets. It checks the selected candidate,
cluster, and role assignments against the archived final panel. This verifies
the selection step only, not the upstream training and screening computation.

## Full training and screening

Full screening requires `data/prepared/candidate_feature_matrix.npy`
(2,450,854 x 829 float32 values; approximately 7.6 GiB). It is deliberately
excluded from this package and Git tracking. **No public download is available
yet, so a checkout alone cannot reproduce full screening.** The expected size
and SHA-256 are recorded in `data/prepare_manifest.json`.

If you already have the matrix, place it at the path above and verify it:

```bash
python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

path = Path("data/prepared/candidate_feature_matrix.npy")
manifest = json.loads(Path("data/prepare_manifest.json").read_text())
expected = manifest["files"][path.as_posix()]
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert path.stat().st_size == expected["bytes"]
assert digest == expected["sha256"]
print("Verified")
PYTHON
python -u code/02_train.py
python -u code/03_screen.py
python code/04_cluster_top400.py
```

The reference setup used 16 enumeration workers, two NVIDIA RTX 4090 GPUs,
at least 32 GB RAM, and approximately 20 GB of additional output storage.
Training selects a GPU when available and otherwise falls back to CPU. CPU/GPU
and software differences can change ties and prevent exact reference matching.
The locked ranking run uses 80,000,000 pairs in batches of 20,000. Changing those
settings changes the computation; preserve defaults for reference reproduction.

Training and screening write verification manifests and return a nonzero exit
status when their implemented checks fail. Key outputs include trained models
under `outputs/models/`, candidate tables under `outputs/screening/`, and the
final clustering workbook. The regression Top20 and clustered panel of 21 are
different outputs.

The small prepared inputs are already included. Maintainers with the original
source workspace can regenerate them using:

```bash
python code/01_prepare.py --source-root /path/to/food_atp
```

This is a migration utility, not a downloader. It needs the original training
tables, embedding cache, structural/context inputs, model configurations,
candidate matrix, and reference screening files. It does not recreate those
external assets from raw sequences. Original source filenames and column names
are retained where needed to read that archive.

## Interpretation and release limitations

- Regression uses 1,497 training and 374 model-selection sequences. The latter
  partition informed model selection; R2 = 0.5789692401885986 is not an
  independent external-test result.
- The ranker repeats each available designated anchor 50 times with assigned
  MIC 0.01 uM before a pair-level split. Its metrics do not establish independent
  generalization to unseen peptide sequences.
- Final clustering uses 486 sequence features, seven K-Means clusters,
  `random_state=328`, `n_init=1`, and `max_iter=500`. Representatives are the
  center medoid, farthest boundary, and lowest predicted MIC in each cluster.
  Historical configuration selection and target-hit audits limit prospective
  interpretation; audit metadata is retained with the locked method.
- The 21-peptide computational panel is distinct from the original experimental
  panel and awaits experimental validation.
- Code/data reuse terms and publication citation metadata have not yet been
  supplied. No license is implied by this package.
