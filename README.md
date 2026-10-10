# TMHF peptide screening

Training, candidate screening, and representative-panel selection for the TMHF
antifungal-peptide workflow.

## Workflow

```text
Sequence enumeration and physicochemical filters: 2,450,854 candidates
  -> LightGBM potency classifier: 21,000 candidates
  -> XGBoost pairwise ranker: 1,000 candidates
  -> LightGBM MIC regression
  -> Seven-cluster selection from the regression Top400
```

The classifier uses MIC <= 2 as its positive class. The ranker uses records
with MIC <= 8. Regression predicts log2(MIC), and clustering selects three
representative roles from each of seven clusters.

## Installation

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Full training and screening

The two small training archives are included under `data/prepared/` and use
ordinary Git. Full screening additionally requires
`data/prepared/candidate_feature_matrix.npy` (2,450,854 x 829 float32 values,
approximately 7.6 GiB). It is excluded from Git and has no public download yet.
Expected file sizes and SHA-256 values are in `data/prepare_manifest.json`.

After obtaining the matrix, verify it:

```bash
python - <<'PYTHON'
import hashlib
import json
from pathlib import Path
path = Path("data/prepared/candidate_feature_matrix.npy")
expected = json.loads(Path("data/prepare_manifest.json").read_text())["files"][path.as_posix()]
with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert path.stat().st_size == expected["bytes"]
assert digest == expected["sha256"]
print("Verified")
PYTHON
```

Run these commands in order, proceeding when the previous command succeeds:

```bash
python -u code/02_train.py
python -u code/03_screen.py
python code/04_cluster_top400.py
```

Screening validates model files against the training manifest. The regression
model is saved as `outputs/models/regression.joblib`.

Training manifests record evaluation metrics and model checksums. Screening
validates row counts, finite predictions, pair-processing completion, and the
prescreened sequence checksum used to align candidate-matrix rows.

The default enumeration uses 16 workers; ranking uses 80,000,000 sampled pairs
and batches of 20,000. The previous setup used two NVIDIA RTX 4090 GPUs, at least
32 GB RAM, and approximately 20 GB additional output storage. CPU fallback is
available for ranker training; results can vary across software and devices.

## Outputs

- `outputs/models/`: trained models.
- `outputs/training_manifest.json`: training protocol, metrics, model hashes.
- `outputs/screening/`: retained candidates, regression scores, and Top20.
- `outputs/screening_manifest.json`: screening checks.
- `outputs/clustering/top400_clustering.xlsx`: `panel_21` (21 role selections),
  `unique_peptides`, `cluster_parameters`, and `readme` sheets.

Each cluster contributes a center medoid, a farthest boundary, and a peptide with
the lowest predicted MIC. The same peptide may fill multiple roles, so the number
of unique peptides can be less than 21 and is reported explicitly. The regression
Top20 and the cluster representatives are separate outputs.

## Data inventory

See the [data directory](data/README.md) for descriptive filenames, supplementary
dataset mappings, and checksums. Screening results, evaluation tables, and model
comparisons are organized by purpose under `data/`. The Top21000 feature matrix is included;
the full candidate feature matrix is distributed separately.

## Data preparation

Prepared training inputs are already supplied. Maintainers with the original
source workspace can regenerate them:

```bash
python code/01_prepare.py --source-root /path/to/food_atp
```

This utility requires the source tables, embedding cache, structural/context
features, candidate matrix, and prescreened sequence table. It uses the checked-in
classifier and regression configuration files.

## Evaluation and reproducibility

Classifier records and ranking pairs use the existing random 80:20 splits.
Regression uses 1,497 training and 374 validation sequences; the validation
partition is used for early stopping and model selection. Reported metrics
therefore describe these evaluation partitions, not independent external tests.

Model hyperparameters and split seeds are recorded in the configuration files.
The seven-cluster configuration uses `random_state=328`, `n_init=1`, and
`max_iter=500`. To run panel selection on the supplied regression scores:

```bash
python code/04_cluster_top400.py --input data/screening/regression_top1000.csv
```

Training and screening write run-specific metrics and candidate results to
`outputs/`. Code/data reuse terms and publication citation metadata remain to
be supplied.
