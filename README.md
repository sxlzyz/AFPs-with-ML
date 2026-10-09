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

The classifier uses MIC <= 2 as its positive class. The ranker uses observed
records with MIC <= 8, without duplicating designated sequences or assigning
artificial MIC values. Candidate retention follows model scores. No named
sequence is required to appear in the retained candidates or final panel.

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

Archive any previous `outputs/` directory before starting a new run. Then run
these commands in order, proceeding only when the previous command succeeds:

```bash
python -u code/02_train.py
python -u code/03_screen.py
python code/04_cluster_top400.py
```

Screening requires matching model files and a successful training manifest from
protocol `observed_labels_v1`. Older models must be retrained. The regression
model is now saved as `outputs/models/regression.joblib`.

Training manifests contain newly calculated evaluation metrics. Their `pass`
field checks successful execution and finite metrics, not a performance target.
Screening checks row counts, finite predictions, and pair-processing completion;
it does not require the previous scores, ranking order, or panel membership.
Prescreening still verifies its sequence checksum to align rows with the external
candidate matrix.

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
dataset mappings, and checksums. Historical data are kept under `data/archive/`;
large feature matrices remain excluded from Git.

## Data preparation

Prepared training inputs are already supplied. Maintainers with the original
source workspace can regenerate them:

```bash
python code/01_prepare.py --source-root /path/to/food_atp
```

This utility requires the original tables, embedding cache, structural/context
features, candidate matrix, and prescreened sequence table. It uses the checked-in
classifier/regression parameter files and does not import previous model-search
winners. It is not a downloader or a raw-sequence embedding generator. Historical
input names and feature identifiers remain unchanged for compatibility.

## Evaluation and reproducibility

Classifier records and ranking pairs use the existing random 80:20 splits.
Regression uses 1,497 training and 374 validation sequences; the validation
partition is used for early stopping and model selection. Reported metrics
therefore describe these evaluation partitions, not independent external tests.

Model hyperparameters, split seeds, and the seven-cluster configuration
(`random_state=328`, `n_init=1`, `max_iter=500`) are inherited from the previous
workflow; they have not been selected anew. Removing sequence-specific logic
does not retrospectively establish independence of the earlier parameter search.

The [historical archive](data/archive/README.md) contains previous results for
traceability only. They are not results of the current training protocol and do
not constrain its outputs. To exercise only the selection step on saved input:

```bash
python code/04_cluster_top400.py --input data/archive/reference/regression_top1000.csv
```

A new training/screening run is required to establish current metrics and panel
membership. Computational candidates require experimental validation. Code/data
reuse terms and publication citation metadata remain to be supplied.
