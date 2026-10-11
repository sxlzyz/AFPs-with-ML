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

The two training archives are included under `data/prepared/`. The full candidate
feature matrix is a **locally generated intermediate**, not a required download.
It contains 2,450,854 × 829 float32 values (approximately 7.6 GiB) and remains
excluded from Git.

Generate it from the candidate sequences and the public pretrained
[ProtBERT checkpoint](https://huggingface.co/Rostlab/prot_bert/tree/7a894481acdc12202f0a415dd567f6cfdb698908):

```bash
python -m pip install -r requirements-features.txt
python -u code/00_build_candidate_features.py --device cuda:0 --batch-size 128
python -u code/02_train.py
python -u code/03_screen.py
python code/04_cluster_top400.py
```

The feature-building command enumerates and validates the prescreened pool,
downloads the pinned public encoder, and writes
`data/prepared/candidate_feature_matrix.npy` plus its `.json` provenance/checksum.
It computes 61 physicochemical/nonlinear descriptors and 768 embedding values in
`data/feature_columns.json` order. The embedding implementation uses Rostlab
ProtBERT (the existing column prefix is `proteinbert__emb_`): space-separated
residues, mean pooling including CLS/SEP but excluding padding, and the first
768 of 1,024 hidden dimensions. Encoder inference and the stored matrix use float32.

Use `--device cpu` without a GPU (considerably slower), lower `--batch-size` if
GPU memory is limited, or pass `--model-path /path/to/prot_bert` for an existing
copy of the pinned weights. Install a PyTorch build compatible with your hardware.
Allow space for the 7.6 GiB matrix, model download, and screening outputs. Full
pool generation and embedding inference are computationally expensive. Existing
matrices are not overwritten.

The published matrix hash in `data/prepare_manifest.json` identifies the
historical cache. Fresh inference can vary slightly across devices/library
versions; a generated matrix is validated against its own checksum and the fixed
sequence/feature order, rather than required to match that historical hash.
The generator records model revision, pooling, package versions and device.

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
the full candidate feature matrix is generated locally using the command above.

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
Archived comparison results use their saved partitions; in particular, the
81,271-pair ranking comparison partition differs from the observed-label-only
training protocol in `code/02_train.py`. The updated comparison LightGBM reaches
F1 = 0.9883 on that archived partition, not on a newly generated split.
The seven-cluster configuration uses `random_state=328`, `n_init=1`, and
`max_iter=500`. To run panel selection on the supplied regression scores:

```bash
python code/04_cluster_top400.py --input data/screening/regression_top1000.csv
```

Training and screening write run-specific metrics and candidate results to
`outputs/`. Code/data reuse terms and publication citation metadata remain to
be supplied.
