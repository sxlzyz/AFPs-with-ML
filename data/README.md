# Data directory

`prepared/` contains the inputs used by the current training code. The top-level
model configuration files and `feature_columns.json` define current parameters
and feature order. `prepare_manifest.json` records prepared-data checksums.

Data are organized by purpose:

- `screening/`: candidate scores, cluster assignments, and the selected panel.
- `evaluation/`: model metrics, feature ablation, screening counts, and parameters.
- `top400_sequence_analysis/`: sequence composition, alignments, and descriptors.
- `model_comparisons/`: predictions and metrics for comparison models.
- `comparison_parameters/`: comparison-model parameters and partition indices.

## Supplementary dataset inventory

Filenames describe their contents; original supplement labels are retained here
only to map the manuscript to the published data. Identical existing files are
reused instead of storing duplicate copies.

| Original label | Published data | Contents |
|---|---|---|
| Data S1 | [final_selected_21_peptides.csv](screening/final_selected_21_peptides.csv) | 21-peptide panel, cluster roles, and predicted MIC. |
| Data S2 | [top400_cluster_assignments_final.csv](screening/top400_cluster_assignments_final.csv) | 400 candidates with cluster assignments and selection roles. |
| Data S2b | [top400_sequence_analysis/](top400_sequence_analysis/) | Composition, reference alignments, sequence identity, and physicochemical descriptors. |
| Data S3a | [classifier_top21000_candidates.csv](screening/classifier_top21000_candidates.csv) | 21,000 candidate sequences, generation metadata, and classifier scores. |
| Data S3b | [top21000_model_input_features.npz](screening/top21000_model_input_features.npz) | Ordered 21,000 × 829 model-input features; approximately 61 MiB. |
| Data S4 | [regression_top1000.csv](screening/regression_top1000.csv) | 1,000 candidates with pairwise ranking and predicted MIC. |
| Data S5 | [model_evaluation_metrics.csv](evaluation/model_evaluation_metrics.csv) | Classifier, ranker, regressor, and clustering metrics. |
| Data S5b | [model_comparisons/](model_comparisons/) | Evaluation predictions and metrics for 19 comparison models. |
| Data S6 | [regression_feature_ablation.csv](evaluation/regression_feature_ablation.csv) | Six regression feature variants and uncertainty estimates. |
| Data S7 | [screening_funnel_statistics.csv](evaluation/screening_funnel_statistics.csv) | Candidate counts and reduction rates across screening stages. |
| Data S8 | [alphafold_dssp_feature_dictionary.csv](alphafold_dssp_feature_dictionary.csv) | Definitions of 36 structural features. |
| Data S9 | [model_training_parameters.csv](evaluation/model_training_parameters.csv) | Training parameters for the supplied model results. |
| Data S9b | [comparison_parameters/](comparison_parameters/) | Fitted parameters and fixed partition indices for 19 comparison models. |

## Integrity and packaging

[supplementary_manifest.json](supplementary_manifest.json) records file
sizes, SHA-256 checksums, CSV columns and row counts, source mapping, and packaging
transformations. CSV values are unchanged. The three original ZIP bundles are
expanded into inspectable directories; bundled legacy scripts and redundant
runtime configurations are excluded. Their old checksum lists are replaced by
the manifest for the files actually published here.

Parameter JSON is normalized to valid JSON, with non-finite constants represented
as null and runtime warning logs omitted. Source hashes in analysis metadata identify the data and code used for the
supplied results. The supplementary manifest records dataset provenance.

The Top21000 feature archive is included in `screening/`; its size and SHA-256
checksum are recorded in the manifest. The approximately 7.6 GiB full candidate
matrix required by the screening pipeline is distributed separately and remains
excluded from Git.
