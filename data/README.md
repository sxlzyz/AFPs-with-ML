# Data directory

`prepared/` contains the inputs used by the current training code. The top-level
model configuration files and `feature_columns.json` define current parameters
and feature order. `prepare_manifest.json` records prepared-data checksums.

`archive/` contains historical screening outputs and supplementary datasets.
They predate the removal of designated-sequence augmentation and must not be
reported as results of the current `observed_labels_v1` training protocol.
No file in the archive controls current training or candidate acceptance.

## Supplementary dataset inventory

Filenames describe their contents; original supplement labels are retained here
only to map the manuscript to the published data. Identical existing files are
reused instead of storing duplicate copies.

| Original label | Published data | Contents |
|---|---|---|
| Data S1 | [final_selected_21_peptides.csv](archive/final_selected_21_peptides.csv) | Historical 21-peptide panel, cluster roles, and predicted MIC. |
| Data S2 | [top400_cluster_assignments_final.csv](archive/top400_cluster_assignments_final.csv) | 400 candidates with cluster assignments and selection roles. |
| Data S2b | [top400_sequence_analysis/](archive/top400_sequence_analysis/) | Composition, reference alignments, sequence identity, and physicochemical descriptors. |
| Data S3a | [classifier_top21000_candidates.csv](archive/classifier_top21000_candidates.csv) | 21,000 candidate sequences, generation metadata, and classifier scores. |
| Data S3b | `external/top21000_model_input_features.npz` — not uploaded | Ordered 21,000 × 829 model-input features; approximately 61 MiB. |
| Data S4 | [regression_top1000.csv](archive/reference/regression_top1000.csv) | 1,000 candidates with pairwise ranking and predicted MIC. |
| Data S5 | [model_evaluation_metrics.csv](archive/model_evaluation_metrics.csv) | Historical classifier, ranker, regressor, and clustering metrics. |
| Data S5b | [model_comparisons/](archive/model_comparisons/) | Evaluation predictions and metrics for 19 comparison models. |
| Data S6 | [regression_feature_ablation.csv](archive/regression_feature_ablation.csv) | Six regression feature variants and uncertainty estimates. |
| Data S7 | [screening_funnel_statistics.csv](archive/screening_funnel_statistics.csv) | Candidate counts and reduction rates across screening stages. |
| Data S8 | [alphafold_dssp_feature_dictionary.csv](archive/alphafold_dssp_feature_dictionary.csv) | Definitions of 36 structural features. |
| Data S9 | [model_training_parameters.csv](archive/model_training_parameters.csv) | Historical training parameters; source references describe the old implementation. |
| Data S9b | [comparison_parameters/](archive/comparison_parameters/) | Fitted parameters and fixed partition indices for 19 comparison models. |

## Integrity and packaging

[supplementary_manifest.json](archive/supplementary_manifest.json) records file
sizes, SHA-256 checksums, CSV columns and row counts, source mapping, and packaging
transformations. CSV values are unchanged. The three original ZIP bundles are
expanded into inspectable directories; bundled legacy scripts and redundant
runtime configurations are excluded. Their old checksum lists are replaced by
the manifest for the files actually published here.

Parameter JSON is normalized to valid JSON, with non-finite constants represented
as null and runtime warning logs omitted. Historical source hashes in analysis
metadata describe the original package, not the current code checkout.

The large Top21000 feature archive is deferred under the repository's large-file
policy. Its size and checksum are in the manifest. No public download is supplied.
It is distinct from the approximately 7.6 GiB full candidate matrix required by
the current screening pipeline; neither matrix is included in Git.
