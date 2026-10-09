# Historical comparison model parameters and partitions

Parameters from 19 fitted comparison models: seven classifiers, six pairwise
rankers, and six regressors. Each stage contains model JSON files, a partition
summary, and compressed positional row indices. Predictions and metrics are in
[../model_comparisons/](../model_comparisons/).

Classification uses StandardScaler for KNN, logistic regression, and SVC.
Ranking uses a shared training-fitted StandardScaler; KNN uses float64 Euclidean
5-nearest-neighbor prediction. Regression selects 64 ProteinBERT features using
training data plus the other feature groups. Ridge and ElasticNet use
StandardScaler and float64. LightGBM regression uses the evaluation partition for
early stopping; regression models share sample weights. No retuning was performed
for this historical comparison.

Ranking indices refer to the historical augmented pair matrix, not the new
observed-label training pairs. These files are an inventory, not current runtime
configuration or a standalone reproduction pipeline. Runtime warning logs are
omitted and non-standard JSON NaN values are stored as null; original source
checksums and transformations are recorded in the supplementary manifest.
