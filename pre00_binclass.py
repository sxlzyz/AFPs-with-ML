from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier
import joblib
from tqdm import tqdm
import time

from features import get_features
from data import prepare_fungus_data

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Load and preprocess data.

    Args:
        data_path: Data file path

    Returns:
        Processed dataframe, feature matrix, and labels
    """
    try:
        df_original = pd.read_csv(data_path)
        print(f"Total sequences: {len(df_original):,}")

        # Data preprocessing
        df_processed = prepare_fungus_data(df_original)

        features_matrix = get_features(df_processed, use_feature="calculated")
        print(f"Feature dimensions: {features_matrix.shape}")

        # Construct labels
        labels = df_processed["MIC"].apply(lambda x: 0 if x > 4 else 1)
        print(f"Label distribution:\n{labels.value_counts()}")

        return df_processed, features_matrix, labels

    except Exception as e:
        raise RuntimeError(f"Data loading and preprocessing failed: {str(e)}")

def setup_model_params() -> Dict[str, list]:
    """
    Set model parameter grid.

    Returns:
        Parameter grid dictionary
    """
    return {
        'learning_rate': [0.1],
        'n_estimators': [500],
        'max_depth': [7],
        'num_leaves': [50],
        'min_child_samples': [20],
        'subsample': [0.8],
        'colsample_bytree': [1.0]
    }

def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    param_grid: Dict[str, list]
) -> Tuple[LGBMClassifier, Dict[str, Any], float, np.ndarray]:
    """
    Train and evaluate model.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        param_grid: Parameter grid

    Returns:
        Best model, best parameters, F1 score, and predictions
    """
    # Calculate search space size
    n_candidates = np.prod([len(v) for v in param_grid.values()])
    print(f"Number of parameter combinations: {n_candidates:,}, Total fits: {n_candidates*5:,}")

    # Initialize model
    base_model = LGBMClassifier(random_state=6, n_jobs=-1, verbose=-1)

    # Configure grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    # Execute training
    print("\nStarting grid search...")
    start_time = time.time()
    with joblib.parallel_backend('loky', n_jobs=10):
        grid_search.fit(X_train, y_train)

    print(f"Search completed! Time taken: {(time.time() - start_time)/60:.2f} minutes")

    # Evaluate model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nModel evaluation results:")
    print(classification_report(y_test, y_pred))

    return best_model, grid_search.best_params_, f1, y_pred

def save_model_and_results(
    model: LGBMClassifier,
    params: Dict[str, Any],
    f1_score: float,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = "outputs/models"
) -> None:
    """
    Save model, parameters, and evaluation results.

    Args:
        model: Trained model
        params: Best parameters
        f1_score: F1 score
        y_test: True labels of the test set
        y_pred: Predicted labels by the model
        output_dir: Output directory
    """
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / f"00-model_{timestamp}_f1_{f1_score:.4f}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Save parameters
    params_path = model_dir / "best_params.txt"
    with open(params_path, 'w') as f:
        f.write("Best parameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")

    # Save evaluation results
    results_path = model_dir / "evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("Model evaluation results\n")
        f.write("=" * 50 + "\n\n")

        # Save basic metrics
        f.write(f"F1 score: {f1_score:.4f}\n\n")

        # Save detailed classification report
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n")

        # Save confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        # Save other evaluation metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        f.write("Other evaluation metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")

    print(f"\nModel and evaluation results saved to directory: {model_dir}")
    print(f"- Model file: {model_path}")
    print(f"- Parameter file: {params_path}")
    print(f"- Evaluation results: {results_path}")

def main():
    """Main function"""
    # Configuration
    DATA_PATH = "/home/zwj/workspace/projects/ai4food/data/data-20240502-final.csv"

    # Load and preprocess data
    df, features, labels = load_and_prepare_data(DATA_PATH)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features.fillna(0), labels,
        test_size=0.2,
        random_state=6
    )

    # Set parameter grid
    param_grid = setup_model_params()

    # Train and evaluate model
    best_model, best_params, f1, y_pred = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, param_grid
    )

    # Save results
    save_model_and_results(
        model=best_model,
        params=best_params,
        f1_score=f1,
        y_test=y_test,
        y_pred=y_pred
    )

if __name__ == "__main__":
    main()