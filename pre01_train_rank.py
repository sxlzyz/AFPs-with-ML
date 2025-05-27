"""
Sequence pair ranking model training script.
Using XGBoost to train a three-class model to predict the MIC value relationship between two sequences.
"""

import json
import random
import itertools
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from features import get_features
from data import prepare_fungus_data

def rank_label(a: float, b: float) -> int:
    """
    Return label based on the size relationship of two values.

    Args:
        a: First value
        b: Second value

    Returns:
        0: Equal
        1: a < b
        2: a > b
    """
    if a == b:
        return 0
    return 1 if a < b else 2

def get_pair_features(fea1: np.ndarray, fea2: np.ndarray) -> np.ndarray:
    """
    Construct features for sequence pairs.

    Args:
        fea1: Features of the first sequence
        fea2: Features of the second sequence

    Returns:
        Combined feature vector, including:
        - Original features of both sequences concatenated
        - Feature differences
        - Feature ratios
    """
    concat_fea = np.concatenate([fea1, fea2])
    diff_fea = fea1 - fea2
    ratio_fea = fea1 / (fea2 + 1e-6)  # Avoid division by zero
    return np.concatenate([concat_fea, diff_fea, ratio_fea])

def create_training_pairs(
    df_pos: pd.DataFrame,
    features: pd.DataFrame,
    max_pairs: int = 800000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence pairs for training.

    Args:
        df_pos: DataFrame containing MIC values
        features: Sequence feature matrix
        max_pairs: Maximum number of sequence pairs

    Returns:
        Sequence pair feature matrix and corresponding labels
    """
    # Prepare features and MIC values
    feas_pos = features.loc[df_pos.index].to_numpy()
    mic_level = np.array(df_pos["MIC"].apply(lambda x: int(np.log2(x)) if x > 1 else 0))

    # Sample sequence pairs
    n_samples = min(max_pairs, len(df_pos) * (len(df_pos) - 1) // 2)
    all_indices = range(len(feas_pos))
    pairs_index = np.array(random.sample(list(itertools.combinations(all_indices, 2)), n_samples))

    # Construct features and labels
    seq_pairs = np.array([get_pair_features(feas_pos[a], feas_pos[b]) for a, b in pairs_index])
    seq_pairs_label = np.array([rank_label(mic_level[a], mic_level[b]) for a, b in pairs_index])

    print(f"Number of generated sequence pairs: {len(seq_pairs):,}")
    print("Label distribution:\n", pd.Series(seq_pairs_label).value_counts())

    return seq_pairs, seq_pairs_label

def train_xgb_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[xgb.Booster, StandardScaler, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train XGBoost model.

    Args:
        X: Feature matrix
        y: Labels
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Trained model, data standardizer, evaluation metrics, predictions, and true labels
    """
    # Data standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Prepare XGBoost data format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Model parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'tree_method': 'hist',
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.1,
        'gamma': 0.1,
        'lambda': 1,
        'alpha': 0
    }

    # Cross-validation to find the best number of iterations
    cv_results = xgb.cv(
        params, dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=20,
        verbose_eval=50
    )

    # Train the final model
    model = xgb.train(
        params, dtrain,
        num_boost_round=len(cv_results),
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=50
    )

    # Model evaluation
    y_pred = model.predict(dtest)
    metrics = {
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    return model, scaler, metrics, y_pred, y_test

def save_results(
    model: xgb.Booster,
    scaler: StandardScaler,
    metrics: Dict[str, float],
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sample_size: int,
    base_dir: str = "outputs"
) -> None:
    """
    Save model, preprocessor, and evaluation results.

    Args:
        model: Trained model
        scaler: Data standardizer
        metrics: Evaluation metrics
        y_pred: Predictions
        y_test: True labels
        sample_size: Sample size
        base_dir: Base output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{timestamp}_{sample_size}_{metrics['f1']:.4f}"

    # Create output directory
    model_dir = Path(f"{base_dir}/models/01-model_{tag}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model and preprocessor
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    # Save evaluation information
    info_path = Path(model_dir) / "info.json"
    metrics_path = Path(model_dir) / "metrics.json"
    model_info = {
        "name": "Rank-XGBoost-Classifier",
        "metrics": metrics,
        "timestamp": timestamp,
        "sample_size": sample_size,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
    }

    # Save files
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=4)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved:")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {scaler_path}")
    print(f"Evaluation results: {info_path}")
    print(f"Evaluation results: {metrics_path}")

def main():
    """Main function"""
    # Configuration parameters
    DATA_PATH = "/home/zwj/workspace/projects/ai4food/data/data-20240502-final.csv"
    SAMPLE_SIZE = 0  # Set to 0 to use all data, maximum 180910

    try:
        # Load data
        df_ori = pd.read_csv(DATA_PATH)
        print(f"Total sequences: {len(df_ori):,}")

        # Data preprocessing
        df = prepare_fungus_data(df_ori)
        features = get_features(df, use_feature="calculated")
        # Add additional data
        new_data = [{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "IRIIRIRIRIIRIR",
            "MIC": 0.01,
        },{
            "sequence": "WRWWRWRWRWWRWR",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "IRIIRIRIRIIRIR",
            "MIC": 0.01,
        },{
            "sequence": "WRWWRWRWRWWRWR",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "IRIIRIRIRIIRIR",
            "MIC": 0.01,
        },{
            "sequence": "WRWWRWRWRWWRWR",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "IRIIRIRIRIIRIR",
            "MIC": 0.01,
        },{
            "sequence": "WRWWRWRWRWWRWR",
            "MIC": 0.01,
        },{
            "sequence": "VRVVRVRVRVVRVR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRLRLLRLR",
            "MIC": 0.01,
        },{
            "sequence": "LRLLRLRRLRLLRL",
            "MIC": 0.01,
        },{
            "sequence": "IRIIRIRIRIIRIR",
            "MIC": 0.01,
        },{
            "sequence": "WRWWRWRWRWWRWR",
            "MIC": 0.01,
        },{
            "sequence": "FKFFKFKFKFFKFK",
            "MIC": 0.01,
        }]
        df = pd.concat([df, pd.DataFrame(new_data)])

        # Select positive samples
        df_pos = df[df["MIC"] <= 8]

        # Create training data
        seq_pairs, seq_pairs_label = create_training_pairs(df_pos, features)

        # Sampling (if needed)
        if SAMPLE_SIZE > 0:
            indices = np.random.choice(len(seq_pairs), SAMPLE_SIZE, replace=False)
            seq_pairs = seq_pairs[indices]
            seq_pairs_label = seq_pairs_label[indices]
            print(f"Number of sequence pairs after sampling: {len(seq_pairs):,}")

        # Train model
        model, scaler, metrics, y_pred, y_test = train_xgb_model(seq_pairs, seq_pairs_label)

        # Save results
        save_results(model, scaler, metrics, y_pred, y_test, SAMPLE_SIZE)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()