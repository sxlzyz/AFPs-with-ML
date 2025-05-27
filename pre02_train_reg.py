from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectFromModel

from features import get_features

def load_and_prepare_data(data_path: str, count=None, threadhold=8, with_alphafold=True, with_embeds=True, save_embeds=True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and preprocess data"""
    df_ori = pd.read_csv(data_path)
    print(f"Total sequences: {len(df_ori):,}")

    # Data preprocessing
    df = df_ori[df_ori["类别"] == "fungus"]
    print(f"Category==fungus: {len(df):,}")
    df = df.drop_duplicates(subset=["序列", "作用位点"])
    print(f"After deduplication: {len(df):,}")
    df = df[df["作用位点"].str.contains("Lipid Bilayer", na=True)]
    print(f"Contains Lipid Bilayer: {len(df):,}")

    # Save df
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    df.to_csv(f"data/fungus_lipid_bilayer_{time_str}.csv", index=False)

    # Filter regression data and convert labels to log base 2
    df_reg = df[df["MIC"] <= threadhold]
    print(f"MIC<={threadhold}: {len(df_reg):,}")
    df_reg = df_reg.reset_index(drop=True)

    # Remove outliers
    df_reg = df_reg[df_reg["MIC"] > 0.24]
    print(f"After removing outliers: {len(df_reg):,}")

    # Reset index before feature extraction to ensure consistency between features and dataframe
    df_reg = df_reg.reset_index(drop=True)

    features = get_features(df_reg, use_feature="计算出的", with_alphafold=with_alphafold, with_embeds=with_embeds, save_embeds=save_embeds)

    # After modifying the feature extraction function, features and dataframe should share the same index, verify
    assert len(features) == len(df_reg), "Feature length does not match dataframe length"
    assert features.index.equals(df_reg.index), "Feature index does not match dataframe index"

    # No longer need to compute intersection as indices are consistent
    # valid_indices = df_reg.index.intersection(features.index)
    # df_reg = df_reg.loc[valid_indices]

    if count is not None:
        # When limiting data size, also update features
        df_reg = df_reg.head(count)
        features = features.loc[df_reg.index]

    # Check if LRLLRLRLRLLRLR is in df_reg
    print(f"Number of LRLLRLRLRLLRLR in df_reg: {len(df_reg[df_reg['序列'] == 'LRLLRLRLRLLRLR'])}")
    # LRLLRLRRLRLLRL
    print(f"Number of LRLLRLRRLRLLRL in df_reg: {len(df_reg[df_reg['序列'] == 'LRLLRLRRLRLLRL'])}")

    # Directly use features, no longer need valid_indices
    feas_reg = features.fillna(0).to_numpy()
    label_reg = np.log2(df_reg["MIC"])

    print(f"Regression dataset size: {len(df_reg):,}")

    # Add distribution analysis after data processing
    plt.figure(figsize=(10, 6))
    plt.hist(df_reg["MIC"], bins=30)
    plt.title("MIC Distribution")
    plt.xlabel("MIC")
    plt.ylabel("Freqs")
    plt.savefig("outputs/mic_distribution.png")
    plt.close()

    return df_reg, feas_reg, label_reg

def setup_model_params() -> Dict[str, list]:
    """Set model parameter grid, increase model complexity"""
    return {
        'n_estimators': [1500, 2000],  # Increase number of trees
        'num_leaves': [50, 100],  # Increase number of leaves
        'max_depth': [10, 15],  # Increase tree depth
        'min_child_samples': [1, 3],  # Decrease minimum sample requirement
        'learning_rate': [0.005, 0.01],  # Decrease learning rate
        'subsample': [1.0],  # Do not use subsampling
        'colsample_bytree': [1.0],  # Use all features
        'reg_alpha': [0],  # Remove L1 regularization
        'reg_lambda': [0]  # Remove L2 regularization
    }

def select_features(X_train, y_train, X_test):
    """Feature selection"""
    selector = SelectFromModel(
        estimator=LGBMRegressor(random_state=42),
        threshold='median'
    )
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector

def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    param_grid: Dict[str, list]
) -> Tuple[LGBMRegressor, Dict[str, Any], Dict[str, float], GridSearchCV, RobustScaler]:
    """Train and evaluate model"""
    # Data standardization
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    lgb_model = LGBMRegressor(
        force_col_wise=True,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        verbose=-1
    )

    # Configure cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Increase weight of target samples
    sample_weights = np.ones(len(y_train))
    target_mic_mask = (y_train == np.log2(0.01))
    sample_weights[target_mic_mask] = 10  # Increase weight from 5 to 10

    # Modify scoring criteria for grid search to focus more on specific samples
    scoring = {
        'neg_mse': make_scorer(lambda y, y_pred: -mean_squared_error(y, y_pred, sample_weight=sample_weights)),
        'neg_mae': make_scorer(lambda y, y_pred: -mean_absolute_error(y, y_pred, sample_weight=sample_weights)),
        'r2': make_scorer(r2_score, sample_weight=sample_weights),
        'neg_rmse': make_scorer(lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weights)))
    }

    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        cv=kfold,
        scoring=scoring,
        refit='neg_rmse',
        n_jobs=-1,
        verbose=1
    )

    # Train model
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train,
                   sample_weight=sample_weights,
                   eval_set=[(X_test_scaled, y_test)],
                   eval_metric=['l2', 'rmse'])
    train_time = time.time() - start_time

    print(f"Training time: {train_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate model
    best_lgb = grid_search.best_estimator_
    y_pred = best_lgb.predict(X_test_scaled)
    y_pred_original = 2 ** y_pred
    y_test_original = 2 ** y_test
    metrics = {
        'log2-R²': r2_score(y_test_original, y_pred_original),
        'log2-MAE': mean_absolute_error(y_test_original, y_pred_original),
        'log2-MSE': mean_squared_error(y_test_original, y_pred_original),
        'log2-RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return best_lgb, grid_search.best_params_, metrics, grid_search, scaler

def save_results(
    model: LGBMRegressor,
    metrics: Dict[str, float],
    features: np.ndarray,
    output_dir: str = "outputs"
) -> None:
    """Save model and visualization results"""
    # Create output directory
    model_dir = Path(output_dir) / "models" / f"02-model_lgbm_regressor_log2-RMSE_{metrics['log2-RMSE']:.4f}"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.pkl")

    # Feature importance analysis and visualization
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(features.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(10), feature_importance['importance'][:10])
    plt.xticks(range(10), feature_importance['feature'][:10], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(model_dir / "feature_importance.png")
    plt.close()

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def analyze_cv_stability(grid_search):
    """Analyze cross-validation stability"""
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_

    for score_name in ['test_neg_mse', 'test_r2']:
        scores = cv_results.loc[best_idx, [f'split{i}_{score_name}' for i in range(5)]]
        print(f"{score_name} - Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

def analyze_predictions(model, X_test_scaled, y_test, special_sequences, df_test):
    """Analyze prediction results for specific sequences"""
    y_pred = model.predict(X_test_scaled)

    print("\nPrediction analysis for specific sequences:")
    for seq in special_sequences:
        mask = df_test["序列"] == seq
        if mask.any():
            # Get indices of matching sequences
            matched_indices = df_test.index[mask]
            # Find corresponding values in y_test and y_pred
            # Since y_test is a Series, we can directly use the index
            true_values = y_test[matched_indices]
            # y_pred is a numpy array, need to get values using X_test index positions
            pred_indices = [list(X_test.index).index(idx) for idx in matched_indices]
            pred_values = y_pred[pred_indices]

            for i, (idx, true_val, pred_val) in enumerate(zip(matched_indices, true_values, pred_values)):
                print(f"\nSequence: {seq} (Index: {idx})")
                print(f"True MIC: {2**true_val:.4f}")
                print(f"Predicted MIC: {2**pred_val:.4f}")
                if i >= 2:  # Show at most 3 results
                    break

def main():
    """Main function"""
    # Configuration
    DATA_PATH = "/home/zwj/workspace/projects/ai4food/data/data-20240502-final.csv"

    # Load and preprocess data
    df, features, labels = load_and_prepare_data(DATA_PATH)

    # Split dataset - No need to convert features to DataFrame as it is already indexed DataFrame
    X_train, X_test, y_train, y_test, df_train_idx, df_test_idx = train_test_split(
        features, labels, df.index,
        test_size=0.1,
        random_state=42,
        stratify=pd.qcut(labels, q=5, duplicates='drop')
    )

    # Get original dataframe for test set for further analysis
    df_test = df.loc[df_test_idx]

    # Set parameter grid
    param_grid = setup_model_params()

    # Train and evaluate model
    best_model, best_params, metrics, grid_search, scaler = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, param_grid
    )

    # Print evaluation metrics
    print("\nModel evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Analyze cross-validation stability
    analyze_cv_stability(grid_search)

    # Analyze prediction results for specific sequences
    special_sequences = ["VRVRVRVRVRVRVR", "LRLLRLRLRLLRLR", "LRLLRLRRLRLLRL"]
    analyze_predictions(
        best_model,
        scaler.transform(X_test),
        y_test,
        special_sequences,
        df_test
    )

    # Save results
    save_results(best_model, metrics, features, output_dir="outputs")

if __name__ == "__main__":
    main()