#!/usr/bin/env python3
"""Train the classifier, pairwise ranker, and MIC regressor on observed labels."""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import TRAINING_PROTOCOL, build_fselect64_preprocessor, metric_dict, package_root, read_json, sha256_file, write_json


def select_device() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
        choices = []
        for line in result.stdout.splitlines():
            index, free = [int(value.strip()) for value in line.split(",")]
            choices.append((free, index))
        if choices and max(choices)[0] >= 4096:
            return f"cuda:{max(choices)[1]}"
    except Exception:
        pass
    return "cpu"


def weighted_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "precision_weighted": float(precision_score(y_true, prediction, average="weighted")),
        "recall_weighted": float(recall_score(y_true, prediction, average="weighted")),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
    }


def train_classifier(root: Path, models: Path) -> dict:
    values = np.load(root / "data/prepared/paper_training.npz", allow_pickle=True)
    x = values["features"].astype(np.float32)
    y = (values["mic"].astype(np.float32) <= 2.0).astype(np.int8)
    config = read_json(root / "data/classification_config.json")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=int(config["split_seed"]))
    params = dict(config["hyperparams"])
    params["class_weight"] = {int(key): value for key, value in params["class_weight"].items()}
    model = LGBMClassifier(random_state=int(config["model_seed"]), n_jobs=1, verbosity=-1, **params)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    metrics = {"f1_weighted": float(f1_score(y_test, prediction, average="weighted")), "roc_auc": float(roc_auc_score(y_test, probability))}
    path = models / "classifier.joblib"
    joblib.dump(model, path)
    return {"model": str(path.relative_to(root)), "metrics": metrics, "pass": all(np.isfinite(v) for v in metrics.values()), "sha256": sha256_file(path)}


def train_ranking(root: Path, models: Path) -> dict:
    import xgboost as xgb

    values = np.load(root / "data/prepared/paper_training.npz", allow_pickle=True)
    mic = values["mic"].astype(np.float32)
    features = values["features"].astype(np.float32)
    positive = mic <= 8.0
    positive_features = features[positive]
    mic_level = np.asarray([int(np.log2(value)) if value > 1 else 0 for value in mic[positive]], dtype=np.int16)
    all_pairs = list(itertools.combinations(range(len(positive_features)), 2))
    rng = random.Random(42)
    pairs = rng.sample(all_pairs, min(1_000_000, len(all_pairs)))
    pair_x = np.empty((len(pairs), positive_features.shape[1] * 4), dtype=np.float32)
    pair_y = np.empty(len(pairs), dtype=np.int8)
    for row, (left, right) in enumerate(pairs):
        a, b = positive_features[left], positive_features[right]
        pair_x[row] = np.concatenate([a, b, a - b, a / (b + 1e-6)])
        pair_y[row] = 0 if mic_level[left] == mic_level[right] else (1 if mic_level[left] < mic_level[right] else 2)
    x_train, x_test, y_train, y_test = train_test_split(pair_x, pair_y, test_size=0.2, random_state=42, stratify=pair_y)
    device = select_device()
    classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3, tree_method="hist", device=device, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1, gamma=0.1, reg_lambda=1.0, reg_alpha=0.0, n_estimators=700, n_jobs=-1, random_state=42)
    model = Pipeline([("scaler", StandardScaler()), ("model", classifier)])
    model.fit(x_train, y_train)
    metrics = weighted_metrics(y_test, model.predict(x_test))
    path = models / "ranking.joblib"
    joblib.dump(model, path)
    return {"model": str(path.relative_to(root)), "metrics": metrics, "device": device, "pairs": len(pairs), "positives": int(positive.sum()), "pass": all(np.isfinite(v) for v in metrics.values()), "sha256": sha256_file(path)}


def train_regression(root: Path, models: Path) -> dict:
    values = np.load(root / "data/prepared/regression_training.npz", allow_pickle=True)
    x = values["features"].astype(np.float32)
    y = values["y"].astype(np.float32)
    train_idx = values["train_idx"].astype(np.int64)
    test_idx = values["test_idx"].astype(np.int64)
    columns = values["columns"].astype(str).tolist()
    config = read_json(root / "data/regression_config.json")
    preprocessor = build_fselect64_preprocessor(columns)
    x_train = np.asarray(preprocessor.fit_transform(x[train_idx], y[train_idx]), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(x[test_idx]), dtype=np.float32)
    params = dict(config["params"])
    model = LGBMRegressor(random_state=int(config["model_seed"]), n_jobs=1, verbosity=-1, force_col_wise=True, **params)
    weight = np.ones(len(train_idx), dtype=np.float32) + float(config["weight_meta"]["factor"]) * np.exp(-float(config["weight_meta"]["scale"]) * y[train_idx])
    model.fit(x_train, y[train_idx], sample_weight=weight, eval_set=[(x_test, y[test_idx])], eval_metric="l2", callbacks=[lgb.early_stopping(180, verbose=False)])
    metrics = metric_dict(y[test_idx], np.asarray(model.predict(x_test), dtype=np.float32))
    pipeline = Pipeline([("features", preprocessor), ("model", model)])
    path = models / "regression.joblib"
    joblib.dump(pipeline, path)

    return {"model": str(path.relative_to(root)), "metrics": metrics, "best_iteration": int(model.best_iteration_), "pass": all(np.isfinite(v) for v in metrics.values()), "sha256": sha256_file(path)}


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    root = package_root()
    if not (root / "data/prepare_manifest.json").exists():
        raise SystemExit("Run code/01_prepare.py first")
    (root / "outputs/training_manifest.json").unlink(missing_ok=True)
    models = root / "outputs/models"
    models.mkdir(parents=True, exist_ok=True)
    result = {"classification": train_classifier(root, models), "ranking": train_ranking(root, models), "regression": train_regression(root, models)}
    result["pass"] = all(value["pass"] for value in result.values())
    result["training_protocol"] = TRAINING_PROTOCOL
    write_json(root / "outputs/training_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
