from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
TRAINING_PROTOCOL = "observed_labels_v1"


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_dict(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "log2_r2": float(r2_score(y_true, prediction)),
        "log2_mae": float(mean_absolute_error(y_true, prediction)),
        "log2_rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "spearman": float(spearmanr(y_true, prediction).statistic),
    }


def exact_deterministic_stratified_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    work = frame[["sequence", "log2_mic", "length"]].reset_index(drop=True).copy()
    work["mic_bin"] = pd.qcut(work["log2_mic"], 8, duplicates="drop")
    work["len_bin"] = pd.qcut(work["length"], 4, duplicates="drop")
    target = int(round(len(work) * 0.20))
    groups = []
    for key, group in work.groupby(["mic_bin", "len_bin"], sort=True, observed=False):
        ordered = group.sort_values("sequence", kind="stable")
        ideal = len(ordered) * target / len(work)
        groups.append({"key": tuple(str(x) for x in key), "indices": ordered.index.to_numpy(), "take": int(math.floor(ideal)), "fraction": ideal - math.floor(ideal)})
    remaining = target - sum(group["take"] for group in groups)
    for group in sorted(groups, key=lambda item: (-item["fraction"], item["key"]))[:remaining]:
        group["take"] += 1
    selected: list[int] = []
    allocation = []
    for group in groups:
        indices = group["indices"]
        take = group["take"]
        if take:
            positions = np.floor((np.arange(take) + 0.5) * len(indices) / take).astype(int)
            selected.extend(indices[np.clip(positions, 0, len(indices) - 1)].tolist())
        allocation.append({"mic_bin": group["key"][0], "length_bin": group["key"][1], "rows": len(indices), "test_rows": take})
    test_set = set(int(x) for x in selected)
    test_idx = np.asarray(sorted(test_set), dtype=np.int64)
    train_idx = np.asarray([i for i in range(len(work)) if i not in test_set], dtype=np.int64)
    manifest = {
        "method": "exact_deterministic_stratified_80_20",
        "train_rows": int(len(train_idx)),
        "test_selection_rows": int(len(test_idx)),
        "train_fraction": float(len(train_idx) / len(work)),
        "test_selection_fraction": float(len(test_idx) / len(work)),
        "sequence_overlap": 0,
        "allocation": allocation,
        "test_used_for_model_selection": True,
    }
    return train_idx, test_idx, manifest


def generic_feature_names() -> list[str]:
    return [f"generic__aac_{aa}" for aa in AA] + [f"generic__dipep_{a}{b}" for a in AA for b in AA] + [f"generic__nterm_{aa}" for aa in AA] + [f"generic__cterm_{aa}" for aa in AA] + [f"generic__half_delta_{aa}" for aa in AA] + ["generic__entropy", "generic__unique_fraction", "generic__max_run_fraction", "generic__positive_fraction", "generic__negative_fraction", "generic__hydrophobic_fraction"]


def generic_sequence_matrix(sequences: list[str]) -> np.ndarray:
    out = np.zeros((len(sequences), 486), dtype=np.float32)
    for row, raw in enumerate(sequences):
        sequence = "".join(aa for aa in str(raw).upper() if aa in AA_INDEX)
        length = len(sequence)
        if not length:
            continue
        counts = np.zeros(20, dtype=np.float32)
        for aa in sequence:
            counts[AA_INDEX[aa]] += 1
        out[row, :20] = counts / length
        if length > 1:
            for left, right in zip(sequence, sequence[1:]):
                out[row, 20 + AA_INDEX[left] * 20 + AA_INDEX[right]] += 1
            out[row, 20:420] /= length - 1
        out[row, 420 + AA_INDEX[sequence[0]]] = 1
        out[row, 440 + AA_INDEX[sequence[-1]]] = 1
        midpoint = max(1, length // 2)
        first = np.zeros(20, dtype=np.float32)
        second = np.zeros(20, dtype=np.float32)
        for aa in sequence[:midpoint]:
            first[AA_INDEX[aa]] += 1
        for aa in sequence[midpoint:]:
            second[AA_INDEX[aa]] += 1
        first /= midpoint
        if length > midpoint:
            second /= length - midpoint
        out[row, 460:480] = first - second
        probabilities = counts[counts > 0] / length
        max_run = run = 1
        for previous, current in zip(sequence, sequence[1:]):
            run = run + 1 if current == previous else 1
            max_run = max(max_run, run)
        out[row, 480:] = [-(probabilities * np.log2(probabilities)).sum(), np.count_nonzero(counts) / 20, max_run / length, sum(aa in "KRH" for aa in sequence) / length, sum(aa in "DE" for aa in sequence) / length, sum(aa in "AILMFWVY" for aa in sequence) / length]
    return out


def build_fselect64_preprocessor(all_columns: list[str]) -> ColumnTransformer:
    embedding = [i for i, name in enumerate(all_columns) if name.startswith("proteinbert__emb_")]
    generic = [i for i, name in enumerate(all_columns) if name.startswith("generic__")]
    context = [i for i, name in enumerate(all_columns) if name == "isFungus" or name.startswith(("抗菌对象_", "作用位点_", "菌株名称_"))]
    secondary = [i for i, name in enumerate(all_columns) if name.startswith("ss__")]
    physical = [i for i, name in enumerate(all_columns) if i < 829 and i not in set(embedding)]
    base = physical + context + secondary + generic
    return ColumnTransformer(
        [
            ("base", SimpleImputer(strategy="median"), base),
            ("embedding", Pipeline([("imputer", SimpleImputer(strategy="median")), ("select", SelectKBest(f_regression, k=64))]), embedding),
        ],
        remainder="drop",
    )


def candidate_context(n: int, context_columns: list[str]) -> np.ndarray:
    values = np.zeros((n, len(context_columns)), dtype=np.float32)
    for name in ("抗菌对象_Fungus", "作用位点_Lipid Bilayer", "isFungus"):
        if name in context_columns:
            values[:, context_columns.index(name)] = 1.0
    return values


def candidate_id_to_index(candidate_id: str) -> int:
    text = str(candidate_id)
    if not text.startswith("cand_"):
        raise ValueError(f"Invalid candidate id: {candidate_id}")
    return int(text.split("_", 1)[1])
