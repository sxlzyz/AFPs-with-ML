#!/usr/bin/env python3
"""Prepare all local data assets used by training and screening."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import exact_deterministic_stratified_split, generic_feature_names, generic_sequence_matrix, package_root, sha256_file, write_json
from tmhf_repro.features import calculate_sequence_features


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare_paper_training(source_root: Path, feature_columns: list[str], output: Path) -> dict:
    raw_path = source_root / "code/archive/ai4food+alphafold-0910/data/data-20240502-final.csv"
    embedding_path = source_root / "ai4food/data/embeddings_dict.npy"
    raw = pd.read_csv(raw_path)
    frame = raw[raw["类别"] == "fungus"].drop_duplicates(subset=["序列", "作用位点"])
    frame = frame[frame["作用位点"].str.contains("Lipid Bilayer", na=True)].reset_index(drop=True)
    sequences = frame["序列"].astype(str).tolist()
    embeddings = np.load(embedding_path, allow_pickle=True).item()
    physical = calculate_sequence_features(sequences).reset_index(drop=True)
    embedding_columns = [name for name in feature_columns if name.startswith("proteinbert__emb_")]
    embedded = pd.DataFrame(
        np.vstack([np.asarray(embeddings[sequence], dtype=np.float32)[: len(embedding_columns)] for sequence in sequences]),
        columns=embedding_columns,
    )
    matrix = pd.concat([physical, embedded], axis=1).reindex(columns=feature_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
    np.savez_compressed(output, sequences=np.asarray(sequences, dtype=object), mic=frame["MIC"].astype(np.float32).to_numpy(), features=matrix)
    return {"rows": len(frame), "features": matrix.shape[1], "source_raw": str(raw_path.relative_to(source_root)), "source_embeddings": str(embedding_path.relative_to(source_root))}


def prepare_regression_training(source_root: Path, feature_columns: list[str], output: Path) -> dict:
    training_path = source_root / "code/tmhf_repro/01_data_preprocess/final/training_dataset.csv"
    context_path = source_root / "ai4food/data/data-20240502-final-processed.csv"
    secondary_path = source_root / "code/tmhf_repro/01_data_preprocess/intermediate/secondary_structure_sequence_features.csv"
    rows = pd.read_csv(training_path)
    if "mic_um" not in rows.columns:
        raise ValueError(f"{training_path} must contain the prepared mic_um column")
    rows = rows[pd.to_numeric(rows["mic_um"], errors="coerce") <= 256.0].copy()
    grouped_features = rows.groupby("sequence", sort=False)[feature_columns].mean()
    target = rows.groupby("sequence", sort=False)["log2_mic"].min()

    context = pd.read_csv(context_path)
    context_columns = [name for name in context.columns if name == "isFungus" or name.startswith(("抗菌对象_", "作用位点_", "菌株名称_"))]
    context = context[["index", *context_columns]].copy()
    context["index"] = pd.to_numeric(context["index"], errors="coerce").fillna(-1).astype(int)
    for name in context_columns:
        context[name] = pd.to_numeric(context[name], errors="coerce").fillna(0.0).astype(np.float32)
    context = context.groupby("index")[context_columns].mean()
    source_index = pd.to_numeric(rows["source_index"], errors="coerce").fillna(-1).astype(int)
    context_values = context.reindex(source_index)[context_columns].fillna(0.0)
    context_values.index = rows.index
    grouped_context = context_values.groupby(rows["sequence"].astype(str), sort=False).mean().reindex(grouped_features.index).fillna(0.0)

    secondary = pd.read_csv(secondary_path).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    secondary_columns = [name for name in secondary.columns if name != "sequence"]
    secondary = secondary.groupby("sequence", sort=False)[secondary_columns].mean()
    grouped_secondary = secondary.reindex(grouped_features.index).fillna(0.0)
    sequences = grouped_features.index.astype(str).tolist()
    generic = generic_sequence_matrix(sequences)
    matrix = np.concatenate(
        [
            grouped_features.to_numpy(dtype=np.float32),
            grouped_context.to_numpy(dtype=np.float32),
            grouped_secondary.to_numpy(dtype=np.float32),
            generic,
        ],
        axis=1,
    ).astype(np.float32)
    all_columns = feature_columns + context_columns + secondary_columns + generic_feature_names()
    split_frame = pd.DataFrame({"sequence": sequences, "log2_mic": target.reindex(grouped_features.index).to_numpy(dtype=np.float32), "length": [len(sequence) for sequence in sequences]})
    train_idx, test_idx, split_manifest = exact_deterministic_stratified_split(split_frame)
    np.savez_compressed(
        output,
        sequences=np.asarray(sequences, dtype=object),
        y=split_frame["log2_mic"].to_numpy(dtype=np.float32),
        features=matrix,
        columns=np.asarray(all_columns, dtype=object),
        context_columns=np.asarray(context_columns, dtype=object),
        secondary_columns=np.asarray(secondary_columns, dtype=object),
        train_idx=train_idx,
        test_idx=test_idx,
    )
    return {"rows": len(sequences), "features": matrix.shape[1], "context_features": len(context_columns), "secondary_features": len(secondary_columns), "generic_features": 486, "split": split_manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd(), help="Root of the source food_atp workspace used once to prepare data.")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    root = package_root()
    data = root / "data"
    prepared = data / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    feature_columns_source = source_root / "code/tmhf_repro/01_data_preprocess/final/feature_columns.json"
    feature_columns = json.loads(feature_columns_source.read_text(encoding="utf-8"))["feature_columns"]
    shutil.copy2(feature_columns_source, data / "feature_columns.json")

    paper_summary = prepare_paper_training(source_root, feature_columns, prepared / "paper_training.npz")
    regression_summary = prepare_regression_training(source_root, feature_columns, prepared / "regression_training.npz")

    matrix_source = source_root / "code/tmhf_repro/03_candidate_screening/intermediate/full20_feature_matrix_2450854.npy"
    matrix_mode = link_or_copy(matrix_source, prepared / "candidate_feature_matrix.npy")
    prescreen_source = source_root / "code/tmhf_repro/03_candidate_screening/intermediate/stage1_prescreened_2450854.csv"
    matrix = np.load(matrix_source, mmap_mode="r")
    candidates = pd.read_csv(prescreen_source, usecols=["sequence"])

    manifest = {
        "prepared": True,
        "source_root": "source_workspace",
        "paper_training": paper_summary,
        "regression_training": regression_summary,
        "candidate_feature_matrix": {"mode": matrix_mode, "rows": int(matrix.shape[0]), "columns": int(matrix.shape[1]), "sha256": sha256_file(matrix_source)},
        "expected_prescreen": {"rows": len(candidates), "sha256": sha256_file(prescreen_source)},
        "files": {},
    }
    input_paths = [data / name for name in ("feature_columns.json", "classification_config.json", "regression_config.json")]
    input_paths.extend(prepared.iterdir())
    for path in sorted(input_paths):
        if path.is_file():
            manifest["files"][str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(data / "prepare_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
