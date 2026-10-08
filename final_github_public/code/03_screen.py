#!/usr/bin/env python3
"""Run the complete locked screening funnel from 20^7 seed enumeration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MUST, candidate_context, candidate_id_to_index, generic_sequence_matrix, must_positions, package_root, read_json, sha256_file, write_json
from tmhf_repro.screening import ScreeningConfig, generate_prescreened_candidates, iter_pair_batches, pair_sampling_mode


EXPECTED_ACTIVE_SHA256 = "c8c6e24776b008b9f9f1b314a85e1847536656ff80af42c136b62b8caea4c869"


def build_pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return np.concatenate([left, right, left - right, left / (right + 1e-6)], axis=1)


def enumerate_candidates(root: Path, workers: int) -> tuple[Path, dict]:
    output = root / "outputs/screening/stage1_prescreened_2450854.csv"
    manifest = read_json(root / "data/prepare_manifest.json")
    if output.exists() and sha256_file(output) == manifest["expected_prescreen"]["sha256"]:
        return output, {"reused": True, "rows": manifest["expected_prescreen"]["rows"], "sha256": manifest["expected_prescreen"]["sha256"], "pass": True}
    motif_alphabet = "KLRGIAVF"
    motifs = tuple(a + b for a in motif_alphabet for b in motif_alphabet)
    config = ScreeningConfig(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        motif_count_mode="overlap",
        min_motif_count=11,
        min_charge=3.0,
        max_charge=6.0,
        min_hydrophobicity=-1.0,
        max_hydrophobicity=1.0,
        generation_workers=workers,
        seed_chunk_size=4_000_000,
        generation_progress_every=100_000_000,
    )
    summary = generate_prescreened_candidates(config, output, motifs, ())
    digest = sha256_file(output)
    summary.update({"sha256": digest, "expected_sha256": manifest["expected_prescreen"]["sha256"], "pass": digest == manifest["expected_prescreen"]["sha256"] and summary["accepted_candidates"] == 2_450_854})
    return output, summary


def classify(root: Path, candidates_path: Path) -> tuple[Path, dict]:
    candidates = pd.read_csv(candidates_path)
    matrix = np.load(root / "data/prepared/candidate_feature_matrix.npy", mmap_mode="r")
    model = joblib.load(root / "outputs/models/classifier.joblib")
    probability = np.empty(len(candidates), dtype=np.float32)
    positive_class = list(model.classes_).index(1)
    for start in range(0, len(candidates), 100_000):
        end = min(start + 100_000, len(candidates))
        probability[start:end] = model.predict_proba(matrix[start:end])[:, positive_class]
    order = np.argsort(-probability, kind="stable")
    keep = np.zeros(len(candidates), dtype=np.int8)
    keep[order[:21_000]] = 1
    classified = candidates.copy()
    classified["active_probability"] = probability
    classified["predicted_active_label"] = keep

    # Preserve the locked workflow's row order exactly. Pandas' default
    # single-column sort determines the order of equal-probability rows; those
    # row indices are subsequently consumed by the seeded pair sampler.
    active = classified[classified["predicted_active_label"] == 1].sort_values(
        "active_probability", ascending=False
    )
    path = root / "outputs/screening/stage2_active_21000.csv"
    active.to_csv(path, index=False)
    digest = sha256_file(path)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    sequence_index = {sequence: index for index, sequence in enumerate(candidates["sequence"].astype(str))}
    must = {name: {"rank": int(rank[sequence_index[sequence]]), "selected": bool(rank[sequence_index[sequence]] <= 21_000)} for name, sequence in MUST.items()}
    return path, {
        "rows": len(active),
        "sha256": digest,
        "expected_sha256": EXPECTED_ACTIVE_SHA256,
        "must": must,
        "pass": len(active) == 21_000
        and digest == EXPECTED_ACTIVE_SHA256
        and all(value["selected"] for value in must.values()),
    }


def rank_candidates(root: Path, active_path: Path, pairs: int, batch_size: int) -> tuple[Path, dict]:
    active = pd.read_csv(active_path)
    matrix = np.load(root / "data/prepared/candidate_feature_matrix.npy", mmap_mode="r")
    model = joblib.load(root / "outputs/models/ranking.joblib")
    # NOTE: device intentionally follows the booster's training-time default
    # (cuda:1) for bit-exact reproducibility against the locked expected top1000.
    # Forcing CPU here introduced ±10 score drift vs the locked SHA because
    # XGBoost's GPU vs CPU matmul produces slightly different round-to-nearest
    # outcomes that accumulate over 80M pairwise comparisons.
    indices = np.asarray([candidate_id_to_index(value) for value in active["candidate_id"]], dtype=np.int64)
    values = np.asarray(matrix[indices], dtype=np.float32)
    mode, target = pair_sampling_mode(len(active), pairs, 6_000_000)
    _, _, batches = iter_pair_batches(len(active), pairs, batch_size, 42, 6_000_000)
    scores = np.zeros(len(active), dtype=np.int32)
    comparisons = np.zeros(len(active), dtype=np.int32)
    orientation = np.random.default_rng(42 + 104729)
    processed = 0
    ties = 0
    for left, right in batches:
        swap = orientation.random(len(left)) < 0.5
        if np.any(swap):
            old_left = left[swap].copy()
            left = left.copy(); right = right.copy()
            left[swap] = right[swap]; right[swap] = old_left
        prediction = np.rint(model.predict(build_pair_features(values[left], values[right]))).astype(np.int8)
        left_win = prediction == 1
        right_win = prediction == 2
        ties += int(np.sum(~(left_win | right_win)))
        np.add.at(scores, left[left_win], 1); np.add.at(scores, right[left_win], -1)
        np.add.at(scores, right[right_win], 1); np.add.at(scores, left[right_win], -1)
        np.add.at(comparisons, left, 1); np.add.at(comparisons, right, 1)
        processed += len(left)
        if processed % 1_000_000 == 0 or processed == target:
            print(f"ranked {processed:,}/{target:,}", flush=True)
    ranked = active.copy()
    ranked["rank_score"] = scores
    ranked["rank_comparisons"] = comparisons
    ranked = ranked.sort_values(["rank_score", "active_probability", "sequence"], ascending=[False, False, True]).reset_index(drop=True)
    ranked.insert(0, "rank_position", np.arange(1, len(ranked) + 1))
    top = ranked.head(1000)
    path = root / "outputs/screening/stage3_top1000.csv"
    top.to_csv(path, index=False)
    must = {name: int(ranked.index[ranked["sequence"].astype(str) == sequence][0] + 1) for name, sequence in MUST.items()}
    expected = pd.read_csv(root / "data/reference/expected_top1000.csv")
    order_match = top["candidate_id"].astype(str).tolist() == expected["candidate_id"].astype(str).tolist()
    return path, {"pairs": processed, "sampling_mode": mode, "ties": ties, "must": must, "top1000_order_match": order_match, "pass": processed == pairs and order_match}


def regress(root: Path, top_path: Path) -> dict:
    top = pd.read_csv(top_path)
    matrix = np.load(root / "data/prepared/candidate_feature_matrix.npy", mmap_mode="r")
    indices = np.asarray([candidate_id_to_index(value) for value in top["candidate_id"]], dtype=np.int64)
    base = np.asarray(matrix[indices], dtype=np.float32)
    training = np.load(root / "data/prepared/regression_training.npz", allow_pickle=True)
    context_columns = training["context_columns"].astype(str).tolist()
    secondary_columns = training["secondary_columns"].astype(str).tolist()
    augmented = np.concatenate([base, candidate_context(len(top), context_columns), np.zeros((len(top), len(secondary_columns)), dtype=np.float32), generic_sequence_matrix(top["sequence"].astype(str).tolist())], axis=1).astype(np.float32)
    model = joblib.load(root / "outputs/models/regression_r2_0p578969.joblib")
    prediction = np.asarray(model.predict(augmented), dtype=np.float32)
    result = top.copy()
    result["pred_log2_mic"] = prediction
    result["pred_mic_um"] = np.power(2.0, np.clip(prediction, -20, 20))
    result = result.sort_values(["pred_log2_mic", "rank_position"], ascending=[True, True]).reset_index(drop=True)
    result.insert(0, "regression_rank", np.arange(1, len(result) + 1))
    scores = root / "outputs/screening/stage4_regression_scores_top1000.csv"
    final = root / "outputs/screening/stage4_top20.csv"
    result.to_csv(scores, index=False)
    result.head(20).to_csv(final, index=False)
    positions = {name: int(result.index[result["sequence"].astype(str) == sequence][0] + 1) for name, sequence in MUST.items()}
    return {"scores": str(scores.relative_to(root)), "top20": str(final.relative_to(root)), "must": positions, "pass": len(result) == 1000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-workers", type=int, default=16)
    parser.add_argument("--ranking-pairs", type=int, default=80_000_000)
    parser.add_argument("--ranking-batch-size", type=int, default=20_000)
    args = parser.parse_args()
    root = package_root()
    if not (root / "data/prepared/candidate_feature_matrix.npy").is_file():
        raise SystemExit(
            "Missing external data: data/prepared/candidate_feature_matrix.npy. "
            "See README.md: Data requirements."
        )
    if not (root / "outputs/training_manifest.json").exists():
        raise SystemExit("Run code/02_train.py first")
    (root / "outputs/screening").mkdir(parents=True, exist_ok=True)
    candidates_path, stage1 = enumerate_candidates(root, args.generation_workers)
    if not stage1["pass"]:
        raise SystemExit(f"Stage 1 mismatch: {stage1}")
    active_path, classification = classify(root, candidates_path)
    top_path, ranking = rank_candidates(root, active_path, args.ranking_pairs, args.ranking_batch_size)
    regression = regress(root, top_path)
    result = {"stage1": stage1, "classification": classification, "ranking": ranking, "regression": regression}
    result["pass"] = all(value["pass"] for value in result.values())
    write_json(root / "outputs/screening_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
