from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .features import AA


AA_INDEX = {aa: index for index, aa in enumerate(AA)}
REQUIRED_COLUMNS = {
    "candidate_id",
    "sequence",
    "seed",
    "mode",
    "regression_rank",
    "rank_position",
    "pred_log2_mic",
    "active_probability",
    "rank_score",
    "charge_ph7",
    "hydrophobicity",
}


def validate_candidates(frame: pd.DataFrame, expected_rows: int | None = None) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate input is empty")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Candidate input must contain exactly {expected_rows} rows")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique")
    sequences = frame["sequence"].astype(str).str.upper()
    if sequences.duplicated().any():
        raise ValueError("sequence values must be unique")
    if sequences.str.len().nunique() != 1:
        raise ValueError("All sequences must have equal length")
    invalid = sorted(set("".join(sequences)) - set(AA))
    if invalid:
        raise ValueError(f"Sequences contain non-canonical amino acids: {invalid}")
    result = frame.copy().reset_index(drop=True)
    result["sequence"] = sequences
    numeric = [
        "regression_rank",
        "rank_position",
        "pred_log2_mic",
        "active_probability",
        "rank_score",
        "charge_ph7",
        "hydrophobicity",
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Candidate numeric columns must contain finite values")
    return result


def pairwise_distance(values: Sequence[str] | np.ndarray, metric: str) -> np.ndarray:
    if metric == "hamming":
        sequences = list(values)
        encoded = np.asarray([[AA_INDEX[residue] for residue in sequence] for sequence in sequences], dtype=np.int8)
        return squareform(pdist(encoded, metric="hamming")).astype(np.float32)
    matrix = np.asarray(values, dtype=np.float64)
    distances = squareform(pdist(matrix, metric=metric))
    distances[~np.isfinite(distances)] = 0.0
    return distances.astype(np.float32)


def _normalize_distance(distance: np.ndarray) -> np.ndarray:
    maximum = float(distance.max())
    return distance / maximum if maximum > 0 else distance.copy()


def cluster_three_role_panel(
    frame: pd.DataFrame,
    distance: np.ndarray,
    labels: np.ndarray,
    fixed_medoids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for output_cluster_id, label in enumerate(sorted(np.unique(labels)), start=1):
        members = np.flatnonzero(labels == label)
        if len(members) < 2:
            raise ValueError(f"Cluster {output_cluster_id} has fewer than two candidates")
        center = int(fixed_medoids[output_cluster_id - 1]) if fixed_medoids is not None else _cluster_representative(frame, distance, members, "medoid")
        non_center = members[members != center]
        maximum = float(distance[non_center, center].max())
        boundary_candidates = non_center[
            np.flatnonzero(np.isclose(distance[non_center, center], maximum, rtol=0.0, atol=1e-12))
        ]
        boundary = int(_preference_order(frame, boundary_candidates)[0])
        regression_best = int(
            sorted(
                members.tolist(),
                key=lambda index: (
                    float(frame.iloc[index]["pred_log2_mic"]),
                    float(frame.iloc[index]["regression_rank"]),
                    str(frame.iloc[index]["candidate_id"]),
                ),
            )[0]
        )
        for role, index in (
            ("cluster_center", center),
            ("cluster_boundary", boundary),
            ("cluster_regression_best", regression_best),
        ):
            row = frame.iloc[index].to_dict()
            row.update(
                {
                    "cluster_id": output_cluster_id,
                    "cluster_size": int(len(members)),
                    "selection_role": role,
                    "distance_to_center": float(distance[index, center]),
                }
            )
            rows.append(row)
    selections = pd.DataFrame(rows)
    role_priority = {"cluster_center": 0, "cluster_boundary": 1, "cluster_regression_best": 2}
    unique_rows = []
    for candidate_id, group in selections.groupby("candidate_id", sort=False):
        first = group.sort_values("selection_role", key=lambda values: values.map(role_priority)).iloc[0].to_dict()
        first["selection_roles"] = "+".join(group["selection_role"].tolist())
        first["role_count"] = int(len(group))
        unique_rows.append(first)
    unique_panel = pd.DataFrame(unique_rows).sort_values(["cluster_id", "regression_rank", "candidate_id"]).reset_index(drop=True)
    unique_panel.insert(0, "panel_position", np.arange(1, len(unique_panel) + 1))
    return selections, unique_panel


def _preference_order(frame: pd.DataFrame, indices: np.ndarray | None = None) -> np.ndarray:
    pool = np.arange(len(frame), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    ordered = sorted(
        pool.tolist(),
        key=lambda index: (
            float(frame.iloc[index]["pred_log2_mic"]),
            float(frame.iloc[index]["regression_rank"]),
            str(frame.iloc[index]["candidate_id"]),
        ),
    )
    return np.asarray(ordered, dtype=np.int64)


def _cluster_representative(
    frame: pd.DataFrame,
    distance: np.ndarray,
    members: np.ndarray,
    representative: str,
) -> int:
    if representative == "medoid":
        means = distance[np.ix_(members, members)].mean(axis=1)
        best = members[np.flatnonzero(np.isclose(means, means.min(), rtol=0.0, atol=1e-12))]
        return int(_preference_order(frame, best)[0])
    raise ValueError(f"Unsupported representative rule: {representative}")
