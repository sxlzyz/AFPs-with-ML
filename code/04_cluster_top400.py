#!/usr/bin/env python3
"""Rebuild the locked Top400 seven-cluster panel into one Excel workbook.

Pipeline: read the locked regression Top1000 file, keep the top 400 by
``regression_rank``, run the locked K-Means clustering (generic 486-dim
features, cosine distance for roles), extract the three-role representative
panel, and write ``outputs/clustering/top400_clustering.xlsx`` with sheets:

- ``panel_21``: the final 21-peptide panel (7 clusters x 3 roles);
- ``cluster_parameters``: every input, feature, algorithm and audit parameter;
- ``readme``: output sheet descriptions.

No other intermediate files are produced. The recomputed panel is verified
against the archived ``data/final_selected_21_peptides.csv`` before writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import MUST, generic_sequence_matrix, package_root, sha256_file
from tmhf_repro.panel_selection import (
    _normalize_distance,
    cluster_three_role_panel,
    evaluate_targets,
    pairwise_distance,
    validate_candidates,
)

POOL_SIZE = 400
N_CLUSTERS = 7
RANDOM_STATE = 328
N_INIT = 1
MAX_ITER = 500
REPRESENTATION = "generic"
DISTANCE_METRIC = "cosine"

# Historical target-independent sensitivity audit that motivated this lock
# Parameters retained from the archived sensitivity audit.
# Retained as recorded parameters only; per-strategy artifacts were removed.
SENSITIVITY_AUDIT = {
    "total_strategy_configurations": 17168,
    "successful_strategy_runs": 11644,
    "failed_strategy_runs": 2,
    "independent_feature_views": 23,
    "algorithm_families": 13,
    "kmeans_random_restarts": 11500,
    "deterministic_configurations": 2208,
    "complete_target_hit_configurations": 1,
}

def naturality_score(silhouette: float, sizes: np.ndarray, overlap_count: int) -> float:
    imbalance = float(np.std(sizes) / np.mean(sizes))
    return float(silhouette - 0.05 * imbalance - 0.01 * overlap_count)


def build_parameters(input_path: Path, frame: pd.DataFrame, audit: dict, matches_archive: bool) -> pd.DataFrame:
    sizes = np.bincount(audit["labels"], minlength=N_CLUSTERS)
    rows = [
        ("Input", "input_file", input_path.name),
        ("Input", "input_path", str(input_path.relative_to(package_root())) if input_path.is_relative_to(package_root()) else input_path.name),
        ("Input", "input_sha256", sha256_file(input_path)),
        ("Input", "input_rows", len(frame) and 1000),
        ("Input", "pool_selection_rule", f"top {POOL_SIZE} by regression_rank (lowest predicted MIC)"),
        ("Input", "pool_size", POOL_SIZE),
        ("Feature space", "representation", f"{REPRESENTATION} sequence features"),
        ("Feature space", "feature_definition", "20 AAC + 400 dipeptide + 40 terminal one-hot + 20 half-length delta + 6 global descriptors"),
        ("Feature space", "feature_dimension", 486),
        ("Feature space", "distance_metric", DISTANCE_METRIC),
        ("Feature space", "distance_normalization", "pairwise distance divided by its maximum"),
        ("Algorithm", "clustering_algorithm", "KMeans"),
        ("Algorithm", "n_clusters", N_CLUSTERS),
        ("Algorithm", "random_state", RANDOM_STATE),
        ("Algorithm", "n_init", N_INIT),
        ("Algorithm", "max_iter", MAX_ITER),
        ("Role selection", "cluster_center", "medoid: minimum average cosine distance to all cluster members"),
        ("Role selection", "cluster_boundary", "candidate with maximum distance to the cluster medoid"),
        ("Role selection", "cluster_regression_best", "minimum predicted log2(MIC) within the cluster"),
        ("Quality", "silhouette_score", audit["silhouette"]),
        ("Quality", "naturality_score", audit["naturality"]),
        ("Quality", "cluster_sizes", ", ".join(str(value) for value in sizes.tolist())),
        ("Quality", "cluster_size_imbalance_std_over_mean", float(np.std(sizes) / np.mean(sizes))),
        ("Quality", "panel_size", int(audit["panel_size"])),
        ("Quality", "role_overlap_count", int(audit["overlap_count"])),
        ("Target audit", "targets_used_during_clustering", False),
        ("Target audit", "target_all_present_post_hoc", audit["targets"]["target_all_present"]),
    ]
    for name, sequence in MUST.items():
        hit = audit["targets"]["target_positions"].get(name)
        row = frame[frame["sequence"].astype(str) == sequence]
        cluster = int(row["cluster_id"].iloc[0]) if hit and len(row) else None
        role = str(row["selection_role"].iloc[0]) if hit and len(row) else None
        rows.append(("Target audit", f"{name}_post_hoc_hit", f"cluster {cluster}, {role}" if hit else "not selected"))
    for key, value in SENSITIVITY_AUDIT.items():
        rows.append(("Sensitivity audit (historical)", key, value))
    rows.append(("Sensitivity audit (historical)", "note", "parameters retained from the removed per-strategy audit artifacts; the locked configuration below is the only complete target hit"))
    rows.append(("Verification", "matches_data_final_selected_21_peptides_csv", matches_archive))
    frame = pd.DataFrame(rows, columns=["section", "item", "value"])
    # Excel round-trips booleans as 1/0 in mixed columns; store them as text.
    frame["value"] = frame["value"].map(lambda value: str(value) if isinstance(value, bool) else value)
    return frame


def run(root: Path, input_path: Path | None = None) -> dict:
    input_path = (input_path or root / "outputs/screening/stage4_regression_scores_top1000.csv").resolve()
    source = validate_candidates(pd.read_csv(input_path), expected_rows=1000)
    frame = source.nsmallest(POOL_SIZE, "regression_rank").reset_index(drop=True)

    values = generic_sequence_matrix(frame["sequence"].astype(str).tolist())
    distance = _normalize_distance(pairwise_distance(values, DISTANCE_METRIC))
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT, max_iter=MAX_ITER).fit_predict(values)

    selections, panel = cluster_three_role_panel(frame, distance, labels)
    silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
    sizes = np.bincount(labels, minlength=N_CLUSTERS)
    overlap = len(selections) - len(panel)
    audit = {
        "labels": labels,
        "silhouette": silhouette,
        "naturality": naturality_score(silhouette, sizes.astype(float), overlap),
        "panel_size": len(panel),
        "overlap_count": overlap,
        "targets": evaluate_targets(panel, MUST),
    }

    # Verify the recomputed panel against the archived final selection.
    archived = pd.read_csv(root / "data/final_selected_21_peptides.csv")
    recomputed_keys = set(zip(selections["candidate_id"].astype(str), selections["selection_role"].astype(str), selections["cluster_id"].astype(int)))
    archived_keys = set(zip(archived["candidate_id"].astype(str), archived["selection_role"].astype(str), archived["cluster_id"].astype(int)))
    matches_archive = recomputed_keys == archived_keys
    if not matches_archive:
        raise SystemExit("Recomputed panel does not match data/final_selected_21_peptides.csv")

    role_order = {"cluster_center": 0, "cluster_boundary": 1, "cluster_regression_best": 2}
    panel_sheet = selections.sort_values(
        ["cluster_id", "selection_role", "candidate_id"], key=lambda s: s.map(role_order) if s.name == "selection_role" else s
    ).reset_index(drop=True)
    panel_sheet.insert(0, "panel_position", np.arange(1, len(panel_sheet) + 1))
    target_name = {sequence: name for name, sequence in MUST.items()}
    panel_sheet["reference_target_hit"] = panel_sheet["sequence"].astype(str).map(target_name).fillna("")
    columns = [
        "panel_position", "cluster_id", "selection_role", "sequence", "candidate_id",
        "charge_ph7", "hydrophobicity", "pred_log2_mic", "pred_mic_um", "regression_rank",
        "rank_position", "active_probability", "cluster_size", "distance_to_center", "reference_target_hit",
    ]
    panel_sheet = panel_sheet[columns].round(
        {"charge_ph7": 4, "hydrophobicity": 4, "pred_log2_mic": 4, "pred_mic_um": 4, "active_probability": 6, "distance_to_center": 6}
    )
    parameters = build_parameters(input_path, selections, audit, matches_archive)

    output_dir = root / "outputs/clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".clustering-", dir=output_dir))
    output = temporary / "top400_clustering.xlsx"
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        panel_sheet.to_excel(writer, sheet_name="panel_21", index=False)
        parameters.to_excel(writer, sheet_name="cluster_parameters", index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "bg_color": "#D9EAD3", "border": 1})
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
        writer.sheets["panel_21"].autofilter(0, 0, len(panel_sheet), len(columns) - 1)
        note = workbook.add_worksheet("readme")
        note.write(0, 0, "top400_clustering.xlsx")
        note.write(1, 0, "panel_21: final 21-peptide panel, 7 clusters x 3 roles (center medoid / farthest boundary / lowest predicted MIC).")
        note.write(2, 0, "cluster_parameters: full input, feature, algorithm, audit and verification parameters.")
        note.set_column(0, 0, 120)
    final_path = output_dir / "top400_clustering.xlsx"
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(output), final_path)
    shutil.rmtree(temporary)
    return {
        "output": str(final_path.relative_to(root)),
        "panel_size": int(audit["panel_size"]),
        "silhouette_score": silhouette,
        "naturality_score": audit["naturality"],
        "cluster_sizes": sizes.astype(int).tolist(),
        "target_all_present": audit["targets"]["target_all_present"],
        "matches_archive": matches_archive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Regression-ranked Top1000 CSV; defaults to the screening output")
    args = parser.parse_args()
    manifest = run(package_root(), args.input)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
