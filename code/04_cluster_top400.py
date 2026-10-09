#!/usr/bin/env python3
"""Select three representative roles per cluster from the regression Top400.

Write role selections, unique peptides, and method parameters to an Excel
workbook. A peptide may fill more than one role; report the unique count.
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

from common import TRAINING_PROTOCOL, generic_sequence_matrix, package_root, read_json, sha256_file
from tmhf_repro.panel_selection import (
    _normalize_distance,
    cluster_three_role_panel,
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

def naturality_score(silhouette: float, sizes: np.ndarray, overlap_count: int) -> float:
    imbalance = float(np.std(sizes) / np.mean(sizes))
    return float(silhouette - 0.05 * imbalance - 0.01 * overlap_count)


def build_parameters(input_path: Path, audit: dict) -> pd.DataFrame:
    sizes = np.bincount(audit["labels"], minlength=N_CLUSTERS)
    rows = [
        ("Input", "input_file", input_path.name),
        ("Input", "input_path", str(input_path.relative_to(package_root())) if input_path.is_relative_to(package_root()) else input_path.name),
        ("Input", "input_sha256", sha256_file(input_path)),
        ("Input", "input_rows", 1000),
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
    ]
    frame = pd.DataFrame(rows, columns=["section", "item", "value"])
    # Excel round-trips booleans as 1/0 in mixed columns; store them as text.
    frame["value"] = frame["value"].map(lambda value: str(value) if isinstance(value, bool) else value)
    return frame


def run(root: Path, input_path: Path | None = None) -> dict:
    if input_path is None:
        manifest_path = root / "outputs/screening_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("Run code/03_screen.py first, or provide --input for a standalone selection")
        manifest = read_json(manifest_path)
        if manifest.get("training_protocol") != TRAINING_PROTOCOL or manifest.get("pass") is not True:
            raise SystemExit("Run code/03_screen.py with the current trained models first")
        record = manifest["regression"]
        input_path = root / record["scores"]
        if not input_path.is_file() or sha256_file(input_path) != record["sha256"]:
            raise SystemExit("Regression scores do not match the screening manifest")
    input_path = input_path.resolve()
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
    }

    role_order = {"cluster_center": 0, "cluster_boundary": 1, "cluster_regression_best": 2}
    panel_sheet = selections.sort_values(
        ["cluster_id", "selection_role", "candidate_id"], key=lambda s: s.map(role_order) if s.name == "selection_role" else s
    ).reset_index(drop=True)
    panel_sheet.insert(0, "panel_position", np.arange(1, len(panel_sheet) + 1))
    columns = [
        "panel_position", "cluster_id", "selection_role", "sequence", "candidate_id",
        "charge_ph7", "hydrophobicity", "pred_log2_mic", "pred_mic_um", "regression_rank",
        "rank_position", "active_probability", "cluster_size", "distance_to_center",
    ]
    panel_sheet = panel_sheet[columns].round(
        {"charge_ph7": 4, "hydrophobicity": 4, "pred_log2_mic": 4, "pred_mic_um": 4, "active_probability": 6, "distance_to_center": 6}
    )
    parameters = build_parameters(input_path, audit)

    output_dir = root / "outputs/clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".clustering-", dir=output_dir))
    output = temporary / "top400_clustering.xlsx"
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        panel_sheet.to_excel(writer, sheet_name="panel_21", index=False)
        panel.to_excel(writer, sheet_name="unique_peptides", index=False)
        parameters.to_excel(writer, sheet_name="cluster_parameters", index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "bg_color": "#D9EAD3", "border": 1})
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 22, header)
        writer.sheets["panel_21"].autofilter(0, 0, len(panel_sheet), len(columns) - 1)
        note = workbook.add_worksheet("readme")
        note.write(0, 0, "top400_clustering.xlsx")
        note.write(1, 0, "panel_21: 21 role selections; roles may share peptides. See unique_peptides for the distinct panel.")
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
        "role_selections": len(selections),
        "role_overlap_count": int(audit["overlap_count"]),
        "silhouette_score": silhouette,
        "naturality_score": audit["naturality"],
        "cluster_sizes": sizes.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Regression-ranked Top1000 CSV; defaults to the screening output")
    args = parser.parse_args()
    manifest = run(package_root(), args.input)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
