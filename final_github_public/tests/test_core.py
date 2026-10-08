from pathlib import Path
import sys
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from tmhf_repro.panel_selection import validate_candidates, pairwise_distance, evaluate_targets, cluster_three_role_panel

def candidate_frame(n: int = 30) -> pd.DataFrame:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for index in range(n):
        sequence = "".join(
            alphabet[((index // (len(alphabet) ** (position % 3))) + position * 5) % len(alphabet)]
            for position in range(14)
        )
        rows.append(
            {
                "regression_rank": index + 1,
                "rank_position": n - index,
                "candidate_id": f"cand_{index:08d}",
                "sequence": sequence,
                "seed": sequence[:7],
                "mode": "mirror" if index % 2 == 0 else "repeat",
                "charge_ph7": 3.0 + (index % 4),
                "hydrophobicity": -0.8 + 0.1 * (index % 12),
                "active_probability": 0.55 + 0.01 * (index % 20),
                "rank_score": 1000 + index * 7,
                "pred_log2_mic": 2.0 + index * 0.03,
            }
        )
    return pd.DataFrame(rows)

def test_validate_candidates_accepts_well_formed_frame() -> None:
    validated = validate_candidates(candidate_frame())
    assert len(validated) == 30
    assert validated["candidate_id"].is_unique

def test_validate_candidates_rejects_unequal_sequence_lengths() -> None:
    frame = candidate_frame()
    frame.loc[0, "sequence"] = "AAAA"
    with pytest.raises(ValueError, match="equal length"):
        validate_candidates(frame)

def test_validate_candidates_enforces_expected_row_count() -> None:
    with pytest.raises(ValueError, match="exactly 1000"):
        validate_candidates(candidate_frame(30), expected_rows=1000)

def test_validate_candidates_rejects_duplicates_after_uppercase_normalization() -> None:
    frame = candidate_frame(3)
    frame.loc[0, "sequence"] = frame.loc[1, "sequence"].lower()
    with pytest.raises(ValueError, match="sequence values must be unique"):
        validate_candidates(frame)

def test_hamming_distance_is_normalized_and_symmetric() -> None:
    matrix = pairwise_distance(["AAAA", "AAAV", "VVVV"], metric="hamming")
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(0.25)
    assert matrix[0, 2] == pytest.approx(1.0)

def test_evaluate_targets_reports_hits_only_after_panel_exists() -> None:
    frame = candidate_frame(30)
    panel = frame.iloc[[0, 4, 8, 12]]
    result = evaluate_targets(
        panel,
        {
            "present_a": frame.iloc[0]["sequence"],
            "present_b": frame.iloc[8]["sequence"],
            "absent": frame.iloc[20]["sequence"],
        },
    )
    assert result["target_hit_count"] == 2
    assert result["target_all_present"] is False
    assert result["target_positions"] == {"present_a": 1, "present_b": 3}

def test_cluster_three_role_panel_tracks_role_overlap_and_unique_candidates() -> None:
    frame = candidate_frame(28)
    distance = pairwise_distance(frame["sequence"].tolist(), "hamming")
    labels = np.repeat(np.arange(7), 4)
    selections, unique_panel = cluster_three_role_panel(frame, distance, labels)
    assert len(selections) == 21
    assert selections["cluster_id"].nunique() == 7
    assert set(selections["selection_role"]) == {"cluster_center", "cluster_boundary", "cluster_regression_best"}
    assert 7 <= len(unique_panel) <= 21
    assert unique_panel["candidate_id"].is_unique
    for _cluster_id, group in selections.groupby("cluster_id"):
        assert set(group["selection_role"]) == {"cluster_center", "cluster_boundary", "cluster_regression_best"}


def test_included_inputs_match_integrity_manifest():
    manifest = json.loads((ROOT / "data/prepare_manifest.json").read_text())
    external = "data/prepared/candidate_feature_matrix.npy"
    for name, record in manifest["files"].items():
        if name == external:
            continue  # This matrix is explicitly distributed separately.
        path = ROOT / name
        assert path.stat().st_size == record["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name


def test_final_panel_matches_archive_without_external_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location("cluster_top400", ROOT / "code/04_cluster_top400.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "data").mkdir()
    archive = ROOT / "data/final_selected_21_peptides.csv"
    (tmp_path / "data/final_selected_21_peptides.csv").write_bytes(archive.read_bytes())
    report = module.run(tmp_path, ROOT / "data/reference/regression_top1000.csv")
    assert report["matches_archive"] is True
    assert report["panel_size"] == 21
    workbook = tmp_path / report["output"]
    assert pd.ExcelFile(workbook).sheet_names == ["panel_21", "cluster_parameters", "readme"]
    panel = pd.read_excel(workbook, sheet_name="panel_21")
    expected = pd.read_csv(archive)
    keys = ["candidate_id", "cluster_id", "selection_role", "sequence"]
    assert set(panel[keys].itertuples(index=False, name=None)) == set(expected[keys].itertuples(index=False, name=None))
    assert panel["candidate_id"].is_unique
    assert panel.groupby("cluster_id").size().eq(3).all()
