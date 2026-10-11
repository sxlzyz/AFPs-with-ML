"""Validate regenerated feature alignment and screening provenance checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / "code" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_features_match_published_physicochemical_values_and_embedding_order():
    module = load_script("00_build_candidate_features.py")
    archive = np.load(ROOT / "data/screening/top21000_model_input_features.npz", allow_pickle=True)
    # The released archive is an independent reference for column/value alignment.
    sequences = archive["sequence"][:3].astype(str).tolist()
    reference = archive["features"][:3]
    columns = json.loads((ROOT / "data/feature_columns.json").read_text())["feature_columns"]
    generated = module.assemble_features(sequences, reference[:, 61:], columns)
    np.testing.assert_array_equal(generated, reference)
    with pytest.raises(ValueError, match="schema"):
        module.assemble_features(sequences, reference[:, 61:], columns[::-1])
    bad = reference[:, 61:].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        module.assemble_features(sequences, bad, columns)


def test_screening_accepts_regeneration_and_rejects_stale_provenance(tmp_path):
    module = load_script("03_screen.py")
    prepared = tmp_path / "data/prepared"
    prepared.mkdir(parents=True)
    columns = tmp_path / "data/feature_columns.json"
    columns.write_text('{"feature_columns": ["a", "b"]}')
    path = prepared / "candidate_feature_matrix.npy"
    np.save(path, np.arange(6, dtype=np.float32).reshape(3, 2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"candidate_feature_matrix": {"rows": 3, "columns": 2, "sha256": "historical-cache"},
                "expected_prescreen": {"sha256": "fixed-sequence-order"}}
    (tmp_path / "data/prepare_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="provenance"):
        module.verify_candidate_matrix(tmp_path)
    record = {"complete": True, "full_pool": True, "matrix_sha256": digest,
              "input_sha256": "fixed-sequence-order",
              "feature_columns_sha256": hashlib.sha256(columns.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(record))
    module.verify_candidate_matrix(tmp_path)
    record["input_sha256"] = "different-row-order"
    path.with_suffix(".json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="row order"):
        module.verify_candidate_matrix(tmp_path)
    record["input_sha256"] = "fixed-sequence-order"
    path.with_suffix(".json").write_text(json.dumps(record))
    np.save(path, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(SystemExit, match="checksum"):
        module.verify_candidate_matrix(tmp_path)
