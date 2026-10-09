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
from tmhf_repro.panel_selection import validate_candidates, pairwise_distance, cluster_three_role_panel

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


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), ROOT / 'code' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_accepts_changed_candidates_without_archive_gate(tmp_path):
    module = load_script('04_cluster_top400.py')
    frame = pd.read_csv(ROOT / 'data/archive/reference/regression_top1000.csv')
    # Change every identity, so comparison with any old panel would fail.
    frame['candidate_id'] = ['new_' + str(i) for i in range(len(frame))]
    path = tmp_path / 'new_scores.csv'
    frame.to_csv(path, index=False)
    report = module.run(tmp_path, path)
    assert report['role_selections'] == 21
    workbook = tmp_path / report['output']
    assert pd.ExcelFile(workbook).sheet_names == ['panel_21', 'unique_peptides', 'cluster_parameters', 'readme']
    roles = pd.read_excel(workbook, sheet_name='panel_21')
    unique = pd.read_excel(workbook, sheet_name='unique_peptides')
    assert roles.candidate_id.str.startswith('new_').all()
    assert unique.candidate_id.is_unique
    assert len(unique) == report['panel_size']
    assert len(roles) - len(unique) == report['role_overlap_count']


def test_classifier_retains_scores_without_designated_sequences(tmp_path, monkeypatch):
    module = load_script('03_screen.py')
    folder = tmp_path / 'outputs/screening'
    folder.mkdir(parents=True)
    prepared = tmp_path / 'data/prepared'
    prepared.mkdir(parents=True)
    count = 21_003
    scores = np.linspace(0.01, 0.99, count, dtype=np.float32)
    np.save(prepared / 'candidate_feature_matrix.npy', scores[:, None])
    source = folder / 'candidates.csv'
    pd.DataFrame({'candidate_id': [f'cand_{i:08d}' for i in range(count)],
                  'sequence': ['SYNTHETIC_TEST_INPUT'] * count}).to_csv(source, index=False)

    class Classifier:
        classes_ = [0, 1]
        def predict_proba(self, matrix):
            return np.column_stack([1 - matrix[:, 0], matrix[:, 0]])

    monkeypatch.setattr(module.joblib, 'load', lambda path: Classifier())
    path, report = module.classify(tmp_path, source)
    assert report['pass']
    retained = pd.read_csv(path)
    assert len(retained) == 21_000
    assert set(retained.candidate_id) == {f'cand_{i:08d}' for i in range(3, count)}


def test_ranker_uses_only_observed_records(tmp_path, monkeypatch):
    module = load_script('02_train.py')
    import xgboost
    from sklearn.base import BaseEstimator, ClassifierMixin
    folder = tmp_path / 'data/prepared'
    folder.mkdir(parents=True)
    mic = np.tile([1., 4., 16.], 12).astype(np.float32)
    features = np.arange(len(mic) * 4, dtype=np.float32).reshape(-1, 4) + 1
    np.savez(folder / 'paper_training.npz', mic=mic, features=features)
    seen = {}

    class Classifier(ClassifierMixin, BaseEstimator):
        def __init__(self, **kwargs):
            pass
        def fit(self, x, y):
            seen['train_rows'] = len(y)
            self.classes_ = np.unique(y)
            return self
        def predict(self, x):
            seen['evaluation_rows'] = len(x)
            return np.resize(self.classes_, len(x))

    monkeypatch.setattr(xgboost, 'XGBClassifier', Classifier)
    monkeypatch.setattr(module, 'select_device', lambda: 'cpu')
    monkeypatch.setattr(module.joblib, 'dump', lambda model, path: path.write_bytes(b'test-model'))
    models = tmp_path / 'outputs/models'
    models.mkdir(parents=True)
    report = module.train_ranking(tmp_path, models)
    assert report['positives'] == 24
    assert report['pairs'] == 276
    assert seen['train_rows'] + seen['evaluation_rows'] == 276
    assert report['pass']
    np.testing.assert_array_equal(np.load(folder / 'paper_training.npz')['mic'], mic)


def test_screening_rejects_legacy_or_mismatched_models(tmp_path):
    module = load_script('03_screen.py')
    folder = tmp_path / 'outputs'
    folder.mkdir()
    path = folder / 'training_manifest.json'
    path.write_text(json.dumps({'pass': True}))
    with pytest.raises(SystemExit, match='Retrain'):
        module.verify_training(tmp_path)
    manifest = {'pass': True, 'training_protocol': module.TRAINING_PROTOCOL}
    for stage in ('classification', 'ranking', 'regression'):
        model = folder / (stage + '.joblib')
        model.write_bytes(stage.encode())
        manifest[stage] = {'model': str(model.relative_to(tmp_path)), 'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
    path.write_text(json.dumps(manifest))
    module.verify_training(tmp_path)
    model.write_bytes(b'changed')
    with pytest.raises(SystemExit, match='does not match'):
        module.verify_training(tmp_path)


def test_ranking_and_regression_allow_new_results_without_reference_tables(tmp_path, monkeypatch):
    module = load_script('03_screen.py')
    prepared = tmp_path / 'data/prepared'
    prepared.mkdir(parents=True)
    folder = tmp_path / 'outputs/screening'
    folder.mkdir(parents=True)
    frame = candidate_frame(1000).drop(columns=['regression_rank', 'rank_position', 'rank_score', 'pred_log2_mic'])
    frame['active_probability'] = np.linspace(0.01, 0.99, len(frame))
    active = folder / 'active.csv'
    frame.to_csv(active, index=False)
    np.save(prepared / 'candidate_feature_matrix.npy', np.arange(2000, dtype=np.float32).reshape(1000, 2))
    np.savez(prepared / 'regression_training.npz', context_columns=np.array([], dtype=str), secondary_columns=np.array([], dtype=str))

    class Ranker:
        def predict(self, x):
            return np.where(x[:, 0] < x[:, 2], 1, 2)

    class Regressor:
        def predict(self, x):
            return -x[:, 0] / 1000

    monkeypatch.setattr(module.joblib, 'load', lambda path: Ranker() if path.name == 'ranking.joblib' else Regressor())
    top, rank_report = module.rank_candidates(tmp_path, active, pairs=100, batch_size=13)
    assert rank_report['pass'] and rank_report['pairs'] == 100
    report = module.regress(tmp_path, top)
    assert report['pass']
    result = pd.read_csv(tmp_path / report['scores'])
    assert result.pred_log2_mic.is_monotonic_increasing
    assert result.iloc[0].candidate_id == 'cand_00000999'
    with pytest.raises(ValueError, match='must be positive'):
        module.rank_candidates(tmp_path, active, pairs=0, batch_size=13)


def test_selection_default_rejects_stale_screening(tmp_path):
    module = load_script('04_cluster_top400.py')
    folder = tmp_path / 'outputs'
    folder.mkdir()
    (folder / 'screening_manifest.json').write_text(json.dumps({'pass': True}))
    with pytest.raises(SystemExit, match='current trained models'):
        module.run(tmp_path)
