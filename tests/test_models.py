"""Models and training: padding handled, deterministic, reloadable, test partition untouched."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from scholartrack.features.specs import load_feature_config
from scholartrack.models.artifacts import load_network, load_run, save_network
from scholartrack.models.baselines import LogisticModel, MajorityBaseline
from scholartrack.models.networks import MLP, LSTMClassifier
from scholartrack.synthetic.generator import generate, load_synthetic_config
from scholartrack.targets import load_target_config
from scholartrack.training.data import prepare_data
from scholartrack.training.experiment import run_experiment
from scholartrack.training.loop import TrainSettings, predict_proba, set_seed, train_network


def test_lstm_ignores_padding():
    set_seed(0)
    model = LSTMClassifier(n_inputs=3, hidden_size=5, num_layers=1, dropout=0.0).eval()
    seq = torch.randn(1, 2, 3)
    zero_padded = torch.cat([seq, torch.zeros(1, 4, 3)], dim=1)
    junk_padded = torch.cat([seq, 100 * torch.randn(1, 4, 3)], dim=1)
    lengths = torch.tensor([2])
    with torch.no_grad():
        exact = model(seq, lengths)
        assert torch.allclose(model(zero_padded, lengths), exact, atol=1e-6)
        assert torch.allclose(model(junk_padded, lengths), exact, atol=1e-6)


def test_lstm_uses_order_and_history():
    set_seed(1)
    model = LSTMClassifier(n_inputs=2, hidden_size=4, num_layers=1, dropout=0.0).eval()
    seq = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    with torch.no_grad():
        forward = model(seq, torch.tensor([2]))
        reversed_ = model(seq.flip(1), torch.tensor([2]))
        last_only = model(seq[:, 1:], torch.tensor([1]))
    assert not torch.allclose(forward, reversed_)
    assert not torch.allclose(forward, last_only)


def test_mlp_shapes():
    out = MLP(n_inputs=6, hidden_layers=[8, 4], dropout=0.2)(torch.randn(10, 6))
    assert out.shape == (10,)


def test_logistic_json_matches_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    ours = LogisticModel.fit(x, y, ["a", "b", "c", "d"], C=1.0, class_weight=None, seed=0)
    reference = LogisticRegression(C=1.0, max_iter=5000, random_state=0).fit(x, y)
    np.testing.assert_allclose(ours.predict_proba(x), reference.predict_proba(x)[:, 1], atol=1e-10)
    reloaded = LogisticModel.from_json(ours.to_json())
    np.testing.assert_array_equal(reloaded.predict_proba(x), ours.predict_proba(x))


def test_majority_baseline():
    model = MajorityBaseline.fit(np.array([0, 0, 0, 1]))
    assert model.predict_proba(np.zeros((3, 2))).tolist() == [0.25, 0.25, 0.25]


def _toy_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, 3)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] > 0).astype(np.float32)
    return torch.as_tensor(x), torch.as_tensor(y)


def test_training_is_deterministic_and_restores_best_epoch():
    x, y = _toy_problem()
    settings = TrainSettings(
        learning_rate=0.01, weight_decay=0.0, batch_size=32, max_epochs=40, patience=5
    )

    def fit(seed):
        set_seed(seed)
        model = MLP(3, [8], dropout=0.0)
        result = train_network(model, (x[:300],), y[:300], (x[300:],), y[300:], settings, seed)
        return model, result

    m1, r1 = fit(7)
    m2, r2 = fit(7)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert r1.best_val_loss == min(h["val_loss"] for h in r1.history)
    p = predict_proba(m1, (x[300:],))
    assert ((p > 0.5) == (y[300:].numpy() > 0.5)).mean() > 0.9  # learns an easy rule


def test_network_save_and_load(tmp_path):
    set_seed(3)
    model = MLP(4, [6], dropout=0.1)
    spec = {"type": "mlp", "n_inputs": 4, "hidden_layers": [6], "dropout": 0.1}
    save_network(tmp_path, "mlp", model, spec)
    loaded, loaded_spec = load_network(tmp_path, "mlp")
    x = torch.randn(5, 4)
    assert loaded_spec == spec
    np.testing.assert_allclose(predict_proba(loaded, (x,)), predict_proba(model, (x,)))


TINY_EXPERIMENT = {
    "name": "smoke",
    "seed": 11,
    "selection_metric": "log_loss",
    "seeds_for_chosen_networks": 2,
    "logistic_regression": {"C": [0.1, 1.0], "class_weight": [None]},
    "mlp": {
        "grid": {"hidden_layers": [[8]], "dropout": [0.1]},
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 64,
            "max_epochs": 4,
            "patience": 2,
        },
    },
    "lstm": {
        "num_layers": 1,
        "grid": {"hidden_size": [8], "dropout": [0.1]},
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 64,
            "max_epochs": 4,
            "patience": 2,
            "grad_clip": 1.0,
        },
    },
}


@pytest.fixture(scope="module")
def small_prepared(curriculum):
    cfg = load_synthetic_config()
    cfg["population"]["cohorts"] = [2021, 2022]
    cfg["population"]["students_per_cohort"] = 60
    target = load_target_config()
    data = generate(cfg, curriculum, target)
    return prepare_data(data.semesters, data.subjects, curriculum, target, load_feature_config())


def test_end_to_end_experiment_saves_reloadable_run(small_prepared, tmp_path):
    small_prepared.preprocessor.save(tmp_path / "preprocessor.json")
    results = run_experiment(small_prepared, TINY_EXPERIMENT, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"test_partition_evaluated": False}))

    assert set(results["chosen"]) == {"majority", "logistic_regression", "mlp", "lstm"}
    assert len(results["chosen"]["mlp"]["seeds"]) == 2

    run = load_run(tmp_path)
    va = small_prepared.partitions["validation"]
    saved = np.genfromtxt(
        tmp_path / "validation_predictions.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    for name, predictor in run.predictors.items():
        p = predictor.predict_proba(va.x, va.seq, va.lengths)
        assert ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(p, saved[name], rtol=1e-5, err_msg=name)


def test_experiment_never_reads_the_test_partition(small_prepared, tmp_path, monkeypatch):
    """Replace the test arrays with values that would crash any model if used."""
    test = small_prepared.partitions["test"]
    monkeypatch.setattr(test, "x", None)
    monkeypatch.setattr(test, "seq", None)
    monkeypatch.setattr(test, "y", None)
    run_experiment(small_prepared, TINY_EXPERIMENT, tmp_path)  # must not raise
