from __future__ import annotations

import json

import pytest
import torch

from heart_nn.features import FEATURE_NAMES, FEATURES
from heart_nn.predict import HeartPredictor

CONTRACT_FILES = [
    "model.pt",
    "preprocessor.joblib",
    "metadata.json",
    "metrics.json",
    "history.json",
]


def test_artifacts_follow_contract(trained_artifacts):
    for name in CONTRACT_FILES:
        assert (trained_artifacts / name).exists(), name
    meta = json.loads((trained_artifacts / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == FEATURE_NAMES
    assert 0.0 <= meta["threshold"] <= 1.0
    assert {"model_config", "trained_at", "seed", "version"} <= meta.keys()
    metrics = json.loads((trained_artifacts / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["test"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for key in ("accuracy", "f1", "roc_auc"):
        assert set(metrics["cv"][key]) == {"mean", "std"}
    assert len(metrics["confusion_matrix"]) == 2
    assert set(metrics["roc_curve"]) == {"fpr", "tpr"}
    assert set(metrics["class_distribution"]) == {"0", "1"}
    assert "logistic_regression" in metrics["baselines"]
    assert set(metrics["n_samples"]) == {"train", "val", "test"}
    history = json.loads((trained_artifacts / "history.json").read_text(encoding="utf-8"))
    assert len(history["train_loss"]) == len(history["val_loss"])
    assert 0 <= history["best_epoch"] < len(history["val_loss"])


def test_predictor_round_trip(trained_artifacts, sample_record):
    p = HeartPredictor.from_artifacts(str(trained_artifacts))
    assert p.feature_names == FEATURE_NAMES
    out = p.predict(sample_record)
    assert set(out) == {"probability", "prediction"}
    assert 0.0 <= out["probability"] <= 1.0
    assert out["prediction"] == int(out["probability"] >= p.threshold)
    batch = p.predict_batch([sample_record, sample_record])
    assert len(batch) == 2
    assert batch[0]["probability"] == pytest.approx(out["probability"])


def test_predictor_is_deterministic(trained_artifacts, sample_record):
    a = HeartPredictor.from_artifacts(trained_artifacts).predict(sample_record)
    b = HeartPredictor.from_artifacts(trained_artifacts).predict(sample_record)
    assert a == b


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("chol"),
        lambda r: r.update(extra=1),
        lambda r: r.update(age=150),
        lambda r: r.update(cp=7),
        lambda r: r.update(oldpeak=-1),
        lambda r: r.update(sex="male"),
        lambda r: r.update(chol=float("nan")),
        lambda r: r.update(fbs=True),
    ],
    ids=["missing", "extra", "age-range", "cp-category", "oldpeak-range", "string", "nan", "bool"],
)
def test_predictor_validation_errors(trained_artifacts, sample_record, mutate):
    p = HeartPredictor.from_artifacts(trained_artifacts)
    mutate(sample_record)
    with pytest.raises(ValueError):
        p.predict(sample_record)


def test_predict_batch_rejects_empty(trained_artifacts):
    with pytest.raises(ValueError):
        HeartPredictor.from_artifacts(trained_artifacts).predict_batch([])


def test_features_metadata_shape():
    assert [f["name"] for f in FEATURES] == FEATURE_NAMES
    for f in FEATURES:
        assert {"name", "label_es", "description_es", "type", "min", "max", "step"} <= f.keys()
        assert f["type"] in {"numeric", "categorical"}
        if f["type"] == "categorical":
            values = [o["value"] for o in f["options"]]
            assert min(values) == f["min"] and max(values) == f["max"]


def test_training_is_reproducible(data_path, fast_cfg):
    from heart_nn.data import load_dataset, split_data
    from heart_nn.train import train_pipeline

    split = split_data(
        load_dataset(data_path), fast_cfg.test_size, fast_cfg.val_size, fast_cfg.seed
    )
    runs = [
        train_pipeline(split.X_train, split.y_train, split.X_val, split.y_val, fast_cfg)
        for _ in range(2)
    ]
    for (_, a), (_, b) in zip(
        runs[0].model.state_dict().items(), runs[1].model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)
    assert runs[0].history == runs[1].history


def test_best_weights_are_restored(data_path, fast_cfg):
    """El modelo devuelto debe reproducir la pérdida de validación de la mejor época."""
    from heart_nn.data import load_dataset, split_data
    from heart_nn.losses import build_loss
    from heart_nn.train import to_tensor, train_pipeline

    split = split_data(
        load_dataset(data_path), fast_cfg.test_size, fast_cfg.val_size, fast_cfg.seed
    )
    pipe = train_pipeline(split.X_train, split.y_train, split.X_val, split.y_val, fast_cfg)
    xv = to_tensor(pipe.preprocessor.transform(split.X_val[FEATURE_NAMES]))
    yv = to_tensor(split.y_val.to_numpy())
    with torch.inference_mode():
        loss = build_loss(fast_cfg.loss)(pipe.model(xv), yv).item()
    assert loss == pytest.approx(min(pipe.history["val_loss"]), rel=1e-5)
    assert pipe.history["val_loss"][pipe.history["best_epoch"]] == pytest.approx(loss, rel=1e-5)
