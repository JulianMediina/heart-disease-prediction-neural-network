from __future__ import annotations

import pandas as pd
import pytest

from heart_nn.data import drop_duplicates, load_dataset, split_data, validate_schema
from heart_nn.features import FEATURE_NAMES, TARGET


def test_load_dataset_removes_duplicates(data_path):
    raw = pd.read_csv(data_path)
    df = load_dataset(data_path)
    assert raw.duplicated().sum() >= 1
    assert not df.duplicated().any()
    assert len(df) == len(raw) - raw.duplicated().sum()


def test_drop_duplicates_keeps_first():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    assert drop_duplicates(df).to_dict("list") == {"a": [1, 2], "b": [3, 4]}


def test_split_is_disjoint_and_complete(data_path):
    df = load_dataset(data_path)
    split = split_data(df, test_size=0.15, val_size=0.15, seed=42)
    idx = [set(split.X_train.index), set(split.X_val.index), set(split.X_test.index)]
    assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2]) and idx[1].isdisjoint(idx[2])
    assert sum(len(s) for s in idx) == len(df)
    # Sin filas idénticas compartidas entre particiones (dedupe antes del split).
    rows = [
        {tuple(r) for r in part.to_numpy()} for part in (split.X_train, split.X_val, split.X_test)
    ]
    full = df[FEATURE_NAMES]
    if not full.duplicated().any():
        assert rows[0].isdisjoint(rows[2]) and rows[1].isdisjoint(rows[2])


def test_split_is_stratified(data_path):
    df = load_dataset(data_path)
    split = split_data(df, test_size=0.15, val_size=0.15, seed=42)
    overall = 1 - df[TARGET].mean()  # y = enfermedad = 1 - output
    for y in (split.y_train, split.y_val, split.y_test):
        assert abs(y.mean() - overall) < 0.05


def test_split_is_reproducible(data_path):
    df = load_dataset(data_path)
    a = split_data(df, 0.15, 0.15, seed=7)
    b = split_data(df, 0.15, 0.15, seed=7)
    assert a.X_test.index.equals(b.X_test.index)


def test_validate_schema_rejects_bad_columns(data_path):
    df = pd.read_csv(data_path).drop(columns=["chol"])
    with pytest.raises(ValueError):
        validate_schema(df)


def test_validate_schema_rejects_unknown_category(data_path):
    df = pd.read_csv(data_path)
    df.loc[0, "cp"] = 9
    with pytest.raises(ValueError):
        validate_schema(df)


def test_preprocessor_fit_only_on_train(data_path, fast_cfg):
    """El escalador debe reflejar solo las estadísticas de train (sin fuga de val/test)."""
    from heart_nn.train import train_pipeline

    df = load_dataset(data_path)
    split = split_data(df, fast_cfg.test_size, fast_cfg.val_size, fast_cfg.seed)
    pipe = train_pipeline(split.X_train, split.y_train, split.X_val, split.y_val, fast_cfg)
    scaler = pipe.preprocessor.named_transformers_["num"]
    cols = list(scaler.feature_names_in_)
    assert scaler.mean_ == pytest.approx(split.X_train[cols].mean().to_numpy())
    assert scaler.mean_ != pytest.approx(df[cols].mean().to_numpy())
