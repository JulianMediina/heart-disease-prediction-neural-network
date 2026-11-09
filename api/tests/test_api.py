from __future__ import annotations

import pytest

from api.schemas import risk_level_for

from .conftest import FAKE_FEATURES, FAKE_HISTORY, FAKE_METRICS, VALID_PAYLOAD, FakePredictor


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "1.0.0",
                        "target_note": "Etiqueta posiblemente invertida."}


def test_features(client):
    r = client.get("/api/features")
    assert r.status_code == 200
    assert r.json() == FAKE_FEATURES


def test_metrics_and_history(client):
    assert client.get("/api/metrics").json() == FAKE_METRICS
    assert client.get("/api/training-history").json() == FAKE_HISTORY


def test_dataset_summary(client):
    body = client.get("/api/dataset/summary").json()
    assert body["n_samples"] == 4
    assert body["n_duplicates"] == 1
    # enfermedad = 1 - output: el CSV de prueba tiene output = [1, 1, 1, 0]
    assert body["class_distribution"] == {"0": 3, "1": 1}
    assert len(body["features"]) == 13
    assert body["features"]["age"]["min"] == 37.0
    assert body["features"]["age"]["max"] == 63.0
    assert set(body["features"]["chol"]) == {"mean", "std", "min", "max"}


def test_predict_ok(client, fake_predictor: FakePredictor):
    r = client.post("/api/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json() == {"probability": 0.8, "prediction": 1, "risk_level": "alto"}
    assert fake_predictor.calls == [VALID_PAYLOAD]


@pytest.mark.parametrize(
    ("p", "level"),
    [(0.0, "bajo"), (0.329, "bajo"), (0.33, "moderado"), (0.659, "moderado"),
     (0.66, "alto"), (1.0, "alto")],
)
def test_risk_level_boundaries(p, level):
    assert risk_level_for(p) == level


@pytest.mark.parametrize(
    "mutation",
    [
        {"age": 150},           # fuera de rango
        {"chol": 600},          # fuera del rango observado
        {"sex": 2},             # categoría inexistente
        {"cp": -1},
        {"caa": 5},
        {"thall": 4},
        {"oldpeak": -0.5},
        {"chol": "alto"},       # tipo incorrecto
        {"age": 50.5},          # entero con decimales
        {"sex": True},          # booleano en lugar de entero (modo estricto)
        {"cp": "1"},            # cadena numérica (modo estricto)
        {"extra_field": 1},     # campos extra prohibidos
    ],
)
def test_predict_invalid_input_returns_422(client, mutation):
    r = client.post("/api/predict", json={**VALID_PAYLOAD, **mutation})
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = dict(VALID_PAYLOAD)
    payload.pop("thall")
    assert client.post("/api/predict", json=payload).status_code == 422


def test_predict_value_error_from_model_returns_422(client):
    r = client.post("/api/predict", json={**VALID_PAYLOAD, "chol": 500})
    assert r.status_code == 422
    assert "entrada inválida" in r.json()["detail"]


def test_cors_allows_frontend_origin(client):
    r = client.options(
        "/api/predict",
        headers={"Origin": "http://localhost:4200", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:4200"


def test_cors_rejects_unknown_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_degraded_mode_returns_503(degraded_client):
    h = degraded_client.get("/api/health")
    assert h.status_code == 200
    assert h.json() == {"status": "ok", "model_loaded": False, "model_version": None,
                        "target_note": None}
    for path in ("/api/features", "/api/metrics", "/api/training-history", "/api/dataset/summary"):
        assert degraded_client.get(path).status_code == 503, path
    r = degraded_client.post("/api/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503
    assert "Modelo no disponible" in r.json()["detail"]


def test_schema_ranges_match_model_feature_catalog():
    """Los rangos del esquema Pydantic deben coincidir con heart_nn.features.FEATURES."""
    features_mod = pytest.importorskip("heart_nn.features")
    from api.schemas import PatientFeatures

    schema = PatientFeatures.model_json_schema()["properties"]
    assert set(schema) == {f["name"] for f in features_mod.FEATURES}
    for spec in features_mod.FEATURES:
        prop = schema[spec["name"]]
        assert prop["minimum"] == spec["min"], spec["name"]
        assert prop["maximum"] == spec["max"], spec["name"]
        if "options" in spec:
            values = sorted(o["value"] for o in spec["options"])
            assert values == list(range(int(spec["min"]), int(spec["max"]) + 1)), spec["name"]


def test_degraded_errors_do_not_leak_paths(degraded_client):
    for path in ("/api/metrics", "/api/dataset/summary"):
        detail = degraded_client.get(path).json()["detail"]
        assert "nope" not in detail and ":\\" not in detail and "/" not in detail.replace("/api", "")
    detail = degraded_client.post("/api/predict", json=VALID_PAYLOAD).json()["detail"]
    assert "nope" not in detail and "model.pt" not in detail
