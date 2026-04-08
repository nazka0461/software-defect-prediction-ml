import threading
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
import requests
from sklearn.linear_model import LogisticRegression
from werkzeug.serving import make_server

from src.app import create_app
from src.data.preprocessing import build_pipeline, fit_resample_train, transform_test


def _build_test_bundle(tmp_path: Path) -> Path:
    X_df = pd.DataFrame(
        {
            "loc": [10, 25, 30, 5, 50, 60, 12, 45],
            "lOComment": [2, 5, 1, 0, 10, 9, 3, 8],
            "v(g)": [1, 4, 3, 1, 7, 8, 2, 6],
        }
    )
    y = pd.Series([0, 1, 0, 0, 1, 1, 0, 1])

    pipeline, _ = build_pipeline(y_train=y)
    fit_resample_train(pipeline, X_df, y)
    X_selected = transform_test(pipeline, X_df)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_selected, y)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": ["loc", "lOComment", "v(g)"],
        "raw_numeric_features": ["loc", "lOComment", "v(g)"],
        "post_filter_features": ["loc", "lOComment", "v(g)"],
        "selected_features": ["loc", "lOComment", "v(g)"],
        "selected_feature_indices": [0, 1, 2],
        "meta": {},
    }
    bundle = {
        "dataset_name": "TESTSET",
        "model_name": "LogReg",
        "model": model,
        "threshold": 0.5,
        "preprocessing_artifact": artifact,
        "raw_feature_order": artifact["raw_numeric_features"],
        "selected_feature_order": artifact["selected_features"],
        "global_feature_importance": [],
    }

    bundle_path = tmp_path / "bundle.pkl"
    joblib.dump(bundle, bundle_path)
    return bundle_path


@pytest.fixture()
def app_client(tmp_path: Path):
    bundle_path = _build_test_bundle(tmp_path)
    app = create_app(bundle_path)
    app.testing = True
    with app.test_client() as client:
        yield client


@pytest.fixture()
def live_server(tmp_path: Path):
    bundle_path = _build_test_bundle(tmp_path)
    app = create_app(bundle_path)
    server = make_server("127.0.0.1", 5055, app)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    time.sleep(0.2)
    try:
        yield "http://127.0.0.1:5055"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_health_returns_bundle_metadata(app_client) -> None:
    response = app_client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model"] == "LogReg"
    assert payload["dataset"] == "TESTSET"
    assert payload["selected_feature_order"] == ["loc", "lOComment", "v(g)"]


def test_predict_accepts_valid_metrics(app_client) -> None:
    response = app_client.post(
        "/predict",
        json={"metrics": {"loc": 42, "lOComment": 5, "v(g)": 3}},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["label"] in {"CLEAN", "DEFECT-PRONE"}
    assert payload["model"] == "LogReg"
    assert payload["dataset"] == "TESTSET"
    assert isinstance(payload["top_features"], list)


def test_predict_handles_missing_metrics_with_imputer(app_client) -> None:
    response = app_client.post("/predict", json={"metrics": {"loc": 18}})
    assert response.status_code == 200
    payload = response.get_json()
    assert 0.0 <= payload["probability"] <= 1.0


def test_predict_rejects_malformed_payload(app_client) -> None:
    response = app_client.post("/predict", json={"bad": "payload"})
    assert response.status_code == 400


def test_live_server_accepts_requests(live_server: str) -> None:
    response = requests.post(
        f"{live_server}/predict",
        json={"metrics": {"loc": 55, "lOComment": 7, "v(g)": 6}},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["dataset"] == "TESTSET"
    assert "probability" in payload
