import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.commit_metrics import extract_source_metrics
from src.serving import (
    default_bundle_path,
    extract_estimator,
    load_deployment_bundle,
    predict_from_metrics,
)

INDIVIDUAL_MODELS_DIR = Path("src/models/individual")
ALLOWED_UPLOAD_SUFFIXES = {".c", ".h", ".cpp", ".cc", ".cxx", ".java", ".py"}
DEFAULT_UPLOAD_SUFFIX = ".c"
SAMPLE_C_CODE = """#include <stdio.h>

int sum_positive(int* values, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        if (values[i] > 0) {
            total += values[i];
        }
    }
    return total;
}

int main() {
    int data[] = {4, -1, 2, 0, 3, -2};
    int result = sum_positive(data, 6);
    if (result > 5) {
        printf("High-risk path check: %d\\n", result);
    } else {
        printf("Low-risk path check: %d\\n", result);
    }
    return 0;
}
"""


def _normalize_suffix(value: Optional[str]) -> str:
    suffix = (value or DEFAULT_UPLOAD_SUFFIX).strip().lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        return DEFAULT_UPLOAD_SUFFIX
    return suffix


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_threshold_input(
    raw_value: Any,
    default_value: float,
) -> Tuple[float, Optional[str]]:
    if raw_value is None or str(raw_value).strip() == "":
        return default_value, None

    try:
        threshold = float(raw_value)
    except (TypeError, ValueError):
        return default_value, "Threshold must be a numeric value between 0 and 1."

    if threshold < 0.0 or threshold > 1.0:
        return default_value, "Threshold must be within the range 0 to 1."

    return threshold, None


def _serialize_global_importance(model: Any, feature_names: List[str]) -> List[Dict[str, float]]:
    estimator = extract_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_[0], dtype=float)
    else:
        return []

    if values.shape[0] != len(feature_names):
        return []

    pairs = sorted(
        zip(feature_names, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    return [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in pairs[:10]
    ]


def _discover_individual_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    if not INDIVIDUAL_MODELS_DIR.exists():
        return registry

    for dataset_dir in sorted(INDIVIDUAL_MODELS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        preprocessing_path = dataset_dir / "preprocessing_bundle.pkl"
        test_results_path = dataset_dir / "test_results.csv"
        if not preprocessing_path.exists() or not test_results_path.exists():
            continue

        with test_results_path.open("r", encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                model_name = str(row.get("Model", "")).strip()
                if not model_name:
                    continue

                model_path = dataset_dir / f"{model_name}.pkl"
                if not model_path.exists():
                    continue

                threshold = _safe_float(row.get("Threshold"), 0.5)
                bundle_key = f"{dataset_dir.name}::{model_name}"
                registry[bundle_key] = {
                    "bundle_key": bundle_key,
                    "dataset_name": dataset_dir.name,
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "preprocessing_path": str(preprocessing_path),
                    "threshold": threshold,
                }

    return registry


def _build_runtime_bundle(entry: Dict[str, Any]) -> Dict[str, Any]:
    model = joblib.load(entry["model_path"])
    preprocessing_artifact = joblib.load(entry["preprocessing_path"])
    raw_feature_order = list(
        preprocessing_artifact.get(
            "raw_numeric_features",
            preprocessing_artifact.get("feature_columns", []),
        )
    )
    selected_feature_order = list(
        preprocessing_artifact.get("selected_features", raw_feature_order)
    )

    return {
        "dataset_name": entry["dataset_name"],
        "model_name": entry["model_name"],
        "model": model,
        "threshold": float(entry["threshold"]),
        "preprocessing_artifact": preprocessing_artifact,
        "raw_feature_order": raw_feature_order,
        "selected_feature_order": selected_feature_order,
        "global_feature_importance": _serialize_global_importance(model, selected_feature_order),
        "_bundle_path": entry["model_path"],
    }


def _build_model_options(registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    options = [
        {
            "key": bundle_key,
            "dataset": value["dataset_name"],
            "model": value["model_name"],
            "label": f"{value['dataset_name']} / {value['model_name']}",
        }
        for bundle_key, value in registry.items()
    ]
    return sorted(options, key=lambda item: (item["dataset"], item["model"]))


def _initialize_registry(
    explicit_bundle_path: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], str]:
    registry = _discover_individual_registry()
    cache: Dict[str, Dict[str, Any]] = {}
    default_bundle_key: Optional[str] = None

    if explicit_bundle_path is not None:
        explicit_bundle = load_deployment_bundle(explicit_bundle_path)
        default_bundle_key = (
            f"{explicit_bundle['dataset_name']}::{explicit_bundle['model_name']}"
        )
        registry[default_bundle_key] = {
            "bundle_key": default_bundle_key,
            "dataset_name": explicit_bundle["dataset_name"],
            "model_name": explicit_bundle["model_name"],
            "bundle_path": explicit_bundle.get("_bundle_path"),
            "deployment_bundle": True,
        }
        cache[default_bundle_key] = explicit_bundle
        return registry, cache, default_bundle_key

    try:
        deployment_bundle = load_deployment_bundle(default_bundle_path())
    except FileNotFoundError:
        deployment_bundle = None

    if deployment_bundle is not None:
        deployment_key = (
            f"{deployment_bundle['dataset_name']}::{deployment_bundle['model_name']}"
        )
        cache[deployment_key] = deployment_bundle
        registry.setdefault(
            deployment_key,
            {
                "bundle_key": deployment_key,
                "dataset_name": deployment_bundle["dataset_name"],
                "model_name": deployment_bundle["model_name"],
                "bundle_path": deployment_bundle.get("_bundle_path"),
                "deployment_bundle": True,
            },
        )
        default_bundle_key = deployment_key

    if not registry:
        raise FileNotFoundError(
            "No model artifacts found. Expected bundles in src/models/deployment "
            "or dataset/model artifacts in src/models/individual."
        )

    if default_bundle_key is None:
        default_bundle_key = sorted(registry.keys())[0]

    return registry, cache, default_bundle_key


def _resolve_bundle(app: Flask, requested_key: Optional[str]) -> Tuple[Dict[str, Any], str]:
    registry: Dict[str, Dict[str, Any]] = app.config["MODEL_REGISTRY"]
    cache: Dict[str, Dict[str, Any]] = app.config["MODEL_BUNDLE_CACHE"]
    default_key: str = app.config["DEFAULT_BUNDLE_KEY"]

    selected_key = requested_key if requested_key in registry else default_key
    if selected_key in cache:
        return cache[selected_key], selected_key

    entry = registry[selected_key]
    if entry.get("deployment_bundle"):
        bundle = load_deployment_bundle(Path(entry["bundle_path"]))
    else:
        bundle = _build_runtime_bundle(entry)
    cache[selected_key] = bundle
    return bundle, selected_key


def _render_ui(
    app: Flask,
    loaded_bundle: Dict[str, Any],
    selected_bundle_key: str,
    source_suffix: str,
    sample_code: str,
    prediction: Optional[Dict[str, Any]],
    prediction_metrics: Optional[Dict[str, float]],
    error: Optional[str],
    threshold_input: float,
    uploaded_filename: Optional[str] = None,
) -> str:
    return render_template(
        "index.html",
        sample_code=sample_code,
        source_suffix=source_suffix,
        prediction=prediction,
        error=error,
        model=loaded_bundle["model_name"],
        dataset=loaded_bundle["dataset_name"],
        threshold=loaded_bundle["threshold"],
        top_features=(prediction or {}).get("top_features", []),
        prediction_metrics=prediction_metrics,
        upload_suffixes=sorted(ALLOWED_UPLOAD_SUFFIXES),
        uploaded_filename=uploaded_filename,
        model_options=app.config["MODEL_OPTIONS"],
        selected_bundle_key=selected_bundle_key,
        threshold_input=threshold_input,
        model_default_threshold=loaded_bundle["threshold"],
    )


def create_app(bundle_path: Optional[Path] = None) -> Flask:
    app = Flask(__name__)
    registry, cache, default_bundle_key = _initialize_registry(bundle_path)
    app.config["MODEL_REGISTRY"] = registry
    app.config["MODEL_OPTIONS"] = _build_model_options(registry)
    app.config["MODEL_BUNDLE_CACHE"] = cache
    app.config["DEFAULT_BUNDLE_KEY"] = default_bundle_key

    @app.get("/")
    def ui_home() -> str:
        bundle_key = request.args.get("bundle_key")
        loaded_bundle, selected_bundle_key = _resolve_bundle(app, bundle_key)
        return _render_ui(
            app=app,
            loaded_bundle=loaded_bundle,
            selected_bundle_key=selected_bundle_key,
            source_suffix=DEFAULT_UPLOAD_SUFFIX,
            sample_code=SAMPLE_C_CODE,
            prediction=None,
            prediction_metrics=None,
            error=None,
            threshold_input=float(loaded_bundle["threshold"]),
        )

    @app.post("/ui/predict")
    def ui_predict() -> tuple:
        bundle_key = request.form.get("bundle_key")
        loaded_bundle, selected_bundle_key = _resolve_bundle(app, bundle_key)
        uploaded_file = request.files.get("source_file")
        source_code = (request.form.get("code") or "").strip()
        filename = None
        threshold_value, threshold_error = _parse_threshold_input(
            request.form.get("threshold_override"),
            float(loaded_bundle["threshold"]),
        )

        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            file_contents = uploaded_file.stream.read().decode("utf-8", errors="ignore")
            if file_contents.strip():
                source_code = file_contents.strip()

        suffix_candidate = request.form.get("source_suffix")
        if filename:
            suffix_candidate = Path(filename).suffix or suffix_candidate
        source_suffix = _normalize_suffix(suffix_candidate)

        if threshold_error:
            return (
                _render_ui(
                    app=app,
                    loaded_bundle=loaded_bundle,
                    selected_bundle_key=selected_bundle_key,
                    source_suffix=source_suffix,
                    sample_code=source_code or SAMPLE_C_CODE,
                    prediction=None,
                    prediction_metrics=None,
                    error=threshold_error,
                    threshold_input=threshold_value,
                ),
                400,
            )

        if not source_code:
            return (
                _render_ui(
                    app=app,
                    loaded_bundle=loaded_bundle,
                    selected_bundle_key=selected_bundle_key,
                    source_suffix=source_suffix,
                    sample_code=SAMPLE_C_CODE,
                    prediction=None,
                    prediction_metrics=None,
                    error="Please paste source code or upload a source file before running prediction.",
                    threshold_input=threshold_value,
                ),
                400,
            )

        try:
            metrics = extract_source_metrics(source_code=source_code, suffix=source_suffix)
            raw_feature_order = loaded_bundle["preprocessing_artifact"]["raw_numeric_features"]
            prediction_metrics = {
                feature_name: float(metrics[feature_name])
                for feature_name in raw_feature_order
                if feature_name in metrics
            }
            prediction = predict_from_metrics(
                loaded_bundle,
                prediction_metrics,
                threshold_override=threshold_value,
            )
            prediction["bundle_key"] = selected_bundle_key
        except Exception as exc:
            return (
                _render_ui(
                    app=app,
                    loaded_bundle=loaded_bundle,
                    selected_bundle_key=selected_bundle_key,
                    source_suffix=source_suffix,
                    sample_code=source_code,
                    prediction=None,
                    prediction_metrics=None,
                    error=f"Unable to analyze source code: {exc}",
                    threshold_input=threshold_value,
                ),
                400,
            )

        return (
            _render_ui(
                app=app,
                loaded_bundle=loaded_bundle,
                selected_bundle_key=selected_bundle_key,
                source_suffix=source_suffix,
                sample_code=source_code,
                prediction=prediction,
                prediction_metrics=prediction_metrics,
                error=None,
                threshold_input=threshold_value,
                uploaded_filename=filename,
            ),
            200,
        )

    @app.get("/health")
    def health() -> tuple:
        bundle_key = request.args.get("bundle_key")
        loaded_bundle, selected_bundle_key = _resolve_bundle(app, bundle_key)
        preprocessing_artifact = loaded_bundle["preprocessing_artifact"]
        return (
            jsonify(
                {
                    "status": "ok",
                    "bundle_key": selected_bundle_key,
                    "model": loaded_bundle["model_name"],
                    "dataset": loaded_bundle["dataset_name"],
                    "threshold": loaded_bundle["threshold"],
                    "bundle_path": loaded_bundle.get("_bundle_path"),
                    "raw_feature_order": preprocessing_artifact["raw_numeric_features"],
                    "selected_feature_order": loaded_bundle["selected_feature_order"],
                    "available_bundle_count": len(app.config["MODEL_OPTIONS"]),
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        metrics = body.get("metrics")
        if not isinstance(metrics, dict):
            return jsonify({"error": "Request JSON must include a 'metrics' object."}), 400

        loaded_bundle, selected_bundle_key = _resolve_bundle(app, body.get("bundle_key"))
        threshold_value, threshold_error = _parse_threshold_input(
            body.get("threshold"),
            float(loaded_bundle["threshold"]),
        )
        if threshold_error:
            return jsonify({"error": threshold_error}), 400

        prediction = predict_from_metrics(
            loaded_bundle,
            metrics,
            threshold_override=threshold_value,
        )
        prediction["bundle_key"] = selected_bundle_key
        return jsonify(prediction), 200

    return app


try:
    app = create_app()
except FileNotFoundError:
    app = Flask(__name__)

    @app.get("/")
    def _bundle_missing_root() -> tuple:
        return (
            jsonify(
                {
                    "status": "bundle-missing",
                    "message": "No model bundles found in src/models/individual or src/models/deployment.",
                }
            ),
            503,
        )

    @app.get("/health")
    def _bundle_missing_health() -> tuple:
        return jsonify({"status": "bundle-missing"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
