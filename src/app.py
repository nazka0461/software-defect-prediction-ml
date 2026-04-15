from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from src.commit_metrics import extract_source_metrics
from src.serving import default_bundle_path, load_deployment_bundle, predict_from_metrics

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


def create_app(bundle_path: Optional[Path] = None) -> Flask:
    app = Flask(__name__)
    bundle = load_deployment_bundle(bundle_path or default_bundle_path())
    app.config["MODEL_BUNDLE"] = bundle

    @app.get("/")
    def ui_home() -> str:
        loaded_bundle = app.config["MODEL_BUNDLE"]
        return render_template(
            "index.html",
            sample_code=SAMPLE_C_CODE,
            source_suffix=DEFAULT_UPLOAD_SUFFIX,
            prediction=None,
            error=None,
            model=loaded_bundle["model_name"],
            dataset=loaded_bundle["dataset_name"],
            threshold=loaded_bundle["threshold"],
            top_features=None,
            prediction_metrics=None,
            upload_suffixes=sorted(ALLOWED_UPLOAD_SUFFIXES),
        )

    @app.post("/ui/predict")
    def ui_predict() -> tuple:
        loaded_bundle = app.config["MODEL_BUNDLE"]
        uploaded_file = request.files.get("source_file")
        source_code = (request.form.get("code") or "").strip()
        filename = None

        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            file_contents = uploaded_file.stream.read().decode("utf-8", errors="ignore")
            if file_contents.strip():
                source_code = file_contents.strip()

        suffix_candidate = request.form.get("source_suffix")
        if filename:
            suffix_candidate = Path(filename).suffix or suffix_candidate
        source_suffix = _normalize_suffix(suffix_candidate)

        if not source_code:
            return (
                render_template(
                    "index.html",
                    sample_code=SAMPLE_C_CODE,
                    source_suffix=source_suffix,
                    prediction=None,
                    error="Please paste source code or upload a source file before running prediction.",
                    model=loaded_bundle["model_name"],
                    dataset=loaded_bundle["dataset_name"],
                    threshold=loaded_bundle["threshold"],
                    top_features=None,
                    prediction_metrics=None,
                    upload_suffixes=sorted(ALLOWED_UPLOAD_SUFFIXES),
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
            prediction = predict_from_metrics(loaded_bundle, prediction_metrics)
        except Exception as exc:
            return (
                render_template(
                    "index.html",
                    sample_code=source_code,
                    source_suffix=source_suffix,
                    prediction=None,
                    error=f"Unable to analyze source code: {exc}",
                    model=loaded_bundle["model_name"],
                    dataset=loaded_bundle["dataset_name"],
                    threshold=loaded_bundle["threshold"],
                    top_features=None,
                    prediction_metrics=None,
                    upload_suffixes=sorted(ALLOWED_UPLOAD_SUFFIXES),
                ),
                400,
            )

        return (
            render_template(
                "index.html",
                sample_code=source_code,
                source_suffix=source_suffix,
                prediction=prediction,
                error=None,
                model=loaded_bundle["model_name"],
                dataset=loaded_bundle["dataset_name"],
                threshold=loaded_bundle["threshold"],
                top_features=prediction.get("top_features", []),
                prediction_metrics=prediction_metrics,
                upload_suffixes=sorted(ALLOWED_UPLOAD_SUFFIXES),
                uploaded_filename=filename,
            ),
            200,
        )

    @app.get("/health")
    def health() -> tuple:
        loaded_bundle = app.config["MODEL_BUNDLE"]
        preprocessing_artifact = loaded_bundle["preprocessing_artifact"]
        return (
            jsonify(
                {
                    "status": "ok",
                    "model": loaded_bundle["model_name"],
                    "dataset": loaded_bundle["dataset_name"],
                    "threshold": loaded_bundle["threshold"],
                    "bundle_path": loaded_bundle.get("_bundle_path"),
                    "raw_feature_order": preprocessing_artifact["raw_numeric_features"],
                    "selected_feature_order": loaded_bundle["selected_feature_order"],
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

        prediction = predict_from_metrics(app.config["MODEL_BUNDLE"], metrics)
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
                    "message": "Set MODEL_BUNDLE_PATH or create src/models/deployment/best_model_bundle.pkl",
                }
            ),
            503,
        )

    @app.get("/health")
    def _bundle_missing_health() -> tuple:
        return jsonify({"status": "bundle-missing"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
