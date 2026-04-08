from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

from src.serving import default_bundle_path, load_deployment_bundle, predict_from_metrics


def create_app(bundle_path: Optional[Path] = None) -> Flask:
    app = Flask(__name__)
    bundle = load_deployment_bundle(bundle_path or default_bundle_path())
    app.config["MODEL_BUNDLE"] = bundle

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

    @app.get("/health")
    def _bundle_missing_health() -> tuple:
        return jsonify({"status": "bundle-missing"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
