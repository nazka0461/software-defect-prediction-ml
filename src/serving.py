import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.data.preprocessing import transform_with_artifact

DEFAULT_MODEL_BUNDLE_PATH = Path("src/models/deployment/best_model_bundle.pkl")
MODEL_BUNDLE_ENV = "MODEL_BUNDLE_PATH"


def default_bundle_path() -> Path:
    return Path(os.environ.get(MODEL_BUNDLE_ENV, DEFAULT_MODEL_BUNDLE_PATH))


def load_deployment_bundle(path: Optional[Path] = None) -> Dict[str, Any]:
    bundle_path = Path(path or default_bundle_path())
    bundle = joblib.load(bundle_path)
    bundle.setdefault("_runtime_cache", {})
    bundle["_bundle_path"] = str(bundle_path)
    return bundle


def extract_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model.named_steps["clf"]
    return model


def get_positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _is_tree_estimator(estimator: Any) -> bool:
    return hasattr(estimator, "feature_importances_")


def _global_feature_importance(bundle: Dict[str, Any]) -> List[Dict[str, float]]:
    importance_rows = bundle.get("global_feature_importance", [])
    return sorted(
        importance_rows,
        key=lambda row: abs(float(row.get("importance", 0.0))),
        reverse=True,
    )


def _compute_shap_contributions(
    bundle: Dict[str, Any],
    estimator: Any,
    X_selected: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        import shap
    except Exception:
        return None

    cache = bundle.setdefault("_runtime_cache", {})
    explainer = cache.get("shap_explainer")
    if explainer is None:
        if _is_tree_estimator(estimator):
            explainer = shap.TreeExplainer(estimator)
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, X_selected)
        else:
            return None
        cache["shap_explainer"] = explainer

    shap_values = explainer.shap_values(X_selected)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[0, :, -1]
    elif shap_array.ndim == 2:
        shap_array = shap_array[0]

    return np.asarray(shap_array, dtype=float)


def explain_prediction(
    bundle: Dict[str, Any],
    X_selected: np.ndarray,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    feature_names = bundle["selected_feature_order"]
    estimator = extract_estimator(bundle["model"])

    shap_contrib = _compute_shap_contributions(bundle, estimator, X_selected)
    if shap_contrib is not None and shap_contrib.shape[0] == len(feature_names):
        pairs = sorted(
            zip(feature_names, shap_contrib),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        return [
            {
                "feature": feature_name,
                "contribution": float(contribution),
                "magnitude": float(abs(contribution)),
            }
            for feature_name, contribution in pairs[:top_n]
        ]

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_[0], dtype=float)
        contrib = coefficients * X_selected[0]
        pairs = sorted(
            zip(feature_names, contrib),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        return [
            {
                "feature": feature_name,
                "contribution": float(contribution),
                "magnitude": float(abs(contribution)),
            }
            for feature_name, contribution in pairs[:top_n]
        ]

    return _global_feature_importance(bundle)[:top_n]


def predict_from_metrics(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    X_selected = transform_with_artifact(bundle["preprocessing_artifact"], metrics)
    probability = float(get_positive_scores(bundle["model"], X_selected)[0])
    threshold = float(bundle["threshold"])
    label = "DEFECT-PRONE" if probability >= threshold else "CLEAN"
    top_features = explain_prediction(bundle, X_selected)

    return {
        "probability": round(probability, 3),
        "label": label,
        "threshold": round(threshold, 3),
        "model": bundle["model_name"],
        "dataset": bundle["dataset_name"],
        "top_features": top_features,
    }
