import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 42
TOP_K = 12


def select_features_mi_rfe(
    X_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
    feature_columns: List[str],
    top_k: int = TOP_K,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Rank features with mutual information and RFE, then keep the union of both top-k sets.
    """
    if len(feature_columns) == 0:
        raise ValueError("Feature selection requires at least one feature.")

    n_features = min(top_k, len(feature_columns))

    mi_scores = mutual_info_classif(
        X_train_resampled,
        y_train_resampled,
        random_state=random_state,
    )
    mi_series = pd.Series(mi_scores, index=feature_columns).sort_values(ascending=False)
    top_mi = mi_series.head(n_features).index.tolist()

    if len(feature_columns) == 1:
        top_rfe = feature_columns[:]
        rfe_support = [True]
        rfe_ranking = [1]
        rf_importances = pd.Series([1.0], index=feature_columns)
    else:
        rfe = RFE(
            estimator=RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
            ),
            n_features_to_select=n_features,
        )
        rfe.fit(X_train_resampled, y_train_resampled)
        top_rfe = [
            feature_columns[i]
            for i, is_selected in enumerate(rfe.support_)
            if is_selected
        ]
        rfe_support = rfe.support_.tolist()
        rfe_ranking = rfe.ranking_.tolist()
        estimator = rfe.estimator_
        rf_importances = (
            pd.Series(estimator.feature_importances_, index=top_rfe).sort_values(
                ascending=False
            )
            if hasattr(estimator, "feature_importances_")
            else pd.Series(dtype=float)
        )

    selected_features = sorted(set(top_mi) | set(top_rfe))
    selected_indices = [
        feature_columns.index(feature_name) for feature_name in selected_features
    ]

    return {
        "selected_features": selected_features,
        "selected_indices": selected_indices,
        "mi_scores": mi_series.to_dict(),
        "top_mi": top_mi,
        "top_rfe": top_rfe,
        "rfe_support": rfe_support,
        "rfe_ranking": rfe_ranking,
        "rf_importances": rf_importances.to_dict(),
    }


def verify_lr_f1(
    X_train_selected: np.ndarray,
    y_train: np.ndarray,
    X_test_selected: np.ndarray,
    y_test: np.ndarray,
    X_train_all: Optional[np.ndarray] = None,
    X_test_all: Optional[np.ndarray] = None,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Compare Logistic Regression on all features versus selected features.
    """
    lr_selected = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_selected.fit(X_train_selected, y_train)
    selected_f1 = f1_score(
        y_test,
        lr_selected.predict(X_test_selected),
        zero_division=0,
    )

    results = {
        "f1_selected_features": float(selected_f1),
        "n_selected": int(X_train_selected.shape[1]),
    }

    if X_train_all is not None and X_test_all is not None:
        lr_all = LogisticRegression(random_state=random_state, max_iter=1000)
        lr_all.fit(X_train_all, y_train)
        all_f1 = f1_score(y_test, lr_all.predict(X_test_all), zero_division=0)
        results["f1_all_features"] = float(all_f1)
        results["n_all"] = int(X_train_all.shape[1])

    return results


def plot_feature_selection_summary(
    selection_result: Dict[str, Any],
    out_path: Path,
    top_k: int = 15,
) -> None:
    """Plot MI and RF-RFE importance views for a selected feature set."""
    mi_series = pd.Series(selection_result.get("mi_scores", {})).sort_values()
    rf_series = pd.Series(selection_result.get("rf_importances", {})).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not mi_series.empty:
        mi_series.tail(min(top_k, len(mi_series))).plot(
            kind="barh",
            ax=axes[0],
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
    axes[0].set_title("Mutual Information Ranking")
    axes[0].set_xlabel("MI score")

    if not rf_series.empty:
        rf_series.tail(min(top_k, len(rf_series))).plot(
            kind="barh",
            ax=axes[1],
            color="coral",
            edgecolor="black",
            linewidth=0.5,
        )
    axes[1].set_title("RandomForest RFE Importance")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def export_selected_features_json(
    selected_features: List[str],
    out_path: Path,
) -> None:
    """Legacy helper to save selected features as JSON when needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected_features, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable feature-selection helpers. "
        "Use preprocessing or the benchmark runner to generate artifacts."
    )
