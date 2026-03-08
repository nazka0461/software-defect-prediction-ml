from pathlib import Path
from typing import List, Optional

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

try:
    from .preprocessing import run_preprocessing, load_pipeline, prepare_and_split, fit_resample_train, transform_test
except ImportError:
    from preprocessing import run_preprocessing, load_pipeline, prepare_and_split, fit_resample_train, transform_test

RANDOM_STATE = 42
TOP_K = 12


def select_features_mi_rfe(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    feature_columns: List[str],
    top_k: int = TOP_K,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Rank by mutual information and by RFE with RandomForest; return union of top-k from each.
    """
    n_features = min(top_k, len(feature_columns))
    # Method 1: mutual information
    mi = mutual_info_classif(X_train_res, y_train_res, random_state=random_state)
    mi_series = pd.Series(mi, index=feature_columns).sort_values(ascending=False)
    top_mi = mi_series.head(top_k).index.tolist()
    # Method 2: RFE with RandomForest
    rfe = RFE(
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        n_features_to_select=n_features,
    )
    rfe.fit(X_train_res, y_train_res)
    top_rfe = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
    # Union of both
    selected = sorted(set(top_mi) | set(top_rfe))
    return selected, mi_series, rfe


def verify_lr_f1(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: List[str],
    selected_features: List[str],
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train LogisticRegression with all features vs selected features; compare F1 on test."""
    # Indices of selected features
    col_to_idx = {c: i for i, c in enumerate(feature_columns)}
    sel_idx = [col_to_idx[c] for c in selected_features if c in col_to_idx]
    X_train_all = X_train_res
    X_train_sel = X_train_res[:, sel_idx]
    X_test_all = X_test
    X_test_sel = X_test[:, sel_idx]

    lr_all = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_sel = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_all.fit(X_train_all, y_train_res)
    lr_sel.fit(X_train_sel, y_train_res)

    f1_all = f1_score(y_test, lr_all.predict(X_test_all), zero_division=0)
    f1_sel = f1_score(y_test, lr_sel.predict(X_test_sel), zero_division=0)
    return {
        "f1_all_features": float(f1_all),
        "f1_selected_features": float(f1_sel),
        "n_all": len(feature_columns),
        "n_selected": len(selected_features),
    }


def run_feature_selection(
    data_path: Path,
    pipeline_path: Optional[Path] = None,
    out_json_path: Path = Path("../models/selected_features.json"),
    out_plot_path: Optional[Path] = Path("../models/feature_importance.png"),
    top_k: int = TOP_K,
    run_verification: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Load data, run preprocessing (or load pipeline), run MI + RFE, save selected_features.json,
    optionally run LR F1 verification and save feature importance plot.
    """
    df = pd.read_csv(data_path)
    if pipeline_path is not None and Path(pipeline_path).exists():
        pipeline, feature_columns, _ = load_pipeline(Path(pipeline_path))
        X_train, X_test, y_train, y_test, feature_columns, _ = prepare_and_split(df, verbose=False)
        X_train_res, y_train_res = fit_resample_train(pipeline, X_train, y_train)
        X_test_scaled = transform_test(pipeline, X_test)
    else:
        X_train_res, y_train_res, X_test_scaled, y_test, feature_columns, _ = run_preprocessing(
            df, pipeline_save_path=pipeline_path, verbose=verbose
        )

    selected, mi_series, rfe = select_features_mi_rfe(
        X_train_res, y_train_res, feature_columns, top_k=top_k
    )

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(selected, f, indent=2)
    if verbose:
        print("Selected features (union of MI + RFE top-k):", selected)
        print(f"Saved to {out_json_path}")

    if run_verification:
        results = verify_lr_f1(
            X_train_res, y_train_res, X_test_scaled, y_test, feature_columns, selected
        )
        if verbose:
            print("LR F1 verification:")
            print(f"  F1 (all {results['n_all']} features): {results['f1_all_features']:.4f}")
            print(f"  F1 (selected {results['n_selected']} features): {results['f1_selected_features']:.4f}")

    if out_plot_path:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # MI ranking
            mi_series.plot(kind="barh", ax=axes[0], color="steelblue")
            axes[0].set_title("Feature importance (mutual information)")
            axes[0].set_xlabel("MI score")
            # RFE ranking (use RF feature_importances_ from the fitted RF inside RFE)
            rf = rfe.estimator_
            imp = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=True)
            imp.tail(len(feature_columns)).plot(kind="barh", ax=axes[1], color="coral")
            axes[1].set_title("Feature importance (RandomForest from RFE)")
            axes[1].set_xlabel("Importance")
            plt.tight_layout()
            out_plot_path = Path(out_plot_path)
            out_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"Feature importance plot saved to {out_plot_path}")
        except Exception as e:
            if verbose:
                print(f"Could not save feature importance plot: {e}")

    return selected


if __name__ == "__main__":
    import sys
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent.parent
    _default_data = _project_root / "data" / "processed" / "promise_nasa_combined_clean.csv"
    _default_models = _project_root / "src" / "models"

    data_path = _default_data
    pipeline_path = _default_models / "pipeline.pkl"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        pipeline_path = Path(sys.argv[2])

    run_feature_selection(
        data_path,
        pipeline_path=pipeline_path,
        out_json_path=_default_models / "selected_features.json",
        out_plot_path=_default_models / "feature_importance.png",
        run_verification=True,
        verbose=True,
    )
