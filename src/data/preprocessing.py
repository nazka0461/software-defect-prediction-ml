from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .feature_selection import (
    RANDOM_STATE as FEATURE_SELECTION_RANDOM_STATE,
    TOP_K as DEFAULT_TOP_K,
    select_features_mi_rfe,
    verify_lr_f1,
)

TARGET_COL = "label"
DATASET_COL = "dataset"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
VIF_THRESHOLD = 10.0
CORR_THRESHOLD = 0.95


def _stratify_or_none(y: pd.Series) -> Optional[pd.Series]:
    counts = y.value_counts()
    if y.nunique() < 2 or counts.min() < 2:
        return None
    return y


def _compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    if df_features.shape[1] <= 1:
        return pd.DataFrame(
            [(col, 1.0) for col in df_features.columns],
            columns=["feature", "vif"],
        )

    X_np = df_features.values.astype(float)
    vif_list = []
    for i, col in enumerate(df_features.columns):
        try:
            vif = float(variance_inflation_factor(X_np, i))
        except Exception:
            vif = float("inf")
        vif_list.append((col, vif))
    return pd.DataFrame(vif_list, columns=["feature", "vif"])


def _prepare_numeric_frame(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = df.copy()
    excluded_cols = {target_col, DATASET_COL}
    feature_df = df.drop(columns=[c for c in excluded_cols if c in df.columns], errors="ignore")
    numeric_feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in feature_df.columns if c not in numeric_feature_cols]
    X_all = feature_df[numeric_feature_cols].copy()
    y_all = df[target_col].astype(int).copy()
    return X_all, y_all, numeric_feature_cols, non_numeric_cols


def _drop_constant_corr_vif_across_splits(
    X_train: pd.DataFrame,
    X_other: Dict[str, pd.DataFrame],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    X_train = X_train.drop(columns=constant_cols, errors="ignore")
    X_other = {
        name: split.drop(columns=constant_cols, errors="ignore")
        for name, split in X_other.items()
    }

    high_corr_cols: List[str] = []
    if X_train.shape[1] > 1:
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_cols = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
        X_train = X_train.drop(columns=high_corr_cols, errors="ignore")
        X_other = {
            name: split.drop(columns=high_corr_cols, errors="ignore")
            for name, split in X_other.items()
        }

    dropped_vif: List[Tuple[str, float]] = []
    while X_train.shape[1] > 1:
        vif_df = _compute_vif(X_train)
        max_vif = float(vif_df["vif"].max())
        if not np.isfinite(max_vif) or max_vif > VIF_THRESHOLD:
            feature_to_drop = str(
                vif_df.sort_values("vif", ascending=False).iloc[0]["feature"]
            )
            dropped_vif.append((feature_to_drop, max_vif))
            X_train = X_train.drop(columns=[feature_to_drop], errors="ignore")
            X_other = {
                name: split.drop(columns=[feature_to_drop], errors="ignore")
                for name, split in X_other.items()
            }
            continue
        break

    meta = {
        "constant_cols": constant_cols,
        "high_corr_cols": high_corr_cols,
        "dropped_vif": dropped_vif,
        "feature_columns": X_train.columns.tolist(),
    }

    if verbose:
        print("Constant columns dropped:", constant_cols)
        print("Highly correlated columns dropped (|r|>0.95):", high_corr_cols)
        print("Features dropped due to VIF > 10 (in order):")
        for feature_name, vif_score in dropped_vif:
            print(f"  {feature_name}: VIF={vif_score:.2f}")

    return X_train, X_other, meta


def prepare_and_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """
    Legacy helper kept for compatibility with the original combined-data flow.
    """
    X_all, y_all, _, non_numeric_cols = _prepare_numeric_frame(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_or_none(y_all),
    )

    X_train, filtered_others, filter_meta = _drop_constant_corr_vif_across_splits(
        X_train,
        {"test": X_test},
        verbose=verbose,
    )
    X_test = filtered_others["test"]

    meta = {
        "non_numeric_cols": non_numeric_cols,
        **filter_meta,
    }
    return X_train, X_test, y_train, y_test, filter_meta["feature_columns"], meta


def prepare_train_val_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a train/validation/test split using train-only feature filtering.
    """
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must both be > 0.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1.")

    X_all, y_all, raw_numeric_features, non_numeric_cols = _prepare_numeric_frame(
        df,
        target_col=target_col,
    )

    holdout_size = validation_size + test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_all,
        y_all,
        test_size=holdout_size,
        random_state=random_state,
        stratify=_stratify_or_none(y_all),
    )

    test_fraction_in_holdout = test_size / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_in_holdout,
        random_state=random_state,
        stratify=_stratify_or_none(y_holdout),
    )

    X_train_filtered, filtered_others, filter_meta = _drop_constant_corr_vif_across_splits(
        X_train,
        {"val": X_val, "test": X_test},
        verbose=verbose,
    )

    return {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "X_train_filtered": X_train_filtered,
        "X_val_filtered": filtered_others["val"],
        "X_test_filtered": filtered_others["test"],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "raw_numeric_features": raw_numeric_features,
        "non_numeric_cols": non_numeric_cols,
        "filter_meta": filter_meta,
    }


def _build_sampler(y_train: pd.Series, random_state: int = RANDOM_STATE) -> Tuple[Any, Dict[str, Any]]:
    class_counts = Counter(y_train.astype(int).tolist())
    if not class_counts:
        return "passthrough", {"sampler_name": "passthrough", "class_counts": {}}

    minority_count = min(class_counts.values())
    if minority_count <= 1:
        sampler = RandomOverSampler(random_state=random_state)
        meta = {
            "sampler_name": "RandomOverSampler",
            "class_counts": dict(class_counts),
            "smote_k_neighbors": None,
        }
        return sampler, meta

    k_neighbors = max(1, min(5, minority_count - 1))
    sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    meta = {
        "sampler_name": "SMOTE",
        "class_counts": dict(class_counts),
        "smote_k_neighbors": k_neighbors,
    }
    return sampler, meta


def build_pipeline(
    y_train: Optional[pd.Series] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[ImbPipeline, Dict[str, Any]]:
    """
    Build the preprocessing pipeline used for train-only fitting.
    """
    sampler, sampler_meta = _build_sampler(y_train, random_state=random_state) if y_train is not None else (
        SMOTE(random_state=random_state),
        {
            "sampler_name": "SMOTE",
            "class_counts": {},
            "smote_k_neighbors": 5,
        },
    )
    pipeline = ImbPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("smote", sampler),
        ]
    )
    return pipeline, sampler_meta


def fit_resample_train(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply imputation, scaling, and sampling on training data only."""
    return pipeline.fit_resample(X_train, y_train)


def transform_test(pipeline: ImbPipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Transform data with the fitted imputer + scaler only."""
    return pipeline[:-1].transform(X_test)


def build_preprocessing_artifact(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None,
    artifact_save_path: Optional[Path] = None,
    top_k: int = DEFAULT_TOP_K,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = FEATURE_SELECTION_RANDOM_STATE,
    run_verification: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Create a dataset-specific preprocessing + feature-selection artifact.
    """
    split_data = prepare_train_val_test_split(
        df=df,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        verbose=verbose,
    )

    pipeline, sampler_meta = build_pipeline(
        y_train=split_data["y_train"],
        random_state=random_state,
    )
    X_train_resampled_all, y_train_resampled = fit_resample_train(
        pipeline,
        split_data["X_train_filtered"],
        split_data["y_train"],
    )
    X_train_all = transform_test(pipeline, split_data["X_train_filtered"])
    X_val_all = transform_test(pipeline, split_data["X_val_filtered"])
    X_test_all = transform_test(pipeline, split_data["X_test_filtered"])

    post_filter_features = split_data["filter_meta"]["feature_columns"]
    selection_result = select_features_mi_rfe(
        X_train_resampled_all,
        y_train_resampled,
        post_filter_features,
        top_k=top_k,
        random_state=random_state,
    )
    selected_indices = selection_result["selected_indices"]
    selected_features = selection_result["selected_features"]

    X_train_selected = X_train_all[:, selected_indices]
    X_val_selected = X_val_all[:, selected_indices]
    X_test_selected = X_test_all[:, selected_indices]
    X_train_resampled_selected = X_train_resampled_all[:, selected_indices]

    verification = {}
    if run_verification:
        verification = verify_lr_f1(
            X_train_selected=X_train_selected,
            y_train=split_data["y_train"].to_numpy(),
            X_test_selected=X_test_selected,
            y_test=split_data["y_test"].to_numpy(),
            X_train_all=X_train_all,
            X_test_all=X_test_all,
            random_state=random_state,
        )

    artifact = {
        "artifact_type": "preprocessing_selection_bundle",
        "bundle_version": 2,
        "dataset_name": dataset_name,
        "pipeline": pipeline,
        "feature_columns": post_filter_features,
        "raw_numeric_features": split_data["raw_numeric_features"],
        "post_filter_features": post_filter_features,
        "selected_features": selected_features,
        "selected_feature_indices": selected_indices,
        "meta": {
            "non_numeric_cols": split_data["non_numeric_cols"],
            **split_data["filter_meta"],
            "sampler": sampler_meta,
            "selection": selection_result,
            "verification": verification,
            "top_k": top_k,
        },
    }

    result = {
        "artifact": artifact,
        "X_train_all": X_train_all,
        "X_val_all": X_val_all,
        "X_test_all": X_test_all,
        "X_train_selected": X_train_selected,
        "X_val_selected": X_val_selected,
        "X_test_selected": X_test_selected,
        "X_train_resampled_all": X_train_resampled_all,
        "X_train_resampled_selected": X_train_resampled_selected,
        "y_train": split_data["y_train"].to_numpy(),
        "y_val": split_data["y_val"].to_numpy(),
        "y_test": split_data["y_test"].to_numpy(),
        "y_train_resampled": y_train_resampled,
        "X_train_filtered_df": split_data["X_train_filtered"].copy(),
        "X_val_filtered_df": split_data["X_val_filtered"].copy(),
        "X_test_filtered_df": split_data["X_test_filtered"].copy(),
    }

    if artifact_save_path is not None:
        artifact_save_path = Path(artifact_save_path)
        artifact_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_save_path)
        if verbose:
            print(f"Saved preprocessing artifact to {artifact_save_path}")

    if verbose:
        print(f"Dataset: {dataset_name or 'unknown'}")
        print("Raw numeric features:", len(split_data["raw_numeric_features"]))
        print("Post-filter features:", len(post_filter_features))
        print("Selected features:", selected_features)
        print(
            "Train/Val/Test shapes:",
            X_train_selected.shape,
            X_val_selected.shape,
            X_test_selected.shape,
        )

    return result


def _coerce_input_frame(
    raw_input: Union[pd.DataFrame, Dict[str, Any]],
    raw_feature_order: List[str],
) -> pd.DataFrame:
    if isinstance(raw_input, dict):
        row = {feature: raw_input.get(feature, np.nan) for feature in raw_feature_order}
        return pd.DataFrame([row], columns=raw_feature_order)

    df = raw_input.copy()
    for feature in raw_feature_order:
        if feature not in df.columns:
            df[feature] = np.nan
    return df[raw_feature_order]


def transform_with_artifact(
    artifact: Dict[str, Any],
    raw_input: Union[pd.DataFrame, Dict[str, Any]],
) -> np.ndarray:
    """
    Transform raw feature inputs into the model-ready selected feature matrix.
    """
    raw_feature_order = artifact["raw_numeric_features"]
    post_filter_features = artifact.get("post_filter_features", artifact["feature_columns"])
    selected_indices = artifact.get(
        "selected_feature_indices",
        list(range(len(post_filter_features))),
    )

    raw_df = _coerce_input_frame(raw_input, raw_feature_order)
    filtered_df = raw_df[post_filter_features]
    scaled = transform_test(artifact["pipeline"], filtered_df)
    return scaled[:, selected_indices]


def load_preprocessing_artifact(path: Path) -> Dict[str, Any]:
    """Load the richer preprocessing artifact."""
    return joblib.load(path)


def run_preprocessing(
    df: pd.DataFrame,
    pipeline_save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], ImbPipeline]:
    """
    Backward-compatible wrapper for the original preprocessing flow.
    """
    result = build_preprocessing_artifact(
        df=df,
        artifact_save_path=pipeline_save_path,
        run_verification=False,
        verbose=verbose,
    )
    artifact = result["artifact"]
    return (
        result["X_train_resampled_all"],
        result["y_train_resampled"],
        result["X_test_all"],
        result["y_test"],
        artifact["feature_columns"],
        artifact["pipeline"],
    )


def load_pipeline(path: Path) -> Tuple[ImbPipeline, List[str], Dict[str, Any]]:
    """
    Load the original compatibility view of the preprocessing artifact.
    """
    artifact = load_preprocessing_artifact(path)
    return artifact["pipeline"], artifact["feature_columns"], artifact.get("meta", {})
