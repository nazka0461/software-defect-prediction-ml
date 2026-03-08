from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET_COL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VIF_THRESHOLD = 10
CORR_THRESHOLD = 0.95


def _compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    X_np = df_features.values
    vif_list = []
    for i, col in enumerate(df_features.columns):
        vif = variance_inflation_factor(X_np, i)
        vif_list.append((col, vif))
    return pd.DataFrame(vif_list, columns=["feature", "vif"])


def _drop_constant_corr_vif(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str], List[Tuple[str, float]]]:
    """Drop constant columns, then |r| > 0.95 correlated, then VIF > 10. Uses train only for decisions."""
    constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    X_train = X_train.drop(columns=constant_cols)
    X_test = X_test.drop(columns=constant_cols)

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_cols = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
    X_train = X_train.drop(columns=high_corr_cols)
    X_test = X_test.drop(columns=high_corr_cols)

    dropped_vif: List[Tuple[str, float]] = []
    while True:
        vif_df = _compute_vif(X_train)
        max_vif = vif_df["vif"].max()
        if max_vif <= VIF_THRESHOLD:
            break
        feature_to_drop = vif_df.sort_values("vif", ascending=False).iloc[0]["feature"]
        dropped_vif.append((feature_to_drop, float(max_vif)))
        X_train = X_train.drop(columns=[feature_to_drop])
        X_test = X_test.drop(columns=[feature_to_drop])

    if verbose:
        print("Features dropped due to VIF > 10 (in order):")
        for name, v in dropped_vif:
            print(f"  {name}: VIF={v:.2f}")
        print("Constant columns dropped:", constant_cols)
        print("Highly correlated columns dropped (|r|>0.95):", high_corr_cols)

    return X_train, X_test, constant_cols, high_corr_cols, X_train.columns.tolist(), dropped_vif


def prepare_and_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """
    Prepare features and target, stratified train/test split, then drop constant/corr/VIF.
    Returns X_train, X_test, y_train, y_test, feature_columns, and meta (drops lists).
    """
    df = df.copy()
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in feature_df.columns if c not in numeric_feature_cols]

    X_all = df[numeric_feature_cols].copy()
    y_all = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    X_train, X_test, constant_cols, high_corr_cols, feature_columns, dropped_vif = _drop_constant_corr_vif(
        X_train, X_test, verbose=verbose
    )

    meta = {
        "non_numeric_cols": non_numeric_cols,
        "constant_cols": constant_cols,
        "high_corr_cols": high_corr_cols,
        "dropped_vif": dropped_vif,
        "feature_columns": feature_columns,
    }
    return X_train, X_test, y_train, y_test, feature_columns, meta


def build_pipeline(random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Pipeline: imputer -> RobustScaler -> SMOTE. SMOTE is applied only during fit_resample (train)."""
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=random_state)),
    ])


def fit_resample_train(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply pipeline (impute, scale, SMOTE) on training data only. Returns resampled X, y."""
    return pipeline.fit_resample(X_train, y_train)


def transform_test(pipeline: ImbPipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Transform test data with imputer + scaler only (no SMOTE)."""
    return pipeline[:-1].transform(X_test)


def run_preprocessing(
    df: pd.DataFrame,
    pipeline_save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], ImbPipeline]:
    """
    Full preprocessing: prepare, split, filter, build pipeline, fit_resample train, transform test.
    Optionally saves pipeline and feature_columns to pipeline_save_path (e.g. models/pipeline.pkl).
    Returns X_train_res, y_train_res, X_test_scaled, y_test, feature_columns, pipeline.
    """
    X_train, X_test, y_train, y_test, feature_columns, meta = prepare_and_split(
        df, verbose=verbose
    )

    pipeline = build_pipeline()
    X_train_res, y_train_res = fit_resample_train(pipeline, X_train, y_train)
    X_test_scaled = transform_test(pipeline, X_test)

    if verbose:
        print("\nFinal selected feature columns:", feature_columns)
        print("Train shape (pre-SMOTE):", X_train.shape, "y:", y_train.shape)
        print("Train shape (post-SMOTE):", X_train_res.shape, "y:", y_train_res.shape)
        print("Test shape:", X_test_scaled.shape, "y:", y_test.shape)
        print("Class balance y_train (post-SMOTE):", pd.Series(y_train_res).value_counts().sort_index().to_dict())

    if pipeline_save_path is not None:
        pipeline_save_path = Path(pipeline_save_path)
        pipeline_save_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "meta": meta,
        }
        joblib.dump(artifact, pipeline_save_path)
        if verbose:
            print(f"Saved pipeline and feature_columns to {pipeline_save_path}")

    return X_train_res, y_train_res, X_test_scaled, y_test, feature_columns, pipeline


def load_pipeline(path: Path) -> Tuple[ImbPipeline, List[str], Dict[str, Any]]:
    """Load artifact from pipeline.pkl; returns pipeline, feature_columns, meta."""
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["feature_columns"], artifact.get("meta", {})
