import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from tabulate import tabulate
from tqdm import tqdm
from xgboost import XGBClassifier

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = SRC_DIR / "models"
RESULTS_DIR = SRC_DIR / "results"
PLOTS_DIR = SRC_DIR / "plots"

DATA_PATH = DATA_DIR / "promise_nasa_combined_clean.csv"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"

TARGET_COL = "label"
DATASET_COL = "dataset"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TRAIN_N_JOBS = 9
RANDOM_SEARCH_ITERS = 40

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)

    required_cols = {TARGET_COL, DATASET_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    log.info("Loaded dataset: %s | shape=%s", path, df.shape)
    log.info("Dataset counts: %s", df[DATASET_COL].value_counts().to_dict())
    return df


def load_pipeline_artifact(path: Path) -> Tuple[Any, List[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(
            "Pipeline artifact must be a dict containing 'pipeline' and 'feature_columns'."
        )

    if "pipeline" not in artifact or "feature_columns" not in artifact:
        raise KeyError(
            "Pipeline artifact must contain keys: 'pipeline' and 'feature_columns'."
        )

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    meta = artifact.get("meta", {})

    if not hasattr(pipeline, "steps"):
        raise TypeError(
            "Loaded pipeline does not appear to be a sklearn-style pipeline."
        )

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("'feature_columns' must be a non-empty list.")

    log.info("Loaded pipeline: steps=%s", [s[0] for s in pipeline.steps])
    log.info("Feature columns (%d): %s", len(feature_columns), feature_columns)
    return pipeline, feature_columns, meta


def get_dataset_xy(
    df: pd.DataFrame,
    dataset_name: str,
    feature_columns: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    subset = df[df[DATASET_COL] == dataset_name]
    if subset.empty:
        log.warning("Dataset '%s' not found in CSV — skipping.", dataset_name)
        return None

    missing_features = [c for c in feature_columns if c not in subset.columns]
    if missing_features:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing feature columns: {missing_features}"
        )

    X = subset[feature_columns].values.astype(float)
    y = subset[TARGET_COL].values.astype(int)
    defect_rate = 100 * y.mean() if len(y) else 0.0
    log.info(
        "[%s] n=%d  defective=%d (%.1f%%)", dataset_name, len(y), y.sum(), defect_rate
    )
    return X, y


def apply_preprocessing(pipeline: Any, X: np.ndarray) -> np.ndarray:
    if len(pipeline.steps) < 2:
        raise ValueError(
            "Pipeline must contain preprocessing step(s) and a final estimator step."
        )
    return pipeline[:-1].transform(X)


def build_models(pos_weight: float) -> Dict[str, Any]:
    return {
        "LogReg": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "RF": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=TRAIN_N_JOBS,
            class_weight="balanced_subsample",
        ),
        "XGB": XGBClassifier(
            n_estimators=400,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            scale_pos_weight=pos_weight,
            n_jobs=TRAIN_N_JOBS,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced",
            n_jobs=TRAIN_N_JOBS,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=TRAIN_N_JOBS,
            class_weight="balanced_subsample",
        ),
        "BalancedRF": BalancedRandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=TRAIN_N_JOBS,
            replacement=True,
        ),
    }


def build_param_distributions() -> Dict[str, Dict[str, List[Any]]]:
    return {
        "LogReg": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [500, 1000],
        },
        "RF": {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 8, 12, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
            "bootstrap": [True],
        },
        "XGB": {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [1, 2, 5, 10],
        },
        "LGBM": {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 5, 8, 12],
            "min_child_samples": [5, 10, 20, 40],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0, 5.0],
        },
        "ExtraTrees": {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 8, 12, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "BalancedRF": {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 8, 12, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "sampling_strategy": ["all", "auto", "not minority"],
        },
    }


def with_fold_safe_smote(
    name: str, model: Any, params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    use_smote = {"LogReg"}
    if name not in use_smote:
        return model, params

    wrapped = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", model),
        ]
    )
    wrapped_params = {f"clf__{k}": v for k, v in params.items()}
    return wrapped, wrapped_params


def normalize_param_names(best_params: Dict[str, Any]) -> Dict[str, Any]:
    return {key.replace("clf__", ""): value for key, value in best_params.items()}


def get_positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return model.predict(X).astype(float)


def safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def tune_and_cv(
    name: str,
    model: Any,
    param_dist: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    skf: StratifiedKFold,
    scoring: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    log.info("  [%s] RandomizedSearchCV (n_iter=%d) …", name, RANDOM_SEARCH_ITERS)
    search_model, search_params = with_fold_safe_smote(name, model, param_dist)

    rscv = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=search_params,
        n_iter=RANDOM_SEARCH_ITERS,
        cv=skf,
        scoring=scoring,
        refit="mcc",
        random_state=RANDOM_STATE,
        n_jobs=TRAIN_N_JOBS,
        error_score="raise",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rscv.fit(X_train, y_train)

    best = rscv.best_estimator_
    log.info("  [%s] Best params: %s", name, normalize_param_names(rscv.best_params_))
    return best, rscv.best_params_


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    uniq = np.unique(y_score)
    if uniq.size > 2000:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, 1000)))
    else:
        thresholds = uniq
    best_threshold = float(np.median(y_score))
    best_mcc = -1.0
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        if (mcc > best_mcc) or (np.isclose(mcc, best_mcc) and f1 > best_f1):
            best_threshold = float(threshold)
            best_mcc = float(mcc)
            best_f1 = float(f1)

    return best_threshold


def calibrate_threshold(model: Any, X_calib: np.ndarray, y_calib: np.ndarray) -> float:
    if np.unique(y_calib).size < 2:
        return 0.5
    y_scores = get_positive_scores(model, X_calib)
    return find_optimal_threshold(y_calib, y_scores)


def evaluate_on_test(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_score = get_positive_scores(model, X_test)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold": threshold,
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": safe_roc_auc_score(y_test, y_score),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def evaluate_cross_dataset(
    name: str,
    model: Any,
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_score = get_positive_scores(model, X_ext)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold": threshold,
        "F1": f1_score(y_ext, y_pred, zero_division=0),
        "Precision": precision_score(y_ext, y_pred, zero_division=0),
        "Recall": recall_score(y_ext, y_pred, zero_division=0),
        "AUC-ROC": safe_roc_auc_score(y_ext, y_score),
        "MCC": matthews_corrcoef(y_ext, y_pred),
    }


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


def plot_cv_bar(
    cv_df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(cv_df)))

    bars = ax.bar(
        cv_df["Model"],
        cv_df[mean_col],
        yerr=cv_df[std_col],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    std_max = float(np.nanmax(cv_df[std_col].values)) if len(cv_df) else 0.0
    for bar, val in zip(bars, cv_df[mean_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_max * 0.05 + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")

    y_min = max(0.0, cv_df[mean_col].min() - std_max - 0.05)
    y_max = min(1.05, cv_df[mean_col].max() + std_max + 0.08)
    ax.set_ylim(y_min, y_max)

    _savefig(out_path)


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        y_score = get_positive_scores(model, X_test)
        if np.unique(y_test).size < 2:
            log.warning(
                "Skipping ROC curve for '%s': test labels contain only one class.", name
            )
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.6)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Held-Out Test Set", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    _savefig(out_path)


def plot_pr_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        y_score = get_positive_scores(model, X_test)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=1.6)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Precision-Recall Curves — Held-Out Test Set", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")
    _savefig(out_path)


def plot_confusion_matrices(
    models_dict: Dict[str, Any],
    thresholds: Dict[str, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    n = len(models_dict)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.2))
    axes = np.atleast_1d(axes).flatten()

    idx = -1
    for idx, (name, model) in enumerate(models_dict.items()):
        threshold = thresholds.get(name, 0.5)
        y_score = get_positive_scores(model, X_test)
        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Clean", "Defective"],
        )
        disp.plot(ax=axes[idx], colorbar=False, cmap="Blues", values_format=".2f")
        axes[idx].set_title(
            f"{name}\nthr={threshold:.3f}", fontsize=9, fontweight="bold"
        )

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Normalised Confusion Matrices — Thresholded Test Predictions",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    _savefig(out_path)


def plot_cross_dataset_heatmap(
    test_df: pd.DataFrame,
    cross_dfs: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    base = test_df.set_index("Model")[["F1", "MCC"]].rename(
        columns={"F1": "KC1_F1", "MCC": "KC1_MCC"}
    )

    for ds_name, ds_df in cross_dfs.items():
        tmp = ds_df.set_index("Model")[["F1", "MCC"]].rename(
            columns={"F1": f"{ds_name}_F1", "MCC": f"{ds_name}_MCC"}
        )
        base = base.join(tmp, how="left")

    n_cols = len(base.columns)
    fig, ax = plt.subplots(
        figsize=(max(7, n_cols * 1.6), max(4, len(base) * 0.75 + 1.5))
    )
    sns.heatmap(
        base.astype(float),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws={"size": 9},
    )
    ax.set_title("Cross-Dataset Generalisation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Model")
    _savefig(out_path)


def plot_feature_importance(
    model: Any,
    feature_columns: List[str],
    model_name: str,
    out_path: Path,
    top_k: int = 15,
) -> None:
    estimator = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        estimator = model.named_steps["clf"]

    if not hasattr(estimator, "feature_importances_"):
        log.warning(
            "Skipping feature importance: '%s' has no feature_importances_.", model_name
        )
        return

    importances = np.asarray(estimator.feature_importances_)
    if importances.shape[0] != len(feature_columns):
        log.warning(
            "Skipping feature importance for '%s': number of importances (%d) does not match feature columns (%d).",
            model_name,
            importances.shape[0],
            len(feature_columns),
        )
        return

    imp_series = pd.Series(importances, index=feature_columns).sort_values(
        ascending=True
    )
    top_imp = imp_series.tail(min(top_k, len(imp_series)))

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_imp) * 0.5 + 1)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_imp)))
    top_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(
        f"Top-{len(top_imp)} Feature Importances ({model_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Importance")
    _savefig(out_path)


def main() -> None:
    warnings.filterwarnings("ignore")

    for d in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("STEP 1 — Loading data and pipeline artifact")
    df = load_data(DATA_PATH)
    pipeline, feature_columns, _ = load_pipeline_artifact(PIPELINE_PATH)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input dataset is missing required feature columns: {missing_features}"
        )

    log.info("STEP 2 — Preparing pooled dataset (all NASA subsets combined)")
    X_all_raw = df[feature_columns].values.astype(float)
    y_all = df[TARGET_COL].values.astype(int)
    X_all = apply_preprocessing(pipeline, X_all_raw)
    defect_rate = 100 * y_all.mean() if len(y_all) else 0.0
    log.info(
        "Pooled dataset shape=%s | defective=%d (%.1f%%)",
        X_all.shape,
        y_all.sum(),
        defect_rate,
    )

    log.info("STEP 3 — Global stratified train/validation/test split")
    holdout_size = TEST_SIZE + VALIDATION_SIZE
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_all,
        y_all,
        test_size=holdout_size,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    test_fraction_in_holdout = TEST_SIZE / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_in_holdout,
        stratify=y_holdout,
        random_state=RANDOM_STATE,
    )
    log.info(
        "Train=%s  Validation=%s  Test=%s", X_train.shape, X_val.shape, X_test.shape
    )

    class_balance = pd.Series(y_train).value_counts().sort_index().to_dict()
    neg_count = int(class_balance.get(0, 0))
    pos_count = int(class_balance.get(1, 1))
    pos_weight = float(neg_count / max(pos_count, 1))
    log.info(
        "STEP 4 — Class imbalance handling | train balance=%s | pos_weight=%.3f",
        class_balance,
        pos_weight,
    )

    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    report_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    )
    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "mcc": make_scorer(matthews_corrcoef),
    }

    models = build_models(pos_weight=pos_weight)
    param_dists = build_param_distributions()

    trained_models: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}
    cv_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    hyperparam_rows: List[Dict[str, Any]] = []

    log.info(
        "STEP 5 — Hyperparameter tuning + repeated CV + threshold calibration (%d models)",
        len(models),
    )

    for name in tqdm(list(models.keys()), desc="Models", ncols=70):
        log.info("\nModel: %s", name)

        best_model, best_params = tune_and_cv(
            name=name,
            model=models[name],
            param_dist=param_dists[name],
            X_train=X_train,
            y_train=y_train,
            skf=tune_cv,
            scoring=scoring,
        )

        cv_scores = cross_validate(
            best_model,
            X_train,
            y_train,
            cv=report_cv,
            scoring=scoring,
            n_jobs=TRAIN_N_JOBS,
            error_score="raise",
        )

        hyperparam_rows.append({"Model": name, **normalize_param_names(best_params)})

        cv_row: Dict[str, Any] = {"Model": name}
        for metric in ("f1", "roc_auc", "precision", "recall", "mcc"):
            vals = cv_scores[f"test_{metric}"]
            cv_row[f"{metric}_mean"] = float(np.mean(vals))
            cv_row[f"{metric}_std"] = float(np.std(vals))
        cv_rows.append(cv_row)

        log.info(
            "  [%s] CV F1=%.4f±%.4f  MCC=%.4f±%.4f",
            name,
            cv_row["f1_mean"],
            cv_row["f1_std"],
            cv_row["mcc_mean"],
            cv_row["mcc_std"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model.fit(X_train, y_train)

        best_threshold = calibrate_threshold(best_model, X_val, y_val)
        thresholds[name] = best_threshold
        log.info("  [%s] Calibrated threshold=%.3f", name, best_threshold)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(best_model, model_path)
        trained_models[name] = best_model
        log.info("  [%s] Saved → %s", name, model_path)

        test_metrics = evaluate_on_test(
            name, best_model, X_test, y_test, best_threshold
        )
        test_rows.append(test_metrics)
        log.info(
            "  [%s] Global test → F1=%.4f  MCC=%.4f",
            name,
            test_metrics["F1"],
            test_metrics["MCC"],
        )

    log.info("STEP 6 — Saving result CSVs")
    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    hp_df = pd.DataFrame(hyperparam_rows)

    cv_df.to_csv(RESULTS_DIR / "cv_results_overall.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "test_results_overall.csv", index=False)
    hp_df.to_csv(RESULTS_DIR / "hyperparam_log_overall.csv", index=False)

    log.info("STEP 7 — Ranking models")
    ranking_df = (
        cv_df[["Model", "mcc_mean", "f1_mean"]]
        .merge(test_df, on="Model", how="left")
        .sort_values(["mcc_mean", "f1_mean", "MCC", "F1"], ascending=False)
        .reset_index(drop=True)
    )
    ranking_df.index += 1
    ranking_df.to_csv(
        RESULTS_DIR / "model_ranking_overall.csv", index=True, index_label="Rank"
    )

    best_row = ranking_df.iloc[0]
    best_name = best_row["Model"]
    log.info(
        "Best model: %s | CV MCC=%.4f | Test F1=%.4f | Test MCC=%.4f",
        best_name,
        best_row["mcc_mean"],
        best_row["F1"],
        best_row["MCC"],
    )

    log.info("STEP 8 — Generating plots")
    plot_cv_bar(
        cv_df,
        "f1_mean",
        "f1_std",
        "Cross-Validation F1 Score — Combined NASA (5x3 Repeated Stratified CV)",
        "Mean F1",
        PLOTS_DIR / "cv_f1_comparison_overall.png",
    )
    plot_cv_bar(
        cv_df,
        "mcc_mean",
        "mcc_std",
        "Cross-Validation MCC — Combined NASA (5x3 Repeated Stratified CV)",
        "Mean MCC",
        PLOTS_DIR / "cv_mcc_comparison_overall.png",
    )
    plot_roc_curves(
        trained_models, X_test, y_test, PLOTS_DIR / "roc_curves_overall.png"
    )
    plot_pr_curves(trained_models, X_test, y_test, PLOTS_DIR / "pr_curves_overall.png")
    plot_confusion_matrices(
        trained_models,
        thresholds,
        X_test,
        y_test,
        PLOTS_DIR / "confusion_matrices_overall.png",
    )

    tree_candidates = ("RF", "XGB", "LGBM", "ExtraTrees", "BalancedRF")
    tree_name = (
        best_name
        if best_name in tree_candidates
        else next(
            (c for c in tree_candidates if c in trained_models),
            None,
        )
    )

    if tree_name is not None:
        plot_feature_importance(
            trained_models[tree_name],
            feature_columns,
            tree_name,
            PLOTS_DIR / "feature_importance.png",
        )

    sep = "=" * 70
    print(f"\n{sep}\nGLOBAL HELD-OUT TEST SET RESULTS (COMBINED NASA DATASET)\n{sep}")
    print(
        tabulate(test_df.round(4), headers="keys", tablefmt="pretty", showindex=False)
    )

    print(
        f"\n{sep}\nCROSS-VALIDATION RESULTS (mean ± std, 5x3 repeated stratified, pooled training)\n{sep}"
    )
    cv_display = cv_df[
        [
            "Model",
            "f1_mean",
            "f1_std",
            "mcc_mean",
            "mcc_std",
            "roc_auc_mean",
            "precision_mean",
            "recall_mean",
        ]
    ].round(4)
    print(tabulate(cv_display, headers="keys", tablefmt="pretty", showindex=False))

    print(f"\n{sep}")
    print(f"  BEST MODEL : {best_name}")
    print(f"  F1         : {best_row['F1']:.4f}")
    print(f"  MCC        : {best_row['MCC']:.4f}")
    print(sep)
    print(f"\n  Models   → {MODELS_DIR}")
    print(f"  Results  → {RESULTS_DIR}")
    print(f"  Plots    → {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()

