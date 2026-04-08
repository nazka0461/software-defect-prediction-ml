import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from tabulate import tabulate
from tqdm import tqdm

import src.pipeline as combined_pipeline
from src.commit_metrics import coverage_for_features
from src.data.feature_selection import (
    export_selected_features_json,
    plot_feature_selection_summary,
)
from src.data.load_promise_nasa import TARGET_COL, load_nasa_datasets
from src.data.preprocessing import build_preprocessing_artifact
from src.serving import extract_estimator

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = SRC_DIR / "models" / "individual"
DEPLOYMENT_DIR = SRC_DIR / "models" / "deployment"
RESULTS_DIR = SRC_DIR / "results" / "individual"
PLOTS_DIR = SRC_DIR / "plots" / "individual"
XAI_DIR = PROJECT_ROOT / "figures" / "xai"

BASELINE_TEST_RESULTS = SRC_DIR / "results" / "test_results_overall.csv"
BASELINE_CV_RESULTS = SRC_DIR / "results" / "cv_results_overall.csv"
BASELINE_RANKING = SRC_DIR / "results" / "model_ranking_overall.csv"

RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


def _safe_cv_splits(y: np.ndarray, desired_splits: int) -> int:
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 2
    return max(2, min(desired_splits, int(counts.min())))


def _resample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    counts = pd.Series(y_train).value_counts()
    minority_count = int(counts.min()) if not counts.empty else 0
    if minority_count <= 1:
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled, "RandomOverSampler"

    sampler = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=max(1, min(5, minority_count - 1)),
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled, "SMOTE"


def _serialize_global_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, float]]:
    estimator = extract_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_[0], dtype=float)
    else:
        return []

    if values.shape[0] != len(feature_names):
        return []

    ranking = pd.Series(values, index=feature_names).sort_values(
        key=np.abs, ascending=False
    )
    return [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in ranking.head(top_n).items()
    ]


def _plot_metric_heatmap(test_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    heatmap_df = test_df.pivot(index="Dataset", columns="Model", values=metric)
    fig, ax = plt.subplots(
        figsize=(max(8, heatmap_df.shape[1] * 1.4), max(5, heatmap_df.shape[0] * 0.65))
    )
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(f"Held-Out {metric} Across Individual PROMISE Datasets", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    _savefig(out_path)


def _plot_delta_vs_combined(comparison_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        comparison_df.groupby("Model", as_index=False)[["Delta_F1", "Delta_MCC", "Delta_AUC_ROC"]]
        .mean()
        .sort_values("Delta_MCC", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    metrics = [
        ("Delta_F1", "Mean F1 delta vs combined baseline"),
        ("Delta_MCC", "Mean MCC delta vs combined baseline"),
        ("Delta_AUC_ROC", "Mean AUC-ROC delta vs combined baseline"),
    ]

    for ax, (column, title) in zip(axes, metrics):
        colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in summary[column]]
        ax.barh(summary["Model"], summary[column], color=colors, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Delta")

    _savefig(out_path)


def _plot_best_model_per_dataset(dataset_winners: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=dataset_winners.sort_values("MCC", ascending=False),
        x="Dataset",
        y="MCC",
        hue="Model",
        dodge=False,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("Best Model Per Dataset (Held-Out MCC)", fontweight="bold")
    ax.set_ylabel("MCC")
    ax.set_xlabel("Dataset")
    ax.legend(title="Winning model", bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig(out_path)


def _plot_average_rank(model_family_ranking: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        model_family_ranking["Model"],
        model_family_ranking["mean_rank"],
        color="#457b9d",
        edgecolor="black",
    )
    ax.invert_yaxis()
    ax.set_title("Average Rank Across 12 Individual Datasets", fontweight="bold")
    ax.set_ylabel("Mean dataset rank (lower is better)")
    ax.set_xlabel("Model")
    _savefig(out_path)


def _select_dataset_winners(test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked = test_df.sort_values(
        ["Dataset", "MCC", "F1", "AUC-ROC"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    ranked["DatasetRank"] = ranked.groupby("Dataset").cumcount() + 1

    winners = (
        ranked[ranked["DatasetRank"] == 1]
        .sort_values(["MCC", "F1", "AUC-ROC"], ascending=False)
        .reset_index(drop=True)
    )
    return winners, ranked


def _choose_deployable_winner(
    test_df: pd.DataFrame,
    dataset_cache: Dict[str, Dict[str, Any]],
) -> pd.Series:
    candidate_rows: List[Dict[str, Any]] = []

    for row in test_df.to_dict(orient="records"):
        artifact = dataset_cache[row["Dataset"]]["artifact"]
        coverage = coverage_for_features(artifact["selected_features"])
        candidate_rows.append(
            {
                **row,
                "supported_feature_count": len(coverage["supported"]),
                "missing_feature_count": len(coverage["missing"]),
                "supported_features": coverage["supported"],
                "missing_features": coverage["missing"],
                "strictly_deployable": int(len(coverage["missing"]) == 0),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        [
            "strictly_deployable",
            "missing_feature_count",
            "MCC",
            "F1",
            "AUC-ROC",
        ],
        ascending=[False, True, False, False, False],
    )
    candidate_df.to_csv(RESULTS_DIR / "deployable_candidates.csv", index=False)
    return candidate_df.iloc[0]


def _build_deployment_bundle(
    deploy_row: pd.Series,
    dataset_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    dataset_name = str(deploy_row["Dataset"])
    model_name = str(deploy_row["Model"])
    cached = dataset_cache[dataset_name]
    artifact = cached["artifact"]
    model = cached["trained_models"][model_name]
    threshold = float(cached["thresholds"][model_name])

    bundle = {
        "bundle_version": 1,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model": model,
        "threshold": threshold,
        "preprocessing_artifact": artifact,
        "raw_feature_order": artifact["raw_numeric_features"],
        "selected_feature_order": artifact["selected_features"],
        "global_feature_importance": _serialize_global_importance(
            model,
            artifact["selected_features"],
        ),
        "benchmark_metrics": {
            "F1": float(deploy_row["F1"]),
            "MCC": float(deploy_row["MCC"]),
            "AUC-ROC": float(deploy_row["AUC-ROC"]),
        },
        "strictly_deployable": bool(deploy_row.get("strictly_deployable", 0)),
        "supported_features": deploy_row.get("supported_features", []),
        "missing_features": deploy_row.get("missing_features", []),
    }

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = DEPLOYMENT_DIR / "best_model_bundle.pkl"
    joblib.dump(bundle, bundle_path)
    (DEPLOYMENT_DIR / "best_model_bundle.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "threshold": threshold,
                "benchmark_metrics": bundle["benchmark_metrics"],
                "strictly_deployable": bundle["strictly_deployable"],
                "supported_features": bundle["supported_features"],
                "missing_features": bundle["missing_features"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Saved deployment bundle: %s", bundle_path)
    return bundle


def _run_shap_and_lime(
    bundle: Dict[str, Any],
    dataset_cache: Dict[str, Dict[str, Any]],
) -> None:
    try:
        import lime.lime_tabular
        import shap
    except Exception as exc:
        log.warning("Skipping XAI generation because SHAP/LIME is unavailable: %s", exc)
        return

    dataset_name = bundle["dataset_name"]
    model_name = bundle["model_name"]
    cache = dataset_cache[dataset_name]
    estimator = cache["trained_models"][model_name]
    X_train = cache["prep"]["X_train_selected"]
    X_test = cache["prep"]["X_test_selected"]
    y_test = cache["prep"]["y_test"]
    feature_names = bundle["selected_feature_order"]

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    XAI_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(extract_estimator(estimator), X_train_df)
        shap_explanation = explainer(X_test_df)

    shap_values = np.asarray(shap_explanation.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap.summary_plot(
        shap_values,
        X_test_df,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    _savefig(XAI_DIR / "shap_summary_bar.png")

    shap.summary_plot(
        shap_values,
        X_test_df,
        feature_names=feature_names,
        show=False,
    )
    _savefig(XAI_DIR / "shap_beeswarm.png")

    top_indices = np.argsort(combined_pipeline.get_positive_scores(estimator, X_test))[-3:][::-1]
    base_values = np.asarray(shap_explanation.base_values)
    for rank, sample_index in enumerate(top_indices, start=1):
        if base_values.ndim == 0:
            sample_base_value = float(base_values)
        elif base_values.ndim == 1:
            sample_base_value = float(base_values[sample_index])
        else:
            sample_base_value = float(base_values[sample_index, -1])
        waterfall_exp = shap.Explanation(
            values=shap_values[sample_index],
            base_values=sample_base_value,
            data=X_test_df.iloc[sample_index],
            feature_names=feature_names,
        )
        shap.waterfall_plot(waterfall_exp, show=False)
        _savefig(XAI_DIR / f"shap_waterfall_{rank}.png")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=float),
        feature_names=feature_names,
        class_names=["CLEAN", "DEFECT-PRONE"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    for rank, sample_index in enumerate(top_indices, start=1):
        explanation = lime_explainer.explain_instance(
            np.asarray(X_test[sample_index], dtype=float),
            estimator.predict_proba,
            num_features=min(10, len(feature_names)),
        )
        explanation.save_to_file(str(XAI_DIR / f"lime_case_{rank}.html"))

    pd.DataFrame(
        {
            "sample_index": top_indices,
            "predicted_probability": combined_pipeline.get_positive_scores(estimator, X_test)[top_indices],
            "actual_label": y_test[top_indices],
        }
    ).to_csv(XAI_DIR / "high_risk_cases.csv", index=False)


def _run_ablation_study(
    bundle: Dict[str, Any],
    dataset_cache: Dict[str, Dict[str, Any]],
) -> None:
    dataset_name = bundle["dataset_name"]
    model_name = bundle["model_name"]
    cache = dataset_cache[dataset_name]

    y_train = cache["prep"]["y_train"]
    y_val = cache["prep"]["y_val"]
    y_test = cache["prep"]["y_test"]

    artifact = cache["artifact"]
    class_balance = pd.Series(y_train).value_counts().sort_index().to_dict()
    neg_count = int(class_balance.get(0, 0))
    pos_count = int(class_balance.get(1, 1))
    pos_weight = float(neg_count / max(pos_count, 1))

    base_estimator = combined_pipeline.build_models(pos_weight=pos_weight)[model_name]
    best_params = cache["best_params"][model_name]
    base_estimator.set_params(**best_params)

    rows = []

    def evaluate_variant(
        comparison_name: str,
        variant_name: str,
        X_train_variant: np.ndarray,
        X_val_variant: np.ndarray,
        X_test_variant: np.ndarray,
        apply_smote: bool,
    ) -> None:
        model = clone(base_estimator)
        fit_X = X_train_variant
        fit_y = y_train
        sampler_name = "none"
        if apply_smote:
            fit_X, fit_y, sampler_name = _resample_training_data(X_train_variant, y_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(fit_X, fit_y)
        threshold = combined_pipeline.calibrate_threshold(model, X_val_variant, y_val)
        metrics = combined_pipeline.evaluate_on_test(
            model_name,
            model,
            X_test_variant,
            y_test,
            threshold,
        )
        rows.append(
            {
                "comparison": comparison_name,
                "variant": variant_name,
                "sampler": sampler_name if apply_smote else "none",
                **metrics,
            }
        )

    evaluate_variant(
        "selected_vs_all",
        "selected_features",
        cache["prep"]["X_train_selected"],
        cache["prep"]["X_val_selected"],
        cache["prep"]["X_test_selected"],
        apply_smote=False,
    )
    evaluate_variant(
        "selected_vs_all",
        "all_post_filter_features",
        cache["prep"]["X_train_all"],
        cache["prep"]["X_val_all"],
        cache["prep"]["X_test_all"],
        apply_smote=False,
    )
    evaluate_variant(
        "smote_vs_no_smote",
        "selected_features_no_smote",
        cache["prep"]["X_train_selected"],
        cache["prep"]["X_val_selected"],
        cache["prep"]["X_test_selected"],
        apply_smote=False,
    )
    evaluate_variant(
        "smote_vs_no_smote",
        "selected_features_with_smote",
        cache["prep"]["X_train_selected"],
        cache["prep"]["X_val_selected"],
        cache["prep"]["X_test_selected"],
        apply_smote=True,
    )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation_study.csv", index=False)


def _run_wilcoxon(model_family_ranking: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if len(model_family_ranking) < 2:
        return

    top_two = model_family_ranking.head(2)["Model"].tolist()
    left = (
        test_df[test_df["Model"] == top_two[0]][["Dataset", "MCC"]]
        .rename(columns={"MCC": top_two[0]})
    )
    right = (
        test_df[test_df["Model"] == top_two[1]][["Dataset", "MCC"]]
        .rename(columns={"MCC": top_two[1]})
    )
    merged = left.merge(right, on="Dataset", how="inner").sort_values("Dataset")

    if len(merged) < 2:
        return

    try:
        statistic, pvalue = wilcoxon(merged[top_two[0]], merged[top_two[1]])
    except Exception as exc:
        log.warning("Could not run Wilcoxon test: %s", exc)
        return

    pd.DataFrame(
        [
            {
                "model_left": top_two[0],
                "model_right": top_two[1],
                "statistic": statistic,
                "pvalue": pvalue,
            }
        ]
    ).to_csv(RESULTS_DIR / "wilcoxon_top_two_models.csv", index=False)


def _load_baseline_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not BASELINE_TEST_RESULTS.exists() or not BASELINE_CV_RESULTS.exists() or not BASELINE_RANKING.exists():
        raise FileNotFoundError("Combined baseline CSVs are required for comparison but were not found.")

    return (
        pd.read_csv(BASELINE_TEST_RESULTS),
        pd.read_csv(BASELINE_CV_RESULTS),
        pd.read_csv(BASELINE_RANKING),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train individual PROMISE dataset benchmarks.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--baseline", default="existing")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--random-search-iters", type=int, default=40)
    parser.add_argument("--train-n-jobs", type=int, default=9)
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--skip-deployment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_pipeline.RANDOM_SEARCH_ITERS = args.random_search_iters
    combined_pipeline.TRAIN_N_JOBS = args.train_n_jobs

    baseline_test_df, baseline_cv_df, baseline_ranking_df = _load_baseline_frames()
    datasets = load_nasa_datasets(data_dir=args.data_dir, mode="individual")
    if args.datasets:
        wanted = {name.upper() for name in args.datasets}
        datasets = {name: df for name, df in datasets.items() if name in wanted}

    log.info("Loaded %d individual datasets.", len(datasets))

    test_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    hyperparam_rows: List[Dict[str, Any]] = []
    dataset_cache: Dict[str, Dict[str, Any]] = {}

    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "mcc": make_scorer(matthews_corrcoef),
    }

    for dataset_name, df in tqdm(sorted(datasets.items()), desc="Datasets", ncols=80):
        dataset_dir = MODELS_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = PLOTS_DIR / dataset_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        log.info("Preparing dataset bundle for %s", dataset_name)
        prep = build_preprocessing_artifact(
            df=df,
            dataset_name=dataset_name,
            artifact_save_path=dataset_dir / "preprocessing_bundle.pkl",
            top_k=args.top_k,
            verbose=False,
        )
        artifact = prep["artifact"]
        export_selected_features_json(
            artifact["selected_features"],
            dataset_dir / "selected_features.json",
        )
        plot_feature_selection_summary(
            artifact["meta"]["selection"],
            plot_dir / "feature_selection_summary.png",
        )

        class_balance = pd.Series(prep["y_train"]).value_counts().sort_index().to_dict()
        neg_count = int(class_balance.get(0, 0))
        pos_count = int(class_balance.get(1, 1))
        pos_weight = float(neg_count / max(pos_count, 1))

        tune_splits = _safe_cv_splits(prep["y_train"], 5)
        report_splits = _safe_cv_splits(prep["y_train"], 5)
        tune_cv = StratifiedKFold(n_splits=tune_splits, shuffle=True, random_state=RANDOM_STATE)
        report_cv = RepeatedStratifiedKFold(
            n_splits=report_splits,
            n_repeats=3,
            random_state=RANDOM_STATE,
        )

        models = combined_pipeline.build_models(pos_weight=pos_weight)
        param_dists = combined_pipeline.build_param_distributions()
        trained_models: Dict[str, Any] = {}
        thresholds: Dict[str, float] = {}
        best_params_per_model: Dict[str, Dict[str, Any]] = {}

        for model_name in models:
            log.info("[%s] Training %s", dataset_name, model_name)
            best_model, best_params = combined_pipeline.tune_and_cv(
                name=model_name,
                model=models[model_name],
                param_dist=param_dists[model_name],
                X_train=prep["X_train_selected"],
                y_train=prep["y_train"],
                skf=tune_cv,
                scoring=scoring,
            )

            cv_scores = cross_validate(
                best_model,
                prep["X_train_selected"],
                prep["y_train"],
                cv=report_cv,
                scoring=scoring,
                n_jobs=args.train_n_jobs,
                error_score="raise",
            )

            normalized_best_params = combined_pipeline.normalize_param_names(best_params)
            best_params_per_model[model_name] = normalized_best_params
            hyperparam_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    **normalized_best_params,
                }
            )

            cv_row: Dict[str, Any] = {
                "Dataset": dataset_name,
                "Model": model_name,
                "raw_feature_count": len(artifact["raw_numeric_features"]),
                "post_filter_feature_count": len(artifact["post_filter_features"]),
                "selected_feature_count": len(artifact["selected_features"]),
            }
            for metric in ("f1", "roc_auc", "precision", "recall", "mcc"):
                vals = cv_scores[f"test_{metric}"]
                cv_row[f"{metric}_mean"] = float(np.mean(vals))
                cv_row[f"{metric}_std"] = float(np.std(vals))
            cv_rows.append(cv_row)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_model.fit(prep["X_train_selected"], prep["y_train"])

            threshold = combined_pipeline.calibrate_threshold(
                best_model,
                prep["X_val_selected"],
                prep["y_val"],
            )
            thresholds[model_name] = threshold
            trained_models[model_name] = best_model

            joblib.dump(best_model, dataset_dir / f"{model_name}.pkl")

            test_metrics = combined_pipeline.evaluate_on_test(
                model_name,
                best_model,
                prep["X_test_selected"],
                prep["y_test"],
                threshold,
            )
            test_rows.append(
                {
                    "Dataset": dataset_name,
                    "raw_feature_count": len(artifact["raw_numeric_features"]),
                    "post_filter_feature_count": len(artifact["post_filter_features"]),
                    "selected_feature_count": len(artifact["selected_features"]),
                    **test_metrics,
                }
            )

        dataset_test_df = pd.DataFrame(
            [row for row in test_rows if row["Dataset"] == dataset_name]
        ).sort_values(["MCC", "F1", "AUC-ROC"], ascending=False)
        dataset_test_df.to_csv(dataset_dir / "test_results.csv", index=False)

        dataset_cv_df = pd.DataFrame(
            [row for row in cv_rows if row["Dataset"] == dataset_name]
        ).sort_values(["mcc_mean", "f1_mean"], ascending=False)
        dataset_cv_df.to_csv(dataset_dir / "cv_results.csv", index=False)

        tree_candidates = dataset_test_df[dataset_test_df["Model"].isin(["RF", "XGB", "LGBM", "ExtraTrees", "BalancedRF"])]
        if not tree_candidates.empty:
            top_tree_model = str(tree_candidates.iloc[0]["Model"])
            combined_pipeline.plot_feature_importance(
                trained_models[top_tree_model],
                artifact["selected_features"],
                top_tree_model,
                plot_dir / "feature_importance.png",
            )

        dataset_cache[dataset_name] = {
            "artifact": artifact,
            "prep": prep,
            "trained_models": trained_models,
            "thresholds": thresholds,
            "best_params": best_params_per_model,
        }

    test_df = pd.DataFrame(test_rows).sort_values(
        ["Dataset", "MCC", "F1", "AUC-ROC"],
        ascending=[True, False, False, False],
    )
    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["Dataset", "mcc_mean", "f1_mean"],
        ascending=[True, False, False],
    )
    hyperparam_df = pd.DataFrame(hyperparam_rows)

    test_df.to_csv(RESULTS_DIR / "individual_test_results.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "individual_cv_results.csv", index=False)
    hyperparam_df.to_csv(RESULTS_DIR / "individual_hyperparam_log.csv", index=False)

    dataset_winners, ranked_test_df = _select_dataset_winners(test_df)
    dataset_winners.to_csv(RESULTS_DIR / "per_dataset_winners.csv", index=False)

    model_family_ranking = (
        ranked_test_df.groupby("Model", as_index=False)
        .agg(
            mean_rank=("DatasetRank", "mean"),
            mean_MCC=("MCC", "mean"),
            mean_F1=("F1", "mean"),
            mean_AUC_ROC=("AUC-ROC", "mean"),
        )
        .sort_values(["mean_rank", "mean_MCC"], ascending=[True, False])
    )
    model_family_ranking.to_csv(RESULTS_DIR / "model_family_ranking.csv", index=False)

    baseline_compare_rows = []
    for row in test_df.to_dict(orient="records"):
        baseline_row = baseline_test_df[baseline_test_df["Model"] == row["Model"]].iloc[0]
        baseline_compare_rows.append(
            {
                **row,
                "Baseline_F1": float(baseline_row["F1"]),
                "Baseline_MCC": float(baseline_row["MCC"]),
                "Baseline_AUC_ROC": float(baseline_row["AUC-ROC"]),
                "Delta_F1": float(row["F1"] - baseline_row["F1"]),
                "Delta_MCC": float(row["MCC"] - baseline_row["MCC"]),
                "Delta_AUC_ROC": float(row["AUC-ROC"] - baseline_row["AUC-ROC"]),
            }
        )
    comparison_df = pd.DataFrame(baseline_compare_rows)
    comparison_df.to_csv(RESULTS_DIR / "individual_vs_combined_long.csv", index=False)

    comparison_summary = (
        comparison_df.groupby("Model", as_index=False)
        .agg(
            individual_mean_F1=("F1", "mean"),
            individual_mean_MCC=("MCC", "mean"),
            individual_mean_AUC_ROC=("AUC-ROC", "mean"),
            combined_baseline_F1=("Baseline_F1", "first"),
            combined_baseline_MCC=("Baseline_MCC", "first"),
            combined_baseline_AUC_ROC=("Baseline_AUC_ROC", "first"),
            mean_delta_F1=("Delta_F1", "mean"),
            mean_delta_MCC=("Delta_MCC", "mean"),
            mean_delta_AUC_ROC=("Delta_AUC_ROC", "mean"),
        )
        .sort_values("mean_delta_MCC", ascending=False)
    )
    comparison_summary.to_csv(RESULTS_DIR / "combined_vs_individual_summary.csv", index=False)

    _plot_metric_heatmap(test_df, "MCC", PLOTS_DIR / "individual_mcc_heatmap.png")
    _plot_metric_heatmap(test_df, "F1", PLOTS_DIR / "individual_f1_heatmap.png")
    _plot_delta_vs_combined(comparison_df, PLOTS_DIR / "delta_vs_combined.png")
    _plot_best_model_per_dataset(dataset_winners, PLOTS_DIR / "best_model_per_dataset.png")
    _plot_average_rank(model_family_ranking, PLOTS_DIR / "average_model_rank.png")

    deploy_row = _choose_deployable_winner(test_df, dataset_cache)
    deployment_bundle = None
    if not args.skip_deployment:
        deployment_bundle = _build_deployment_bundle(deploy_row, dataset_cache)
        try:
            _run_ablation_study(deployment_bundle, dataset_cache)
        except Exception as exc:
            log.warning("Ablation study failed: %s", exc)
        try:
            _run_wilcoxon(model_family_ranking, test_df)
        except Exception as exc:
            log.warning("Wilcoxon test failed: %s", exc)
        if not args.skip_xai:
            try:
                _run_shap_and_lime(deployment_bundle, dataset_cache)
            except Exception as exc:
                log.warning("XAI generation failed: %s", exc)

    summary_payload = {
        "baseline_mode": args.baseline,
        "dataset_count": len(datasets),
        "overall_model_family_winner": model_family_ranking.iloc[0].to_dict(),
        "deployable_winner": deploy_row.to_dict(),
        "baseline_reference": {
            "test_results": str(BASELINE_TEST_RESULTS),
            "cv_results": str(BASELINE_CV_RESULTS),
            "ranking": str(BASELINE_RANKING),
        },
    }
    (RESULTS_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nINDIVIDUAL DATASET WINNERS")
    print(tabulate(dataset_winners.round(4), headers="keys", tablefmt="pretty", showindex=False))
    print("\nMODEL FAMILY RANKING")
    print(tabulate(model_family_ranking.round(4), headers="keys", tablefmt="pretty", showindex=False))
    print("\nCOMBINED BASELINE COMPARISON SUMMARY")
    print(tabulate(comparison_summary.round(4), headers="keys", tablefmt="pretty", showindex=False))

    if deployment_bundle is not None:
        print(
            f"\nDeployable winner: {deployment_bundle['model_name']} on {deployment_bundle['dataset_name']}"
        )


if __name__ == "__main__":
    main()
