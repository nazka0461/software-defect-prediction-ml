# Software Defect Prediction Project: Results-Driven Experimental Summary and Reproducibility Guide

This document is the consolidated experimental report for the project. It is written to support the FPR/IPR submission, the thesis implementation chapter, and final reproducibility. It explains what changed across the three experiments, what each experiment achieved, which models performed best under different criteria, how explainability and CI/CD were integrated, and how the final deployed model can be used again.

## Purpose

The work progressed through three major stages:

1. **Experiment 1** established the pooled combined-data baseline.
2. **Experiment 2** refined the evaluation protocol with a proper train/validation/test split on the same combined dataset.
3. **Experiment 3** expanded the study to 12 individual PROMISE datasets and added deployment, CI/CD automation, and explainable AI.

The key thesis-level conclusion is that the "best research model" and the "best deployable model" are not necessarily the same thing. In this project, `ExtraTrees` emerged as the strongest overall benchmark family across the 12-dataset study, while `RF` on `KC1` became the final operational model because it offered strong predictive performance and full compatibility with the CI/CD metric-mapping pipeline.

## Project Timeline Overview

| Experiment | Focus | Dataset scope | Main change introduced | Main outcome |
| --- | --- | --- | --- | --- |
| Experiment 1 | Initial benchmark | Combined cleaned dataset | Pooled training baseline | `ExtraTrees` ranked first overall |
| Experiment 2 | Evaluation refinement | Same combined cleaned dataset | 60/20/20 train/validation/test split | `RF` ranked first overall |
| Experiment 3 | Generalization + deployment | 12 raw ARFF PROMISE datasets | Individual-dataset benchmarking + API + CI/CD + XAI | `ExtraTrees` strongest overall family, `RF` on `KC1` selected for deployment |

## Experiment 1: Combined Dataset Baseline

Experiment 1 corresponds to the first combined-data benchmark on the cleaned pooled PROMISE/NASA dataset.

### Methodology

- Dataset scope: combined dataset only.
- Training data source: `data/processed/promise_nasa_combined_clean.csv`.
- Preprocessing artifact: `src/models/pipeline.pkl`.
- Split strategy: `TEST_SIZE = 0.15`, so the run was effectively an 85/15 train/test setup.
- No dedicated validation split was used.
- Threshold calibration was based on out-of-fold train predictions rather than a separate validation set.
- Models evaluated:
  - `LogReg`
  - `RF`
  - `XGB`
  - `LGBM`
  - `ExtraTrees`
  - `BalancedRF`

### Saved Outputs

- Results:
  - `src/results/cv_results_overall.csv`
  - `src/results/test_results_overall.csv`
  - `src/results/model_ranking_overall.csv`
- Plots:
  - `src/plots/cv_f1_comparison_overall.png`
  - `src/plots/cv_mcc_comparison_overall.png`
  - `src/plots/roc_curves_overall.png`
  - `src/plots/pr_curves_overall.png`
  - `src/plots/confusion_matrices_overall.png`
  - `src/plots/feature_importance.png`

## Experiment 2: Combined Dataset with Explicit 60/20/20 Split

Experiment 2 kept the same combined cleaned dataset and same six-model family, but changed the evaluation protocol.

### What Changed from Experiment 1

- Same combined dataset.
- Same overall model family.
- Same pooled-data training goal.
- New split design:
  - `TEST_SIZE = 0.20`
  - `VALIDATION_SIZE = 0.20`
  - Effective split: 60/20/20 train/validation/test.
- Threshold calibration changed from train-based out-of-fold predictions to a true validation split.

### Saved Outputs

Experiment 2 wrote its results into the same combined result and plot locations:

- `src/results/cv_results_overall.csv`
- `src/results/test_results_overall.csv`
- `src/results/model_ranking_overall.csv`
- `src/plots/cv_f1_comparison_overall.png`
- `src/plots/cv_mcc_comparison_overall.png`
- `src/plots/roc_curves_overall.png`
- `src/plots/pr_curves_overall.png`
- `src/plots/confusion_matrices_overall.png`
- `src/plots/feature_importance.png`

## Quantitative Results Across Experiments 1 and 2

The table below quantifies the exact held-out metrics saved for Experiments 1 and 2.

| Experiment | Model | Threshold | F1 | Precision | Recall | AUC-ROC | MCC | Ranking position |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Experiment 1 | LogReg | 0.4853 | 0.4486 | 0.3811 | 0.5449 | 0.6753 | 0.2580 | 6 |
| Experiment 1 | RF | 0.5011 | 0.4498 | 0.4297 | 0.4719 | 0.6786 | 0.2812 | 2 |
| Experiment 1 | XGB | 0.5485 | 0.4447 | 0.4060 | 0.4916 | 0.6830 | 0.2657 | 4 |
| Experiment 1 | LGBM | 0.5822 | 0.4169 | 0.4333 | 0.4017 | 0.6777 | 0.2565 | 5 |
| Experiment 1 | ExtraTrees | 0.5163 | 0.4354 | 0.4505 | 0.4213 | 0.6908 | 0.2791 | 1 |
| Experiment 1 | BalancedRF | 0.5863 | 0.4152 | 0.4507 | 0.3848 | 0.6799 | 0.2639 | 3 |
| Experiment 2 | LogReg | 0.5062 | 0.4275 | 0.3861 | 0.4789 | 0.6720 | 0.2409 | 4 |
| Experiment 2 | RF | 0.6595 | 0.2684 | 0.5985 | 0.1730 | 0.6884 | 0.2363 | 1 |
| Experiment 2 | XGB | 0.5143 | 0.4441 | 0.3711 | 0.5527 | 0.6869 | 0.2494 | 6 |
| Experiment 2 | LGBM | 0.6789 | 0.3305 | 0.5180 | 0.2426 | 0.6862 | 0.2413 | 5 |
| Experiment 2 | ExtraTrees | 0.5005 | 0.4551 | 0.4610 | 0.4494 | 0.7017 | 0.3003 | 3 |
| Experiment 2 | BalancedRF | 0.6619 | 0.3343 | 0.4918 | 0.2532 | 0.6883 | 0.2318 | 2 |

### Interpretation of the Combined-Data Results

- **Experiment 1 ranking winner**: `ExtraTrees`
- **Experiment 1 highest held-out MCC**: `RF = 0.2812`
- **Experiment 2 ranking winner**: `RF`
- **Experiment 2 highest held-out MCC**: `ExtraTrees = 0.3003`

This difference matters. "Best ranked" and "best held-out MCC" are not identical because the ranking files combined repeated cross-validation behavior with held-out performance. In other words, the ranking rewarded consistency and general training behavior, while the held-out test table shows the strongest single final-set outcome.

## Experiment 3: Individual-Dataset Benchmarking, Deployment, CI/CD, and XAI

Experiment 3 extended the project from pooled benchmarking into a wider research-and-engineering system.

### Expanded Scope

- 12 PROMISE/NASA ARFF datasets were loaded individually from `data/raw`.
- Preprocessing and feature selection were merged into a richer dataset-specific artifact.
- The same six-model family was trained separately on each dataset.
- A unified benchmark suite was produced for all `dataset x model` combinations.
- A deployable winner was selected based on both predictive quality and feature compatibility for commit-level prediction.
- The best deployable model was wrapped in a Flask API.
- A GitHub Actions workflow was added for automated commit-level risk scoring.
- SHAP and LIME were applied to the final deployed model for interpretability.

### Main Saved Outputs

#### Benchmark results

- `src/results/individual/benchmark_summary.json`
- `src/results/individual/individual_test_results.csv`
- `src/results/individual/individual_cv_results.csv`
- `src/results/individual/model_family_ranking.csv`
- `src/results/individual/per_dataset_winners.csv`
- `src/results/individual/combined_vs_individual_summary.csv`
- `src/results/individual/individual_vs_combined_long.csv`
- `src/results/individual/deployable_candidates.csv`
- `src/results/individual/ablation_study.csv`
- `src/results/individual/wilcoxon_top_two_models.csv`

#### Plots

- `src/plots/individual/individual_mcc_heatmap.png`
- `src/plots/individual/individual_f1_heatmap.png`
- `src/plots/individual/delta_vs_combined.png`
- `src/plots/individual/best_model_per_dataset.png`
- `src/plots/individual/average_model_rank.png`

#### Deployment and inference

- API: `src/app.py`
- Serving helpers: `src/serving.py`
- Deployment bundle: `src/models/deployment/best_model_bundle.pkl`
- Deployment summary: `src/models/deployment/best_model_bundle.json`
- Commit predictor: `scripts/predict_commit.py`
- Workflow: `.github/workflows/defect_check.yml`
- Tests: `tests/test_api.py`

#### XAI

- `figures/xai/shap_summary_bar.png`
- `figures/xai/shap_beeswarm.png`
- `figures/xai/shap_waterfall_1.png`
- `figures/xai/shap_waterfall_2.png`
- `figures/xai/shap_waterfall_3.png`
- `figures/xai/lime_case_1.html`
- `figures/xai/lime_case_2.html`
- `figures/xai/lime_case_3.html`
- `figures/xai/high_risk_cases.csv`

## Best Individual Performance by Model Family

For Experiment 3, the "best individual run per model family" is defined as:

1. highest `MCC`
2. tie-break by `F1`
3. tie-break by `AUC-ROC`

Using that rule, the best individual run achieved by each model family is:

| Model | Best dataset | F1 | AUC-ROC | MCC | Selected feature count |
| --- | --- | ---: | ---: | ---: | ---: |
| BalancedRF | PC4 | 0.5783 | 0.9324 | 0.5191 | 15 |
| ExtraTrees | PC4 | 0.6757 | 0.9349 | 0.6300 | 15 |
| LGBM | PC4 | 0.6753 | 0.9424 | 0.6299 | 15 |
| LogReg | MW1 | 0.6000 | 0.6522 | 0.5565 | 12 |
| RF | PC4 | 0.5938 | 0.9451 | 0.5473 | 15 |
| XGB | PC4 | 0.6506 | 0.9463 | 0.6044 | 15 |

### Interpretation

The best individual-dataset runs show that the strongest single-dataset results came largely from the `PC4` dataset. This is important because it separates two different questions:

- **Which model family can achieve the strongest per-dataset result?**
- **Which trained model should actually be deployed in the CI/CD pipeline?**

For example, `RF`'s best individual MCC came from `PC4`, but the final deployed model is still `RF` on `KC1`, because deployment was chosen using both predictive strength and feature compatibility with the automated commit-level workflow.

## Overall Individual Benchmark Results

The overall benchmark aggregates results across all 12 individual datasets.

### Aggregate Model-Family Results

| Model | Mean rank | Mean F1 | Mean AUC-ROC | Mean MCC | Mean delta vs combined baseline MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees | 2.9167 | 0.3904 | 0.7863 | 0.3000 | -0.0002 |
| LogReg | 2.9167 | 0.3753 | 0.7470 | 0.2922 | 0.0513 |
| RF | 3.1667 | 0.3820 | 0.7838 | 0.2988 | 0.0625 |
| BalancedRF | 3.7500 | 0.3818 | 0.7604 | 0.2994 | 0.0677 |
| XGB | 4.0833 | 0.3698 | 0.7461 | 0.2793 | 0.0299 |
| LGBM | 4.1667 | 0.3457 | 0.7451 | 0.2337 | -0.0076 |

### Benchmark Conclusions

- **Overall family winner**: `ExtraTrees`
- **Deployable winner**: `RF` on `KC1`
- `RF` has the highest positive mean `F1` delta vs the combined baseline.
- `BalancedRF` has the largest positive mean `MCC` delta vs the combined baseline.
- `ExtraTrees` is essentially flat on mean `MCC` delta relative to the combined baseline, but still wins overall by mean rank.

### Deployable Winner Details

From `src/models/deployment/best_model_bundle.json`:

- Final deployed model: `RF`
- Dataset: `KC1`
- Threshold: `0.45388100311700624`
- Benchmark metrics:
  - `F1 = 0.5672`
  - `MCC = 0.4859`
  - `AUC-ROC = 0.8262`
- Strictly deployable: `true`
- Missing mapped features: none

Supported deployable feature list:

- `e`
- `ev(g)`
- `i`
- `l`
- `lOBlank`
- `lOComment`
- `locCodeAndComment`
- `uniq_Op`

## SHAP and XAI

The final deployable model was explained using both SHAP and LIME.

### What SHAP Was Used For

SHAP was used to explain the final deployable model, which is `RF` on `KC1`.

- `shap_summary_bar.png` shows **global importance**: which features mattered most overall.
- `shap_beeswarm.png` shows **direction and spread**: whether higher or lower feature values pushed predictions toward higher defect risk.
- `shap_waterfall_1.png`, `shap_waterfall_2.png`, and `shap_waterfall_3.png` show **local decision paths** for the top-risk individual test cases.

### What LIME Was Used For

LIME was used as a second local-explanation method for the same deployable model.

- `lime_case_1.html`
- `lime_case_2.html`
- `lime_case_3.html`

These files provide instance-level feature-attribution reports in a human-readable format, which is useful when discussing individual high-risk predictions in the thesis or implementation defense.

### High-Risk Case Study Notes

The three highest-risk predictions are listed in `figures/xai/high_risk_cases.csv`:

- sample `323`: predicted probability `0.8916`, actual label `0`
- sample `419`: predicted probability `0.8799`, actual label `1`
- sample `153`: predicted probability `0.8668`, actual label `1`

This is useful analytically because one of the top-risk predictions was actually labeled clean. That gives us a concrete false-positive case study: the model assigned very high risk, SHAP/LIME can be used to inspect why, and the result helps us discuss both interpretability and the practical cost of conservative defect-risk screening.

### XAI Interpretation

SHAP and LIME together make the final model auditable rather than just accurate. SHAP explains both global feature importance and local decision behavior, while LIME gives a second, more narrative-style view of why a particular file or instance was marked as risky. Together, they strengthen the argument that the deployed model can be examined, justified, and discussed in a thesis setting rather than treated as a black box.

## Comparative Analysis

### Three-Experiment Comparison

| Category | Experiment 1 | Experiment 2 | Experiment 3 |
| --- | --- | --- | --- |
| Dataset scope | Combined cleaned dataset | Same combined cleaned dataset | 12 individual raw PROMISE datasets |
| Split strategy | 85/15-style train/test | 60/20/20 train/validation/test | Per-dataset split inside each benchmark run |
| Threshold strategy | Out-of-fold train-based calibration | Validation-based calibration | Validation-based calibration per dataset |
| Best ranked model | `ExtraTrees` | `RF` | `ExtraTrees` as overall family winner |
| Strongest held-out test model | `RF` by MCC | `ExtraTrees` by MCC | Varies by dataset and model family |
| Deployment decision | Not in scope | Not in scope | `RF` on `KC1` |
| Main strength | Strong pooled baseline | More realistic evaluation | Strongest overall research + engineering integration |
| Main limitation | No explicit validation holdout | Still only pooled combined data | Benchmark winner and deployment winner differ |
| Main takeaway | Combined data is enough to establish a baseline | Split design can change ranking outcomes | Practical deployment requires both predictive quality and schema compatibility |

### What Changed and Why It Mattered

#### Experiment 1 -> Experiment 2

The move from a simple pooled split to a true 60/20/20 train/validation/test design improved evaluation realism. It separated threshold selection from the final test set and directly changed the ranking outcome from `ExtraTrees` to `RF`.

#### Experiment 2 -> Experiment 3

The move from one combined benchmark to 12 individual-dataset benchmarks broadened the study into a generalization analysis. It showed that model behavior is heterogeneous across datasets and that the best benchmark family is not automatically the best model to deploy inside a CI/CD system.

## CI/CD Integration

The CI/CD integration is implemented through `.github/workflows/defect_check.yml`.

### Operational Flow

1. GitHub Actions starts on `push` or `pull_request`.
2. The repository is checked out.
3. Python dependencies are installed from `requirements.txt`.
4. The Flask API is started with `MODEL_BUNDLE_PATH=src/models/deployment/best_model_bundle.pkl`.
5. `scripts/predict_commit.py` runs.
6. The script identifies changed `.py` and `.java` files.
7. `lizard`-derived metrics are extracted from those files.
8. The extracted values are mapped into the PROMISE-style feature schema expected by the deployed model.
9. The script calls `/predict` on the running API.
10. The resulting `defect_report.json` file is uploaded as a workflow artifact.
11. On pull requests, the workflow posts a summary comment containing the risk predictions.

### Why Feature Compatibility Matters

The deployment bundle confirms that the final model is **strictly deployable**:

- file: `src/models/deployment/best_model_bundle.json`
- missing mapped features: none
- supported features: 8

This matters because the commit-level workflow cannot use arbitrary PROMISE features. It can only deploy a model whose required features can be derived or proxied from changed source files in CI/CD. That is why `RF` on `KC1` became the operational model even though `ExtraTrees` was the strongest family overall in the research benchmark.

## How to Get a Prediction from the Final Model

The final deployed model is:

- model: `RF`
- dataset: `KC1`
- threshold: `0.45388100311700624`
- bundle path: `src/models/deployment/best_model_bundle.pkl`

### 1. Start the API

```powershell
$env:MODEL_BUNDLE_PATH="src/models/deployment/best_model_bundle.pkl"
python -m src.app
```

### 2. Check That the API Is Healthy

```powershell
curl http://127.0.0.1:5000/health
```

Expected response fields include:

- `status`
- `model`
- `dataset`
- `threshold`
- `raw_feature_order`
- `selected_feature_order`

### 3. Request a Prediction Directly

```powershell
curl -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d "{\"metrics\":{\"locCodeAndComment\":120,\"ev(g)\":4,\"lOBlank\":10,\"lOComment\":6,\"uniq_Op\":20,\"e\":500,\"i\":30,\"l\":0.2}}"
```

The response includes:

- `probability`: predicted defect probability
- `label`: `DEFECT-PRONE` or `CLEAN`
- `threshold`: decision threshold used
- `model`: deployed model name
- `dataset`: source benchmark dataset of the deployed model
- `top_features`: most important contributing features for that prediction

### 4. Run Commit-Level Batch Prediction

With the API running:

```powershell
python scripts/predict_commit.py --output defect_report.json
```

Optional commit range:

```powershell
python scripts/predict_commit.py --output defect_report.json --base-ref HEAD~1 --head-ref HEAD
```

This is the path used by CI/CD. The API is the general reusable inference interface; `scripts/predict_commit.py` is the automation layer that transforms changed source files into a prediction-ready report.

## Figures and Artifact Reference Table

| Artifact | Role in the project | Path |
| --- | --- | --- |
| Combined CV F1 comparison | Experiment 1 and 2 pooled CV F1 comparison | `src/plots/cv_f1_comparison_overall.png` |
| Combined CV MCC comparison | Experiment 1 and 2 pooled MCC comparison | `src/plots/cv_mcc_comparison_overall.png` |
| Combined ROC curves | Experiment 1 and 2 held-out discrimination view | `src/plots/roc_curves_overall.png` |
| Combined PR curves | Experiment 1 and 2 precision-recall view | `src/plots/pr_curves_overall.png` |
| Combined confusion matrices | Experiment 1 and 2 thresholded prediction behavior | `src/plots/confusion_matrices_overall.png` |
| Combined feature importance | Experiment 1 and 2 strongest tree-model feature view | `src/plots/feature_importance.png` |
| Individual MCC heatmap | Experiment 3 per-dataset MCC view | `src/plots/individual/individual_mcc_heatmap.png` |
| Individual F1 heatmap | Experiment 3 per-dataset F1 view | `src/plots/individual/individual_f1_heatmap.png` |
| Delta vs combined | Experiment 3 change relative to pooled baseline | `src/plots/individual/delta_vs_combined.png` |
| Best model per dataset | Experiment 3 winner map | `src/plots/individual/best_model_per_dataset.png` |
| Average model rank | Experiment 3 family ranking | `src/plots/individual/average_model_rank.png` |
| SHAP summary bar | Global feature importance for final deployable model | `figures/xai/shap_summary_bar.png` |
| SHAP beeswarm | Direction and spread of feature effects | `figures/xai/shap_beeswarm.png` |
| SHAP waterfalls | Local explanation for top-risk cases | `figures/xai/shap_waterfall_1.png`, `figures/xai/shap_waterfall_2.png`, `figures/xai/shap_waterfall_3.png` |
| LIME case files | Local case reports for high-risk predictions | `figures/xai/lime_case_1.html`, `figures/xai/lime_case_2.html`, `figures/xai/lime_case_3.html` |
| Benchmark summary | Main summary of Experiment 3 | `src/results/individual/benchmark_summary.json` |
| Combined vs individual summary | Aggregate benchmark-vs-baseline comparison | `src/results/individual/combined_vs_individual_summary.csv` |

## Reproducibility Guide

### Environment Setup

From the project root:

```powershell
uv venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The raw ARFF files should be present in `data/raw`, and the combined cleaned CSV should be present in `data/processed`.

### Reproduce Experiment 1

```powershell
git switch --detach origin/main
python -m src.pipeline
```

Generated outputs appear in:

- `src/results/`
- `src/plots/`
- `src/models/`

### Reproduce Experiment 2

```powershell
git switch --detach origin/req/split
python -m src.pipeline
```

Generated outputs appear in:

- `src/results/`
- `src/plots/`
- `src/models/`

### Reproduce Experiment 3

From the current local working tree:

```powershell
python -m src.train_individual_benchmark --data-dir data/raw --baseline existing --top-k 12
```

Outputs appear in:

- `src/models/individual/`
- `src/results/individual/`
- `src/plots/individual/`
- `src/models/deployment/`
- `figures/xai/`

### Exact Command Used for the Current Saved Local Benchmark

The current saved Experiment 3 outputs in the working tree were generated with:

```powershell
python -m src.train_individual_benchmark --random-search-iters 1 --train-n-jobs 1
```

This is the exact command path that produced the current saved local benchmark tables, deployment bundle, and XAI outputs.

## Implementation Inventory Added in Experiment 3

- dataset-loader support for both combined-baseline and individual-dataset modes
- richer preprocessing bundles with embedded feature-selection metadata
- new individual benchmark runner
- deployment bundle generation
- Flask inference API
- commit-level prediction script
- GitHub Actions workflow
- API test suite
- SHAP and LIME explanation outputs

## Limitations and Next Steps

- The strongest overall benchmark family and the best deployable model are different by design:
  - benchmark winner: `ExtraTrees`
  - deployable winner: `RF` on `KC1`
- The CI/CD pipeline uses commit-level proxy metrics derived from `lizard`, so some PROMISE features are approximated rather than observed exactly.
- Experiments 1 and 2 are described from saved branch artifacts, while Experiment 3 is described from the current local benchmark outputs.
- A future refinement would be to rerun the full 12-dataset benchmark with the preferred higher-iteration tuning budget and preserve those outputs as a final frozen results snapshot.

## Reproducibility Matrix

| Experiment | Git reference |
| --- | --- |
| Experiment 1 | `origin/main` |
| Experiment 2 | `origin/req/split` |
| Experiment 3 | current local `req/split` working tree |

## Final Conclusions

The full experimentation pipeline shows a clear methodological progression. Combined-data experiments were valuable for establishing the classical machine learning baseline, and they confirmed that tree ensembles and logistic regression were viable defect-prediction candidates. However, the transition from Experiment 1 to Experiment 2 showed that evaluation design matters: once thresholding was tied to a dedicated validation split, the model ranking changed.

The final individual-dataset benchmark showed that performance is heterogeneous across PROMISE datasets. No single model is universally best under every criterion. Across the 12-dataset benchmark, `ExtraTrees` was the strongest overall family by mean rank. At the same time, the final deployment decision had to consider more than benchmark rank alone. `RF` on `KC1` became the operational choice because it combined strong predictive performance with full CI/CD feature compatibility and a clean serving path through the API and commit predictor.

### Final Answer to the Research Question

The project concludes that there is no single universally best model under all criteria. Benchmark superiority and deployment suitability must be treated separately. For this project, `ExtraTrees` is the strongest research winner across the full individual-dataset benchmark, while `RF` on `KC1` is the strongest engineering winner for practical automated defect-risk prediction. This is not a contradiction; it is the central result of the whole experimentation process.
