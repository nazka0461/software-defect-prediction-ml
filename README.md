## AI/ML Software Defect Prediction

**Student**: Azka Noor (ID: 24091601)  
**Module**: 7COM1040 — Computer Science Masters Project  
**University**: University of Hertfordshire  
**Project Duration**: 25 February 2026 – 10 April 2026 (6.5 weeks)

This repository contains the research code and assets for a masters project on **software defect prediction using classical machine learning and explainable AI (XAI)**. The work focuses on predicting defect-prone modules from static code metrics and analysing model behaviour using modern explainability techniques.

### Project Objectives

- **Defect Prediction**: Build and evaluate machine learning models that predict whether a software module is defect-prone using static code metrics.
- **Benchmarking**: Use widely-cited PROMISE / NASA datasets as the primary benchmark for training and evaluation.
- **Explainability**: Apply XAI methods (e.g., SHAP, LIME) to interpret model predictions for both global feature importance and individual modules.
- **Automation & Reproducibility**: Structure the project with a clear directory layout and automation hooks so that experiments can be reproduced reliably.

### Datasets

The project is based on the **PROMISE Repository** family of **NASA defect datasets**, which provide static code metrics and binary defect labels at the module level. The primary datasets used are:

- **KC1** — C++ flight software
- **CM1** — NASA spacecraft instrument
- **PC1** — NASA flight software
- **JM1** — NASA real-time predictive system
- **KC2** — C++ scientific software

Each dataset includes Halstead metrics, McCabe complexity measures, size metrics (e.g., LOC), and a binary target label indicating whether a module is defective. Mirrors of these datasets are also available in platforms such as **OpenML**, which can be used if the primary PROMISE sources are slow or unavailable.

> Note: Implementation details (data loading, preprocessing, model training, CI/CD, and XAI pipelines) are tracked in separate documentation and notebooks, and are intentionally not described here yet.

### High-Level Project Phases

1. **Phase 1 — Data Acquisition & Exploratory Analysis**  
   Collect PROMISE / NASA datasets, perform basic exploratory data analysis (EDA), and understand class imbalance and feature distributions.
2. **Phase 2 — Feature Engineering & Preprocessing**  
   Prepare the data for modelling (label binarisation, handling multicollinearity, scaling, and imbalance treatment) and define a reusable preprocessing pipeline.
3. **Phase 3 — Model Development & Evaluation**  
   Train and compare multiple classical ML models for defect prediction, using robust validation and appropriate performance metrics.
4. **Phase 4 — CI/CD Integration**  
   Wrap the best-performing model in a lightweight API and integrate it into a continuous integration workflow for automated risk assessment on code changes.
5. **Phase 5 — Explainability & Thesis Write-Up**  
   Apply XAI techniques, analyse feature importance, and consolidate experimental results into the final thesis chapter and appendices.

### Repository Structure

The project follows a standard, experiment-friendly directory layout:

- **`data/`**: Data files (not all are necessarily committed to version control)
  - **`data/raw/`**: Original PROMISE / NASA datasets in their downloaded format
  - **`data/processed/`**: Cleaned and transformed datasets ready for modelling
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, experimentation, and reporting
- **`src/`**: Source code for the core project logic
  - **`src/data/`**: Data loading and dataset management utilities
  - **`src/features/`**: Feature engineering and preprocessing components
  - **`src/models/`**: Model training, evaluation, and selection code
  - **`src/api/`**: Application code for serving predictions (e.g., web or REST API)
- **`models/`**: Saved model artefacts and related metadata (e.g., selected feature lists, pipelines)
- **`results/`**: Experiment outputs such as metrics tables and comparison summaries
- **`figures/`**: Generated plots, diagrams, and visualisations for the thesis and reports
- **`scripts/`**: Command-line scripts for running end-to-end workflows (e.g., training, evaluation)
- **`tests/`**: Automated tests for critical functionality
- **`.github/workflows/`**: Continuous integration workflows (e.g., automated checks on pushes/PRs)

### Getting Started

This project uses a **uv-managed virtual environment** and an unpinned `requirements.txt` to keep the setup simple and reproducible.

#### 1. Install uv (Windows PowerShell)

Run the following command in **PowerShell** to install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal if `uv` is not immediately available.

#### 2. Create a virtual environment with uv

From the project root:

```powershell
uv venv
```

This will create a `.venv` folder in the repository.

#### 3. Allow local scripts and activate the environment

In PowerShell, ensure local scripts can run and then activate the environment:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You should see the virtual environment name (e.g. `.venv`) appear in your prompt.

#### 4. Install Python dependencies

With the environment activated, install all required libraries from `requirements.txt` using uv’s pip interface:

```powershell
pip install -r requirements.txt
.\.venv\Scripts\activate
```

At this stage, the environment will contain the full stack of libraries needed for data loading, preprocessing, model training, evaluation, explainability, and API / CI work described in the project plan. Detailed implementation steps are documented separately and are not covered here.