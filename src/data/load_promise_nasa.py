from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from scipy.io import arff

TARGET_COL = "label"
DATASET_COL = "dataset"
BASELINE_CANONICAL_DATASET = "JM1"


def load_arff(path: str) -> pd.DataFrame:
    """Load a PROMISE / NASA .arff file into a pandas DataFrame."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )

    return df


def binarise_label(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Binarise the target label column for a given PROMISE / NASA dataset.

    Maps each dataset's native label encoding to a shared binary ``label`` column
    where ``1 = defective`` and ``0 = clean``.
    """
    df = df.copy()
    dataset_key = dataset.upper()

    dataset_label_specs = {
        "CM1": ("class", {"true"}),
        "KC1": ("class", {"true"}),
        "PC1": ("class", {"true"}),
        "JM1": ("defects", {"true"}),
        "KC2": ("class", {"yes"}),
        "KC3": ("Defective", {"y"}),
        "MC1": ("Defective", {"y"}),
        "MC2": ("Defective", {"y"}),
        "MW1": ("Defective", {"y"}),
        "PC2": ("Defective", {"y"}),
        "PC3": ("Defective", {"y"}),
        "PC4": ("c", {"true"}),
    }

    if dataset_key in dataset_label_specs:
        label_col, positive_tokens = dataset_label_specs[dataset_key]
        if label_col not in df.columns:
            raise KeyError(
                f"Expected label column '{label_col}' not found for dataset '{dataset_key}'."
            )
        values = df[label_col].astype(str).str.strip().str.lower()
        df[TARGET_COL] = values.isin({t.lower() for t in positive_tokens}).astype(int)
    else:
        candidates = ["Defective", "defects", "class", "c"]
        lower_map = {c.lower(): c for c in df.columns}
        label_col = None

        for name in candidates:
            if name.lower() in lower_map:
                label_col = lower_map[name.lower()]
                break

        if label_col is None:
            bool_tokens = {"Y", "N", "YES", "NO", "TRUE", "FALSE", "T", "F", "1", "0"}
            for col in reversed(df.columns.tolist()):
                uniques = set(str(v).upper() for v in df[col].dropna().unique())
                if uniques and uniques.issubset(bool_tokens) and len(uniques) <= 2:
                    label_col = col
                    break

        if label_col is None:
            raise KeyError(
                f"Could not infer a binary defect label column in DataFrame for dataset '{dataset_key}'."
            )

        values = df[label_col].astype(str).str.strip().str.lower()
        df[TARGET_COL] = values.isin({"y", "yes", "true", "t", "1"}).astype(int)

    label_like_names = {
        "defects",
        "defective",
        "defect",
        "class",
        "c",
        "buggy",
        "label",
    }
    to_drop = [c for c in df.columns if c.lower() in label_like_names and c != TARGET_COL]
    return df.drop(columns=to_drop, errors="ignore")


def _iter_arff_files(data_dir: str) -> Iterable[Path]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")
    return sorted(base.glob("*.arff"))


def load_individual_datasets(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load all PROMISE / NASA datasets individually.

    Each returned DataFrame keeps its native numeric feature schema and only adds
    the shared ``label`` and ``dataset`` columns.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    for path in _iter_arff_files(data_dir):
        dataset_name = path.stem.upper()
        df = binarise_label(load_arff(str(path)), dataset=dataset_name)
        df[DATASET_COL] = dataset_name
        datasets[dataset_name] = df

    if not datasets:
        raise ValueError(f"No .arff files found in {data_dir}.")

    return datasets


def load_compatible_nasa_datasets(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load datasets that align 1-to-1 with the canonical JM1 schema.

    This preserves the historical combined-baseline behavior used by the current
    repo-level benchmark and processed combined CSVs.
    """
    all_datasets = load_individual_datasets(data_dir=data_dir)
    if BASELINE_CANONICAL_DATASET not in all_datasets:
        raise FileNotFoundError(
            f"{BASELINE_CANONICAL_DATASET}.arff not found in {data_dir}; cannot establish canonical schema."
        )

    jm1_df = all_datasets[BASELINE_CANONICAL_DATASET].copy()
    canonical_features = [
        c for c in jm1_df.columns if c not in (TARGET_COL, DATASET_COL)
    ]

    compatible: Dict[str, pd.DataFrame] = {}
    for name, df in all_datasets.items():
        if all(col in df.columns for col in canonical_features):
            aligned = df[canonical_features].copy()
            aligned[TARGET_COL] = df[TARGET_COL].values
            aligned[DATASET_COL] = name
            compatible[name] = aligned

    return compatible


def load_nasa_datasets(
    data_dir: str = "data/raw",
    mode: str = "combined-baseline",
) -> Dict[str, pd.DataFrame]:
    """
    Load PROMISE / NASA datasets.

    Supported modes:
    - ``combined-baseline``: keep only datasets compatible with the canonical
      JM1 feature schema. This matches the historical combined benchmark logic.
    - ``individual``: return every dataset independently with its native schema.
    """
    mode_key = mode.strip().lower()
    if mode_key == "combined-baseline":
        return load_compatible_nasa_datasets(data_dir=data_dir)
    if mode_key == "individual":
        return load_individual_datasets(data_dir=data_dir)
    raise ValueError(
        "Unsupported mode. Use 'combined-baseline' or 'individual'."
    )


def build_master_dataframe(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple dataset DataFrames into a single master DataFrame."""
    if not datasets:
        raise ValueError("No datasets provided to combine.")
    return pd.concat(datasets.values(), axis=0, ignore_index=True)


def load_and_combine(
    data_dir: str = "data/raw",
    mode: str = "combined-baseline",
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Convenience helper that returns both the per-dataset dict and a combined DataFrame."""
    datasets = load_nasa_datasets(data_dir=data_dir, mode=mode)
    return datasets, build_master_dataframe(datasets)


if __name__ == "__main__":
    datasets, master_df = load_and_combine(data_dir="data/raw", mode="combined-baseline")

    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_path = processed_dir / "promise_nasa_combined.csv"
    master_df.to_csv(combined_path, index=False)

    for name, df in datasets.items():
        print(f"{name}: shape={df.shape}, label distribution=")
        print(df[TARGET_COL].value_counts())
        print("-" * 40)

    print("Master DataFrame shape:", master_df.shape)
    print("Datasets present:", master_df[DATASET_COL].unique().tolist())
    print(f"Combined dataset saved to: {combined_path}")
