from pathlib import Path
from typing import Dict, Tuple

from scipy.io import arff
import pandas as pd


def load_arff(path: str) -> pd.DataFrame:
    """Load a PROMISE / NASA .arff file into a pandas DataFrame."""
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Decode bytes columns (arff stores strings as bytes)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    return df


def binarise_label(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Binarise the target label column for a given PROMISE / NASA dataset.

    Uses explicit knowledge of each dataset's label column and value
    encoding, based on the ARFF definitions, and maps all of them to a
    unified binary column called 'label' where 1 = defective and 0 = clean.
    """
    df = df.copy()
    dataset_key = dataset.upper()

    # Explicit mapping from dataset name -> (label column, set of positive tokens)
    # Tokens are specified in lowercase; the column values are normalised before comparison.
    dataset_label_specs = {
        # class {false,true}
        "CM1": ("class", {"true"}),
        "KC1": ("class", {"true"}),
        "PC1": ("class", {"true"}),

        # defects {false,true}
        "JM1": ("defects", {"true"}),

        # class {no,yes}
        "KC2": ("class", {"yes"}),

        # Defective {N,Y}
        "KC3": ("Defective", {"y"}),
        "MC1": ("Defective", {"y"}),
        "MC2": ("Defective", {"y"}),
        "MW1": ("Defective", {"y"}),
        "PC2": ("Defective", {"y"}),
        "PC3": ("Defective", {"y"}),

        # c {FALSE,TRUE}
        "PC4": ("c", {"true"}),
    }

    if dataset_key in dataset_label_specs:
        label_col, positive_tokens = dataset_label_specs[dataset_key]
        if label_col not in df.columns:
            raise KeyError(
                f"Expected label column '{label_col}' not found for dataset '{dataset_key}'."
            )
        values = df[label_col].astype(str).str.strip().str.lower()
        df["label"] = values.isin({t.lower() for t in positive_tokens}).astype(int)
    else:
        # Fallback: generic handling based on common label column names and boolean-like values.
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
        df["label"] = values.isin({"y", "yes", "true", "t", "1"}).astype(int)

    # Drop any original categorical label columns so that only 'label' remains.
    label_like_names = {
        "defects",
        "defective",
        "defect",
        "class",
        "c",
        "buggy",
        "label",
    }
    to_drop = [c for c in df.columns if c.lower() in label_like_names and c != "label"]
    df = df.drop(columns=to_drop, errors="ignore")

    return df


def load_nasa_datasets(
    data_dir: str = "data/raw",
) -> Dict[str, pd.DataFrame]:
    """
    Load PROMISE / NASA defect datasets into individual DataFrames, aligned to JM1.

    JM1 is treated as the canonical schema. Only datasets whose columns map
    1-to-1 to the JM1 feature set are included; others are skipped. All
    returned DataFrames share the same set of feature columns plus a unified
    'label' and 'dataset' column.
    """
    base = Path(data_dir)

    datasets: Dict[str, pd.DataFrame] = {}

    # --- 1. Load JM1 as the canonical schema ---
    jm1_path = None
    for path in base.glob("*.arff"):
        if path.stem.upper() == "JM1":
            jm1_path = path
            break

    if jm1_path is None:
        raise FileNotFoundError("JM1.arff not found in data/raw; cannot establish canonical schema.")

    jm1_df = load_arff(str(jm1_path))
    jm1_df = binarise_label(jm1_df, dataset="JM1")
    jm1_df["dataset"] = "JM1"

    # Canonical feature columns = all columns except 'label' and 'dataset'
    canonical_features = [c for c in jm1_df.columns if c not in ("label", "dataset")]
    jm1_df = jm1_df[canonical_features + ["label", "dataset"]]
    datasets["JM1"] = jm1_df

    # --- 2. Load and align other datasets to JM1 schema ---
    for path in sorted(base.glob("*.arff")):
        name = path.stem.upper()
        if name == "JM1":
            continue

        df = load_arff(str(path))
        df = binarise_label(df, dataset=name)

        # If this dataset does not contain all canonical JM1 feature columns,
        # it is considered structurally incompatible and is skipped.
        if not all(col in df.columns for col in canonical_features):
            continue

        # Keep only the JM1-aligned features plus label and dataset.
        features_df = df[canonical_features].copy()
        features_df["label"] = df["label"].values
        features_df["dataset"] = name

        datasets[name] = features_df

    return datasets


def build_master_dataframe(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple PROMISE / NASA datasets into a single master DataFrame.

    Assumes each input DataFrame already has a 'dataset' column and a
    binarised 'label' column.
    """
    if not datasets:
        raise ValueError("No datasets provided to combine.")

    master = pd.concat(datasets.values(), axis=0, ignore_index=True)
    return master


def load_and_combine(
    data_dir: str = "data/raw",
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Convenience helper: load all datasets and return both the per-dataset
    dict and the combined master DataFrame.
    """
    datasets = load_nasa_datasets(data_dir=data_dir)
    master = build_master_dataframe(datasets)
    return datasets, master


if __name__ == "__main__":
    # Example usage when running this module directly.
    datasets, master_df = load_and_combine(data_dir="data/raw")

    # Ensure processed directory exists and save combined dataset.
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_path = processed_dir / "promise_nasa_combined.csv"
    master_df.to_csv(combined_path, index=False)

    # Print basic sanity checks
    for name, df in datasets.items():
        print(f"{name}: shape={df.shape}, label distribution=")
        print(df["label"].value_counts())
        print("-" * 40)

    print("Master DataFrame shape:", master_df.shape)
    print("Datasets present:", master_df["dataset"].unique().tolist())
    print(f"Combined dataset saved to: {combined_path}")

