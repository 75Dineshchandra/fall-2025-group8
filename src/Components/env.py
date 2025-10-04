
# single_script.py
# Run: python3 single_script.py
# Builds item mapping, action matrix, and feature matrix from sales.csv and saves CSV outputs.

import os
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd

# ===================== Saving Helpers =====================

def save_action_matrix(
    action_matrix: np.ndarray,
    save_path: str,
    item_to_idx: Dict[str, int]
) -> None:
    if not isinstance(action_matrix, np.ndarray):
        raise TypeError("action_matrix must be a numpy.ndarray")

    num_time_slots, num_items = action_matrix.shape

    action_df = pd.DataFrame(
        action_matrix,
        columns=[f"item_{i}" for i in range(num_items)]
    )
    action_df["time_slot_id"] = range(num_time_slots)

    cols = ["time_slot_id"] + [f"item_{i}" for i in range(num_items)]
    action_df = action_df[cols]

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    action_df.to_csv(save_path, index=False)
    print(f"✓ Action matrix saved to: {save_path}")


def save_feature_matrix(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    rows_df: pd.DataFrame,
    save_path: str
) -> None:
    if not isinstance(feature_matrix, np.ndarray):
        raise TypeError("feature_matrix must be a numpy.ndarray")

    n_rows, n_feats = feature_matrix.shape
    if len(feature_names) != n_feats:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != number of features ({n_feats})"
        )

    required_cols = {"t", "description", "item_idx"}
    missing = required_cols - set(rows_df.columns)
    if missing:
        raise KeyError(f"rows_df missing required columns: {sorted(missing)}")

    feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
    feature_df["time_slot_id"] = rows_df["t"].values
    feature_df["item"] = rows_df["description"].values
    feature_df["item_idx"] = rows_df["item_idx"].values

    meta_cols = ["time_slot_id", "item", "item_idx"]
    other_cols = [c for c in feature_df.columns if c not in meta_cols]
    feature_df = feature_df[meta_cols + other_cols]

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    feature_df.to_csv(save_path, index=False)
    print(f"✓ Feature matrix saved to: {save_path}")

# ===================== IO & Pathing =====================

def load_data(file_path: str) -> pd.DataFrame:
    # Use low_memory=False to avoid DtypeWarning fragmentation across chunks
    # Optionally coerce common text columns to string dtype
    dtype_overrides = {
        "school_name": "string",
        "time_of_day": "string",
        "description": "string",
    }
    return pd.read_csv(file_path, low_memory=False, dtype=dtype_overrides)

def _find_sales_csv() -> str:
    """
    Try a few likely locations for sales.csv.
    Priority: repo_root/data, then src/data, then local cwd variants.
    """
    base_dir = Path(os.path.dirname(os.path.dirname(__file__))) if "__file__" in globals() else Path(os.getcwd())
    candidates = [
        base_dir / "data" / "sales.csv",
        base_dir / "src" / "data" / "sales.csv",
        Path("data/sales.csv"),
        Path("src/data/sales.csv"),
        Path("sales.csv"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    raise FileNotFoundError(f"Could not find sales.csv in any of: {', '.join(str(p) for p in candidates)}")

# ===================== Mappings & Matrices =====================

def build_item_mapping(
    dataframe: pd.DataFrame,
    item_col: str = "description",
    save_path: str = "item_mapping.csv"
) -> Tuple[Dict[str, int], List[str]]:
    valid_items = sorted(dataframe[item_col].astype(str).unique())
    item_to_idx = {item: idx for idx, item in enumerate(valid_items)}
    all_items = valid_items

    mapping_df = pd.DataFrame({
        'item': all_items,
        'item_idx': range(len(all_items))
    })
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    mapping_df.to_csv(save_path, index=False)

    print(f"✓ Created item mapping with {len(item_to_idx)} items")
    print(f"✓ Saved mapping to: {save_path}")
    return item_to_idx, all_items


def build_action_matrix(
    merged: pd.DataFrame,
    item_to_idx: Dict[str, int],
    *,
    item_col: str = "description",
    time_slot_col: str = "time_slot_id"
) -> np.ndarray:
    if time_slot_col not in merged.columns:
        raise KeyError(f"Required column '{time_slot_col}' not found.")
    if item_col not in merged.columns:
        raise KeyError(f"Required item column '{item_col}' not found.")

    num_items = len(item_to_idx)
    num_time_slots = int(pd.to_numeric(merged[time_slot_col], errors="coerce").max()) + 1

    action_matrix = np.zeros((num_time_slots, num_items), dtype=int)

    grouped = merged.groupby(time_slot_col)[item_col].unique()

    for time_slot_id, items_in_slot in grouped.items():
        t = int(time_slot_id)
        for item in items_in_slot:
            item_str = str(item)
            idx = item_to_idx.get(item_str, None)
            if idx is not None:
                action_matrix[t, idx] = 1

    total_possible = num_time_slots * num_items
    actual_available = int(action_matrix.sum())
    coverage_percent = 100 * actual_available / total_possible if total_possible > 0 else 0

    print(f"Action matrix: {num_time_slots} time slots × {num_items} items")
    print(f"Coverage: {coverage_percent:.1f}% of (t, item) slots are available")
    if num_time_slots > 0:
        print(f"Average {actual_available/num_time_slots:.1f} items available per time slot")
    return action_matrix


def build_feature_matrix(
    df_merged: pd.DataFrame,
    item_to_idx: Dict[str, int],
    *,
    item_col: str = "description",
    use_time_slot_id: bool = True,
    cat_cols: List[str] = None,
    cyc7_cols: List[str] = None,
    num_cols: List[str] = None,
    default_nutrients: List[str] = None,
    add_bias: bool = False,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, Dict[int, np.ndarray], Dict[str, Any]]:
    if cat_cols is None:
        cat_cols = ["school_name", "time_of_day"]
    if cyc7_cols is None:
        cyc7_cols = ["day_of_week"]
    if num_cols is None:
        num_cols = ["HealthScore"]
    if default_nutrients is None:
        default_nutrients = [
            "GramsPerServing", "Calories", "Protein", "Total Carbohydrate", "Dietary Fiber",
            "Total Sugars", "Added Sugars", "Total Fat", "Saturated Fat", "Trans Fat",
            "Cholesterol", "Sodium", "Vitamin D (D2 + D3)", "Calcium", "Iron", "Potassium",
            "Vitamin A", "Vitamin C"
        ]

    def _lower_map(df: pd.DataFrame) -> Dict[str, str]:
        return {c.lower(): c for c in df.columns}

    def _zscore_1d(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        return (x - mu) / sd, mu, sd

    df = df_merged.copy()
    df[item_col] = df[item_col].astype(str)

    # filter to mapped items
    original_count = len(df)
    df = df[df[item_col].isin(item_to_idx.keys())].copy()
    filtered_count = len(df)
    if filtered_count < original_count:
        print(f"Note: Filtered out {original_count - filtered_count} rows with unmapped items")

    # timestep col
    if use_time_slot_id and "time_slot_id" in df.columns:
        df = df.rename(columns={"time_slot_id": "t"})
    elif "t" not in df.columns:
        raise KeyError("No timestep found. Provide 'time_slot_id' or a precomputed 't' column.")
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(int)

    cmap = _lower_map(df)
    cat_used = [c for c in cat_cols if c in df.columns]
    cyc7_used = [c for c in cyc7_cols if c in df.columns]
    num_used = [c for c in num_cols if c in df.columns]
    nutr_used_actual = [cmap[n.lower()] for n in default_nutrients if n.lower() in cmap]

    blocks, names = [], []

    if add_bias:
        blocks.append(np.ones((len(df), 1), dtype=float))
        names.append("bias")

    # categorical one-hot
    for col in cat_used:
        ohe = pd.get_dummies(df[col].astype(str), prefix=col)
        if ohe.shape[1] > 0:
            blocks.append(ohe.to_numpy(dtype=float))
            names.extend(ohe.columns.tolist())

    # cyclical (7-day) sin/cos
    for col in cyc7_used:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        v = np.where((v >= 1) & (v <= 7), v - 1, v)  # 1..7 -> 0..6
        ang = 2.0 * np.pi * (v / 7.0)
        blocks.append(np.sin(ang).reshape(-1, 1))
        blocks.append(np.cos(ang).reshape(-1, 1))
        names.extend([f"{col}_sin", f"{col}_cos"])

    # numeric z-scores
    for col in num_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")

    # nutrient z-scores
    for col in nutr_used_actual:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")

    X_all = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(df), 0), dtype=np.float32)

    rows_df = df[["t", item_col]].copy()
    rows_df.columns = ["t", "description"]
    rows_df["item_idx"] = rows_df["description"].map(item_to_idx).astype(int)

    groups = {int(t): idx.to_numpy(dtype=int) for t, idx in rows_df.groupby("t").groups.items()}

    meta = {
        "items": list(item_to_idx.keys()),
        "item2idx": item_to_idx,
        "used_nutrients": nutr_used_actual,
        "used_cat": cat_used,
        "used_cyc7": cyc7_used,
        "used_num": num_used,
        "n_features": X_all.shape[1],
        "n_samples": X_all.shape[0],
        "n_timesteps": len(groups),
        "feature_names": names,
    }

    print(f"Built feature matrix with {X_all.shape[0]} samples and {X_all.shape[1]} features")
    print(f"Features include: {len(cat_used)} categorical, {len(cyc7_used)} cyclical, "
          f"{len(num_used)} numerical, {len(nutr_used_actual)} nutritional")

    return X_all, names, rows_df, groups, meta

# ===================== Main =====================

if __name__ == "__main__":
    DATA_PATH = _find_sales_csv() 
    data = load_data(DATA_PATH)
    print("Loaded sales.csv from:", DATA_PATH)
    print(data.head())

    merged_data = data  # reuse; do not re-read

    print("Building item mapping...")
    item_to_idx, all_items = build_item_mapping(
        dataframe=merged_data,
        item_col="description",
        save_path="out/item_mapping.csv"
    )
    print(f"Total unique items: {len(all_items)}")

    print("Building action matrix...")
    action_matrix = build_action_matrix(
        merged=merged_data,
        item_to_idx=item_to_idx,
        item_col="description",
        time_slot_col="time_slot_id"
    )

    print("Building feature matrix...")
    feature_matrix, feature_names, rows_df, groups, meta = build_feature_matrix(
        df_merged=merged_data,
        item_to_idx=item_to_idx,
        item_col="description",
        use_time_slot_id=True,
        cat_cols=[],      # customize as needed, e.g., ["school_name","time_of_day"]
        cyc7_cols=[],     # e.g., ["day_of_week"]
        num_cols=[],      # e.g., ["HealthScore"]
        default_nutrients=None,  # or leave None to auto-pick common nutrient columns
        add_bias=False
    )
    print("Feature matrix shape:", feature_matrix.shape)
    print("Feature names (first 10):", feature_names[:10])

    os.makedirs("out", exist_ok=True)
    save_action_matrix(action_matrix, "out/action_matrix.csv", item_to_idx)
    save_feature_matrix(feature_matrix, feature_names, rows_df, "out/feature_matrix.csv")
    print("✓ All matrices saved successfully!")
