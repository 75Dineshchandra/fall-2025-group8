import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os



def save_action_matrix(
    action_matrix: np.ndarray,
    save_path: str,
    item_to_idx: Dict[str, int]
) -> None:
    """
    Save action matrix to CSV file.

    Args:
        action_matrix: Binary matrix from build_action_matrix (shape: [time_slots, items]).
        save_path: Path to save the CSV file.
        item_to_idx: Dictionary mapping items to indices (kept for compatibility; not written).
    """
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
    """
    Save feature matrix to CSV file.

    Args:
        feature_matrix: Array of shape [n_rows, n_features].
        feature_names: List of length n_features.
        rows_df: DataFrame with columns: 't' (time_slot_id), 'description' (item), 'item_idx' (int).
        save_path: Output CSV path.
    """
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

# ===================== IO =====================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def build_item_mapping(
    dataframe: pd.DataFrame,
    item_col: str = "description",
    save_path: str = "item_mapping.csv"
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build item-to-index mapping from all unique items in the data.
    Save mapping to CSV for reproducibility.
    
    Args:
        dataframe: DataFrame containing the item column
        item_col: Column name containing item identifiers
        save_path: Path to save the mapping CSV file
        
    Returns:
        item_to_idx: Dictionary mapping item -> index
        all_items: List of items in order of indices
    """
    # Get all unique items and sort for consistency
    valid_items = sorted(dataframe[item_col].astype(str).unique())
    
    # Create mapping
    item_to_idx = {item: idx for idx, item in enumerate(valid_items)}
    all_items = valid_items
    
    # Save to CSV
    mapping_df = pd.DataFrame({
        'item': all_items,
        'item_idx': range(len(all_items))
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    mapping_df.to_csv(save_path, index=False)
    
    print(f"Created mapping with {len(item_to_idx)} items")
    print(f"Saved mapping to: {save_path}")
    
    return item_to_idx, all_items


def build_action_matrix(
    merged: pd.DataFrame,
    item_to_idx: Dict[str, int],
    *,
    item_col: str = "description",
    time_slot_col: str = "time_slot_id"
) -> np.ndarray:
    """
    Build a (time_slots x items) binary matrix indicating available items per time slot.
    
    Args:
        merged: DataFrame containing time slot and item information
        item_to_idx: Dictionary mapping items to indices
        item_col: Column name containing item identifiers
        time_slot_col: Column name containing time slot identifiers
        
    Returns:
        action_matrix: Binary matrix where action_matrix[t, i] = 1 if item i is available at time t
    """
    if time_slot_col not in merged.columns:
        raise KeyError(f"Required column '{time_slot_col}' not found.")
    if item_col not in merged.columns:
        raise KeyError(f"Required item column '{item_col}' not found.")
    
    num_items = len(item_to_idx)
    num_time_slots = int(merged[time_slot_col].max()) + 1
    
    # Initialize matrix with zeros
    action_matrix = np.zeros((num_time_slots, num_items), dtype=int)
    
    # Group by time slot and get unique items available
    grouped = merged.groupby(time_slot_col)[item_col].unique()
    
    # Fill the matrix
    for time_slot_id, items_in_slot in grouped.items():
        for item in items_in_slot:
            item_str = str(item)
            if item_str in item_to_idx:
                action_matrix[int(time_slot_id), item_to_idx[item_str]] = 1
    
    # Report statistics
    total_possible = num_time_slots * num_items
    actual_available = np.sum(action_matrix)
    coverage_percent = 100 * actual_available / total_possible if total_possible > 0 else 0
    
    print(f"Action matrix: {num_time_slots} time slots × {num_items} items")
    print(f"Coverage: {coverage_percent:.1f}% slots have items available")
    print(f"Average {actual_available/num_time_slots:.1f} items available per time slot")
    
    return action_matrix

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any

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
    """
    Build feature matrix for CMAB/LinUCB with contextual and nutritional features.
    
    Args:
        df_merged: DataFrame containing all features and items
        item_to_idx: Dictionary mapping items to indices
        item_col: Column name containing item identifiers
        use_time_slot_id: Whether to use time_slot_id as timestep
        cat_cols: List of categorical columns to one-hot encode
        cyc7_cols: List of cyclical columns (7-day) to encode as sin/cos
        num_cols: List of numerical columns to standardize
        default_nutrients: List of nutrient columns to include
        add_bias: Whether to add a bias term
        
    Returns:
        X_all: Feature matrix (n_samples x n_features)
        feature_names: List of feature names
        rows_df: DataFrame linking timesteps to items
        groups: Dictionary mapping timestep to row indices
        meta: Metadata about the features
    """
    
    # Set default parameters
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
    
    # Copy data and ensure string type for items
    df = df_merged.copy()
    df[item_col] = df[item_col].astype(str)
    
    # Filter to only include items in our mapping
    original_count = len(df)
    df = df[df[item_col].isin(item_to_idx.keys())].copy()
    filtered_count = len(df)
    
    if filtered_count < original_count:
        print(f"Note: Filtered out {original_count - filtered_count} rows with unmapped items")
    
    # Timestep setup
    if use_time_slot_id and "time_slot_id" in df.columns:
        df = df.rename(columns={"time_slot_id": "t"})
    elif "t" not in df.columns:
        raise KeyError("No timestep found. Provide 'time_slot_id' or a precomputed 't' column.")
    
    df["t"] = df["t"].astype(int)
    
    # Choose columns that exist
    cmap = _lower_map(df)
    cat_used = [c for c in cat_cols if c in df.columns]
    cyc7_used = [c for c in cyc7_cols if c in df.columns]
    num_used = [c for c in num_cols if c in df.columns]
    nutr_used_actual = [cmap[n.lower()] for n in default_nutrients if n.lower() in cmap]
    
    # Build feature blocks
    blocks, names = [], []
    
    if add_bias:
        blocks.append(np.ones((len(df), 1), dtype=float))
        names.append("bias")
    
    # Categorical features (one-hot encoding)
    for col in cat_used:
        ohe = pd.get_dummies(df[col].astype(str), prefix=col)
        if ohe.shape[1] > 0:
            blocks.append(ohe.to_numpy(dtype=float))
            names.extend(ohe.columns.tolist())
    
    # Cyclical features (7-day cycle)
    for col in cyc7_used:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        v = np.where((v >= 1) & (v <= 7), v - 1, v)  # map 1..7 → 0..6
        ang = 2.0 * np.pi * (v / 7.0)
        blocks.extend([np.sin(ang).reshape(-1, 1), np.cos(ang).reshape(-1, 1)])
        names.extend([f"{col}_sin", f"{col}_cos"])
    
    # Numerical features (standardized)
    for col in num_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")
    
    # Nutrient features (standardized)
    for col in nutr_used_actual:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")
    
    # Combine all features
    if blocks:
        X_all = np.concatenate(blocks, axis=1).astype(np.float32)
    else:
        X_all = np.zeros((len(df), 0), dtype=np.float32)
    
    # Create rows dataframe with shared mapping
    rows_df = df[["t", item_col]].copy()
    rows_df.columns = ["t", "description"]
    rows_df["item_idx"] = rows_df["description"].map(item_to_idx).astype(int)
    
    # Create groups dictionary
    groups = {int(t): idx.to_numpy(dtype=int) for t, idx in rows_df.groupby("t").groups.items()}
    
    # Metadata
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
    # Resolve base path (works in scripts and notebooks)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

    data = load_data(DATA_PATH)
    print(data.head())
    merged_data = pd.read_csv("/Users/ganeshkumarboini/Documents/testrepo/fall-2025-group8/src/data/data_with_timestamps.csv")

    print("Building item mapping...")
    item_to_idx, all_items = build_item_mapping(
        dataframe=merged_data,
        item_col="description")   
    print(f"Total unique items: {len(all_items)}")
     
    # Step 2: Build action matrix
    print("Building action matrix...")
    action_matrix = build_action_matrix(
        merged=merged_data,
        item_to_idx=item_to_idx,
        item_col="description"
    )
    
    # Step 3: Build feature matrix  
    print("Building feature matrix...")
    feature_matrix, feature_names, rows_df, groups, meta = build_feature_matrix(
        df_merged=merged_data,
        item_to_idx=item_to_idx,
        item_col="description",
        use_time_slot_id=True,
        cat_cols=[],
        cyc7_cols=[],
        num_cols=[],
        default_nutrients=None,
        add_bias=False
    )
    print("Feature matrix shape:", feature_matrix.shape)
    print("Feature names:", feature_names)
# ===================== Actions & Features (complex) =====================
    save_action_matrix(action_matrix, "out/action_matrix.csv", item_to_idx)
    save_feature_matrix(feature_matrix, feature_names, rows_df, "out/feature_matrix.csv")
    print("✓ All matrices saved successfully!")
