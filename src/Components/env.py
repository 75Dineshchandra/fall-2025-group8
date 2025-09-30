import os
import json
import pickle
import numpy as np
import pandas as pd

# ===================== IO =====================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

# ===================== Actions & Features (simple) =====================

def get_actions(data: pd.DataFrame) -> np.ndarray:
    """
    Return unique actions from either 'sales_item' or 'description'.
    """
    data = pd.DataFrame(data)
    col = "sales_item" if "sales_item" in data.columns else "description"
    if col not in data.columns:
        raise KeyError("Neither 'sales_item' nor 'description' column was found.")
    return data[col].astype(str).unique()

def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return the feature frame (kept simple here: returns the input df).
    """
    return pd.DataFrame(data)

# ===================== Health score =====================

def health_score(row: pd.Series) -> float:
    """
    Simple health score = sum(%DV for good) - sum(%DV for bad), capped at 100 per nutrient.
    """
    DV = {
        "elementary": {
            "Calories": 1600, "Protein": 19, "Total Carbohydrate": 130,
            "Dietary Fiber": 25, "Added Sugars": 25, "Total Fat": 40,
            "Saturated Fat": 20, "Sodium": 1500, "Vitamin D": 20,
            "Calcium": 1000, "Iron": 10, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "middle": {
            "Calories": 2200, "Protein": 34, "Total Carbohydrate": 130,
            "Dietary Fiber": 31, "Added Sugars": 50, "Total Fat": 77,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "high": {
            "Calories": 2600, "Protein": 46, "Total Carbohydrate": 130,
            "Dietary Fiber": 38, "Added Sugars": 50, "Total Fat": 91,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        }
    }
    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium",
            "Iron", "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    school_group = str(row.get("school_group", "high")).lower()
    if "elementary" in school_group:
        dv = DV["elementary"]
    elif "middle" in school_group:
        dv = DV["middle"]
    else:
        dv = DV["high"]

    good_score = 0.0
    bad_score = 0.0
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val / ref) * 100)
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val / ref) * 100)
    return good_score - bad_score

# ===================== Action matrix =====================

def build_action_matrix(
    merged: pd.DataFrame,
    *,
    item_col: str = "description",
    strict: bool = False,
    mapping_json: str = "item_to_idx.json",
):
    """
    Build a (time_slots x items) binary matrix indicating available items per time slot.
    Requires columns: 'time_slot_id' and <item_col>.
    """
    if "time_slot_id" not in merged.columns:
        raise KeyError("Required column 'time_slot_id' not found.")
    if item_col not in merged.columns:
        raise KeyError(f"Required item column '{item_col}' not found.")

    # Load existing mapping (if any)
    try:
        with open(mapping_json, "r") as f:
            item_to_idx = json.load(f)
    except FileNotFoundError:
        item_to_idx = {}

    # Extend mapping with new items
    for item in merged[item_col].astype(str).unique():
        if item not in item_to_idx:
            if strict:
                raise ValueError(f"New item '{item}' not in mapping and strict=True.")
            item_to_idx[item] = len(item_to_idx)

    # Ordered list of items
    all_items = [None] * len(item_to_idx)
    for item, idx in item_to_idx.items():
        all_items[idx] = item

    num_items = len(all_items)
    num_time_slots = int(merged["time_slot_id"].max()) + 1

    # Fill matrix
    action_matrix = np.zeros((num_time_slots, num_items), dtype=int)
    grouped = merged.groupby("time_slot_id")[item_col].unique()
    for time_slot_id, items_in_slot in grouped.items():
        for item in items_in_slot:
            action_matrix[int(time_slot_id), item_to_idx[str(item)]] = 1

    return action_matrix, all_items, item_to_idx

# ===================== Feature matrix (nutrients + optional context) =====================

def build_feature_matrix(
    df_merged: pd.DataFrame,
    *,
    item_col: str = "description",
    use_time_slot_id: bool = True,
    cat_cols: list[str] = (),
    cyc7_cols: list[str] = ("day_of_week",),
    num_cols: list[str] = (),
    default_nutrients: list[str] = (
        "GramsPerServing","Calories","Protein","Total Carbohydrate","Dietary Fiber",
        "Total Sugars","Added Sugars","Total Fat","Saturated Fat","Trans Fat",
        "Cholesterol","Sodium","Vitamin D (D2 + D3)","Calcium","Iron","Potassium",
        "Vitamin A","Vitamin C"
    ),
    add_bias: bool = True,
):
    """
    Build feature matrix for CMAB/LinUCB. Returns (X_all, feature_names, rows_df, groups, meta).
    """
    def _lower_map(df: pd.DataFrame) -> dict[str, str]:
        return {c.lower(): c for c in df.columns}

    def _zscore_1d(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        return (x - mu) / sd, mu, sd

    df = df_merged.copy()
    if item_col not in df.columns:
        raise ValueError(f"'{item_col}' not found in dataframe columns: {list(df.columns)}")
    df[item_col] = df[item_col].astype(str)

    # timestep column
    if use_time_slot_id and "time_slot_id" in df.columns:
        df = df.rename(columns={"time_slot_id": "t"})
    elif "t" not in df.columns:
        raise KeyError("No timestep found. Provide 'time_slot_id' or a precomputed 't' column.")
    df["t"] = df["t"].astype(int)

    # choose columns that exist
    cat_used = [c for c in cat_cols if c in df.columns]
    cyc7_used = [c for c in cyc7_cols if c in df.columns]
    num_used = [c for c in num_cols if c in df.columns]

    cmap = _lower_map(df)
    nutr_used_actual = [cmap[n.lower()] for n in default_nutrients if n.lower() in cmap]

    blocks, names = [], []

    if add_bias:
        blocks.append(np.ones((len(df), 1), dtype=float))
        names.append("bias")

    for col in cat_used:
        ohe = pd.get_dummies(df[col].astype(str), prefix=col)
        if ohe.shape[1] > 0:
            blocks.append(ohe.to_numpy(dtype=float))
            names.extend(ohe.columns.tolist())

    for col in cyc7_used:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        v = np.where((v >= 1) & (v <= 7), v - 1, v)  # map 1..7 → 0..6
        ang = 2.0 * np.pi * (v / 7.0)
        blocks.extend([np.sin(ang).reshape(-1, 1), np.cos(ang).reshape(-1, 1)])
        names.extend([f"{col}_sin", f"{col}_cos"])

    for col in num_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")

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

    item_ids = sorted(df[item_col].unique())
    item2idx = {it: i for i, it in enumerate(item_ids)}
    rows_df["item_idx"] = rows_df["description"].map(item2idx).astype(int)

    groups = {int(t): idx.to_numpy(dtype=int) for t, idx in rows_df.groupby("t").groups.items()}

    meta = {
        "items": item_ids,
        "item2idx": item2idx,
        "used_nutrients": nutr_used_actual,
        "used_cat": cat_used,
        "used_cyc7": cyc7_used,
        "used_num": num_used,
    }

    return X_all, names, rows_df, groups, meta

# ===================== Main =====================

if __name__ == "__main__":
    # Resolve base path (works in scripts and notebooks)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

    data = load_data(DATA_PATH)
    print(data.head())

    features = get_features(data)
    print("Features shape:", features.shape)

    actions = get_actions(data)
    print("Unique actions:", actions)
    print("Number of unique actions:", len(actions))

    # Compute and save health scores
    df = data.copy()
    df["HealthScore"] = df.apply(health_score, axis=1)
    output_file = "scored_data.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Health scores calculated and saved to {output_file}")

    # ---------- Optional: build action/feature matrices if columns exist ----------
    # Prepare dataframe expected by the matrix builders
    merged = data.copy()
    if "sales_item" in merged.columns and "description" not in merged.columns:
        merged = merged.rename(columns={"sales_item": "description"})

    # Action matrix (requires time_slot_id + description)
    if all(c in merged.columns for c in ["time_slot_id", "description"]):
        action_matrix, all_items, item_to_idx = build_action_matrix(merged, item_col="description")
        np.save("action_matrix.npy", action_matrix)
        with open("action_matrix.pkl", "wb") as f:
            pickle.dump(action_matrix, f)
        pd.DataFrame(action_matrix, columns=all_items).to_csv("action_matrix.csv", index=False)
        with open("item_to_idx.json", "w") as f:
            json.dump(item_to_idx, f)
        with open("item_to_idx.pkl", "wb") as f:
            pickle.dump(item_to_idx, f)
        time_cols = [c for c in ["time_slot_id", "date", "school_code"] if c in merged.columns]
        if time_cols:
            merged[time_cols].drop_duplicates().sort_values(time_cols).to_csv("time_slots_info.csv", index=False)
        print("✅ Action matrix artifacts saved.")
    else:
        print("ℹ️ Skipping action matrix (need columns: 'time_slot_id' and 'description').")

    # Feature matrix (requires nutrient/context columns + time_slot_id or t)
    if ("time_slot_id" in merged.columns or "t" in merged.columns) and "description" in merged.columns:
        X_all, feature_names, rows_df, groups, meta = build_feature_matrix(
            merged,
            item_col="description",
            cat_cols=[],   # customize if needed
            cyc7_cols=[],  # customize if needed
            num_cols=[],
            add_bias=False
        )
        np.save("feature_matrix.npy", X_all)
        with open("feature_matrix.pkl", "wb") as f:
            pickle.dump({
                "X_all": X_all,
                "feature_names": feature_names,
                "groups": groups,
                "meta": meta
            }, f)
        rows_df.to_csv("feature_rows.csv", index=False)
        pd.DataFrame({"feature_names": feature_names}).to_csv("feature_names.csv", index=False)
        print("✅ Feature matrix artifacts saved.")
    else:
        print("ℹ️ Skipping feature matrix (need 'description' and 'time_slot_id' or 't').")