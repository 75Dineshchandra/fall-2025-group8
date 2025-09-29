import numpy as np
import pandas as pd
import json
import pickle


# %%
import numpy as np
import pandas as pd
import json
import pickle

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pandas.DataFrame: The loaded data as a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data


# %%
def build_action_matrix(merged, strict=False):
    """
    Build an action matrix (time_slots x items) and item mapping.

    Args:
        merged (pd.DataFrame): Must include 'time_slot_id' and 'description'.
        strict (bool): If True, raises error if new items are found outside existing mapping.

    Returns:
        action_matrix (np.ndarray): Binary matrix indicating available items per time slot.
        all_items (list): Ordered list of all unique items.
        item_to_idx (dict): Mapping of item name -> column index in action matrix.
        time_slots (pd.DataFrame): Unique time slot information (time_slot_id, date, school_code).
    """

    # -------------------- 1) Load or initialize item_to_idx --------------------
    mapping_json = "item_to_idx.json"

    try:
        with open(mapping_json, "r") as f:
            item_to_idx = json.load(f)
    except FileNotFoundError:
        item_to_idx = {}

    # -------------------- 2) Extend mapping with new items --------------------
    for item in merged["description"].unique():
        if item not in item_to_idx:
            if strict:
                raise ValueError(f"New item '{item}' not in mapping and strict=True.")
            item_to_idx[item] = len(item_to_idx)

    # -------------------- 3) Build ordered list of items --------------------
    all_items = [None] * len(item_to_idx)
    for item, idx in item_to_idx.items():
        all_items[idx] = item

    num_items = len(all_items)
    num_time_slots = merged["time_slot_id"].max() + 1

    # -------------------- 4) Initialize action matrix --------------------
    action_matrix = np.zeros((num_time_slots, num_items), dtype=int)

    # -------------------- 5) Fill matrix using groupby --------------------
    grouped = merged.groupby("time_slot_id")["description"].unique()
    for time_slot_id, items_in_slot in grouped.items():
        for item in items_in_slot:
            action_matrix[time_slot_id, item_to_idx[item]] = 1

    return action_matrix, all_items, item_to_idx


# %%
def build_feature_matrix(
    df_merged: pd.DataFrame,
    *,
    item_col: str = "description",            # Column holding the item/arm name
    use_time_slot_id: bool = True,            # Reuse existing 'time_slot_id' as timestep id 't'
    cat_cols: list[str] = (),                 # Categorical context columns (one-hot)
    cyc7_cols: list[str] = ("day_of_week",),  # Weekly cyclical context columns (sin/cos)
    num_cols: list[str] = (),                 # Numeric context columns (z-score)
    default_nutrients: list[str] = (          # Nutrient columns to include (z-score)
        "GramsPerServing","Calories","Protein","Total Carbohydrate","Dietary Fiber",
        "Total Sugars","Added Sugars","Total Fat","Saturated Fat","Trans Fat",
        "Cholesterol","Sodium","Vitamin D (D2 + D3)","Calcium","Iron","Potassium",
        "Vitamin A","Vitamin C"
    ),
    add_bias: bool = True                     # Whether to prepend a bias column
):
    """
    Build feature matrix for CMAB / LinUCB using nutrients and optional context.
    Assumes 'time_slot_id' exists if use_time_slot_id=True.
    """

    # ------------------ Helper Functions ------------------

    def _lower_map(df: pd.DataFrame) -> dict[str, str]:
        """
        Map lowercase column names to actual df columns.
        Useful for case-insensitive nutrient matching.
        """
        return {c.lower(): c for c in df.columns}

    def _zscore_1d(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Z-score a 1D array safely. Returns (z, mean, std).
        If std is zero or invalid, uses 1 to avoid division by zero.
        """
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        return (x - mu) / sd, mu, sd

    # ------------------ 0) Copy & Validate ------------------

    df = df_merged.copy()

    # Ensure the item column exists and is string
    if item_col not in df.columns:
        raise ValueError(f"'{item_col}' not found in dataframe columns: {list(df.columns)}")
    df[item_col] = df[item_col].astype(str)

    # ------------------ 1) Reuse existing time_slot_id ------------------

    if use_time_slot_id and "time_slot_id" in df.columns:
        # Rename to standard 't' for timestep
        df = df.rename(columns={"time_slot_id": "t"})
    elif "t" not in df.columns:
        raise KeyError(
            "No 't' column found. Provide 'time_slot_id' in your dataframe or a 't' column."
        )
    df["t"] = df["t"].astype(int)  # enforce integer type

    # ------------------ 2) Determine which columns exist ------------------

    # Only use columns that actually exist in df
    cat_used = [c for c in cat_cols if c in df.columns]
    cyc7_used = [c for c in cyc7_cols if c in df.columns]
    num_used = [c for c in num_cols if c in df.columns]

    # Resolve nutrient columns case-insensitively
    cmap = _lower_map(df)
    nutr_used_actual = [cmap[n.lower()] for n in default_nutrients if n.lower() in cmap]

    # ------------------ 3) Assemble feature blocks ------------------

    blocks, names = [], []

    # 3a) Bias term
    if add_bias:
        blocks.append(np.ones((len(df), 1), dtype=float))
        names.append("bias")

    # 3b) One-hot encode categorical context
    for col in cat_used:
        ohe = pd.get_dummies(df[col].astype(str), prefix=col)
        if ohe.shape[1] > 0:
            blocks.append(ohe.to_numpy(dtype=float))
            names.extend(ohe.columns.tolist())

    # 3c) Encode cyclical weekly context as sin/cos
    for col in cyc7_used:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        # Shift 1..7 to 0..6 if necessary
        v = np.where((v >= 1) & (v <= 7), v - 1, v)
        ang = 2.0 * np.pi * (v / 7.0)
        blocks.extend([np.sin(ang).reshape(-1, 1), np.cos(ang).reshape(-1, 1)])
        names.extend([f"{col}_sin", f"{col}_cos"])

    # 3d) Z-score numeric context columns
    for col in num_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")

    # 3e) Z-score nutrient columns
    for col in nutr_used_actual:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any():
            x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _zscore_1d(x)
        blocks.append(z.reshape(-1, 1))
        names.append(f"{col}_z")

    # Concatenate all feature blocks into final matrix
    X_all = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(df), 0), dtype=np.float32)

    # ------------------ 4) Build row mapping & groups ------------------

    # rows_df aligns with X_all rows: timestep, item, item index
    rows_df = df[["t", item_col]].copy()
    rows_df.columns = ["t", "description"]

    # Map items to stable integer indices
    item_ids = sorted(df[item_col].unique())
    item2idx = {it: i for i, it in enumerate(item_ids)}
    rows_df["item_idx"] = rows_df["description"].map(item2idx).astype(int)

    # For each timestep, store indices of available items
    groups = {int(t): idx.to_numpy(dtype=int) for t, idx in rows_df.groupby("t").groups.items()}

    # ------------------ 5) Package metadata ------------------

    meta = {
        "items": item_ids,
        "item2idx": item2idx,
        "used_nutrients": nutr_used_actual,
        "used_cat": cat_used,
        "used_cyc7": cyc7_used,
        "used_num": num_used,
    }

    return X_all, names, rows_df, groups, meta


#%%

def health_score(row):
    # Daily Values (DV) per school group
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

    # Nutrients classification
    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium",
            "Iron", "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    # Determine school group (default high school)
    school_group = str(row.get("school_group", "high")).lower()
    if "elementary" in school_group:
        dv = DV["elementary"]
    elif "middle" in school_group:
        dv = DV["middle"]
    elif "high" in school_group:
        dv = DV["high"]
    else:
        dv = DV["high"]

    good_score = 0
    bad_score = 0

    # Calculate %DV for good nutrients
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val / ref) * 100)

    # Calculate %DV for bad nutrients
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val / ref) * 100)

    return good_score - bad_score




# %%
import os

# Get the directory of the current file (env.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  

# Construct path to data/sales.csv
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

data = load_data(DATA_PATH)
# --- Build the action matrix ---
action_matrix, all_items, item_to_idx = build_action_matrix(merged)

# --- Save matrix ---
np.save("action_matrix.npy", action_matrix)
with open("action_matrix.pkl", "wb") as f:
    pickle.dump(action_matrix, f)
pd.DataFrame(action_matrix, columns=all_items).to_csv("action_matrix.csv", index=False)

# --- Save item_to_idx mapping ---
with open("item_to_idx.json", "w") as f:
    json.dump(item_to_idx, f)
with open("item_to_idx.pkl", "wb") as f:
    pickle.dump(item_to_idx, f)

    # -------------------- 6) Prepare time slots --------------------
time_slots = merged[["time_slot_id", "date", "school_code"]].drop_duplicates().sort_values(["school_code", "date"])
# --- Save time slots ---
time_slots.to_csv("time_slots_info.csv", index=False)

print("All action matrix files saved successfully!")

print(data.head())



# --- Step 1: Build the feature matrix ---
X_all, feature_names, rows_df, groups, meta = build_feature_matrix(
    merged,
    cat_cols=[],       # no categorical context
    cyc7_cols=[],      # no cyclical context
    num_cols=[],
    add_bias=False     # nutrient-only benchmark, no bias
)

# --- Step 2: Save the outputs ---
# Save matrix as .npy
np.save("feature_matrix.npy", X_all)

# Save matrix + metadata as .pkl
with open("feature_matrix.pkl", "wb") as f:
    pickle.dump({
        "X_all": X_all,
        "feature_names": feature_names,
        "groups": groups,
        "meta": meta
    }, f)

# Save row-to-timestep mapping as CSV
rows_df.to_csv("feature_rows.csv", index=False)

# Save feature names separately as CSV
pd.DataFrame({"feature_names": feature_names}).to_csv("feature_names.csv", index=False)

print("All feature files saved successfully!")


# %%
num_actions = len(actions)
print("Number of unique actions:", num_actions)

#%%
output_file = "scored_data.csv"
df=data.copy()
# Compute health score
df["HealthScore"] = df.apply(health_score, axis=1)
# Save results
df.to_csv(output_file, index=False)
print(f"✅ Health scores calculated and saved to {output_file}")




























# %%
"""def health_score(row, min_score=None, max_score=None):
    # Daily Values (DV) per school group
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

    good_score = 0
    bad_score = 0

    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val / ref) * 100)

    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val / ref) * 100)

    raw_score = good_score - bad_score

    # If min/max scores are not provided, use default range
    if min_score is None:
        min_score = -300  # worst case negative score
    if max_score is None:
        max_score = 800   # approximate best-case total for good nutrients

    # Scale to 0-10 range
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(10, scaled_score))  # clamp to [0,10]

    return round(scaled_score, 1)
"""