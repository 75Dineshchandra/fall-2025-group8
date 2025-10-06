# single_script.py
# Run: python3 single_script.py
# Unifies outputs under src/data/, reuses existing matrices if present, and
# writes rewards.csv with extra context columns.

import os
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd

# ===================== Config =====================
DATA_DIR = Path("src/data")
SALES_NUTR_CSV = DATA_DIR / "data_sales_nutrition.csv"
MERGED_CSV     = DATA_DIR / "fcps_data_with_timestamps.csv"

ITEM_MAP_CSV   = DATA_DIR / "item_mapping.csv"
ACTION_CSV     = DATA_DIR / "action_matrix.csv"
FEATURE_CSV    = DATA_DIR / "feature_matrix.csv"
REWARD_CSV     = DATA_DIR / "rewards.csv"

OVERWRITE = False       # set True to force rebuild of matrices if files exist
LAMBDA_H  = 0.1         # weight for HealthScore_z in reward

# ===================== IO Helpers =====================

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def load_data(file_path: Path, text_col: str = "description") -> pd.DataFrame:
    dtype_overrides = {"school_name": "string", "time_of_day": "string"}
    dtype_overrides[text_col] = "string"
    df = pd.read_csv(file_path, low_memory=False, dtype=dtype_overrides, quotechar='"')
    if text_col in df.columns:
        df[text_col] = df[text_col].str.strip().replace({"": pd.NA})
    return df

# ===================== Health score (optional) =====================

def health_score(row: pd.Series) -> float:
    DV = {
        "elementary": {"Calories":1600,"Protein":19,"Total Carbohydrate":130,"Dietary Fiber":25,"Added Sugars":25,
                       "Total Fat":40,"Saturated Fat":20,"Sodium":1500,"Vitamin D":20,"Calcium":1000,"Iron":10,
                       "Potassium":4700,"Vitamin A":900,"Vitamin C":90},
        "middle":     {"Calories":2200,"Protein":34,"Total Carbohydrate":130,"Dietary Fiber":31,"Added Sugars":50,
                       "Total Fat":77,"Saturated Fat":20,"Sodium":2300,"Vitamin D":20,"Calcium":1300,"Iron":18,
                       "Potassium":4700,"Vitamin A":900,"Vitamin C":90},
        "high":       {"Calories":2600,"Protein":46,"Total Carbohydrate":130,"Dietary Fiber":38,"Added Sugars":50,
                       "Total Fat":91,"Saturated Fat":20,"Sodium":2300,"Vitamin D":20,"Calcium":1300,"Iron":18,
                       "Potassium":4700,"Vitamin A":900,"Vitamin C":90}
    }
    GOOD = ["Protein","Dietary Fiber","Vitamin D","Calcium","Iron","Potassium","Vitamin A","Vitamin C"]
    BAD  = ["Added Sugars","Saturated Fat","Sodium"]

    school_group = str(row.get("school_group", "high")).lower()
    dv = DV["elementary"] if "elementary" in school_group else (DV["middle"] if "middle" in school_group else DV["high"])

    good_score, bad_score = 0.0, 0.0
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val/ref)*100)
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val/ref)*100)
    return good_score - bad_score

# ===================== Build mapping & matrices =====================

def build_item_mapping(df: pd.DataFrame, item_col: str = "description", save_path: Path = ITEM_MAP_CSV) -> Tuple[Dict[str,int], List[str]]:
    valid_items = sorted(df[item_col].astype(str).unique())
    item_to_idx = {item: idx for idx, item in enumerate(valid_items)}
    mapping_df = pd.DataFrame({"item": valid_items, "item_idx": range(len(valid_items))})
    ensure_dir(save_path)
    mapping_df.to_csv(save_path, index=False)
    print(f"✓ Item mapping: {len(valid_items)} items → {save_path}")
    return item_to_idx, valid_items

def build_action_matrix(merged: pd.DataFrame, item_to_idx: Dict[str,int],
                        *, item_col: str = "description", time_slot_col: str = "time_slot_id") -> np.ndarray:
    if time_slot_col not in merged.columns: raise KeyError(f"Missing '{time_slot_col}'")
    if item_col not in merged.columns:      raise KeyError(f"Missing '{item_col}'")

    num_items = len(item_to_idx)
    tmax = int(pd.to_numeric(merged[time_slot_col], errors="coerce").max())
    num_time_slots = tmax + 1
    action = np.zeros((num_time_slots, num_items), dtype=int)

    for t, items in merged.groupby(time_slot_col)[item_col].unique().items():
        ti = int(t)
        for it in items:
            idx = item_to_idx.get(str(it))
            if idx is not None: action[ti, idx] = 1

    total_possible = num_time_slots * num_items
    coverage = 100 * action.sum() / total_possible if total_possible else 0
    print(f"✓ Action matrix: {num_time_slots}×{num_items} (coverage {coverage:.1f}%)")
    return action

def save_action_matrix(action: np.ndarray, save_path: Path, item_to_idx: Dict[str,int]) -> None:
    nT, nI = action.shape
    df = pd.DataFrame(action, columns=[f"item_{i}" for i in range(nI)])
    df["time_slot_id"] = range(nT)
    df = df[["time_slot_id"] + [c for c in df.columns if c.startswith("item_")]]
    ensure_dir(save_path)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved action matrix → {save_path}")

def build_feature_matrix(df_merged: pd.DataFrame, item_to_idx: Dict[str,int], *,
                         item_col: str = "description",
                         use_time_slot_id: bool = True,
                         cat_cols: List[str] = None,
                         cyc7_cols: List[str] = None,
                         num_cols: List[str] = None,
                         default_nutrients: List[str] = None,
                         add_bias: bool = False
                         ) -> Tuple[np.ndarray, List[str], pd.DataFrame, Dict[int,np.ndarray], Dict[str,Any]]:
    if cat_cols is None:  cat_cols = []
    if cyc7_cols is None: cyc7_cols = []
    if num_cols is None:  num_cols = []
    if default_nutrients is None:
        default_nutrients = ["GramsPerServing","Calories","Protein","Total Carbohydrate","Dietary Fiber",
                             "Total Sugars","Added Sugars","Total Fat","Saturated Fat","Trans Fat",
                             "Cholesterol","Sodium","Vitamin D (D2 + D3)","Calcium","Iron","Potassium",
                             "Vitamin A","Vitamin C"]

    def _lower_map(df: pd.DataFrame) -> Dict[str,str]:
        return {c.lower(): c for c in df.columns}

    def _z(x: np.ndarray) -> Tuple[np.ndarray,float,float]:
        mu = float(np.nanmean(x)); sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < 1e-8: sd = 1.0
        return (x - mu)/sd, mu, sd

    df = df_merged.copy()
    df[item_col] = df[item_col].astype(str)
    df = df[df[item_col].isin(item_to_idx.keys())].copy()

    if use_time_slot_id and "time_slot_id" in df.columns:
        df = df.rename(columns={"time_slot_id":"t"})
    elif "t" not in df.columns:
        raise KeyError("Need 'time_slot_id' or precomputed 't'")
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(int)

    cmap = _lower_map(df)
    cat_used = [c for c in (cat_cols or []) if c in df.columns]
    cyc_used = [c for c in (cyc7_cols or []) if c in df.columns]
    num_used = [c for c in (num_cols or []) if c in df.columns]
    nutr_used = [cmap[n.lower()] for n in default_nutrients if n.lower() in cmap]

    blocks, names = [], []
    if add_bias:
        blocks.append(np.ones((len(df),1), dtype=float)); names.append("bias")

    # categorical OHE
    for col in cat_used:
        ohe = pd.get_dummies(df[col].astype(str), prefix=col)
        if ohe.shape[1] > 0:
            blocks.append(ohe.to_numpy(dtype=float)); names.extend(ohe.columns.tolist())

    # cyclical 7-day
    for col in cyc_used:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy()
        v = np.where((v>=1) & (v<=7), v-1, v)
        ang = 2*np.pi*(v/7.0)
        blocks.append(np.sin(ang).reshape(-1,1)); names.append(f"{col}_sin")
        blocks.append(np.cos(ang).reshape(-1,1)); names.append(f"{col}_cos")

    # numeric z-scores
    for col in num_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any(): x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _z(x); blocks.append(z.reshape(-1,1)); names.append(f"{col}_z")

    # nutrient z-scores
    for col in nutr_used:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isnan(x).any(): x = np.where(np.isnan(x), float(np.nanmedian(x)), x)
        z, _, _ = _z(x); blocks.append(z.reshape(-1,1)); names.append(f"{col}_z")

    X = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((len(df),0), dtype=np.float32)

    rows_df = df[["t", item_col]].copy()
    rows_df.columns = ["t","description"]
    rows_df["item_idx"] = rows_df["description"].map(item_to_idx).astype(int)

    groups = {int(t): idx.to_numpy(dtype=int) for t, idx in rows_df.groupby("t").groups.items()}

    meta = {"n_features": X.shape[1], "n_samples": X.shape[0], "n_timesteps": len(groups),
            "used_cat": cat_used, "used_cyc7": cyc_used, "used_num": num_used,
            "used_nutrients": nutr_used, "feature_names": names}
    print(f"✓ Feature matrix: {X.shape[0]} rows × {X.shape[1]} features "
          f"({len(cat_used)} cat, {len(cyc_used)} cyc, {len(num_used)} num, {len(nutr_used)} nutr)")
    return X, names, rows_df, groups, meta

def save_feature_matrix(X: np.ndarray, feature_names: List[str], rows_df: pd.DataFrame, save_path: Path) -> None:
    n_rows, n_feats = X.shape
    if len(feature_names) != n_feats:
        raise ValueError(f"feature_names ({len(feature_names)}) != n_features ({n_feats})")
    need = {"t","description","item_idx"}
    if need - set(rows_df.columns):
        raise KeyError(f"rows_df missing: {need - set(rows_df.columns)}")
    df = pd.DataFrame(X, columns=feature_names)
    df["time_slot_id"] = rows_df["t"].values
    df["item"] = rows_df["description"].values
    df["item_idx"] = rows_df["item_idx"].values
    meta_cols = ["time_slot_id","item","item_idx"]
    df = df[meta_cols + [c for c in df.columns if c not in meta_cols]]
    ensure_dir(save_path)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved feature matrix → {save_path}")

# ===================== Reward builder (with extra columns) =====================

def build_and_save_rewards(feature_csv: Path, merged_csv: Path, out_csv: Path, lambda_h: float = 0.1) -> None:
    feat = pd.read_csv(feature_csv, low_memory=False)
    rows_df = feat[["time_slot_id","item","item_idx"]].rename(columns={"time_slot_id":"t","item":"description"}).copy()
    rows_df["t"] = pd.to_numeric(rows_df["t"], errors="coerce").astype(int)
    rows_df["description"] = rows_df["description"].astype(str)

    merged = pd.read_csv(merged_csv, low_memory=False)
    merged = merged.rename(columns={"time_slot_id":"t"})
    merged["t"] = pd.to_numeric(merged["t"], errors="coerce").astype(int)
    merged["description"] = merged["description"].astype(str)

    # Aggregate to avoid dupes per (t, description); also collect context you asked for
    agg = (merged.groupby(["t","description"], as_index=False)
                 .agg(total=("total","sum"),
                      HealthScore=("HealthScore","median"),
                      school_code=("school_code","first"),
                      school_name=("school_name","first"),
                      time_of_day=("time_of_day","first"),
                      day_name=("day_name","first")))

    aligned = rows_df.merge(agg, on=["t","description"], how="left", validate="m:1")
    aligned["total"] = pd.to_numeric(aligned["total"], errors="coerce").fillna(0.0)
    aligned["HealthScore"] = pd.to_numeric(aligned["HealthScore"], errors="coerce")
    hs = aligned["HealthScore"].to_numpy(dtype=float)
    hs = np.where(np.isnan(hs), np.nanmedian(hs), hs)

    mu = float(np.nanmean(hs)); sd = float(np.nanstd(hs))
    if not np.isfinite(sd) or sd < 1e-8: sd = 1.0
    hs_z = (hs - mu) / sd

    reward = aligned["total"].to_numpy(dtype=float) + (lambda_h * hs_z)

    out = aligned[["t","item_idx","description","school_code","school_name","time_of_day","day_name"]].copy()
    out["reward"] = reward

    ensure_dir(out_csv)
    out.to_csv(out_csv, index=False)
    print(f"✓ Rewards saved → {out_csv}  (cols: {list(out.columns)})")

# ===================== Main =====================

def main():
    ensure_dir(DATA_DIR)

    # Load merged once
    merged = load_data(MERGED_CSV, text_col="description")

    # Item mapping
    if ITEM_MAP_CSV.exists() and not OVERWRITE:
        item_map_df = pd.read_csv(ITEM_MAP_CSV)
        item_to_idx = dict(zip(item_map_df["item"].astype(str), item_map_df["item_idx"].astype(int)))
        print(f"• Using existing item mapping: {ITEM_MAP_CSV}")
    else:
        item_to_idx, _ = build_item_mapping(merged, item_col="description", save_path=ITEM_MAP_CSV)

    # Action matrix
    if ACTION_CSV.exists() and not OVERWRITE:
        print(f"• Using existing action matrix: {ACTION_CSV}")
    else:
        action = build_action_matrix(merged, item_to_idx, item_col="description", time_slot_col="time_slot_id")
        save_action_matrix(action, ACTION_CSV, item_to_idx)

    # Feature matrix
    if FEATURE_CSV.exists() and not OVERWRITE:
        print(f"• Using existing feature matrix: {FEATURE_CSV}")
    else:
        # Keep your earlier minimal feature set (nutrients only).
        X, names, rows_df, groups, meta = build_feature_matrix(
            df_merged=merged,
            item_to_idx=item_to_idx,
            item_col="description",
            use_time_slot_id=True,
            cat_cols=[],            # set to ["school_name","time_of_day"] if you want those included
            cyc7_cols=[],           # set to ["day_of_week"] if you want sin/cos
            num_cols=[],            # set to ["HealthScore"] to include it directly
            default_nutrients=None, # uses default list inside
            add_bias=False
        )
        save_feature_matrix(X, names, rows_df, FEATURE_CSV)

    # Rewards with extra columns (school_code/name/time_of_day/day_name)
    build_and_save_rewards(FEATURE_CSV, MERGED_CSV, REWARD_CSV, lambda_h=LAMBDA_H)

if __name__ == "__main__":
    main()
'''
Because we want each item’s healthiness to be measured relative to the whole population of items, not just what was served in one cafeteria on one day.
Step-by-step what actually happens
Let’s say your full fcps_data_with_timestamps.csv has 224,000 rows.
When the code runs:
hs = aligned["HealthScore"].to_numpy(dtype=float)
hs_mu = np.nanmean(hs)
hs_sd = np.nanstd(hs)
hs_z = (hs - hs_mu) / hs_sd
Here’s what that means:
It collects all HealthScores for all rows (each row = one (time_slot_id, item) pair).
So it might look like: [3.1, 3.3, 2.9, 4.8, 3.7, …]
Then it computes:
mean = 3.6
std = 0.5
Then it transforms every HealthScore:
For example: z = (3.8 – 3.6) / 0.5 = +0.4
So each item now says, “I’m +0.4 healthier than the average item in the whole dataset.”
So every HealthScore becomes a relative health value, comparable across all contexts.
But what happens within Monday or a specific school?
The context (Monday / school / time of day) still matters —
but it lives in your feature_matrix.csv, not in the z-score computation.
So when you compute rewards, the model still knows:
This row is Colvin Run Elementary
This time is Monday breakfast
This item is Cereal Meal
Health z-score = +0.4 (healthier than average)
Total sold = 19
→ reward = 19 + 0.1 × 0.4 = 19.04
At another time slot (say, Tuesday lunch at a different school), the same item might have the same z-score (+0.4) but a different sales count, so its reward will change (e.g., 28 + 0.1 × 0.4 = 28.04).
So:
The z-score part is global (how healthy an item is overall)
The sales (total) part is context-specific (how well it performed that day, at that school, during that meal)
When you combine them:
Reward = context-specific popularity + small boost/penalty for global healthiness.
 Analogy
Think of it like movie ratings 🎬:
The z-score is like a movie’s IMDB rating (global quality across all viewers).
The sales are like its ticket sales at a particular cinema on Monday.
The reward combines both — a movie that sells well and is high-rated gets the highest reward.
TL;DR
Concept	Scope	Why
HealthScore mean/std	Global (across all data)	So each item’s health value is relative to all items
total sales	Local (per time slot / school / day)	Reflects actual performance that day
reward = total + λ × z(HealthScore)	Combines global health + local popularity	Gives one numeric signal to the bandit
Context (school, day, time)	Stored in feature_matrix.csv	Keeps reward aligned with its scenario
So when you see each number in reward.csv,
it represents “How this specific item performed in its specific time slot, plus a small health boost based on its overall nutritional standing.”

'''