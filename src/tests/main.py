# main.py — load trained model and recommend for a given (date, school, meal)
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

# --- make sure src/Components is importable ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Components.model import LinUCB  # keep case consistent with folder name

# --- correct file paths ---
BASE = Path(__file__).resolve().parents[1] / "data"
FEATURE_PATH = BASE / "feature_matrix.csv"
ACTION_PATH = BASE / "action_matrix.csv"
MERGED_PATH = BASE / "fcps_data_with_timestamps.csv"   # used to locate t for a context
MODEL_PATH = Path(__file__).resolve().parent / "trained_linucb.joblib"

# --- helper functions ---
def load_features(path: Path):
    df = pd.read_csv(path, low_memory=False)
    meta_cols = ["time_slot_id", "item", "item_idx"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    rows = df[meta_cols].copy()
    # group map for quick access
    groups = rows.groupby("time_slot_id").groups
    return X, rows, feat_cols, groups

def load_availability(path: Path):
    A = pd.read_csv(path, low_memory=False)
    item_cols = [c for c in A.columns if c.startswith("item_")]
    return A[item_cols].to_numpy(dtype=np.int32)

# --- main recommendation function ---
def recommend_for(date_str: str, school_name: str, time_of_day: str, topk: int = 5):
    # 1️⃣ load data & model
    if not MODEL_PATH.exists():
        print(f" Trained model not found at {MODEL_PATH}. Train and save first.")
        return

    X, rows_df, _, groups = load_features(FEATURE_PATH)
    avail = load_availability(ACTION_PATH)
    merged = pd.read_csv(MERGED_PATH, low_memory=False)
    model = LinUCB.load(str(MODEL_PATH))

    # 2️⃣ filter by context (day, school, meal)
    mask = (
        (merged["date"].astype(str) == date_str)
        & (merged["school_name"].astype(str) == school_name)
        & (merged["time_of_day"].astype(str) == time_of_day)
    )
    slot_ids = sorted(merged.loc[mask, "time_slot_id"].dropna().astype(int).unique())
    if not slot_ids:
        print(f" No time slots found for {date_str} | {school_name} | {time_of_day}")
        return

    # 3️⃣ generate recommendations per slot
    for t in slot_ids:
        if t not in groups:
            print(f"(t={t}) not present in feature_matrix; skipping.")
            continue
        if t >= avail.shape[0]:
            print(f"(t={t}) exceeds availability matrix rows; skipping.")
            continue

        ridxs = list(groups[t])
        available = np.where(avail[t] == 1)[0].tolist()

        x_by_arm = {}
        for ridx in ridxs:
            a = int(rows_df.iloc[ridx]["item_idx"])
            if a in available:
                # rebuild feature matrix subset row-by-row
                # X[ridx] aligns with rows_df row ridx
                x_by_arm[a] = X[ridx]

        if not x_by_arm:
            print(f"(t={t}) no available items; skipping.")
            continue

        recs = model.recommend(x_by_arm, topk=topk)

        print(f"\nTop {topk} for {date_str} | {school_name} | {time_of_day} | t={t}")
        for rank, (arm, score) in enumerate(recs, 1):
            item = rows_df[rows_df["item_idx"] == arm]["item"].iloc[0]
            print(f"  {rank}. {item} (arm={arm}, UCB={score:.3f})")

# --- example usage ---
if __name__ == "__main__":
    recommend_for(
        date_str="2025-03-03",
        school_name="COLVIN_RUN_ELEMENTARY",
        time_of_day="lunch",
        topk=5
    )
