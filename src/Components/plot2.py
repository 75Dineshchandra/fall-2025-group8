# src/plots/linucb_learning_performance_full4.py
"""
LinUCB Learning Performance (VARIANT1, λ=0.3) — 4 panels:
1) Rolling Avg Reward
2) Rolling Avg Regret
3) Rolling Avg Sales (raw)
4) Rolling Avg Health (raw)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- repo path ---
HERE = Path(__file__).resolve().parent
SRC  = HERE if HERE.name == "src" else HERE.parent
sys.path.insert(0, str(SRC))

# --- project imports ---
try:
    from Components.model import (
        LinUCB, load_feature_matrix, load_action_matrix,
        compute_rewards_for_lambda, _build_bandit_tensors
    )
except Exception:
    from model import (
        LinUCB, load_feature_matrix, load_action_matrix,
        compute_rewards_for_lambda, _build_bandit_tensors
    )

DATA_DIR = SRC / "data"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"
ACTION_MATRIX  = DATA_DIR / "action_matrix.csv"
MERGED_DATA    = DATA_DIR / "data_healthscore_mapped.csv"
TIMESLOTS      = DATA_DIR / "time_slot_mapping.csv"

RESULTS_DIR = SRC / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.80
ROLL_W = 50

# ---------- helpers ----------
def rolling_mean(x, w):
    s = pd.Series(x)
    return s.expanding().mean().to_numpy()



def build_time_item_lookup(merged_csv, timeslots_csv):
    """(time_slot_id, item_name) -> (raw_sales_sum, raw_health_median)"""
    ts = pd.read_csv(timeslots_csv, low_memory=False)
    key2id = dict(zip(
        zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
        ts["time_slot_id"].astype(int)
    ))
    m = pd.read_csv(merged_csv, low_memory=False)
    for c in ["date","school_name","time_of_day","description"]:
        m[c] = m[c].astype(str).str.strip()
    m["time_slot_key"] = list(zip(m["date"], m["school_name"], m["time_of_day"]))
    m["time_slot_id"]  = m["time_slot_key"].map(key2id)
    m = m.dropna(subset=["time_slot_id"]).copy()
    m["time_slot_id"] = m["time_slot_id"].astype(int)

    agg = m.groupby(["time_slot_id","description"], as_index=False).agg(
        sales=("total","sum"),
        health=("HealthScore","median"),
    )
    return {
        (int(r.time_slot_id), str(r.description)): (float(r.sales), float(r.health))
        for r in agg.itertuples(index=False)
    }

def make_meta_item_map(metadata_df):
    """Map (t, arm) -> item_name for fast lookup."""
    return {
        (int(r.time_slot_id), int(r.item_idx)): str(r.item)
        for r in metadata_df.itertuples(index=False)
    }

# ---------- training + tracking ----------
def train_linucb_and_track(
    feature_array, action_matrix, metadata_df, rewards, alpha=1.0, ridge=1.0
):
    data_TAD, rewards_TA, mask_TA = _build_bandit_tensors(
        feature_array, metadata_df, rewards, action_matrix
    )
    T, A, d = data_TAD.shape
    bandit = LinUCB(d=d, n_arms=A, alpha=alpha, l2=ridge, seed=42)

    # lookups for raw sales/health
    meta_item = make_meta_item_map(metadata_df)
    time_item_lookup = build_time_item_lookup(MERGED_DATA, TIMESLOTS)

    rows = []
    for t in range(T):
        avail = np.where(mask_TA[t])[0]
        if avail.size == 0:
            continue
        feats = {a: data_TAD[t, a] for a in avail}
        a = bandit.select_action(list(avail), feats)

        r = float(rewards_TA[t, a])
        oracle = float(np.max(rewards_TA[t, avail]))
        regret = oracle - r

        # raw sales/health for the actually chosen (t, item_name)
        item_name = meta_item.get((t, a))
        if item_name is not None and (t, item_name) in time_item_lookup:
            sales_raw, health_raw = time_item_lookup[(t, item_name)]
        else:
            sales_raw, health_raw = 0.0, 5.0

        rows.append({
            "t": t,
            "reward": r,
            "oracle": oracle,
            "regret": regret,
            "sales": sales_raw,
            "health": health_raw,
        })

        bandit.update_arm(a, feats[a], r)

        if (t+1) % 2000 == 0:
            print(f"[t={t+1}/{T}] avg reward last2k = {np.mean([rr['reward'] for rr in rows[-2000:]]):.3f}")

    df = pd.DataFrame(rows)

    # Rolling avgs (short-term)
    df["roll_reward"] = rolling_mean(df["reward"].to_numpy(), ROLL_W)
    df["roll_regret"] = rolling_mean(df["regret"].to_numpy(), ROLL_W)
    df["roll_sales"]  = rolling_mean(df["sales"].to_numpy(),  ROLL_W)
    df["roll_health"] = rolling_mean(df["health"].to_numpy(), ROLL_W)
    return df

# ---------- plotting (8 panels) ----------

def plot_rolling4(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LinUCB Learning Performance (λ={LAMBDA})", fontsize=16, fontweight="bold", y=0.995)

    # 1) Rolling Avg Reward
    ax = axes[0,0]
    ax.plot(df["t"], df["roll_reward"], lw=2.2, color="tab:blue")
    ax.set_title(f"Rolling Avg Reward (window={ROLL_W})"); ax.set_ylabel("Avg Reward"); ax.grid(True, alpha=0.3)

    # 2) Rolling Avg Regret
    ax = axes[0,1]
    ax.plot(df["t"], df["roll_regret"], lw=2.2, color="tab:red")
    ax.axhline(0, ls="--", lw=1.2, color="tab:green", alpha=0.6)
    ax.set_title(f"Rolling Avg Regret (window={ROLL_W})"); ax.set_ylabel("Avg Regret/Step"); ax.grid(True, alpha=0.3)

    # 3) Rolling Avg Sales (raw)
    ax = axes[1,0]
    ax.plot(df["t"], df["roll_sales"], lw=2.2, color="tab:purple")
    ax.set_title(f"Rolling Avg Sales (raw, window={ROLL_W})"); ax.set_ylabel("Avg Sales"); ax.grid(True, alpha=0.3)

    # 4) Rolling Avg Health (raw)
    ax = axes[1,1]
    ax.plot(df["t"], df["roll_health"], lw=2.2, color="tab:orange")
    ax.set_title(f"Rolling Avg Health (raw, window={ROLL_W})"); ax.set_ylabel("Avg Health"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / f"linucb_learning_performance_lambda_{LAMBDA}_rolling4.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✓ Saved rolling 4-panel plot: {out_png}")
    plt.show()


def main():
    print("="*72)
    print("LINUCB LEARNING PERFORMANCE — ROLLING 4 PANELS")
    print("="*72)

    X, meta, _ = load_feature_matrix(str(FEATURE_MATRIX))
    A = load_action_matrix(str(ACTION_MATRIX))
    print(f"Feature matrix: {X.shape} | Action matrix: {A.shape}")

    rewards_vec = compute_rewards_for_lambda(
        lambda_value=LAMBDA,
        feature_matrix_file=str(FEATURE_MATRIX),
        merged_data_file=str(MERGED_DATA),
        time_slot_mapping_file=str(TIMESLOTS),
    )

    df = train_linucb_and_track(X, A, meta, rewards_vec, alpha=1.0, ridge=1.0)

    # save data + plot
    out_csv = RESULTS_DIR / f"linucb_performance_lambda_{LAMBDA}_rolling4.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved performance CSV: {out_csv}")

    plot_rolling4(df)

if __name__ == "__main__":
    main()
