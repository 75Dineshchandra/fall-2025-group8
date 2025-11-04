# src/plots/linucb_learning_performance_full8.py
"""
LinUCB Learning Performance (VARIANT1, λ=0.3) — 8 panels:
1) Cumulative Reward          2) Rolling Avg Reward
3) Cumulative Regret          4) Rolling Avg Regret
5) Cumulative Sales (raw)     6) Rolling Avg Sales (raw)
7) Cumulative Health (raw)    8) Rolling Avg Health (raw)
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
    return s.rolling(window=w, min_periods=1).mean().to_numpy()

def expanding_sum(x):
    s = pd.Series(x)
    return s.cumsum().to_numpy()

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

    # Cumulative (long-term)
    df["cum_reward"] = expanding_sum(df["reward"].to_numpy())
    df["cum_regret"] = expanding_sum(df["regret"].to_numpy())
    df["cum_sales"]  = expanding_sum(df["sales"].to_numpy())
    df["cum_health"] = expanding_sum(df["health"].to_numpy())

    return df

# ---------- plotting (8 panels) ----------
def plot_full8(df):
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f"LinUCB Learning Performance (λ={LAMBDA})", fontsize=16, fontweight="bold", y=0.995)

    # 1) Cumulative Reward
    ax = axes[0,0]
    ax.plot(df["t"], df["cum_reward"], lw=2.2, label="Cumulative Reward", color="tab:blue")
    ax.plot(df["t"], df["cum_reward"] + df["cum_regret"], lw=1.8, ls="--", color="tab:green", label="Oracle (cum)")
    ax.fill_between(df["t"], df["cum_reward"], df["cum_reward"] + df["cum_regret"], color="tab:red", alpha=0.25, label="Cum Regret Gap")
    ax.set_title("Cumulative Reward — LinUCB vs Oracle"); ax.set_ylabel("Cumulative Reward"); ax.grid(True, alpha=0.3); ax.legend()

    # 2) Rolling Avg Reward
    ax = axes[0,1]
    ax.plot(df["t"], df["roll_reward"], lw=2.2, color="tab:blue")
    ax.set_title(f"Rolling Avg Reward (window={ROLL_W})"); ax.set_ylabel("Avg Reward"); ax.grid(True, alpha=0.3)

    # 3) Cumulative Regret
    ax = axes[1,0]
    ax.plot(df["t"], df["cum_regret"], lw=2.2, color="tab:red")
    ax.set_title("Cumulative Regret"); ax.set_ylabel("Cumulative Regret"); ax.grid(True, alpha=0.3)

    # 4) Rolling Avg Regret
    ax = axes[1,1]
    ax.plot(df["t"], df["roll_regret"], lw=2.2, color="tab:red")
    ax.axhline(0, ls="--", lw=1.2, color="tab:green", alpha=0.6)
    ax.set_title(f"Rolling Avg Regret (window={ROLL_W})"); ax.set_ylabel("Avg Regret/Step"); ax.grid(True, alpha=0.3)

    # 5) Cumulative Sales (raw)
    ax = axes[2,0]
    ax.plot(df["t"], df["cum_sales"], lw=2.2, color="tab:purple")
    ax.set_title("Cumulative Sales (raw)"); ax.set_ylabel("Cumulative Sales"); ax.grid(True, alpha=0.3)

    # 6) Rolling Avg Sales (raw)
    ax = axes[2,1]
    ax.plot(df["t"], df["roll_sales"], lw=2.2, color="tab:purple")
    ax.set_title(f"Rolling Avg Sales (raw, window={ROLL_W})"); ax.set_ylabel("Avg Sales"); ax.grid(True, alpha=0.3)

    # 7) Cumulative Health (raw)
    ax = axes[3,0]
    ax.plot(df["t"], df["cum_health"], lw=2.2, color="tab:orange")
    ax.set_title("Cumulative Health (raw)"); ax.set_ylabel("Cumulative Health"); ax.set_xlabel("Time Step"); ax.grid(True, alpha=0.3)

    # 8) Rolling Avg Health (raw)
    ax = axes[3,1]
    ax.plot(df["t"], df["roll_health"], lw=2.2, color="tab:orange")
    ax.set_title(f"Rolling Avg Health (raw, window={ROLL_W})"); ax.set_ylabel("Avg Health"); ax.set_xlabel("Time Step"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / f"linucb_learning_performance_lambda_{LAMBDA}_full8.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✓ Saved full 8-panel plot: {out_png}")
    plt.show()

def main():
    print("="*72)
    print("LINUCB LEARNING PERFORMANCE — FULL 8 PANELS")
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

    # quick console summary
    total_regret = df["regret"].sum()
    oracle_total = (df["cum_reward"].iloc[-1] + df["cum_regret"].iloc[-1])
    print(f"Total regret: {total_regret:.2f} ({100*total_regret/max(oracle_total,1):.2f}%)")

    # save data + plot
    out_csv = RESULTS_DIR / f"linucb_performance_lambda_{LAMBDA}_full8.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved performance CSV: {out_csv}")

    plot_full8(df)

if __name__ == "__main__":
    main()
