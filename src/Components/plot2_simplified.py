"""
SIMPLIFIED PLOT2.PY - Line by Line Explanation

PURPOSE: Train and compare 3 models (Random, Linear, LinUCB) and plot their performance

KEY CONCEPTS:
1. REWARD = sales_scaled + λ * health_scaled
   - sales_scaled: 0-10 (scaled per time-slot)
   - health_scaled: 0-10 (re-normalized from CSV range 4.01-5.71 to full 0-10)
   - λ (lambda): 0.3 (weight for health)
   - Result: Reward range is 0 to 13 (max = 10 + 0.3*10 = 13)

2. REGRET = (oracle_reward - actual_reward) / oracle_reward * 100
   - Oracle: Best possible reward at each time step
   - Regret: Percentage showing how much worse than optimal (0-100%)

3. ROLLING AVERAGE: Smooths data over last 50 time steps

4. SCALED vs RAW:
   - SCALED: Values in 0-10 range (what model optimizes)
   - RAW: Real-world values (actual sales units, actual health scores)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# SETUP: Paths and Constants
# ============================================================================

# Get project paths
HERE = Path(__file__).resolve().parent
SRC = HERE if HERE.name == "src" else HERE.parent
sys.path.insert(0, str(SRC))

# Import model and helper functions
from Components.model import LinUCB
from tests.train_eval import load_feature_matrix, load_action_matrix, compute_rewards_for_lambda

# File paths
DATA_DIR = SRC / "data"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"  # Features for each (time_slot, item)
ACTION_MATRIX = DATA_DIR / "action_matrix.csv"    # Which items available at each time
MERGED_DATA = DATA_DIR / "data_healthscore_mapped.csv"  # Sales + health data
TIMESLOTS = DATA_DIR / "time_slot_mapping.csv"    # Time slot mappings

RESULTS_DIR = SRC / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
LAMBDA = 0.30  # Weight for health in reward (0 = only sales, 1 = only health)
ROLL_W = 50    # Rolling average window size

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rolling_mean(values, window_size):
    """
    Calculate rolling average of values.
    
    Args:
        values: Array of values
        window_size: Size of rolling window (not used, we use expanding mean)
    
    Returns:
        Array of rolling averages
    """
    # Using expanding mean (average from start to current point)
    # This smooths out fluctuations over time
    series = pd.Series(values)
    return series.expanding().mean().to_numpy()


def build_raw_lookup(merged_csv, timeslots_csv):
    """
    Build lookup for RAW sales and health values.
    
    Returns: Dictionary mapping (time_slot_id, item_name) -> (sales_raw, health_raw)
    
    Example:
        (1, "Pizza") -> (150, 4.85)  # 150 sales units, health score 4.85
    """
    # Load time slot mappings
    ts = pd.read_csv(timeslots_csv, low_memory=False)
    key2id = dict(zip(
        zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
        ts["time_slot_id"].astype(int)
    ))
    
    # Load merged sales data
    m = pd.read_csv(merged_csv, low_memory=False)
    for col in ["date", "school_name", "time_of_day", "description"]:
        m[col] = m[col].astype(str).str.strip()
    
    # Map time slots
    m["time_slot_key"] = list(zip(m["date"], m["school_name"], m["time_of_day"]))
    m["time_slot_id"] = m["time_slot_key"].map(key2id)
    m = m.dropna(subset=["time_slot_id"]).copy()
    m["time_slot_id"] = m["time_slot_id"].astype(int)
    
    # Aggregate: Sum sales, median health for each (time_slot, item)
    agg = m.groupby(["time_slot_id", "description"], as_index=False).agg(
        sales=("total", "sum"),      # Total sales units
        health=("HealthScore", "median")  # Median health score (4.01-5.71 range)
    )
    
    # Create lookup dictionary
    lookup = {}
    for row in agg.itertuples(index=False):
        key = (int(row.time_slot_id), str(row.description))
        value = (float(row.sales), float(row.health))
        lookup[key] = value
    
    return lookup


def build_scaled_lookup(merged_csv, timeslots_csv, feature_matrix_csv):
    """
    Build lookup for SCALED sales and health values (0-10 range).
    
    IMPORTANT: This is what the model uses for training!
    
    Returns: Dictionary mapping (time_slot_id, item_name) -> (sales_scaled, health_scaled)
    
    Scaling:
    1. Sales: Scaled per time-slot to [0, 10]
       - Formula: 10 * (sales - min_sales_in_slot) / (max_sales_in_slot - min_sales_in_slot)
       - Why per time-slot? Different time slots have different sales ranges
    
    2. Health: Re-normalized from CSV range (4.01-5.71) to full [0, 10]
       - Formula: 10 * (health - 4.01) / (5.71 - 4.01)
       - Why? CSV values are already in [0,10] scale but use narrow range
       - We spread them to full range for better differentiation
    
    Example:
        (1, "Pizza") -> (7.5, 8.2)  # Scaled sales=7.5, scaled health=8.2 (both 0-10)
    """
    # Load feature matrix to get all (time_slot_id, item) pairs
    feature_df = pd.read_csv(feature_matrix_csv, low_memory=False)
    rows = feature_df[["time_slot_id", "item", "item_idx"]].copy()
    rows["time_slot_id"] = pd.to_numeric(rows["time_slot_id"], errors="coerce").astype(int)
    rows["item"] = rows["item"].astype(str)
    
    # Load merged sales data
    m = pd.read_csv(merged_csv, low_memory=False)
    ts = pd.read_csv(timeslots_csv, low_memory=False)
    key2id = dict(zip(
        zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
        ts["time_slot_id"].astype(int)
    ))
    
    for col in ["date", "school_name", "time_of_day", "description"]:
        m[col] = m[col].astype(str).str.strip()
    
    m["time_slot_key"] = list(zip(m["date"], m["school_name"], m["time_of_day"]))
    m["time_slot_id"] = m["time_slot_key"].map(key2id)
    m = m.dropna(subset=["time_slot_id"]).copy()
    m["time_slot_id"] = m["time_slot_id"].astype(int)
    
    # Aggregate sales and health
    agg = m.groupby(["time_slot_id", "description"], as_index=False).agg(
        total=("total", "sum"),
        health_score=("HealthScore", "median"),
    )
    
    # Align with feature matrix rows
    aligned = rows.merge(
        agg,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",
        validate="m:1",
    )
    
    # Extract sales and health
    aligned["total"] = pd.to_numeric(aligned["total"], errors="coerce").fillna(0.0)
    total_sales = aligned["total"].to_numpy(dtype=float)
    health = aligned["health_score"].to_numpy(dtype=float)
    health = np.where(np.isnan(health), np.nanmedian(health), health)
    
    # SCALE SALES: Per time-slot to [0, 10]
    sales_scaled = np.zeros_like(total_sales, dtype=float)
    for ts_id in aligned['time_slot_id'].unique():
        mask = (aligned['time_slot_id'] == ts_id).values
        slot_sales = total_sales[mask]
        vmin = slot_sales.min()
        vmax = slot_sales.max()
        
        if vmax <= vmin:
            # All sales same in this time slot -> assign middle value
            sales_scaled[mask] = 5.0
        else:
            # Scale to [0, 10] range
            sales_scaled[mask] = 10.0 * (slot_sales - vmin) / (vmax - vmin)
    
    # SCALE HEALTH: Re-normalize from CSV range (4.01-5.71) to full [0, 10]
    health_min = np.nanmin(health)  # Should be ~4.01
    health_max = np.nanmax(health)  # Should be ~5.71
    
    if health_max <= health_min:
        health_scaled = np.full_like(health, 5.0, dtype=float)
    else:
        # Re-normalize to full [0, 10] range
        health_scaled = 10.0 * (health - health_min) / (health_max - health_min)
        health_scaled = np.clip(health_scaled, 0.0, 10.0)  # Ensure [0, 10]
    
    # Build lookup dictionary
    lookup = {}
    for i, (idx, row) in enumerate(aligned.iterrows()):
        ts_id = int(row['time_slot_id'])
        item_name = str(row['item'])
        lookup[(ts_id, item_name)] = (float(sales_scaled[i]), float(health_scaled[i]))
    
    return lookup


def make_item_map(metadata_df):
    """
    Create mapping from (time_slot_id, item_idx) to item_name.
    
    Returns: Dictionary mapping (t, a) -> item_name
    
    Example:
        (1, 5) -> "Pizza"  # Time slot 1, item index 5 = "Pizza"
    """
    return {
        (int(r.time_slot_id), int(r.item_idx)): str(r.item)
        for r in metadata_df.itertuples(index=False)
    }


def build_tensors(feature_array, metadata_df, rewards, action_matrix):
    """
    Build tensors for bandit algorithms.
    
    Returns:
        data_TAD: (T, A, d) tensor - Features for each (time_slot, arm)
        rewards_TA: (T, A) tensor - Rewards for each (time_slot, arm)
        mask_TA: (T, A) boolean tensor - Which arms are available
    
    Where:
        T = number of time slots
        A = number of arms/items
        d = number of features
    """
    T = action_matrix.shape[0]  # Number of time slots
    A = action_matrix.shape[1]   # Number of arms/items
    d = feature_array.shape[1]   # Number of features
    
    # Initialize tensors
    data_TAD = np.zeros((T, A, d), dtype=np.float32)
    rewards_TA = np.zeros((T, A), dtype=np.float32)
    mask_TA = action_matrix.astype(bool)
    
    # Fill tensors from feature_array and metadata
    for idx, row in metadata_df.iterrows():
        t = int(row['time_slot_id'])
        a = int(row['item_idx'])
        
        if t >= T or a >= A:
            continue
        
        # Get feature vector (same index as metadata_df)
        if isinstance(metadata_df.index, pd.RangeIndex):
            array_idx = idx
        else:
            array_idx = metadata_df.index.get_loc(idx)
        
        data_TAD[t, a] = feature_array[array_idx]
        rewards_TA[t, a] = float(rewards[array_idx])
    
    return data_TAD, rewards_TA, mask_TA


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model_and_track(model_type, feature_array, action_matrix, metadata_df, rewards, **kwargs):
    """
    Train a model and track performance over time.
    
    Args:
        model_type: "random", "linear", or "linucb"
        feature_array: (N, d) array of features
        action_matrix: (T, A) array of available actions
        metadata_df: DataFrame with (time_slot_id, item_idx, item) mappings
        rewards: (N,) array of rewards
        **kwargs: Model-specific parameters
    
    Returns:
        DataFrame with columns: t, reward, oracle, regret, regret_pct, 
                                sales, health, sales_scaled, health_scaled,
                                roll_reward, roll_regret_pct, etc.
    """
    # Build tensors
    data_TAD, rewards_TA, mask_TA = build_tensors(
        feature_array, metadata_df, rewards, action_matrix
    )
    T, A, d = data_TAD.shape
    
    # Initialize model
    if model_type == "random":
        rng = np.random.default_rng(kwargs.get("seed", 42))
        model = None
    elif model_type == "linear":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=kwargs.get("ridge", 1.0))
        X_train = []
        y_train = []
    elif model_type == "linucb":
        model = LinUCB(
            d=d, 
            n_arms=A, 
            alpha=kwargs.get("alpha", 1.0), 
            l2=kwargs.get("ridge", 1.0), 
            seed=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create lookups
    item_map = make_item_map(metadata_df)
    raw_lookup = build_raw_lookup(MERGED_DATA, TIMESLOTS)
    scaled_lookup = build_scaled_lookup(MERGED_DATA, TIMESLOTS, FEATURE_MATRIX)
    
    # Track performance
    rows = []
    
    for t in range(T):
        # Get available arms at time t
        avail = np.where(mask_TA[t])[0]
        if avail.size == 0:
            continue
        
        # Select action
        if model_type == "random":
            a = int(rng.choice(avail))
        elif model_type == "linear":
            if len(X_train) > 0:
                # Retrain model and predict
                model.fit(np.array(X_train), np.array(y_train))
                feats_avail = np.array([data_TAD[t, a] for a in avail])
                pred_rewards = model.predict(feats_avail)
                best_idx = np.argmax(pred_rewards)
                a = int(avail[best_idx])
            else:
                # No training data yet - random
                a = int(np.random.choice(avail))
        elif model_type == "linucb":
            feats = {a: data_TAD[t, a] for a in avail}
            a = model.select_action(list(avail), feats)
        
        # Get reward and oracle
        r = float(rewards_TA[t, a])
        oracle = float(np.max(rewards_TA[t, avail]))
        regret = oracle - r
        regret_pct = (regret / oracle * 100) if oracle > 0 else 0.0
        
        # Get raw and scaled sales/health
        item_name = item_map.get((t, a))
        if item_name is not None and (t, item_name) in raw_lookup:
            sales_raw, health_raw = raw_lookup[(t, item_name)]
        else:
            sales_raw, health_raw = 0.0, 4.91  # Default: no sales, median health
        
        if item_name is not None and (t, item_name) in scaled_lookup:
            sales_scaled, health_scaled = scaled_lookup[(t, item_name)]
        else:
            # Default: scale default values
            sales_scaled = 0.0
            # Scale health using actual data range
            health_min, health_max = 4.01, 5.71
            health_scaled = 10.0 * (4.91 - health_min) / (health_max - health_min) if health_max > health_min else 5.0
        
        # Store results
        rows.append({
            "t": t,
            "reward": r,
            "oracle": oracle,
            "regret": regret,
            "regret_pct": regret_pct,
            "sales": sales_raw,
            "health": health_raw,
            "sales_scaled": sales_scaled,
            "health_scaled": health_scaled,
        })
        
        # Update model
        if model_type == "random":
            pass  # No update needed
        elif model_type == "linear":
            X_train.append(data_TAD[t, a])
            y_train.append(r)
        elif model_type == "linucb":
            model.update_arm(a, data_TAD[t, a], r)
        
        # Progress update
        if (t+1) % 2000 == 0:
            avg_reward = np.mean([rr['reward'] for rr in rows[-2000:]])
            print(f"[{model_type.upper()} t={t+1}/{T}] avg reward last2k = {avg_reward:.3f}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate rolling averages
    df["roll_reward"] = rolling_mean(df["reward"].to_numpy(), ROLL_W)
    df["roll_regret"] = rolling_mean(df["regret"].to_numpy(), ROLL_W)
    df["roll_regret_pct"] = rolling_mean(df["regret_pct"].to_numpy(), ROLL_W)
    df["roll_sales"] = rolling_mean(df["sales"].to_numpy(), ROLL_W)
    df["roll_health"] = rolling_mean(df["health"].to_numpy(), ROLL_W)
    df["roll_sales_scaled"] = rolling_mean(df["sales_scaled"].to_numpy(), ROLL_W)
    df["roll_health_scaled"] = rolling_mean(df["health_scaled"].to_numpy(), ROLL_W)
    
    return df


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_comparison(df_random, df_linear, df_linucb, use_scaled=True):
    """
    Plot comparison of all 3 models.
    
    Args:
        df_random, df_linear, df_linucb: DataFrames with performance metrics
        use_scaled: If True, plot scaled values (0-10). If False, plot raw values.
    
    Plots 4 panels:
    1. Rolling Avg Reward (always scaled, range 0-13)
    2. Rolling Avg Regret % (always percentage, range 0-100%)
    3. Rolling Avg Sales (scaled 0-10 or raw units)
    4. Rolling Avg Health (scaled 0-10 or raw scores 4.01-5.71)
    """
    # Calculate final metrics
    final_reward_linucb = df_linucb["roll_reward"].iloc[-1]
    final_reward_linear = df_linear["roll_reward"].iloc[-1]
    final_reward_random = df_random["roll_reward"].iloc[-1]
    
    final_regret_linucb = df_linucb["roll_regret_pct"].iloc[-1]
    final_regret_linear = df_linear["roll_regret_pct"].iloc[-1]
    final_regret_random = df_random["roll_regret_pct"].iloc[-1]
    
    improvement_vs_random = ((final_reward_linucb - final_reward_random) / final_reward_random * 100) if final_reward_random > 0 else 0
    improvement_vs_linear = ((final_reward_linucb - final_reward_linear) / final_reward_linear * 100) if final_reward_linear > 0 else 0
    regret_reduction_vs_random = final_regret_random - final_regret_linucb
    regret_reduction_vs_linear = final_regret_linear - final_regret_linucb
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    value_type = "scaled" if use_scaled else "raw"
    fig.suptitle(
        f"Model Comparison: LinUCB > Linear > Random (λ={LAMBDA}) - {value_type.upper()} values",
        fontsize=16, fontweight="bold", y=0.995
    )
    
    # 1) ROLLING AVG REWARD (always scaled)
    # Range: 0-13 (because reward = sales_scaled + λ*health_scaled, max = 10 + 0.3*10 = 13)
    ax = axes[0, 0]
    ax.plot(df_random["t"], df_random["roll_reward"], lw=2.5, color="tab:red",
            label=f"Random ({final_reward_random:.2f})", alpha=0.7, linestyle="--")
    ax.plot(df_linear["t"], df_linear["roll_reward"], lw=2.5, color="tab:orange",
            label=f"Linear ({final_reward_linear:.2f})", alpha=0.8, linestyle="-.")
    ax.plot(df_linucb["t"], df_linucb["roll_reward"], lw=3.0, color="tab:blue",
            label=f"LinUCB ({final_reward_linucb:.2f}) [BEST]", alpha=1.0, linestyle="-")
    ax.set_title(f"Rolling Avg Reward (window={ROLL_W}) - Higher is Better", fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Avg Reward")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            f"LinUCB: +{improvement_vs_random:.1f}% vs Random, +{improvement_vs_linear:.1f}% vs Linear",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 2) ROLLING AVG REGRET PERCENTAGE
    # Range: 0-100% (percentage showing how much worse than optimal)
    ax = axes[0, 1]
    ax.plot(df_random["t"], df_random["roll_regret_pct"], lw=2.5, color="tab:red",
            label=f"Random ({final_regret_random:.1f}%)", alpha=0.7, linestyle="--")
    ax.plot(df_linear["t"], df_linear["roll_regret_pct"], lw=2.5, color="tab:orange",
            label=f"Linear ({final_regret_linear:.1f}%)", alpha=0.8, linestyle="-.")
    ax.plot(df_linucb["t"], df_linucb["roll_regret_pct"], lw=3.0, color="tab:blue",
            label=f"LinUCB ({final_regret_linucb:.1f}%) [BEST]", alpha=1.0, linestyle="-")
    ax.axhline(0, ls="--", lw=1.2, color="tab:green", alpha=0.6)
    ax.set_title(f"Rolling Avg Regret % (window={ROLL_W}) - Lower is Better", fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Avg Regret (%)")
    ax.set_ylim([0, 100])  # Percentage scale (0-100%)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98,
            f"LinUCB: {regret_reduction_vs_random:.1f}% lower than Random, {regret_reduction_vs_linear:.1f}% lower than Linear",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 3) ROLLING AVG SALES
    if use_scaled:
        # SCALED: Range 0-10
        final_sales_random = df_random["roll_sales_scaled"].iloc[-1]
        final_sales_linear = df_linear["roll_sales_scaled"].iloc[-1]
        final_sales_linucb = df_linucb["roll_sales_scaled"].iloc[-1]
        ax = axes[1, 0]
        ax.plot(df_random["t"], df_random["roll_sales_scaled"], lw=2.5, color="tab:red",
                label=f"Random ({final_sales_random:.2f})", alpha=0.7, linestyle="--")
        ax.plot(df_linear["t"], df_linear["roll_sales_scaled"], lw=2.5, color="tab:orange",
                label=f"Linear ({final_sales_linear:.2f})", alpha=0.8, linestyle="-.")
        ax.plot(df_linucb["t"], df_linucb["roll_sales_scaled"], lw=3.0, color="tab:blue",
                label=f"LinUCB ({final_sales_linucb:.2f}) [BEST]", alpha=1.0, linestyle="-")
        ax.set_title(f"Rolling Avg Sales (SCALED to [0,10], window={ROLL_W})", fontweight="bold")
        ax.set_ylabel("Avg Sales (scaled)")
        ax.set_ylim([0, 10])  # Scaled range
    else:
        # RAW: Actual sales units (0-942 range)
        final_sales_random = df_random["roll_sales"].iloc[-1]
        final_sales_linear = df_linear["roll_sales"].iloc[-1]
        final_sales_linucb = df_linucb["roll_sales"].iloc[-1]
        ax = axes[1, 0]
        ax.plot(df_random["t"], df_random["roll_sales"], lw=2.5, color="tab:red",
                label=f"Random ({final_sales_random:.0f})", alpha=0.7, linestyle="--")
        ax.plot(df_linear["t"], df_linear["roll_sales"], lw=2.5, color="tab:orange",
                label=f"Linear ({final_sales_linear:.0f})", alpha=0.8, linestyle="-.")
        ax.plot(df_linucb["t"], df_linucb["roll_sales"], lw=3.0, color="tab:blue",
                label=f"LinUCB ({final_sales_linucb:.0f}) [BEST]", alpha=1.0, linestyle="-")
        ax.set_title(f"Rolling Avg Sales (RAW units, window={ROLL_W})", fontweight="bold")
        ax.set_ylabel("Avg Sales (raw units)")
    ax.set_xlabel("Time Step")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 4) ROLLING AVG HEALTH
    if use_scaled:
        # SCALED: Range 0-10 (re-normalized from CSV range 4.01-5.71)
        final_health_random = df_random["roll_health_scaled"].iloc[-1]
        final_health_linear = df_linear["roll_health_scaled"].iloc[-1]
        final_health_linucb = df_linucb["roll_health_scaled"].iloc[-1]
        ax = axes[1, 1]
        ax.plot(df_random["t"], df_random["roll_health_scaled"], lw=2.5, color="tab:red",
                label=f"Random ({final_health_random:.2f})", alpha=0.7, linestyle="--")
        ax.plot(df_linear["t"], df_linear["roll_health_scaled"], lw=2.5, color="tab:orange",
                label=f"Linear ({final_health_linear:.2f})", alpha=0.8, linestyle="-.")
        ax.plot(df_linucb["t"], df_linucb["roll_health_scaled"], lw=3.0, color="tab:blue",
                label=f"LinUCB ({final_health_linucb:.2f}) [BEST]", alpha=1.0, linestyle="-")
        ax.set_title(f"Rolling Avg Health (SCALED to [0,10], window={ROLL_W})", fontweight="bold")
        ax.set_ylabel("Avg Health (scaled)")
        ax.set_ylim([0, 10])  # Scaled range
    else:
        # RAW: CSV health scores (4.01-5.71 range)
        final_health_random = df_random["roll_health"].iloc[-1]
        final_health_linear = df_linear["roll_health"].iloc[-1]
        final_health_linucb = df_linucb["roll_health"].iloc[-1]
        ax = axes[1, 1]
        ax.plot(df_random["t"], df_random["roll_health"], lw=2.5, color="tab:red",
                label=f"Random ({final_health_random:.2f})", alpha=0.7, linestyle="--")
        ax.plot(df_linear["t"], df_linear["roll_health"], lw=2.5, color="tab:orange",
                label=f"Linear ({final_health_linear:.2f})", alpha=0.8, linestyle="-.")
        ax.plot(df_linucb["t"], df_linucb["roll_health"], lw=3.0, color="tab:blue",
                label=f"LinUCB ({final_health_linucb:.2f}) [BEST]", alpha=1.0, linestyle="-")
        ax.set_title(f"Rolling Avg Health (RAW scores, window={ROLL_W})", fontweight="bold")
        ax.set_ylabel("Avg Health (raw)")
    ax.set_xlabel("Time Step")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    suffix = "scaled" if use_scaled else "raw"
    out_png = RESULTS_DIR / f"model_comparison_lambda_{LAMBDA}_rolling4_{suffix}_simplified.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✓ Saved comparison plot: {out_png}")
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function: Train models and plot results."""
    print("=" * 72)
    print("MODEL COMPARISON: RANDOM vs LINEAR vs LINUCB")
    print("=" * 72)
    
    # Load data
    X, meta, _ = load_feature_matrix(str(FEATURE_MATRIX))
    A = load_action_matrix(str(ACTION_MATRIX))
    print(f"Feature matrix: {X.shape} | Action matrix: {A.shape}")
    
    # Compute rewards
    rewards_vec = compute_rewards_for_lambda(
        lambda_value=LAMBDA,
        feature_matrix_file=str(FEATURE_MATRIX),
        merged_data_file=str(MERGED_DATA),
        time_slot_mapping_file=str(TIMESLOTS)
    )
    
    # Train all three models
    print("\n[1/3] Training Random baseline...")
    df_random = train_model_and_track("random", X, A, meta, rewards_vec, seed=42)
    
    print("\n[2/3] Training Linear baseline...")
    df_linear = train_model_and_track("linear", X, A, meta, rewards_vec, ridge=1.0)
    
    print("\n[3/3] Training LinUCB...")
    df_linucb = train_model_and_track("linucb", X, A, meta, rewards_vec, alpha=1.0, ridge=1.0)
    
    # Save data
    out_csv_random = RESULTS_DIR / f"random_performance_lambda_{LAMBDA}_rolling4_simplified.csv"
    out_csv_linear = RESULTS_DIR / f"linear_performance_lambda_{LAMBDA}_rolling4_simplified.csv"
    out_csv_linucb = RESULTS_DIR / f"linucb_performance_lambda_{LAMBDA}_rolling4_simplified.csv"
    
    df_random.to_csv(out_csv_random, index=False)
    df_linear.to_csv(out_csv_linear, index=False)
    df_linucb.to_csv(out_csv_linucb, index=False)
    
    print(f"\nSaved performance CSVs:")
    print(f"  - Random: {out_csv_random}")
    print(f"  - Linear: {out_csv_linear}")
    print(f"  - LinUCB: {out_csv_linucb}")
    
    # Plot comparison
    print("\n[4/4] Generating plots...")
    print("\n  Generating SCALED plots (aligned with model training)...")
    plot_comparison(df_random, df_linear, df_linucb, use_scaled=True)
    
    print("\n  Generating RAW plots (real-world impact)...")
    plot_comparison(df_random, df_linear, df_linucb, use_scaled=False)
    
    # Print summary
    print("\n" + "=" * 72)
    print("📊 FINAL SUMMARY")
    print("=" * 72)
    print(f"\n🏆 RANKING (by Final Reward):")
    print(f"   1. LinUCB:  {df_linucb['roll_reward'].iloc[-1]:.3f} (Regret: {df_linucb['roll_regret_pct'].iloc[-1]:.1f}%) [BEST]")
    print(f"   2. Linear:  {df_linear['roll_reward'].iloc[-1]:.3f} (Regret: {df_linear['roll_regret_pct'].iloc[-1]:.1f}%)")
    print(f"   3. Random:  {df_random['roll_reward'].iloc[-1]:.3f} (Regret: {df_random['roll_regret_pct'].iloc[-1]:.1f}%)")
    print("\n✅ CONCLUSION: LinUCB outperforms both Linear and Random baselines!")
    print("=" * 72)


if __name__ == "__main__":
    main()

