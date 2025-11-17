# train_eval.py
# Purpose: Train LinUCB model with different lambda values and compare results
# Lambda (λ) = Health weight parameter (0 = only popularity, 1 = only health)


import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json


# Add src to path so we can import our custom model
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from Components.model import LinUCB  # Our custom Contextual Bandit model


# ===== FILE PATHS =====

data_dir = src_dir / "data"
feature_matrix_file = data_dir / "feature_matrix.csv"        # Nutritional features for each meal
action_matrix_file = data_dir / "action_matrix.csv"          # Which meals were available when
merged_data_file = data_dir / "data_healthscore_mapped.csv" # Raw sales data with timestamps
time_slot_mapping_file = data_dir / "time_slot_mapping.csv"  # Time slot definitions

# Directory to save trained models and results
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)

# ===== CONFIGURATION =====

# Lambda values to test: Health weight parameter
# Lower lambda = more focus on popularity, Higher lambda = more focus on health
lambda_values_to_test = [0.05, 0.25, 0.50,0.75, 0.9]


# ===== HELPER FUNCTIONS =====

def load_feature_matrix(file_path):
    """
    Load the feature matrix containing nutritional information for each meal.
    
    Input: CSV file with meal features (from env.py)
    Output: 
      - feature_array: numpy array of nutritional features
      - metadata_df: dataframe with meal metadata (time_slot_id, item, item_idx)
      - feature_cols: list of feature column names
    """
    print("Loading feature matrix.")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Separate metadata columns from nutritional feature columns
    metadata_cols = ["time_slot_id", "item", "item_idx"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Convert to numpy array for efficient processing
    feature_array = df[feature_cols].to_numpy(dtype=np.float32)
    metadata_df = df[metadata_cols].copy()
    
    return feature_array, metadata_df, feature_cols

def load_action_matrix(file_path):
    """
    Load the action matrix showing which meals were available at each time slot.
    
    Input: CSV file with availability data (from env.py)
    Output: numpy array where action_matrix[time_slot][item] = 1 if available, 0 if not
    """
    print("Loading action matrix...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Extract columns that represent meal items (item_0, item_1, etc.)
    item_cols = [c for c in df.columns if c.startswith("item_")]
    action_array = df[item_cols].to_numpy(dtype=np.int32)
    
    return action_array

def standardize_text_column(data, column_name):
    """
    Clean text columns by converting to string and removing extra whitespace.
    This ensures consistent text matching across datasets.
    """
    data[column_name] = data[column_name].astype(str).str.strip()
    return data

def compute_rewards_for_lambda(lambda_value):
    """
    Compute reward values with specified lambda (health weight).
    
    REWARD = Total Sales * (1 + λ * Health_Score_Z)
    
    PLUS: 20% bonus for items that are BOTH popular AND healthy ("sweet spot" items)
    
    Input: lambda_value (float) - how much to weight health vs popularity
    Output: numpy array of rewards for each meal serving in historical data
    """
    print(f"Computing rewards with lambda = {lambda_value}...")
    
    # Load feature matrix to get the structure of our data
    feature_df = pd.read_csv(feature_matrix_file, low_memory=False)
    rows_metadata = feature_df[["time_slot_id", "item", "item_idx"]].copy()
    rows_metadata["time_slot_id"] = pd.to_numeric(rows_metadata["time_slot_id"], errors="coerce").astype(int)
    rows_metadata["item"] = rows_metadata["item"].astype(str)
    
    # Load raw sales data to get actual sales numbers and health scores
    merged_data = pd.read_csv(merged_data_file, low_memory=False)
    
    # Clean text data for consistent matching
    merged_data = standardize_text_column(merged_data, 'date')
    merged_data = standardize_text_column(merged_data, 'school_name')
    merged_data = standardize_text_column(merged_data, 'time_of_day')
    merged_data = standardize_text_column(merged_data, 'description')
    
    # Load time slot mapping to connect dates to time slot IDs
    time_slot_df = pd.read_csv(time_slot_mapping_file, low_memory=False)
    time_slot_map = dict(zip(
        zip(time_slot_df['date'].astype(str), 
            time_slot_df['school_name'].astype(str), 
            time_slot_df['time_of_day'].astype(str)),
        time_slot_df['time_slot_id'].astype(int)
    ))
    
    # Assign time slot IDs to each row in the sales data
    merged_data['time_slot_key'] = list(zip(
        merged_data['date'], 
        merged_data['school_name'], 
        merged_data['time_of_day']
    ))
    merged_data['time_slot_id'] = merged_data['time_slot_key'].map(time_slot_map)
    
    # Remove rows without valid time slot IDs
    merged_data = merged_data.dropna(subset=['time_slot_id'])
    merged_data['time_slot_id'] = merged_data['time_slot_id'].astype(int)
    
    # Aggregate sales data: sum total sales and take median health score for each meal
    aggregated = merged_data.groupby(
        ["time_slot_id", "description"], 
        as_index=False
    ).agg(
        total=("total", "sum"),           # Total number sold
        health_score=("HealthScore", "median")  # Health score (1-5 scale)
    )
    
    # Merge aggregated sales data with feature matrix structure
    aligned_data = rows_metadata.merge(
        aggregated,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",      # Keep all rows from feature matrix
        validate="m:1"   # One meal can appear in multiple time slots
    )
    
    # Process total sales: convert to numeric and fill missing values with 0
    aligned_data["total"] = pd.to_numeric(aligned_data["total"], errors="coerce").fillna(0.0)
    
    # Process health scores: fill missing values with median health score
    health_scores = aligned_data["health_score"].to_numpy(dtype=float)
    median_health = np.nanmedian(health_scores)
    health_scores = np.where(np.isnan(health_scores), median_health, health_scores)
    
    # Standardize health scores to z-scores (mean=0, std=1)
    # This puts all health scores on the same scale for fair comparison
    health_mean = np.nanmean(health_scores)
    health_std = np.nanstd(health_scores)
    if not np.isfinite(health_std) or health_std < 1e-8:
        health_std = 1.0  # Prevent division by zero
    health_scores_z = (health_scores - health_mean) / health_std
    
    total_sales = aligned_data["total"].to_numpy(dtype=float)
    
    # ===== CORE REWARD CALCULATION =====
    # Base reward: Balance between sales and health based on lambda
    rewards = total_sales + (lambda_value * health_scores_z)
    
    # ===== HEALTH-POPULARITY BOOST ENHANCEMENT =====
    # Find "sweet spot" items that are BOTH popular AND healthy
    # These are items that students already like AND are good for them
    
    # Thresholds: Top 40% in health AND top 40% in popularity
    health_threshold = np.percentile(health_scores_z, 60)    # 60th percentile = top 40%
    popularity_threshold = np.percentile(total_sales, 60)    # 60th percentile = top 40%
    
    # Create mask: True for items that meet BOTH criteria
    health_popularity_mask = (health_scores_z > health_threshold) & (total_sales > popularity_threshold)
    
    # Apply 20% boost to these "sweet spot" items
    # Why 20%? Gentle but noticeable nudge without distorting rankings
    boost_factor = 1.2
    rewards[health_popularity_mask] *= boost_factor
    
    print(f"  Boosted {health_popularity_mask.sum()} health-popularity 'sweet spot' items")
    
    return rewards







def run_random_baseline(
    metadata_df: pd.DataFrame,
    rewards: np.ndarray,
    *,
    seed: int = 42,
    top_k: int = 1,
    lambda_value: float = None,   # <-- added so the result includes "lambda"
    verbose: bool = False,
):
    """
    Simple Random baseline assuming PERFECT alignment:
      Every (time_slot_id, item_idx) pair in metadata_df is valid and has a reward.

    For each time slot:
      - Randomly pick `top_k` distinct items from the rows in that slot
      - Sum their rewards for the slot
      - Track the oracle (best possible) reward for regret

    Returns a dict with:
      steps, total_reward, oracle_reward, regret, avg_reward, lambda, seed, top_k
    """
    rng = np.random.default_rng(seed)

    # Group row indices by time slot once (fast + clear)
    rows_by_slot = metadata_df.groupby("time_slot_id", sort=True).groups

    total_reward = 0.0
    oracle_reward = 0.0
    slot_reward_history = []

    # Number of decision points (time slots)
    steps = len(rows_by_slot)

    for slot_id in rows_by_slot.keys():
        slot_rows = rows_by_slot[slot_id]

        # Oracle reward for this slot (best possible among rows in this slot)
        slot_rewards = rewards[slot_rows]
        oracle_reward += float(np.max(slot_rewards))

        if top_k == 1:
            # Pick one random row for this slot
            chosen_row = int(rng.choice(slot_rows))
            chosen_reward = float(rewards[chosen_row])
        else:
            # Enforce unique items if there are duplicates in this slot
            item_ids = metadata_df.iloc[slot_rows]["item_idx"].to_numpy(dtype=int)
            _, unique_positions = np.unique(item_ids, return_index=True)
            unique_rows = np.array(slot_rows)[unique_positions]

            k = min(top_k, len(unique_rows))
            chosen_rows = rng.choice(unique_rows, size=k, replace=False)
            chosen_reward = float(np.sum(rewards[chosen_rows]))

        total_reward += chosen_reward
        slot_reward_history.append(chosen_reward)

        if verbose and (slot_id % 500 == 0):
            print(f"[Random] slot={slot_id:5d}  reward={chosen_reward:.2f}  total={total_reward:.2f}")

    regret = oracle_reward - total_reward
    avg_reward = total_reward / max(steps, 1)

    result = {
        "steps": int(steps),
        "total_reward": float(total_reward),
        "oracle_reward": float(oracle_reward),
        "regret": float(regret),
        "avg_reward": float(avg_reward),
        "lambda": float(lambda_value) if lambda_value is not None else None,
        "seed": int(seed),
        "top_k": int(top_k),
    }
    return result

def train_linucb_model(feature_array, action_matrix, metadata_df, rewards, lambda_value, verbose=False):
    """
    Train a LinUCB model with the given data and lambda value.
    
    Input:
      - feature_array: Nutritional features for each meal serving
      - action_matrix: Availability of meals at each time slot
      - metadata_df: Metadata about each serving (time_slot_id, item, item_idx)
      - rewards: Computed reward values for each serving
      - lambda_value: Health weight used to compute these rewards
      - verbose: Whether to print detailed training progress
    
    Output:
      - training_results: Dictionary with performance metrics
      - model: Trained LinUCB model
    """
    print(f"\nTraining LinUCB with lambda = {lambda_value}...")
    
    # Get dimensions of our data
    num_samples, num_features = feature_array.shape  # e.g., 224,536 samples × 18 features
    num_arms = action_matrix.shape[1]               # e.g., 160 different meal items
    
    print(f"  Features: {num_features}")
    print(f"  Arms (items): {num_arms}")
    print(f"  Samples: {num_samples}")
    
    # Initialize LinUCB model
    # d = number of features, n_arms = number of meal items
    # alpha = exploration parameter, l2 = regularization to prevent overfitting
    model = LinUCB(d=num_features, n_arms=num_arms, alpha=1.0, l2=1.0, seed=42)
    
    # Train the model on historical data
    # The model learns which meals to recommend in different contexts
    training_results = model.train(
        X=feature_array,        # Nutritional context
        rows_df=metadata_df,    # Metadata for grouping
        rewards=rewards,        # What we're optimizing for
        avail_mat=action_matrix, # What was actually available
        verbose=verbose         # Progress reporting
    )
    
    # Add lambda value to results for tracking
    training_results['lambda'] = lambda_value
    
    return training_results, model

def plot_bar_random_vs_linucb(rand: dict, lin: dict, lambda_value: float):
    """
    Plots side-by-side bar charts for Total Reward and Regret.
    Works with your current results dicts that don't include cum_rewards.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    totals = [float(rand["total_reward"]), float(lin["total_reward"])]
    regrets = [float(rand["regret"]), float(lin["regret"])]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Random", "LinUCB"], totals)
    axes[0].set_title(f"Total Reward (λ={lambda_value:.2f})")
    axes[0].set_ylabel("Total Reward")

    axes[1].bar(["Random", "LinUCB"], regrets)
    axes[1].set_title(f"Regret (λ={lambda_value:.2f})")
    axes[1].set_ylabel("Regret")

    fig.suptitle("Random vs LinUCB")
    plt.tight_layout()
    plt.show()

def print_ablation_summary(name: str, results: list):
    """
    Pretty-prints an ablation table for a list of per-λ results dicts.
    Expected keys in each dict: 'lambda', 'total_reward', 'oracle_reward', 'regret'
    """
    if not results:
        print(f"[{name}] No results to display.")
        return

    # Best = min regret
    best = min(results, key=lambda r: r["regret"])
    best_lambda = best["lambda"]
    best_regret = best["regret"]
    best_regret_pct = 100 * best_regret / max(best["oracle_reward"], 1)

    print("SUMMARY: LAMBDA ABLATION RESULTS —", name)
    print("Lambda = Health Weight | Lower values favor popularity, Higher values favor health")
    print("Regret = How much worse than perfect choices | Lower is better")
    print("─" * 80)
    print("Lambda     Total Reward       Oracle Reward      Regret          Regret %")
    print("─" * 80)

    for r in sorted(results, key=lambda x: x["lambda"]):
        lam = float(r["lambda"])
        tot = float(r["total_reward"])
        orc = float(r["oracle_reward"])
        reg = float(r["regret"])
        reg_pct = 100 * reg / max(orc, 1)
        marker = "-" if lam == best_lambda else " "
        print(f"{marker} {lam:<8.2f} {tot:>12.2f} {orc:>16.2f} {reg:>15.2f} {reg_pct:>12.1f}%")

    print()
    print("ANALYSIS")
    print(f" Best performing lambda: λ = {best_lambda:.2f}")
    print(f"  Regret: {best_regret:.2f} ({best_regret_pct:.1f}% of optimal)")
    print()

def print_side_by_side(rand_results: list, lin_results: list):
    """
    Prints a compact side-by-side comparison per λ.
    Assumes both lists contain dicts with the same set of λ values.
    """
    # Index by lambda for quick join
    r_by_lam = {float(r["lambda"]): r for r in rand_results}
    l_by_lam = {float(r["lambda"]): r for r in lin_results}
    common = sorted(set(r_by_lam.keys()) & set(l_by_lam.keys()))
    if not common:
        print("[Compare] No overlapping lambda values between Random and LinUCB.")
        return

    print("Random vs LinUCB — per λ")
    print("─" * 100)
    print("λ      RandTot      RandRegret    LinTot       LinRegret      ΔTotal        ΔRegret     Lin Regret%")
    print("─" * 100)
    for lam in common:
        rr = r_by_lam[lam]; ll = l_by_lam[lam]
        d_tot = float(ll["total_reward"]) - float(rr["total_reward"])
        d_reg = float(ll["regret"]) - float(rr["regret"])
        lin_reg_pct = 100 * float(ll["regret"]) / max(float(ll["oracle_reward"]), 1)
        print(f"{lam:<5.2f}  {rr['total_reward']:>12.2f}  {rr['regret']:>12.2f}  "
              f"{ll['total_reward']:>12.2f}  {ll['regret']:>12.2f}  "
              f"{d_tot:>12.2f}  {d_reg:>12.2f}   {lin_reg_pct:>9.1f}%")
    print()

def compute_health_scores_aligned(
    feature_matrix_file: Path,
    merged_data_file: Path,
    time_slot_mapping_file: Path,
) -> np.ndarray:
    """
    Returns a numpy array 'health_scores' aligned 1:1 with the rows in feature_matrix.csv
    (i.e., aligned to metadata_df), using the median HealthScore per (time_slot_id, item).

    Alignment logic mirrors your reward computation:
      - read feature_matrix (for row order + [time_slot_id, item, item_idx])
      - map raw merged data to time_slot_id using the saved mapping
      - aggregate HealthScore per (time_slot_id, description) with median
      - left-merge onto feature rows to preserve order (1:1 alignment)
      - fill missing with global median
    """
    # 1) Row skeleton from feature_matrix to preserve order
    fm = pd.read_csv(feature_matrix_file, low_memory=False)
    rows_meta = fm[["time_slot_id", "item", "item_idx"]].copy()
    rows_meta["time_slot_id"] = pd.to_numeric(rows_meta["time_slot_id"], errors="coerce").astype(int)
    rows_meta["item"] = rows_meta["item"].astype(str)

    # 2) Load raw merged and map to time slots
    merged = pd.read_csv(merged_data_file, low_memory=False)

    # clean text for consistent matching
    merged["date"] = merged["date"].astype(str).str.strip()
    merged["school_name"] = merged["school_name"].astype(str).str.strip()
    merged["time_of_day"] = merged["time_of_day"].astype(str).str.strip()
    merged["description"] = merged["description"].astype(str).str.strip()

    # time-slot mapping
    ts = pd.read_csv(time_slot_mapping_file, low_memory=False)
    ts_map = dict(zip(
        zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
        ts["time_slot_id"].astype(int)
    ))

    merged["time_slot_key"] = list(zip(merged["date"], merged["school_name"], merged["time_of_day"]))
    merged["time_slot_id"] = merged["time_slot_key"].map(ts_map)
    merged = merged.dropna(subset=["time_slot_id"]).copy()
    merged["time_slot_id"] = merged["time_slot_id"].astype(int)

    # 3) Aggregate HealthScore per (time_slot_id, description)
    agg = merged.groupby(["time_slot_id", "description"], as_index=False).agg(
        health_score=("HealthScore", "median")
    )

    # 4) Align to feature_matrix row order
    aligned = rows_meta.merge(
        agg,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",
        validate="m:1",
    )

    # 5) Fill missing with global median
    hs = aligned["health_score"].to_numpy(dtype=float)
    median_h = np.nanmedian(hs)
    hs = np.where(np.isnan(hs), median_h, hs)

    return hs

def run_health_first_eval_on_reward(
    metadata_df: pd.DataFrame,
    health_scores: np.ndarray,   # aligned 1:1 with metadata_df rows
    rewards: np.ndarray,         # λ-reward aligned 1:1 with metadata_df rows
    *,
    top_k: int = 1,
    seed: int = 42,
    lambda_value: float = None,
) -> dict:
    """
    Health-first rule:
      • For each time slot (decision point), select the top-K rows by HealthScore (descending).
      • Break ties by λ-reward (descending) to avoid row-order bias.
      • Evaluate totals and regret using λ-reward so results are comparable to Random/LinUCB.

    Returns:
      {steps, total_reward, oracle_reward, regret, avg_reward, lambda, seed, top_k}
    """
    rng = np.random.default_rng(seed)

    # Group row indices by time slot once
    rows_by_time_slot = metadata_df.groupby("time_slot_id", sort=True).groups
    num_time_slots = len(rows_by_time_slot)

    total_reward_sum = 0.0
    oracle_reward_sum = 0.0

    for time_slot_id, row_indices_in_slot in rows_by_time_slot.items():
        slot_row_indices = np.asarray(row_indices_in_slot, dtype=int)
        if slot_row_indices.size == 0:
            continue

        # Per-slot vectors
        slot_health_scores = health_scores[slot_row_indices]
        slot_lambda_rewards = rewards[slot_row_indices]

        # OPTIONAL: random tie-breaks (keep commented to remain deterministic)
        # slot_lambda_rewards = slot_lambda_rewards + rng.normal(0.0, 1e-12, size=slot_lambda_rewards.size)

        # Selection order: primary = Health desc, secondary = λ-reward desc
        # np.lexsort uses last key as primary → pass (-reward, -health) → primary is -health (i.e., Health desc)
        selection_order = np.lexsort((-slot_lambda_rewards, -slot_health_scores))
        k = min(top_k, slot_row_indices.size)
        chosen_row_indices = slot_row_indices[selection_order[:k]]

        # Evaluate chosen set under λ-reward
        total_reward_sum += float(np.sum(rewards[chosen_row_indices]))

        # Oracle under λ-reward: top-K by reward desc for the same slot
        oracle_order = np.argsort(-slot_lambda_rewards)
        oracle_row_indices = slot_row_indices[oracle_order[:k]]
        oracle_reward_sum += float(np.sum(rewards[oracle_row_indices]))

    regret_value = oracle_reward_sum - total_reward_sum
    avg_reward_per_slot = total_reward_sum / max(num_time_slots, 1)

    return {
        "steps": int(num_time_slots),
        "total_reward": float(total_reward_sum),
        "oracle_reward": float(oracle_reward_sum),
        "regret": float(regret_value),
        "avg_reward": float(avg_reward_per_slot),
        "lambda": float(lambda_value) if lambda_value is not None else None,
        "seed": int(seed),
        "top_k": int(top_k),
    }






def main():
    """
    Main function: Train models with different lambda values and compare results.
    Finds the optimal balance between student preferences and health goals.
    """
    print("=" * 70)
    print("LINUCB TRAINING WITH LAMBDA ABLATION")
    print("=" * 70)
    print("Testing different health-popularity trade-offs")
    print()
    
    # ===== STEP 1: LOAD DATA =====
    print("[1/3] Loading data.")
    feature_array, metadata_df, feature_cols = load_feature_matrix(feature_matrix_file)
    action_matrix = load_action_matrix(action_matrix_file)
    print(f"Loaded {len(metadata_df)} feature samples")
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Action matrix shape: {action_matrix.shape}")
    print()
    # After you have loaded:
# feature_array, metadata_df, action_matrix, rewards

    # ===== STEP 2: TRAIN MODELS WITH DIFFERENT LAMBDAS =====
    # ===== STEP 2: TRAIN MODELS =====
    print("[2/3] Training models with different lambda values.")
    print(f"Testing lambda values: {lambda_values_to_test}")
    print("Lower lambda = more popularity focused, Higher lambda = more health focused\n")


    # Build aligned health scores once (re-use for all comparisons)
    health_scores = compute_health_scores_aligned(
        feature_matrix_file=feature_matrix_file,
        merged_data_file=merged_data_file,
        time_slot_mapping_file=time_slot_mapping_file,
    )
    health_results = []
    rand_results = []
    all_results = []
    all_models = {}

    for lambda_value in lambda_values_to_test:
        print(f"\n--- Lambda = {lambda_value} ---")

        # Compute rewards for this λ
        rewards = compute_rewards_for_lambda(lambda_value)
 

        # Run Random baseline (K=1 unless you serve multiple items per slot)
        
        rand = run_random_baseline(
            metadata_df=metadata_df,
            rewards=rewards,
            seed=42,
            top_k=1,
            lambda_value=lambda_value,  # <-- include λ so it's recorded
            verbose=False,
        )
        rand["lambda"] = lambda_value
        rand_results.append(rand)

            # Health-first selection, evaluated on λ-reward
        health_first = run_health_first_eval_on_reward(
            metadata_df=metadata_df,
            health_scores=health_scores,   # selection metric (fixed)
            rewards=rewards,               # evaluation metric (changes per λ)
            top_k=1,
            seed=42,
            lambda_value=lambda_value,
        )
        health_results.append(health_first)

        # Train LinUCB
        results, model = train_linucb_model(
            feature_array,
            action_matrix,
            metadata_df,
            rewards,
            lambda_value,
            verbose=False
        )
        all_results.append(results)
        all_models[lambda_value] = model

        # Save model
        model_filename = f"model_lambda_{lambda_value:.2f}.joblib"
        model_filepath = results_dir / model_filename
        model.save(str(model_filepath))
        print(f"Saved model to {model_filename}")

    
      # LinUCB summary
    print_ablation_summary("LinUCB", all_results)

    # Random summary (requires you filled rand_results in your loop)
    print_ablation_summary("Random", rand_results)

    print_ablation_summary("Health-First", health_results)

#    Optional: side-by-side comparison
    print_side_by_side(rand_results, all_results)

    plot_bar_random_vs_linucb(rand, results, lambda_value)


    # lengths & key order
    assert len(health_scores) == len(metadata_df)
    fm_keys = pd.read_csv(feature_matrix_file, usecols=["time_slot_id","item","item_idx"])
    assert (metadata_df[["time_slot_id","item","item_idx"]].values == fm_keys.values).all()

    # no weird NaNs or inversions
    print("health_scores stats:", np.nanmin(health_scores), np.nanmax(health_scores), np.nanmean(health_scores))
    print("unique health values (first 20):", np.unique(health_scores)[:20])

    # ===== STEP 3: ANALYZE RESULTS =====
    print("\n" + "=" * 70)
    print("HEALTH-POPULARITY TRADE-OFF ANALYSIS")
    print("=" * 70)
    print()
    
    # Lambda value interpretation guide
    print("LAMBDA VALUE GUIDE:")
    print("0.05-0.15: Popularity Focused (student preferences dominate)")
    print("0.20-0.30: Balanced Approach (mix of popularity and health)")
    print("0.40-0.50: Health Focused (nutritional goals prioritized)")
    print()
    
    if len(all_results) > 1:
        # Find best and worst performing lambda values
        # Lower regret = better performance (closer to optimal choices)
        best_result = min(all_results, key=lambda r: r['regret'])
        worst_result = max(all_results, key=lambda r: r['regret'])
        
        best_lambda = best_result['lambda']
        best_regret = best_result['regret']
        best_regret_pct = 100 * best_regret / max(best_result['oracle_reward'], 1)
        
        
    
    # Save detailed results to JSON file for further analysis
    results_json_file = results_dir / "ablation_results.json"
    with open(results_json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n results saved to: {results_json_file}")

    # Save Random baseline results
    rand_json_file = results_dir / "ablation_results_random.json"
    with open(rand_json_file, 'w') as f:
        json.dump(rand_results, f, indent=2)
    print(f"Random results saved to: {rand_json_file}")

    
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models saved in: {results_dir}")
    print("Next: Use main.py with optimal model for recommendations")

if __name__ == "__main__":
    main()