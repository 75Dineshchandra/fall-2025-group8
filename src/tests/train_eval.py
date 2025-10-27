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
results_dir.mkdir(parents=True, exist_ok=True)

# ===== CONFIGURATION =====

# Lambda values to test: Health weight parameter
# Lower lambda = more focus on popularity, Higher lambda = more focus on health
lambda_values_to_test = [0.05, 0.50, 0.8]

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
    rewards = total_sales * (1.0 + lambda_value * health_scores_z)
    
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
    # reproducibility
    np.random.seed(42)

    try:
        feature_array, metadata_df, feature_cols = load_feature_matrix(feature_matrix_file)
        action_matrix = load_action_matrix(action_matrix_file)
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    print(f"Loaded {len(metadata_df)} feature samples")
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Action matrix shape: {action_matrix.shape}")
    print()
    
    # ===== STEP 2: TRAIN MODELS =====
    print("[2/3] Training models with different lambda values.")
    print(f"Testing lambda values: {lambda_values_to_test}")
    print("Lower lambda = more popularity focused, Higher lambda = more health focused")
    print()
    
    all_results = []    # Store results for each lambda
    all_models = {}     # Store trained models
    
    for lambda_value in lambda_values_to_test:
        print(f"\n--- Lambda = {lambda_value} ---")
        
        # Compute rewards using this lambda value
        # Different lambda = different balance of health vs popularity
        rewards = compute_rewards_for_lambda(lambda_value)
        
        # Train model with these rewards
        results, model = train_linucb_model(
            feature_array,
            action_matrix,
            metadata_df,
            rewards,
            lambda_value,
            verbose=False
        )
        
        # Store results and model
        all_results.append(results)
        all_models[lambda_value] = model
        
        # Save trained model for later use
        model_filename = f"model_lambda_{lambda_value:.2f}.joblib"
        model_filepath = results_dir / model_filename
        model.save(str(model_filepath))
        print(f"Saved model to {model_filename}")
    
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
        
        # ===== RESULTS SUMMARY TABLE =====
        print("SUMMARY: LAMBDA ABLATION RESULTS")
        print("Lambda = Health Weight | Lower values favor popularity, Higher values favor health")
        print("Regret = How much worse than perfect choices | Lower is better")
        print("─" * 80)
        print("Lambda     Total Reward       Oracle Reward      Regret          Regret %")
        print("─" * 80)
        
        for result in sorted(all_results, key=lambda r: r['lambda']):
            lambda_val = result['lambda']
            total_reward = result['total_reward']    # What our model achieved
            oracle_reward = result['oracle_reward']  # Best possible (perfect choices)
            regret = result['regret']               # Oracle - Our performance
            regret_pct = 100 * regret / max(oracle_reward, 1)  # Regret as percentage
            
            # Mark the best performing lambda
            marker = "-" if lambda_val == best_lambda else " "
            
            print(f"{marker} {lambda_val:<8.2f} {total_reward:>12.2f} {oracle_reward:>16.2f} {regret:>15.2f} {regret_pct:>12.1f}%")
        
        print()
        print("ANALYSIS")
        print(f" Best performing lambda: λ = {best_lambda:.2f}")
        print(f"  Regret: {best_regret:.2f} ({best_regret_pct:.1f}% of optimal)")
        print()
        
        
    
    # Save detailed results to JSON file for further analysis
    results_json_file = results_dir / "ablation_results.json"
    with open(results_json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n results saved to: {results_json_file}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models saved in: {results_dir}")
    print("Next: Use main.py with optimal model for recommendations")

if __name__ == "__main__":
    main()