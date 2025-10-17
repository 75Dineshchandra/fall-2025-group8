# env.py
# Purpose: Build feature matrix, action matrix, and time slot mapping from raw FCPS sales data
# This file transforms raw cafeteria sales data into structured formats for machine learning


import os
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

# Import LinUCB when env functions need to load a saved model
from Components.model import LinUCB

# Module-level globals that will be populated by load_env_data()
merged_data = None
feature_df = None
item_to_index = None
index_to_item = None

# ===== FILE PATHS =====

data_dir = Path("src/data")
input_csv = data_dir / "fcps_data_with_timestamps.csv"  # Raw sales data from FCPS

# Output files that will be created:
item_mapping_file = data_dir / "item_mapping.csv"           # Maps meal names to numbers
time_slot_mapping_file = data_dir / "time_slot_mapping.csv" # Maps time periods to IDs
action_matrix_file = data_dir / "action_matrix.csv"         # What was available when
feature_matrix_file = data_dir / "feature_matrix.csv"       # Nutritional information

# ===== CONFIGURATION =====

overwrite_existing = False  # Set to True to rebuild all matrices from scratch

# Nutritional columns to include in feature matrix
# These are the health dimensions our model will learn from
nutrient_columns = ["GramsPerServing", "Calories", "Protein", "Total Carbohydrate", 
                    "Dietary Fiber", "Total Sugars", "Added Sugars", "Total Fat", 
                    "Saturated Fat", "Trans Fat", "Cholesterol", "Sodium", 
                    "Vitamin D (D2 + D3)", "Calcium", "Iron", "Potassium", 
                    "Vitamin A", "Vitamin C"]

# ===== HELPER FUNCTIONS =====
def get_day_of_week_name(date_str):
    """Convert date to day name (Monday, Tuesday, etc)"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%A")

def get_features_for_context(target_date, school, meal_time):
    """
    Get nutritional features for items available in similar historical context
    Returns features_by_arm with ITEM INDICES (0-159) as keys
    """
    day_name = get_day_of_week_name(target_date)
    
    # Find items that were typically available in similar contexts
    similar_data = merged_data[
        (merged_data['school_name'] == school) &
        (merged_data['time_of_day'] == meal_time) & 
        (merged_data['day_name'] == day_name)
    ]
    
    # Get unique items from similar contexts
    typical_items = similar_data['description'].unique()
    
    # Get their features from feature matrix - using ITEM INDICES
    features_by_arm = {}
    for item_name in typical_items:
        # Convert item name to item index
        item_idx = item_to_index.get(item_name)
        if item_idx is not None:
            item_features = feature_df[feature_df['item'] == item_name]
            if len(item_features) > 0:
                # Get the feature columns (exclude metadata)
                feature_cols = [col for col in item_features.columns if col not in ['time_slot_id', 'item', 'item_idx']]
                features = item_features[feature_cols].iloc[0].values
                features_by_arm[item_idx] = features  # Use item index as key
    
    return features_by_arm

def recommend_with_trained_model(target_date, school, meal_time, model_path, top_k=5):
    """
    Get recommendations using your TRAINED LinUCB model
    This uses the optimal lambda = 0.05 balance we found
    """
    print(f"Getting SMART recommendations using trained model...")
    print(f"Date: {target_date}, School: {school}, Meal: {meal_time}")
    print()
    
    # Load the optimal model (lambda = 0.05)
    model = LinUCB.load(model_path)
    print(f" Loaded trained model: {model_path}")
    print(f" Using optimal lambda balance found in training")
    print()
    
    # Get features for available items (with ITEM INDICES as keys)
    features_by_arm = get_features_for_context(target_date, school, meal_time)
    
    if not features_by_arm:
        print("No items found for this context")
        return None
    
    print(f"Found {len(features_by_arm)} typically available items")
    
    # Get model recommendations (returns item indices)
    recommendations = model.get_recommendations(features_by_arm, top_k=top_k)
    
    # Convert item indices back to item names and get sales/health info
    enhanced_recommendations = []
    for item_idx, ucb_score in recommendations:
        # Convert index back to item name
        item_name = index_to_item.get(item_idx, f"Unknown_Item_{item_idx}")
        
        # Get sales and health info from historical data
        item_data = merged_data[
            (merged_data['description'] == item_name) &
            (merged_data['school_name'] == school) &
            (merged_data['time_of_day'] == meal_time)
        ]
        
        if len(item_data) > 0:
            avg_sales = item_data['total'].mean()
            avg_health = item_data['HealthScore'].mean()
            
            enhanced_recommendations.append({
                'meal': item_name,
                'avg_sales': avg_sales,
                'avg_health': avg_health,
                'ucb_score': ucb_score,  # The model's confidence score
                'combined_score': ucb_score  # Use model's score for ranking
            })
        else:
            # Fallback if no historical data found
            enhanced_recommendations.append({
                'meal': item_name,
                'avg_sales': 0,
                'avg_health': 3.0,  # Default average health
                'ucb_score': ucb_score,
                'combined_score': ucb_score
            })
    
    return enhanced_recommendations

def print_enhanced_recommendations(recommendations):
    """
    Enhanced display that shows health-popularity balance
    Helps cafeteria staff understand WHY items are recommended
    """
    print("TOP RECOMMENDATIONS - Using Trained Model (Optimal Balance)")
    print("=" * 80)
    
    for rank, rec in enumerate(recommendations, 1):
        # Health classification
        if rec['avg_health'] >= 4.5:
            health_status = "HEALTH STAR "
        elif rec['avg_health'] >= 4.0:
            health_status = "Very Healthy "
        elif rec['avg_health'] >= 3.5:
            health_status = "Moderately Healthy "
        else:
            health_status = "Less Healthy "
            
        # Popularity classification  
        if rec['avg_sales'] >= 150:
            popularity_status = "VERY POPULAR "
        elif rec['avg_sales'] >= 100:
            popularity_status = "Popular "
        elif rec['avg_sales'] >= 50:
            popularity_status = "Moderate "
        else:
            popularity_status = "Less Popular "
        
        print(f"{rank}. {rec['meal']}")
        print(f"   Sales: {rec['avg_sales']:.0f} ({popularity_status})")
        print(f"   Health: {rec['avg_health']:.1f}")
        print(f"   Model Score: {rec['ucb_score']:.2f} (confidence)")
        print()
    



def create_directory_if_needed(file_path):
    """Create parent directory if it doesn't exist"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

def load_sales_data(file_path):
    """Load the raw sales data CSV file"""
    print("Loading raw sales data...")
    data = pd.read_csv(file_path, low_memory=False)
    return data

def standardize_text_column(data, column_name):
    """
    Convert column to string and remove extra whitespace.
    This ensures consistent text matching (e.g., "Pizza" vs "PIZZA" vs "pizza")
    """
    data[column_name] = data[column_name].astype(str).str.strip()
    return data

# ===== BUILD ITEM MAPPING =====

def create_item_mapping(data, output_file):
    """
    Create unique list of all meal items and assign each an index number.
    
    Why we need this: Machine learning models understand numbers, not text.
    This creates a dictionary that translates meal names to numbers.
    
    Input: Raw sales data with 'description' column containing meal names
    Output: CSV file mapping meal names to unique indices (0, 1, 2, ...)
    """
    print("Creating item mapping.")
    
    # Get unique items from description column
    # Example: ["CHICKEN TENDERS", "PIZZA", "SALAD", "CHICKEN TENDERS"] → ["CHICKEN TENDERS", "PIZZA", "SALAD"]
    unique_items = sorted(data['description'].unique())
    
    # Create mapping dictionary: meal_name -> index_number
    item_to_index = {}
    for index, item_name in enumerate(unique_items):
        item_to_index[item_name] = index  # "CHICKEN TENDERS" - 0, "PIZZA" - 1, etc.
    
    # Save mapping to CSV file
    create_directory_if_needed(output_file)
    mapping_df = pd.DataFrame({
        "item": unique_items,        # Meal names
        "item_idx": range(len(unique_items))  # Corresponding numbers
    })
    mapping_df.to_csv(output_file, index=False)
    
    num_items = len(unique_items)
    print(f"Created item mapping: {num_items} unique meal items")
    print(f"Saved to {output_file}")
    
    return item_to_index, unique_items

# ===== BUILD TIME SLOT MAPPING =====

def create_time_slot_mapping(data, output_file):
    """
    Create unique ID for each (date, school, meal_time) combination.
    
    Why we need this: Each unique combination represents one decision point
    where the cafeteria had to choose which meals to serve.
    
    Input: Raw data with date, school_name, and time_of_day columns
    Output: CSV mapping time slots to unique IDs and lookup dictionary
    """
    print("Creating time slot mapping...")
    
    # Standardize text columns to ensure consistent matching
    data = data.copy()
    data = standardize_text_column(data, 'date')
    data = standardize_text_column(data, 'school_name')
    data = standardize_text_column(data, 'time_of_day')
    
    # Get unique combinations of date + school + meal time
    # Example: (2025-03-03, COLVIN_RUN_ELEMENTARY, breakfast)
    unique_combinations = data[['date', 'school_name', 'time_of_day']].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(['date', 'school_name', 'time_of_day'])
    unique_combinations = unique_combinations.reset_index(drop=True)
    
    # Assign time slot ID to each combination (0, 1, 2, ...)
    unique_combinations['time_slot_id'] = range(len(unique_combinations))
    
    # Save to CSV
    create_directory_if_needed(output_file)
    unique_combinations.to_csv(output_file, index=False)
    
    num_slots = len(unique_combinations)
    print(f"Created time slot mapping: {num_slots} unique time slots")
    print(f"Saved to {output_file}")
    
    # Create lookup dictionary for quick access
    # Key: (date, school, time_of_day), Value: time_slot_id
    time_slot_lookup = {}
    for idx, row in unique_combinations.iterrows():
        key = (row['date'], row['school_name'], row['time_of_day'])
        time_slot_lookup[key] = int(row['time_slot_id'])
    
    return time_slot_lookup, unique_combinations

# ===== BUILD ACTION MATRIX =====

def create_action_matrix(data, item_to_index, time_slot_lookup, output_file):
    """
    Create availability matrix showing which items were served at each time slot.
    
    Why we need this: The model needs to know which meals were actually available
    to choose from at each decision point.
    
    Matrix format: action_matrix[time_slot][item] = 1 if available, 0 if not
    
    Input: 
      - data: Raw sales data
      - item_to_index: Mapping from meal names to numbers
      - time_slot_lookup: Mapping from time slots to IDs
    Output: CSV file with availability matrix
    """
    print("Creating action matrix")
    
    # Clean and standardize data
    data = data.copy()
    data = standardize_text_column(data, 'date')
    data = standardize_text_column(data, 'school_name')
    data = standardize_text_column(data, 'time_of_day')
    data = standardize_text_column(data, 'description')
    
    num_items = len(item_to_index)        # Total number of unique meal items
    num_time_slots = len(time_slot_lookup) # Total number of time slots
    
    # Create empty matrix: time_slots * items, initialized to 0 (not available)
    action_matrix = np.zeros((num_time_slots, num_items), dtype=int)
    
    # Assign time slot ID to each row in the data
    data['time_slot_key'] = list(zip(data['date'], data['school_name'], data['time_of_day']))
    data['time_slot_id'] = data['time_slot_key'].map(time_slot_lookup)
    
    # Fill matrix: for each time slot, mark which items were available
    for time_slot_id, group in data.groupby('time_slot_id'):
        time_slot_id = int(time_slot_id)
        # For each unique meal served in this time slot
        for item_name in group['description'].unique():
            item_idx = item_to_index.get(str(item_name))
            if item_idx is not None:
                # Mark this item as available at this time slot
                action_matrix[time_slot_id, item_idx] = 1
    
    # Convert to DataFrame for saving
    create_directory_if_needed(output_file)
    action_df = pd.DataFrame(action_matrix, columns=[f"item_{i}" for i in range(num_items)])
    action_df['time_slot_id'] = range(num_time_slots)  # Add time slot IDs
    
    # Reorder columns: time_slot_id first, then items
    column_order = ['time_slot_id'] + [f"item_{i}" for i in range(num_items)]
    action_df = action_df[column_order]
    action_df.to_csv(output_file, index=False)
    
    # Calculate coverage: percentage of cells that are 1 (items available)
    coverage_percent = 100 * action_matrix.sum() / (num_time_slots * num_items)
    print(f"Created action matrix: {num_time_slots} time slots * {num_items} items")
    print(f"Coverage: {coverage_percent:.1f}% (percentage of possible servings)")
    print(f"Saved to {output_file}")
    
    return action_matrix

# ===== BUILD FEATURE MATRIX =====

def standardize_features(values):
    """
    Convert values to z-scores (standardized scores).
    
    Why we do this: Puts all nutritional features on the same scale (mean=0, std=1).
    This prevents features with large values (like Calories) from dominating
    features with small values (like Vitamin C).
    
    Formula: z-score = (value - mean) / standard_deviation
    """
    mean = np.nanmean(values)
    std_dev = np.nanstd(values)
    
    # Handle edge case where std_dev is 0 or NaN (all values same)
    if not np.isfinite(std_dev) or std_dev < 1e-8:
        std_dev = 1.0
    
    # Compute z-scores
    z_scores = (values - mean) / std_dev
    
    return z_scores

def create_feature_matrix(data, item_to_index, time_slot_lookup, output_file):
    """
    Create feature matrix with nutritional values for each meal serving.
    
    Why we need this: Provides the "context" for machine learning - the nutritional
    characteristics that might influence meal popularity.
    
    Each row represents one meal serving at one time slot.
    Each column represents one nutritional feature (standardized).
    
    Input: Raw data with nutritional information
    Output: CSV with feature matrix for machine learning
    """
    print("Creating feature matrix...")
    
    # Clean and standardize data
    data = data.copy()
    data = standardize_text_column(data, 'date')
    data = standardize_text_column(data, 'school_name')
    data = standardize_text_column(data, 'time_of_day')
    data = standardize_text_column(data, 'description')
    
    # Assign time slot IDs
    data['time_slot_key'] = list(zip(data['date'], data['school_name'], data['time_of_day']))
    data['time_slot_id'] = data['time_slot_key'].map(time_slot_lookup)
    
    # Keep only meals that have valid time slot IDs and are in our item mapping
    data = data.dropna(subset=['time_slot_id'])
    data['time_slot_id'] = data['time_slot_id'].astype(int)
    data = data[data['description'].isin(item_to_index.keys())].copy()
    
    # Build feature matrix from nutritional data
    feature_data = {}  # Dictionary to store each feature column
    
    for nutrient in nutrient_columns:
        if nutrient in data.columns:
            # Convert to numeric, handle missing values
            values = pd.to_numeric(data[nutrient], errors='coerce').to_numpy()
            
            # Replace NaN values with median (middle value)
            if np.isnan(values).any():
                median_val = np.nanmedian(values)
                values = np.where(np.isnan(values), median_val, values)
            
            # Standardize to z-scores
            z_scores = standardize_features(values)
            feature_data[f"{nutrient}_z"] = z_scores  # _z indicates z-scored
    
    # Create final feature matrix by stacking all feature columns
    feature_matrix = np.column_stack(list(feature_data.values())).astype(np.float32)
    
    # Prepare output dataframe with metadata and features
    feature_df = pd.DataFrame(feature_matrix, columns=list(feature_data.keys()))
    feature_df['time_slot_id'] = data['time_slot_id'].values    # Which time slot
    feature_df['item'] = data['description'].values             # Which meal
    feature_df['item_idx'] = data['description'].map(item_to_index).astype(int).values  # Meal number
    
    # Reorder columns: metadata first, then nutritional features
    metadata_cols = ['time_slot_id', 'item', 'item_idx']
    feature_cols = list(feature_data.keys())
    feature_df = feature_df[metadata_cols + feature_cols]
    
    # Save to CSV
    create_directory_if_needed(output_file)
    feature_df.to_csv(output_file, index=False)
    
    num_rows = len(feature_df)
    num_features = len(feature_cols)
    print(f"Created feature matrix: {num_rows} meal servings * {num_features} nutritional features")
    print(f"Saved to {output_file}")
    
    return feature_matrix

# ===== MAIN EXECUTION =====

def main():
    """
    Main function: Transform raw FCPS sales data into structured matrices for machine learning.
    
    This pipeline creates 4 essential files:
    1. item_mapping.csv - Translates meal names to numbers
    2. time_slot_mapping.csv - Organizes serving times into slots  
    3. action_matrix.csv - Shows what was available when
    4. feature_matrix.csv - Nutritional context for machine learning
    """
    print("=" * 70)
    print("BUILDING FCPS DATA MATRICES")
    print("=" * 70)
    print("Transforming raw cafeteria data into machine learning format")
    print()
    
    # ===== STEP 1: LOAD RAW DATA =====
    print("[1/5] Loading raw sales data.")
    sales_data = load_sales_data(input_csv)
    print(f"Loaded {len(sales_data)} rows of raw sales data")
    print("Columns available:", list(sales_data.columns))
    print()
    
    # ===== STEP 2: BUILD ITEM MAPPING =====
    print("[2/5] Building item mapping.")
    if item_mapping_file.exists() and not overwrite_existing:
        print("Item mapping already exists, skipping rebuild")
        # Load existing mapping
        item_map_df = pd.read_csv(item_mapping_file)
        item_to_index = dict(zip(item_map_df['item'].astype(str), item_map_df['item_idx'].astype(int)))
    else:
        print("Building new item mapping.")
        item_to_index, unique_items = create_item_mapping(sales_data, item_mapping_file)
    print()
    
    # ===== STEP 3: BUILD TIME SLOT MAPPING =====
    print("[3/5] Building time slot mapping.")
    if time_slot_mapping_file.exists() and not overwrite_existing:
        print("Time slot mapping already exists, skipping rebuild")
        # Load existing mapping
        time_slot_df = pd.read_csv(time_slot_mapping_file)
        time_slot_lookup = dict(zip(
            zip(time_slot_df['date'].astype(str), 
                time_slot_df['school_name'].astype(str), 
                time_slot_df['time_of_day'].astype(str)),
            time_slot_df['time_slot_id'].astype(int)
        ))
    else:
        print("Building new time slot mapping.")
        time_slot_lookup, time_slot_combos = create_time_slot_mapping(sales_data, time_slot_mapping_file)
    print()
    
    # ===== STEP 4: BUILD ACTION MATRIX =====
    print("[4/5] Building action matrix.")
    if action_matrix_file.exists() and not overwrite_existing:
        print("Action matrix already exists, skipping rebuild")
    else:
        print("Building new action matrix.")
        action_matrix = create_action_matrix(sales_data, item_to_index, time_slot_lookup, action_matrix_file)
    print()
    
    # ===== STEP 5: BUILD FEATURE MATRIX =====
    print("[5/5] Building feature matrix.")
    if feature_matrix_file.exists() and not overwrite_existing:
        print("Feature matrix already exists, skipping rebuild")
    else:
        print("Building new feature matrix.")
        feature_matrix = create_feature_matrix(sales_data, item_to_index, time_slot_lookup, feature_matrix_file)
    print()
    
    
if __name__ == "__main__":
    main()


def load_env_data(repo_root: str = ""):
    """
    Load commonly used CSV files and populate module globals:
    - merged_data
    - feature_df
    - item_to_index
    - index_to_item

    repo_root: optional path prefix (default: current working directory)
    """
    global merged_data, feature_df, item_to_index, index_to_item
    data_dir = Path(repo_root) / "src" / "data" if repo_root else Path("src") / "data"

    merged_data_file = data_dir / "fcps_data_with_timestamps.csv"
    feature_matrix_file = data_dir / "feature_matrix.csv"
    item_mapping_file = data_dir / "item_mapping.csv"

    print(f"Loading merged data from: {merged_data_file}")
    merged_data = pd.read_csv(merged_data_file, low_memory=False)

    print(f"Loading feature matrix from: {feature_matrix_file}")
    feature_df = pd.read_csv(feature_matrix_file, low_memory=False)

    print(f"Loading item mapping from: {item_mapping_file}")
    item_mapping_df = pd.read_csv(item_mapping_file, low_memory=False)
    item_to_index = dict(zip(item_mapping_df['item'], item_mapping_df['item_idx']))
    index_to_item = dict(zip(item_mapping_df['item_idx'], item_mapping_df['item']))

    print("Environment data loaded into env module globals.")
