#  main_predict.py
# Predict for dates NOT in the historical data USING TRAINED MODEL

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import joblib

# Add this to import your LinUCB model
sys.path.append(str(Path(__file__).parent.parent))
from Components.model import LinUCB

# Load data
merged_data_file = Path("src/data/fcps_data_with_timestamps.csv")
merged_data = pd.read_csv(merged_data_file, low_memory=False)

# Load feature matrix to get nutritional features
feature_matrix_file = Path("src/data/feature_matrix.csv")
feature_df = pd.read_csv(feature_matrix_file, low_memory=False)

# Load item mapping to convert between item names and indices
item_mapping_file = Path("src/data/item_mapping.csv")
item_mapping_df = pd.read_csv(item_mapping_file, low_memory=False)

# Create mapping dictionaries
item_to_index = dict(zip(item_mapping_df['item'], item_mapping_df['item_idx']))
index_to_item = dict(zip(item_mapping_df['item_idx'], item_mapping_df['item']))

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
        print(f"   Health: {rec['avg_health']:.1f}/5.0 ({health_status})")
        print(f"   Model Score: {rec['ucb_score']:.2f} (confidence)")
        print()
    

# Main execution
if __name__ == "__main__":
    # Use the OPTIMAL model (lambda = 0.05 from your training)
    optimal_model_path = "src/tests/results/model_lambda_0.05.joblib"
    
    recommendations = recommend_with_trained_model(
        target_date='2025-10-15',
        school="COLVIN_RUN_ELEMENTARY",
        meal_time="lunch",
        model_path=optimal_model_path,
        top_k=5
    )
    
    if recommendations is None:
        print("Could not generate recommendations")
    else:
        print_enhanced_recommendations(recommendations)