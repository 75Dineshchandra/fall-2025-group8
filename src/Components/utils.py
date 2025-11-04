# health_score.py
# Purpose: Calculate health scores for meal items based on nutritional content
import numpy as np
import pandas as pd
from pathlib import Path

def infer_school_group(school_name):
    """Infer school group (elementary, middle, high) from school name"""
    school_lower = str(school_name).lower()
    if 'elementary' in school_lower:
        return 'elementary'
    elif 'middle' in school_lower:
        return 'middle'
    else:
        return 'high'

def health_score(row: pd.Series) -> float:
    """
    Calculate health score for a meal item based on nutritional content
    Returns a score from 0-10 where higher is healthier
    """
    DV = {
        "elementary": {
            "Calories": 2000, "Protein": 19, "Total Carbohydrate": 130, 
            "Dietary Fiber": 28, "Added Sugars": 50, "Total Fat": 78, 
            "Saturated Fat": 22, "Sodium": 1500, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 400, "Vitamin C": 25
        },
        "middle": {
            "Calories": 2600, "Protein": 34, "Total Carbohydrate": 130, 
            "Dietary Fiber": 36.4, "Added Sugars": 65, "Total Fat": 101, 
            "Saturated Fat": 29, "Sodium": 1800, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 600, "Vitamin C": 45
        },
        "high": {
            "Calories": 3200, "Protein": 52, "Total Carbohydrate": 130, 
            "Dietary Fiber": 44.8, "Added Sugars": 80, "Total Fat": 124, 
            "Saturated Fat": 36, "Sodium": 2300, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 900, "Vitamin C": 75
        }
    }
    
    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium", "Iron", 
            "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    school_group = str(row.get("school_group", "high")).lower()
    dv = DV["elementary"] if "elementary" in school_group else (
        DV["middle"] if "middle" in school_group else DV["high"]
    )

    good_score, bad_score = 0.0, 0.0
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val/ref)*100)
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val/ref)*100)

    raw_score = good_score - bad_score
    return raw_score  # Return raw unscaled score

def scale_health_score(raw_score: float) -> float:
    """
    Scale a raw health score from its natural range (-300 to 800) to 0-10.
    
    Args:
        raw_score: Raw health score from health_score() function
        
    Returns:
        float: Score scaled to 0-10 range with 2 decimal precision
    """
    min_score = -300  # worst-case negative score
    max_score = 800   # best-case score
    
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(10, scaled_score))  # clamp to [0,10]
    return np.round(scaled_score, 2)

def merge_sales_nutrition_data(sales_data, nutrition_data):
    """
    Merge sales data with nutrition data using description/sales_item columns
    
    Args:
        sales_data: DataFrame with sales data (must have 'description' column)
        nutrition_data: DataFrame with nutrition data (must have 'sales_item' column)
    
    Returns:
        merged_df: Merged DataFrame
        unmatched_items: DataFrame of items that couldn't be matched
    """
    print("Merging sales data with nutrition data...")
    
    # Align column names for join: sales.description ↔ mapping.sales_item
    if "description" not in sales_data.columns:
        raise ValueError(f"'description' not found in sales columns: {sales_data.columns.tolist()}")
    if "sales_item" not in nutrition_data.columns:
        raise ValueError(f"'sales_item' not found in nutrition columns: {nutrition_data.columns.tolist()}")

    merged_raw = sales_data.merge(
        nutrition_data,
        left_on="description",
        right_on="sales_item",
        how="left",
        suffixes=("", "_nut")
    )

    # Rows whose description did NOT find a nutrient row (strict exact match)
    unmatched_exact = (merged_raw[merged_raw["sales_item"].isna()]
                       [["description"]].drop_duplicates().sort_values("description"))
    print(f"Unmatched menu names (exact): {len(unmatched_exact)}")
    
    if len(unmatched_exact) > 0:
        print("Sample of unmatched items:")
        print(unmatched_exact.head(20))
    
    return merged_raw, unmatched_exact

def calculate_health_scores(sales_file: Path, nutrition_file: Path, output_file: Path = None):
    """
    Calculate health scores for all meals by merging sales and nutrition data
    
    Args:
        sales_file: Path to sales data CSV (must have 'description' column)
        nutrition_file: Path to nutrition data CSV (must have 'sales_item' column)  
        output_file: Path to save results (optional)
    
    Returns:
        DataFrame with health scores
        unmatched_items: DataFrame of items that couldn't be matched
    """
    print("Calculating health scores...")
    
    # Load data
    sales_data = pd.read_csv(sales_file, low_memory=False)
    nutrition_data = pd.read_csv(nutrition_file, low_memory=False)
    
    print(f"Loaded {len(sales_data)} rows from {sales_file}")
    print(f"Loaded {len(nutrition_data)} rows from {nutrition_file}")
    
    # Merge sales and nutrition data
    merged_data, unmatched_items = merge_sales_nutrition_data(sales_data, nutrition_data)
    
    # Add school_group column
    merged_data['school_group'] = merged_data['school_name'].apply(infer_school_group)
    
    # Calculate health scores only for matched items
    print("Calculating health scores for matched items...")
    matched_mask = merged_data["sales_item"].notna()
    merged_data.loc[matched_mask, "HealthScore"] = merged_data[matched_mask].apply(health_score, axis=1)
    
    # For unmatched items, set HealthScore to NaN or default value
    merged_data.loc[~matched_mask, "HealthScore"] = np.nan
    
    match_count = matched_mask.sum()
    unmatch_count = (~matched_mask).sum()
    print(f"Health scores calculated for {match_count} matched rows")
    print(f"Health scores not calculated for {unmatch_count} unmatched rows")
    
    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_file, index=False)
        print(f"Health scores calculated and saved to {output_file}")
        
        # Also save unmatched items for debugging
        unmatched_file = output_file.parent / "unmatched_items.csv"
        unmatched_items.to_csv(unmatched_file, index=False)
        print(f"Unmatched items saved to {unmatched_file}")
    
    return merged_data, unmatched_items

def calculate_health_scores_from_merged(merged_file: Path, output_file: Path = None):
    """
    Calculate health scores from already merged data file
    
    Args:
        merged_file: Path to already merged sales+nutrition data
        output_file: Path to save results (optional)
    
    Returns:
        DataFrame with health scores
    """
    print("Calculating health scores from merged data...")
    
    # Load already merged data
    merged_data = pd.read_csv(merged_file, low_memory=False)
    print(f"Loaded {len(merged_data)} rows from {merged_file}")
    
    # Add school_group column if not present
    if 'school_group' not in merged_data.columns:
        merged_data['school_group'] = merged_data['school_name'].apply(infer_school_group)
    
    # Calculate health scores
    merged_data["HealthScore"] = merged_data.apply(health_score, axis=1)
    
    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_file, index=False)
        print(f"Health scores calculated and saved to {output_file}")
    
    return merged_data

def main():
    """Main function for standalone health score calculation"""
    data_dir = Path("src/data")
    sales_file = data_dir / "sales.csv"
    nutrition_file = data_dir / "data_sales_nutrition.csv"
    output_file = data_dir / "data_healthscore_mapped.csv"

    
    #  separate sales and nutrition files
    calculate_health_scores(sales_file, nutrition_file, output_file)
    
    
    # calculate_health_scores_from_merged(nutrition_file, output_file)

if __name__ == "__main__":
    main()