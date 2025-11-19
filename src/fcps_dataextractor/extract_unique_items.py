#!/usr/bin/env python3
"""
Script to extract unique menu items with nutrition information from a CSV file.
Deduplicates items by RecipeID and creates clean CSV/JSON output files.
"""

import pandas as pd
import json
import argparse
from datetime import datetime
import os


def load_csv_data(csv_file_path):
    """
    Load nutrition data from CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded data as DataFrame
    """
    try:
        print(f"Reading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df):,} records")
        return df
    except Exception as e:
        print(f" Error reading CSV file: {e}")
        return None


def extract_unique_items(df):
    """
    Extract unique menu items from the DataFrame.
    
    Args:
        df (pandas.DataFrame): Input data with nutrition information
    
    Returns:
        pandas.DataFrame: DataFrame with unique menu items
    """
    if df is None or df.empty:
        print("  No data available")
        return None
    
    print(" Extracting unique menu items...")
    
    # Group by RecipeID and keep the first occurrence of each unique item
    unique_items = df.drop_duplicates(subset=['RecipeID'], keep='first')
    
    print(f" Found {len(unique_items):,} unique menu items")
    return unique_items


def create_clean_nutrition_df(unique_items_df):
    """
    Create a clean DataFrame with only nutrition information.
    
    Args:
        unique_items_df (pandas.DataFrame): DataFrame with unique items
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame with nutrition info only
    """
    # Columns to keep (nutrition information and item metadata)
    nutrition_columns = [
        # Item identification
        'RecipeID', 'RecipeName', 'ItemID', 'ServingSize', 'GramsPerServing',
        'HasNutrients', 'Allergens', 'DietaryRestrictions', 'ReligiousRestrictions',
        
        # Nutrition values
        'Calories', 'Calories_Unit', 'Total Fat', 'Total Fat_Unit',
        'Saturated Fat', 'Saturated Fat_Unit', 'Trans Fat', 'Trans Fat_Unit',
        'Cholesterol', 'Cholesterol_Unit', 'Sodium', 'Sodium_Unit',
        'Total Carbohydrate', 'Total Carbohydrate_Unit',
        'Dietary Fiber', 'Dietary Fiber_Unit',
        'Total Sugars', 'Total Sugars_Unit',
        'Added Sugars', 'Added Sugars_Unit',
        'Protein', 'Protein_Unit',
        'Vitamin D (D2 + D3)', 'Vitamin D (D2 + D3)_Unit',
        'Calcium', 'Calcium_Unit',
        'Iron', 'Iron_Unit',
        'Potassium', 'Potassium_Unit',
        'Vitamin A', 'Vitamin A_Unit',
        'Vitamin C', 'Vitamin C_Unit'
    ]
    
    # Only keep columns that actually exist in the DataFrame
    available_columns = [col for col in nutrition_columns if col in unique_items_df.columns]
    
    # Create the clean DataFrame
    clean_df = unique_items_df[available_columns].copy()
    
    # Reorder columns for better readability
    preferred_order = [
        # Basic info
        'RecipeID', 'RecipeName', 'ServingSize', 'GramsPerServing',
        
        # Macronutrients
        'Calories', 'Calories_Unit',
        'Total Fat', 'Total Fat_Unit',
        'Saturated Fat', 'Saturated Fat_Unit',
        'Trans Fat', 'Trans Fat_Unit',
        'Cholesterol', 'Cholesterol_Unit',
        'Sodium', 'Sodium_Unit',
        'Total Carbohydrate', 'Total Carbohydrate_Unit',
        'Dietary Fiber', 'Dietary Fiber_Unit',
        'Total Sugars', 'Total Sugars_Unit',
        'Added Sugars', 'Added Sugars_Unit',
        'Protein', 'Protein_Unit',
        
        # Micronutrients
        'Vitamin D (D2 + D3)', 'Vitamin D (D2 + D3)_Unit',
        'Calcium', 'Calcium_Unit',
        'Iron', 'Iron_Unit',
        'Potassium', 'Potassium_Unit',
        'Vitamin A', 'Vitamin A_Unit',
        'Vitamin C', 'Vitamin C_Unit',
        
        # Dietary info
        'HasNutrients', 'Allergens', 'DietaryRestrictions', 'ReligiousRestrictions',
        'ItemID'
    ]
    
    # Only include columns that exist in our clean DataFrame
    final_columns = [col for col in preferred_order if col in clean_df.columns]
    
    return clean_df[final_columns]


def save_output_files(clean_df, output_prefix):
    """
    Save the cleaned data to CSV and JSON files.
    
    Args:
        clean_df (pandas.DataFrame): Cleaned DataFrame
        output_prefix (str): Prefix for output filenames
    
    Returns:
        tuple: Paths to the saved CSV and JSON files
    """
    if clean_df is None or clean_df.empty:
        print(" No data to save")
        return None, None
    
    # Save to CSV
    csv_filename = f"{output_prefix}.csv"
    clean_df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    # Save to JSON
    json_filename = f"{output_prefix}.json"
    
    # Convert DataFrame to list of dictionaries for JSON
    json_data = clean_df.to_dict('records')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return csv_filename, json_filename


def print_summary(clean_df, input_file):
    """
    Print summary statistics about the unique menu items.
    
    Args:
        clean_df (pandas.DataFrame): Cleaned DataFrame with unique items
        input_file (str): Input file name
    """
    if clean_df is None or clean_df.empty:
        print("No data available for summary")
        return
    
    print("\n" + "="*60)
    print("UNIQUE MENU ITEMS SUMMARY")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Total unique items: {len(clean_df):,}")
    
    # Count items with nutrition data
    if 'HasNutrients' in clean_df.columns:
        with_nutrients = clean_df['HasNutrients'].sum()
        print(f"Items with nutrition data: {with_nutrients:,} ({with_nutrients/len(clean_df)*100:.1f}%)")
    
    # Show nutrient availability statistics
    nutrient_columns = [
        'Calories', 'Total Fat', 'Saturated Fat', 'Trans Fat', 'Cholesterol',
        'Sodium', 'Total Carbohydrate', 'Dietary Fiber', 'Total Sugars',
        'Added Sugars', 'Protein', 'Vitamin D (D2 + D3)', 'Calcium', 'Iron',
        'Potassium', 'Vitamin A', 'Vitamin C'
    ]
    
    print(f"\n Nutrient availability:")
    for nutrient in nutrient_columns:
        if nutrient in clean_df.columns:
            non_zero = (clean_df[nutrient] > 0).sum()
            percentage = non_zero / len(clean_df) * 100
            print(f"  {nutrient}: {non_zero:,} items ({percentage:.1f}%)")
    
    # Show sample items
    print(f"\n  Sample unique items:")
    sample_cols = ['RecipeName', 'Calories', 'Protein', 'Total Sugars']
    available_cols = [col for col in sample_cols if col in clean_df.columns]
    
    if available_cols:
        sample_items = clean_df[available_cols].head(5)
        for _, item in sample_items.iterrows():
            item_desc = f"  - {item['RecipeName']}"
            if 'Calories' in available_cols:
                item_desc += f": {item['Calories']} cal"
            if 'Protein' in available_cols:
                item_desc += f", {item['Protein']}g protein"
            if 'Total Sugars' in available_cols:
                item_desc += f", {item['Total Sugars']}g sugar"
            print(item_desc)
    
    print("="*60)


def main():
    """Main function to extract unique items from CSV file."""
    parser = argparse.ArgumentParser(description='Extract unique menu items with nutrition information from CSV file')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', help='Output filename prefix', default='fairfax_unique_menu_items')
    
    args = parser.parse_args()
    
    print(" Extracting Unique Menu Items with Nutrition Information")
    print("="*70)
    
    # Load the CSV data
    df = load_csv_data(args.input)
    if df is None:
        return
    
    # Extract unique items
    unique_items_df = extract_unique_items(df)
    if unique_items_df is None:
        return
    
    # Create clean nutrition DataFrame
    clean_df = create_clean_nutrition_df(unique_items_df)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_prefix = f"{args.output}_{timestamp}"
    
    # Save output files
    print(f"\n Saving output files...")
    csv_path, json_path = save_output_files(clean_df, output_prefix)
    
    if csv_path and json_path:
        print(f" CSV saved: {csv_path}")
        print(f"JSON saved: {json_path}")
        
        # Print summary
        print_summary(clean_df, args.input)
        
        print(f"\n Success! Created unique items list with complete nutrition information.")
        print(f"Files contain {len(clean_df)} unique menu items.")
    else:
        print("Failed to save output files")


if __name__ == "__main__":
    main()