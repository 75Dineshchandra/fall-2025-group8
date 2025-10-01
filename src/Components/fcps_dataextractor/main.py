"""
Main script for Fairfax County Public Schools nutrition data collection.
Orchestrates the data fetching and processing pipeline.
"""

# --- add these two lines first ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds /.../src to sys.path


import json
from datetime import datetime

from fairfax.config import SCHOOLS_JSON_PATH, OUTPUT_CSV_FILENAME, OUTPUT_JSON_FILENAME
from fairfax.data_fetcher import get_menu_data, generate_date_ranges, respectful_delay
from fairfax.data_processor import extract_nutrition_data, save_data_to_files, print_summary_statistics


def load_schools_data(file_path):
    """
    Load school data from JSON file.
    
    Args:
        file_path (str): Path to the schools JSON file
    
    Returns:
        dict: School data dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" Schools file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f" Invalid JSON in schools file: {file_path}")
        return None


def collect_data_for_school(school, date_ranges):
    """
    Collect nutrition data for a single school across all date ranges.
    
    Args:
        school (dict): School information with BuildingId and Name
        date_ranges (list): List of date range dictionaries
    
    Returns:
        list: Nutrition data records for the school
    """
    building_id = school['BuildingId']
    building_name = school['Name']
    school_data = []
    
    print(f" Processing: {building_name}")
    
    for date_range in date_ranges:
        print(f"  {date_range['month']}")
        
        # Fetch menu data from API
        menu_data = get_menu_data(
            building_id, building_name, 
            date_range["start"], date_range["end"]
        )
        
        if menu_data:
            # Extract nutrition data from API response
            nutrition_data = extract_nutrition_data(
                menu_data, building_id, building_name, date_range
            )
            school_data.extend(nutrition_data)
            print(f" Found {len(nutrition_data)} food items")
        else:
            print(f" No data for this period")
        
        # Add delay between requests
        respectful_delay()
    
    print(f"  Total for {building_name}: {len(school_data)} records")
    return school_data


def main():
    """Main execution function for the nutrition data collection pipeline."""
    print(" Fairfax County Public Schools Nutrition Data Collection")
    print("="*60)
    
    # Step 1: Load school data
    print(" Loading school data...")
    schools_data = load_schools_data(SCHOOLS_JSON_PATH)
    if not schools_data:
        return
    
    schools = schools_data['Buildings']
    print(f" Loaded {len(schools)} schools")
    
    # Step 2: Generate date ranges based on config
    print(" Generating date ranges...")
    date_ranges = generate_date_ranges()
    print(f" Generated {len(date_ranges)} date ranges: {date_ranges[0]['month']} to {date_ranges[-1]['month']}")
    
    # Step 3: Collect data
    print("\n Starting data collection...")
    all_nutrition_data = []
    
    # For demonstration, process only first 3 schools
    #sample_schools = schools[:3]
    
    for school in schools:
        school_data = collect_data_for_school(school, date_ranges)
        all_nutrition_data.extend(school_data)
    
    # Step 4: Save results
    if all_nutrition_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_filename = f"fairfax_nutrition_data_{timestamp}"
        
        print(f"\n Saving data to files...")
        csv_path, json_path = save_data_to_files(all_nutrition_data, base_filename)
        
        if csv_path and json_path:
            print(f" CSV saved: {csv_path}")
            print(f" JSON saved: {json_path}")
            
            # Print summary statistics
            print_summary_statistics(all_nutrition_data)
        else:
            print("Failed to save data")
    else:
        print(" No data was collected")


if __name__ == "__main__":
    # Execute the main function
    main()