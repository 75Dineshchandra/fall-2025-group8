"""
Module for processing and extracting nutrition data from API responses.
Handles data transformation and structure normalization.
"""

import pandas as pd


def extract_nutrition_data(menu_data, building_id, building_name, date_range_info):
    """
    Extract and flatten nutrition data from the API response.
    
    Args:
        menu_data (dict): Raw JSON response from the API
        building_id (str): School building identifier
        building_name (str): School name
        date_range_info (dict): Information about the date range
    
    Returns:
        list: List of dictionaries containing flattened nutrition data
    """
    if not menu_data or 'FamilyMenuSessions' not in menu_data:
        return []
    
    nutrition_records = []
    
    # Iterate through all meal sessions (Breakfast, Lunch, etc.)
    for session in menu_data['FamilyMenuSessions']:
        serving_session = session.get('ServingSession', 'Unknown')
        
        # Iterate through menu plans
        for menu_plan in session.get('MenuPlans', []):
            menu_plan_name = menu_plan.get('MenuPlanName', 'Unknown')
            
            # Iterate through days in the menu
            for day in menu_plan.get('Days', []):
                date = day.get('Date', 'Unknown')
                
                # Iterate through meals (Main Entree, Side Dish, etc.)
                for meal in day.get('MenuMeals', []):
                    meal_name = meal.get('MenuMealName', 'Unknown')
                    
                    # Iterate through food categories
                    for category in meal.get('RecipeCategories', []):
                        category_name = category.get('CategoryName', 'Unknown')
                        
                        # Iterate through individual recipes/food items
                        for recipe in category.get('Recipes', []):
                            record = create_nutrition_record(
                                recipe, building_id, building_name, 
                                date_range_info, serving_session, 
                                menu_plan_name, date, meal_name, category_name
                            )
                            nutrition_records.append(record)
    
    return nutrition_records


def create_nutrition_record(recipe, building_id, building_name, date_range_info, 
                           serving_session, menu_plan_name, date, meal_name, category_name):
    """
    Create a standardized nutrition record from recipe data.
    
    Args:
        recipe (dict): Individual recipe/food item data
        building_id (str): School identifier
        building_name (str): School name
        date_range_info (dict): Date range metadata
        serving_session (str): Meal time (Breakfast, Lunch, etc.)
        menu_plan_name (str): Menu plan name
        date (str): Specific date
        meal_name (str): Meal category name
        category_name (str): Food category name
    
    Returns:
        dict: Standardized nutrition record
    """
    # Base record with metadata
    record = {
        # School information
        'SchoolID': building_id,
        'SchoolName': building_name,
        'DistrictID': date_range_info.get('district_id', ''),
        'DistrictName': 'Fairfax County Public Schools',
        
        # Date information
        'Month': date_range_info['month'],
        'MonthNumber': date_range_info['month_number'],
        'Year': date_range_info['year'],
        'StartDate': date_range_info['start'],
        'EndDate': date_range_info['end'],
        'Date': date,
        
        # Meal information
        'MealTime': serving_session,
        'MenuPlan': menu_plan_name,
        'MealCategory': meal_name,
        'FoodCategory': category_name,
        
        # Recipe information
        'RecipeName': recipe.get('RecipeName', 'Unknown'),
        'RecipeID': recipe.get('RecipeIdentifier', 'Unknown'),
        'ItemID': recipe.get('ItemId', 'Unknown'),
        'ServingSize': recipe.get('ServingSize', 'Unknown'),
        'GramsPerServing': recipe.get('GramPerServing', 0),
        'HasNutrients': recipe.get('HasNutrients', False),
        
        # Dietary information
        'Allergens': ', '.join(recipe.get('Allergens', [])),
        'DietaryRestrictions': ', '.join(recipe.get('DietaryRestrictions', [])),
        'ReligiousRestrictions': ', '.join(recipe.get('ReligiousRestrictions', []))
    }
    
    # Add nutrition values
    for nutrient in recipe.get('Nutrients', []):
        nutrient_name = nutrient.get('Name', 'Unknown')
        record[nutrient_name] = nutrient.get('Value', 0)
        record[f'{nutrient_name}_Unit'] = nutrient.get('Unit', '')
    
    return record


def save_data_to_files(data, base_filename):
    """
    Save nutrition data to both CSV and JSON files.
    
    Args:
        data (list): List of nutrition records
        base_filename (str): Base filename without extension
    
    Returns:
        tuple: Paths to the saved CSV and JSON files
    """
    if not data:
        print(" No data to save")
        return None, None
    
    # Convert to DataFrame for CSV export
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_filename = f"{base_filename}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    # Save to JSON
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        import json
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return csv_filename, json_filename


def print_summary_statistics(data):
    """
    Print summary statistics about the collected data.
    
    Args:
        data (list): List of nutrition records
    """
    if not data:
        print("No data available for summary")
        return
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print(" DATA COLLECTION SUMMARY")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"Schools: {df['SchoolName'].nunique()}")
    print(f"Months: {df['Month'].nunique()}")
    print(f"Meal times: {df['MealTime'].nunique()}")
    print(f"Unique food items: {df['RecipeName'].nunique()}")
    
    if 'Calories' in df.columns:
        print(f"Average calories: {df['Calories'].mean():.1f}")
    
    print("="*50)
