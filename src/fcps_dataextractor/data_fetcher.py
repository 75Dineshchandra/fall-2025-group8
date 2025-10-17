"""
Module for fetching nutrition data from the LINQ Connect API.
Handles API requests, error handling, and rate limiting.
"""

import time
import requests
from datetime import datetime, timedelta

from config import (
    BASE_URL, HEADERS, DISTRICT_ID, REQUEST_TIMEOUT, REQUEST_DELAY,
    DATE_RANGE_TYPE, START_YEAR, END_YEAR, START_MONTH, END_MONTH,
    CUSTOM_START_DATE, CUSTOM_END_DATE
)


def get_menu_data(building_id, building_name, start_date, end_date):
    """
    Fetch menu data for a specific school and date range from the API.
    
    Args:
        building_id (str): Unique identifier for the school
        building_name (str): Name of the school for logging
        start_date (str): Start date in format "M-D-YYYY"
        end_date (str): End date in format "M-D-YYYY"
    
    Returns:
        dict: JSON response from API or None if request fails
    """
    params = {
        "buildingId": building_id,
        "districtId": DISTRICT_ID,
        "startDate": start_date,
        "endDate": end_date
    }
    
    try:
        response = requests.get(
            BASE_URL, 
            headers=HEADERS, 
            params=params, 
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"HTTP {response.status_code} for {building_name}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Timeout for {building_name}")
        return None
    except requests.exceptions.ConnectionError:
        print(f" Connection error for {building_name}")
        return None
    except requests.exceptions.RequestException as e:
        print(f" Request error for {building_name}: {e}")
        return None


def generate_date_ranges():
    """
    Generate date ranges based on configuration in config.py.
    Supports calendar year, school year, or custom date ranges.
    
    Returns:
        list: List of dictionaries with date range information
    """
    date_ranges = []
    
    if DATE_RANGE_TYPE == "calendar_year":
        date_ranges = generate_calendar_year_date_ranges()
    elif DATE_RANGE_TYPE == "school_year":
        date_ranges = generate_school_year_date_ranges()
    elif DATE_RANGE_TYPE == "custom":
        date_ranges = generate_custom_date_ranges()
    else:
        print(f" Unknown DATE_RANGE_TYPE: {DATE_RANGE_TYPE}. Using calendar year.")
        date_ranges = generate_calendar_year_date_ranges()
    
    return date_ranges


def generate_calendar_year_date_ranges():
    """
    Generate monthly date ranges for calendar year(s) (January to December).
    
    Returns:
        list: List of dictionaries with date range information
    """
    date_ranges = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(START_MONTH, END_MONTH + 1):
            # Create date objects
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            # Ensure we don't go beyond current date if collecting current year
            if end_date > datetime.now():
                end_date = datetime.now()
                # If we reached current month, break the loop
                if month == datetime.now().month and year == datetime.now().year:
                    date_ranges.append({
                        "start": start_date.strftime("%-m-%-d-%Y"),
                        "end": end_date.strftime("%-m-%-d-%Y"),
                        "month": start_date.strftime("%B %Y"),
                        "year": year,
                        "month_number": month
                    })
                    break
            
            date_ranges.append({
                "start": start_date.strftime("%-m-%-d-%Y"),
                "end": end_date.strftime("%-m-%-d-%Y"),
                "month": start_date.strftime("%B %Y"),
                "year": year,
                "month_number": month
            })
    
    return date_ranges


def generate_school_year_date_ranges():
    """
    Generate monthly date ranges for school year (August to May).
    
    Returns:
        list: List of dictionaries with date range information
    """
    date_ranges = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        # School year typically runs August to May
        for month_offset in range(10):  # 10 months: Aug-May
            month = (8 + month_offset) % 12
            if month == 0: 
                month = 12
            
            current_year = year if month >= 8 else year + 1
            
            # Create date objects
            start_date = datetime(current_year, month, 1)
            if month == 12:
                end_date = datetime(current_year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(current_year, month + 1, 1) - timedelta(days=1)
            
            date_ranges.append({
                "start": start_date.strftime("%-m-%-d-%Y"),
                "end": end_date.strftime("%-m-%-d-%Y"),
                "month": start_date.strftime("%B %Y"),
                "year": current_year,
                "month_number": month
            })
    
    return date_ranges


def generate_custom_date_ranges():
    """
    Generate date ranges based on custom start and end dates.
    
    Returns:
        list: List of dictionaries with date range information
    """
    date_ranges = []
    
    try:
        start_date = datetime.strptime(CUSTOM_START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(CUSTOM_END_DATE, "%Y-%m-%d")
        
        current_date = start_date
        while current_date <= end_date:
            month_start = datetime(current_date.year, current_date.month, 1)
            if current_date.month == 12:
                month_end = datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
            
            # Don't go beyond the custom end date
            if month_end > end_date:
                month_end = end_date
            
            date_ranges.append({
                "start": month_start.strftime("%-m-%-d-%Y"),
                "end": month_end.strftime("%-m-%-d-%Y"),
                "month": month_start.strftime("%B %Y"),
                "year": month_start.year,
                "month_number": month_start.month
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
                
    except ValueError as e:
        print(f" Invalid custom date format: {e}")
        return []
    
    return date_ranges


def respectful_delay(seconds=REQUEST_DELAY):
    """
    Add a delay between requests to be respectful to the API server.
    
    Args:
        seconds (float): Number of seconds to delay
    """
    time.sleep(seconds)