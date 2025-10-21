"""
Configuration constants for Fairfax County nutrition data collection.
This file contains all API endpoints, headers, and settings.
"""

# API Configuration
BASE_URL = "https://api.linqconnect.com/api/FamilyMenu"
DISTRICT_ID = "019aa8c9-09f1-ee11-a85e-985e1ae32d4b"

# Request Headers
HEADERS = {
    "Linq-Nutrition-Url": "d24b5f15-3a9c-4795-b4ad-5d7cf440704d",
    "Origin": "https://linqconnect.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3.1 Safari/605.1.15"
    )
}

# Data Collection Settings
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 1.0   # seconds between requests to be respectful
SCHOOLS_JSON_PATH = "src/fairfax/schools.json"


# Date Range Settings - UPDATED FOR CALENDAR YEAR
DATE_RANGE_TYPE = "calendar_year"  # Options: "calendar_year", "school_year", "custom"
START_YEAR = 2025
END_YEAR = 2025     # Same as START_YEAR for single year
START_MONTH = 1     # January
END_MONTH = 12      # December

# For custom date range (if DATE_RANGE_TYPE = "custom")
CUSTOM_START_DATE = "2025-01-01"  # YYYY-MM-DD format
CUSTOM_END_DATE = "2025-12-31"    # YYYY-MM-DD format

# Output Settings
OUTPUT_CSV_FILENAME = "fairfax_nutrition_data_{timestamp}.csv"
OUTPUT_JSON_FILENAME = "fairfax_nutrition_data_{timestamp}.json"