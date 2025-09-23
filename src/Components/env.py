# %%
import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pandas.DataFrame: The loaded data as a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data


# %%
def get_actions(data):
    """
    Extract unique actions from the data.

    Parameters:
    - data (pandas.DataFrame): The input data.

    Returns:
    - numpy.ndarray: Unique actions from the 'description' column.
    """
    data = pd.DataFrame(data)
    actions = data['sales_item'].unique()   # ✅ directly get unique descriptions
    actions_list = actions.tolist()  # convert numpy array → python list
    #print("Unique actions as list:")
    #print(actions_list)
    return (actions)


# %%
def get_features(data):
    """
    Extract features from the data (all columns except 'description').

    Parameters:
    - data (pandas.DataFrame): The input data.

    Returns:
    - numpy.ndarray: The features extracted from the data.
    """
    data = pd.DataFrame(data)
    return data
#%%

def health_score(row):
    # Daily Values (DV) per school group
    DV = {
        "elementary": {
            "Calories": 1600, "Protein": 19, "Total Carbohydrate": 130,
            "Dietary Fiber": 25, "Added Sugars": 25, "Total Fat": 40,
            "Saturated Fat": 20, "Sodium": 1500, "Vitamin D": 20,
            "Calcium": 1000, "Iron": 10, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "middle": {
            "Calories": 2200, "Protein": 34, "Total Carbohydrate": 130,
            "Dietary Fiber": 31, "Added Sugars": 50, "Total Fat": 77,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "high": {
            "Calories": 2600, "Protein": 46, "Total Carbohydrate": 130,
            "Dietary Fiber": 38, "Added Sugars": 50, "Total Fat": 91,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        }
    }

    # Nutrients classification
    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium",
            "Iron", "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    # Determine school group (default high school)
    school_group = str(row.get("school_group", "high")).lower()
    if "elementary" in school_group:
        dv = DV["elementary"]
    elif "middle" in school_group:
        dv = DV["middle"]
    elif "high" in school_group:
        dv = DV["high"]
    else:
        dv = DV["high"]

    good_score = 0
    bad_score = 0

    # Calculate %DV for good nutrients
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val / ref) * 100)

    # Calculate %DV for bad nutrients
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val / ref) * 100)

    return good_score - bad_score




# %%
import os

# Get the directory of the current file (env.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  

# Construct path to data/sales.csv
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

data = load_data(DATA_PATH)

print(data.head())

features = get_features(data)
print("Features shape:", features.shape)

actions = get_actions(data)
print("Unique actions:", actions)  # show first 10 actions

# %%
num_actions = len(actions)
print("Number of unique actions:", num_actions)

#%%
output_file = "scored_data.csv"
df=data.copy()
# Compute health score
df["HealthScore"] = df.apply(health_score, axis=1)
# Save results
df.to_csv(output_file, index=False)
print(f"✅ Health scores calculated and saved to {output_file}")




























# %%
"""def health_score(row, min_score=None, max_score=None):
    # Daily Values (DV) per school group
    DV = {
        "elementary": {
            "Calories": 1600, "Protein": 19, "Total Carbohydrate": 130,
            "Dietary Fiber": 25, "Added Sugars": 25, "Total Fat": 40,
            "Saturated Fat": 20, "Sodium": 1500, "Vitamin D": 20,
            "Calcium": 1000, "Iron": 10, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "middle": {
            "Calories": 2200, "Protein": 34, "Total Carbohydrate": 130,
            "Dietary Fiber": 31, "Added Sugars": 50, "Total Fat": 77,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        },
        "high": {
            "Calories": 2600, "Protein": 46, "Total Carbohydrate": 130,
            "Dietary Fiber": 38, "Added Sugars": 50, "Total Fat": 91,
            "Saturated Fat": 20, "Sodium": 2300, "Vitamin D": 20,
            "Calcium": 1300, "Iron": 18, "Potassium": 4700,
            "Vitamin A": 900, "Vitamin C": 90
        }
    }

    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium",
            "Iron", "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    school_group = str(row.get("school_group", "high")).lower()
    if "elementary" in school_group:
        dv = DV["elementary"]
    elif "middle" in school_group:
        dv = DV["middle"]
    else:
        dv = DV["high"]

    good_score = 0
    bad_score = 0

    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val / ref) * 100)

    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val / ref) * 100)

    raw_score = good_score - bad_score

    # If min/max scores are not provided, use default range
    if min_score is None:
        min_score = -300  # worst case negative score
    if max_score is None:
        max_score = 800   # approximate best-case total for good nutrients

    # Scale to 0-10 range
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(10, scaled_score))  # clamp to [0,10]

    return round(scaled_score, 1)
"""