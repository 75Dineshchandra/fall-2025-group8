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
    actions = data['description'].unique()   # ✅ directly get unique descriptions
    actions_list = actions.tolist()  # convert numpy array → python list
    print("Unique actions as list:")
    print(actions_list)


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
    features = data.drop(columns=['description']).values
    return features

def health_score(row):
    """
    Calculate a health score based on nutritional values.

    Parameters:
    - row (pandas.Series): A row of the DataFrame containing nutritional values.

    Returns:
    - float: The calculated health score.
    """
    # Example scoring logic (customize as needed)
    score = 0
    if row['calories'] < 500:
        score += 1
    if row['sugar'] < 10:
        score += 1
    if row['fiber'] > 5:
        score += 1
    return score


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




























