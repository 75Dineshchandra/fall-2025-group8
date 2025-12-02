#  main_predict.py
# Predict for dates NOT in the historical data USING TRAINED MODEL

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import joblib

# Ensure src/ is on sys.path so Components can be imported
sys.path.append(str(Path(__file__).parent))

import Components.env as env

# Load environment data into env module (defaults look in data/)
env.load_env_data()

# Main execution
if __name__ == "__main__":
    # Use the OPTIMAL model (lambda = 0.30 from training)
    optimal_model_path = "data/results/model_lambda_0.30.joblib"
    
    recommendations = env.recommend_with_trained_model(
        target_date='2025-12-03',
        school="HERNDON_HIGH",
        meal_time="lunch",
        model_path=optimal_model_path,
        top_k=5
    )
    
    if recommendations is None:
        print("Could not generate recommendations")
    else:
        env.print_enhanced_recommendations(recommendations)