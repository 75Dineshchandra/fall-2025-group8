
#%%
# run_training.py (place this in your project root)
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from components.main import train_fcps_model

if __name__ == "__main__":
    train_fcps_model()
    #%%