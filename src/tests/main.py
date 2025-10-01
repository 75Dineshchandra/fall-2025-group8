#%%
# --- add these two lines first ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # adds /.../src to sys.path

# --- make all three imports consistent ---
from Components.model import EpsilonGreedy
from Components.env import *
from Components.plot import *

import warnings

warnings.filterwarnings("ignore")

environment = "bernoulli"
n_arms = 2
arm_means = [[0.9, 0.1],[0.1, 0.9]]
obs = 500
random_seed = 123
epsilon = 0.3

data = create_environment(env = environment,
                          n_arms = n_arms,
                          arm_means = arm_means, 
                          observations = obs, 
                          random_seed = random_seed)

model = EpsilonGreedy(n_arms = n_arms, epsilon = epsilon, random_seed=random_seed)

table, rewards, matrix = model.train(data)

violinplot_environment(data, arm_means)

data_average_plot(data, arm_means)

data_cumulative_plot(data, arm_means)

model_average_plot(data, rewards, matrix, arm_means)

model_cumulative_plot(data, rewards, matrix, arm_means)

param = model.save()
# %%
#  Sudo Code for main.py 
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Add project root to Python path
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import your custom modules
try:
    from src.components.model import LinUCB
    from src.data_preprocessing import load_data, health_score, build_feature_matrix, build_action_matrix
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating mock functions for testing...")
    
    # Mock functions for testing
    def load_data(filepath):
        print(f"Mock: Loading data from {filepath}")
        return pd.DataFrame({
            'description': ['pizza', 'salad', 'burger'] * 10,
            'sales_item': ['pizza', 'salad', 'burger'] * 10,
            'school_group': ['elementary'] * 30,
            'time_slot_id': list(range(10)) * 3
        })
    
    def health_score(row):
        return np.random.uniform(60, 95)
    
    def build_feature_matrix(data, **kwargs):
        n_samples = len(data)
        n_features = 5
        X = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = [f'feature_{i}' for i in range(n_features)]
        rows_df = data[['time_slot_id', 'description']].copy()
        rows_df['item_idx'] = np.random.randint(0, 3, n_samples)
        groups = {i: np.array([i]) for i in range(10)}
        meta = {'items': ['pizza', 'salad', 'burger']}
        return X, feature_names, rows_df, groups, meta
    
    def build_action_matrix(data, **kwargs):
        n_time_slots = 10
        n_items = 3
        action_matrix = np.random.randint(0, 2, (n_time_slots, n_items))
        all_items = ['pizza', 'salad', 'burger']
        item_to_idx = {item: i for i, item in enumerate(all_items)}
        return action_matrix, all_items, item_to_idx

def create_synthetic_data():
    """Create synthetic data for testing"""
    n_time_slots = 50
    n_items = 10
    context_dim = 8
    
    # Synthetic feature matrix
    X_all = np.random.normal(0, 1, (n_time_slots, context_dim))
    
    # Synthetic action matrix
    action_matrix = np.random.binomial(1, 0.7, (n_time_slots, n_items))
    
    # Synthetic item names
    all_items = [f"meal_{i}" for i in range(n_items)]
    
    return X_all, action_matrix, all_items

def plot_training_progress(model):
    """Plot training performance metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot cumulative average reward
    plt.subplot(1, 2, 1)
    if model.rewards_list:
        cumulative_avg = [np.mean(model.rewards_list[:i+1]) for i in range(len(model.rewards_list))]
        plt.plot(cumulative_avg)
        plt.title('Cumulative Average Reward')
        plt.xlabel('Time Step')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    # Plot arm selection distribution
    plt.subplot(1, 2, 2)
    top_arms = np.argsort(-model.arm_counts)[:min(8, len(model.arm_counts))]
    plt.bar(range(len(top_arms)), model.arm_counts[top_arms])
    plt.title('Top Most Chosen Foods')
    plt.xlabel('Food Index')
    plt.ylabel('Selection Count')
    plt.xticks(range(len(top_arms)), top_arms)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print(" FCPS School Meal Recommendation System")
    print("=" * 50)
    
    # 1. LOAD AND PREPROCESS DATA
    print(" Step 1: Loading and preprocessing data...")
    
    try:
        # Load data
        data = load_data("data/sales.csv")
        print(f"   Loaded {len(data)} records")
        
        # Calculate health scores
        print("   Calculating health scores...")
        data['health_score'] = data.apply(health_score, axis=1)
        
        # Build feature matrix
        print("   Building feature matrix...")
        X_all, feature_names, rows_df, groups, meta = build_feature_matrix(
            data, 
            item_col="description",
            add_bias=True
        )
        
        # Build action matrix
        print("   Building action matrix...")
        action_matrix, all_items, item_to_idx = build_action_matrix(
            data, 
            item_col="description"
        )
        
        print(f"   Features: {X_all.shape[1]} dimensions")
        print(f"   Food items: {len(all_items)} unique meals")
        print(f"   Time slots: {action_matrix.shape[0]} periods")
        
    except Exception as e:
        print(f"   Using synthetic data: {e}")
        X_all, action_matrix, all_items = create_synthetic_data()
        feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]
        rows_df = pd.DataFrame({
            't': range(X_all.shape[0]),
            'description': [all_items[i % len(all_items)] for i in range(X_all.shape[0])],
            'item_idx': [i % len(all_items) for i in range(X_all.shape[0])]
        })

    # 2. INITIALIZE LINUCB MODEL
    print("\n Step 2: Initializing LinUCB model...")
    n_arms = len(all_items)
    context_dim = X_all.shape[1]
    
    model = LinUCB(
        n_arms=n_arms,
        context_dim=context_dim,
        alpha=1.0,
        lambda_reg=1.0,
        random_seed=42
    )
    
    print(f"   Model: {n_arms} arms, {context_dim} context features")

    # 3. PREPARE TRAINING DATA
    print("\n Step 3: Preparing training sequences...")
    
    contexts_list = []
    available_actions_list = []
    
    # Use first 30 time slots for training
    for t in range(min(30, action_matrix.shape[0])):
        if 'rows_df' in locals():
            slot_data = rows_df[rows_df['t'] == t]
            if len(slot_data) > 0:
                context = X_all[slot_data.index[0]]
            else:
                context = X_all[t % X_all.shape[0]]
        else:
            context = X_all[t]
        
        available_actions = action_matrix[t]
        contexts_list.append(context)
        available_actions_list.append(available_actions)
    
    print(f"   Prepared {len(contexts_list)} training sequences")

    # 4. SETUP REWARDS
    print("\n Step 4: Setting up reward simulation...")
    
    health_scores = {}
    for i in range(n_arms):
        health_scores[i] = np.random.uniform(60, 95)
    
    # Simulate sales data
    sales_data = []
    for t in range(len(contexts_list)):
        slot_sales = np.random.poisson(lam=50, size=n_arms)
        sales_data.append(slot_sales)

    # 5. TRAIN THE MODEL
    print("\n Step 5: Training LinUCB model...")
    
    model.train(
        contexts=contexts_list,
        available_actions_list=available_actions_list,
        sales_data=sales_data,
        health_scores=health_scores,
        lambda_param=0.3
    )
    
    # 6. EVALUATE PERFORMANCE
    print("\n Step 6: Model Evaluation")
    print(f"   Training completed!")
    print(f"   Total rewards: {len(model.rewards_list)}")
    if model.rewards_list:
        print(f"   Average reward: {np.mean(model.rewards_list):.2f}")
        print(f"   Best reward: {np.max(model.rewards_list):.2f}")

    # 7. GENERATE RECOMMENDATIONS
    print("\n Step 7: Sample Recommendations")
    
    if contexts_list:
        sample_context = contexts_list[0]
        sample_available = available_actions_list[0]
        
        recommendations = model.recommend(
            context=sample_context,
            available_actions=sample_available,
            top_k=3
        )
        
        print("   Top 3 recommendations:")
        for i, (food_idx, score) in enumerate(recommendations):
            health = health_scores.get(food_idx, 0)
            food_name = all_items[food_idx] if food_idx < len(all_items) else f"Food_{food_idx}"
            print(f"     {i+1}. {food_name}: score={score:.2f}, health={health:.1f}")

    # 8. SAVE RESULTS
    print("\n Step 8: Saving results...")
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    try:
        model.save("models/trained_linucb_model.json")
        print("   Model saved successfully")
    except Exception as e:
        print(f"   Model save failed: {e}")
    
    # Generate plot
    try:
        plot_training_progress(model)
        print("   Training plot generated")
    except Exception as e:
        print(f"   Plot generation failed: {e}")

    print("\n PROCESS COMPLETED!")

if __name__ == "__main__":
    main()