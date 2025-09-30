# main.py
import os
import numpy as np
import pandas as pd

from Components.env import load_data, health_score, build_action_matrix, build_feature_matrix
from Components.model import train_linucb, evaluate
import Components.plot as plot

if __name__ == "__main__":
    # ---------- Load data ----------
    DATA_PATH = os.path.join("data", "sales.csv")
    df = load_data(DATA_PATH)
    print("✅ Data loaded:", df.shape)

    # ---------- Add health score ----------
    df["HealthScore"] = df.apply(health_score, axis=1)

    # ---------- Build matrices ----------
    action_matrix, all_items, item_to_idx = build_action_matrix(df, item_col="description")
    X_all, feature_names, rows_df, groups, meta = build_feature_matrix(df, item_col="description")

    # Reward definition: here we use health score
    rewards = df["HealthScore"].to_numpy()

    # ---------- Train model ----------
    agent, actions_taken, rewards_received = train_linucb(
        X_all, groups, rewards, alpha=1.0
    )

    # ---------- Evaluate ----------
    metrics = evaluate(actions_taken, rewards_received)
    print("📊 Evaluation:", metrics)

    # ---------- Plots ----------
    plot.model_average_plot(
        data=None, 
        rewards=np.array(rewards_received), 
        action_matrix=action_matrix, 
        arm_means=[df.groupby("description")["HealthScore"].mean().to_numpy()],
        top=5
    )
    plot.model_cumulative_plot(
        data=None,
        rewards=np.array(rewards_received),
        action_matrix=action_matrix,
        arm_means=[df.groupby("description")["HealthScore"].mean().to_numpy()],
        top=5
    )