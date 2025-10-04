# model.py - OPTIMIZED FOR FCPS DATA STRUCTURE
import numpy as np
import pandas as pd
import json

class LinUCB:
    def __init__(self, n_arms=160, context_dim=19, alpha=1.0, lambda_reg=1.0):
        # FCPS-SPECIFIC: 160 food items, 19 nutritional features
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Per-food linear models
        self.A = [np.eye(context_dim) * lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        self.theta = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        
        # Tracking
        self.arm_counts = np.zeros(n_arms)
        self.rewards_list = []
        
    def action(self, context, available_actions):
        """Choose from available foods using UCB"""
        scores = []
        
        for food_idx in range(160):  # All 160 possible foods
            if available_actions[food_idx] == 0:
                scores.append(-np.inf)  # Skip unavailable
                continue
                
            # UCB: prediction + confidence
            prediction = float(self.theta[food_idx].T @ context)
            try:
                confidence = self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[food_idx]) @ context)
                scores.append(prediction + float(confidence))
            except:
                scores.append(prediction)  # Fallback
                
        return np.argmax(scores)  # Returns 0-159

    def update(self, context, chosen_food, reward):
        """Learn from this food choice"""
        x = context.reshape(-1, 1)
        
        self.A[chosen_food] += x @ x.T
        self.b[chosen_food] += reward * x
        self.theta[chosen_food] = np.linalg.inv(self.A[chosen_food]) @ self.b[chosen_food]
        
        self.arm_counts[chosen_food] += 1
        self.rewards_list.append(reward)

    def calculate_reward(self, chosen_food, sales_count, health_score, lambda_param=0.3):
        """FCPS-specific reward: sales + health balance"""
        # Sales are actual counts (1-100+ students)
        # Health scores are 3.1-4.8 scale from your data
        popularity = sales_count  # Already good scale (0-100+)
        health = health_score * 20  # Scale 3.1-4.8 → 62-96 for better balance
        
        return popularity + lambda_param * health

    def train(self, contexts, available_actions_list, sales_data, health_scores):
        """Train on FCPS time series data"""
        for time_slot in range(len(contexts)):
            context = contexts[time_slot]
            available_actions = available_actions_list[time_slot]
            
            # Choose food
            chosen_food = self.action(context, available_actions)
            
            # Get actual results from FCPS data
            sales_count = sales_data[time_slot][chosen_food]
            health_score_val = health_scores[chosen_food]
            
            # Calculate reward
            reward = self.calculate_reward(chosen_food, sales_count, health_score_val)
            
            # Learn
            self.update(context, chosen_food, reward)

    def recommend(self, context, available_actions, top_k=3):
        """Recommend foods for cafeteria staff"""
        scores = []
        for food_idx in range(160):
            if available_actions[food_idx] == 1:
                score = float(self.theta[food_idx].T @ context)
                scores.append((food_idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]