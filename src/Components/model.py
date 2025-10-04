#%%
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional

class LinUCB:
    def __init__(self, n_arms: int = 160, context_dim: int = 19, alpha: float = 1.0, lambda_reg: float = 1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize model parameters for each food item
        self.A = [np.eye(context_dim) * lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        self.theta = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        
        # Tracking
        self.arm_counts = np.zeros(n_arms)
        self.rewards_list = []
        self.regret_list = []
        
    def action(self, context: np.ndarray, available_actions: np.ndarray) -> int:
        """Choose from available foods using UCB"""
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        scores = []
        valid_arms = []
        
        for food_idx in range(self.n_arms):
            if available_actions[food_idx] == 1:  # Only consider available foods
                valid_arms.append(food_idx)
                try:
                    # UCB: prediction + confidence
                    prediction = float(self.theta[food_idx].T @ context)
                    A_inv = np.linalg.inv(self.A[food_idx])
                    confidence = self.alpha * np.sqrt(context.T @ A_inv @ context)
                    scores.append(prediction + float(confidence))
                except np.linalg.LinAlgError:
                    scores.append(prediction)  # Fallback if matrix inversion fails
        
        if not valid_arms:
            raise ValueError("No available actions to choose from")
            
        return valid_arms[np.argmax(scores)]

    def update(self, context: np.ndarray, chosen_food: int, reward: float):
        """Learn from this food choice"""
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        x = context
        
        # Update matrices using ridge regression
        self.A[chosen_food] += x @ x.T
        self.b[chosen_food] += reward * x
        self.theta[chosen_food] = np.linalg.inv(self.A[chosen_food]) @ self.b[chosen_food]
        
        self.arm_counts[chosen_food] += 1
        self.rewards_list.append(reward)

    def calculate_reward(self, chosen_food: int, sales_count: float, health_score: float, 
                        lambda_param: float = 0.3, max_sales: float = 100.0) -> float:
        """FCPS-specific reward: sales + health balance with proper normalization"""
        # Normalize sales count (0 to max_sales) -> [0, 1]
        popularity = min(sales_count / max_sales, 1.0)
        
        # Normalize health scores (3.1-4.8 from your data) -> [0, 1]
        health_min, health_max = 3.1, 4.8
        health_norm = (health_score - health_min) / (health_max - health_min)
        health_norm = max(0.0, min(1.0, health_norm))  # Clamp to [0,1]
        
        # Combined reward
        return popularity + lambda_param * health_norm

    def train(self, contexts: List[np.ndarray], available_actions_list: List[np.ndarray], 
              sales_data: List[np.ndarray], health_scores: Dict[int, float], 
              lambda_param: float = 0.3) -> List[float]:
        """Train on FCPS time series data"""
        rewards = []
        
        for time_slot in range(len(contexts)):
            context = contexts[time_slot]
            available_actions = available_actions_list[time_slot]
            
            # Choose food
            chosen_food = self.action(context, available_actions)
            
            # Get actual results from FCPS data
            sales_count = sales_data[time_slot][chosen_food]
            health_score_val = health_scores.get(chosen_food, 3.5)  # Default health score
            
            # Calculate reward
            reward = self.calculate_reward(chosen_food, sales_count, health_score_val, lambda_param)
            rewards.append(reward)
            
            # Learn from this experience
            self.update(context, chosen_food, reward)
            
            # Calculate regret (optional)
            best_possible_reward = self._calculate_best_possible_reward(
                context, available_actions, sales_data[time_slot], health_scores, lambda_param
            )
            regret = best_possible_reward - reward
            self.regret_list.append(regret)
        
        return rewards

    def _calculate_best_possible_reward(self, context: np.ndarray, available_actions: np.ndarray,
                                      sales_vector: np.ndarray, health_scores: Dict[int, float],
                                      lambda_param: float) -> float:
        """Calculate the best possible reward for regret calculation"""
        best_reward = 0
        for food_idx in range(self.n_arms):
            if available_actions[food_idx] == 1:
                sales_count = sales_vector[food_idx]
                health_score = health_scores.get(food_idx, 3.5)
                reward = self.calculate_reward(food_idx, sales_count, health_score, lambda_param)
                best_reward = max(best_reward, reward)
        return best_reward

    def recommend(self, context: np.ndarray, available_actions: np.ndarray, top_k: int = 3) -> List[tuple]:
        """Recommend foods for cafeteria staff"""
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        scores = []
        for food_idx in range(self.n_arms):
            if available_actions[food_idx] == 1:
                score = float(self.theta[food_idx].T @ context)
                scores.append((food_idx, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, filepath: str):
        """Save model parameters"""
        model_data = {
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'A': [A.tolist() for A in self.A],
            'b': [b.tolist() for b in self.b],
            'theta': [theta.tolist() for theta in self.theta],
            'arm_counts': self.arm_counts.tolist(),
            'rewards_list': self.rewards_list,
            'regret_list': self.regret_list
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to: {filepath}")

    def load(self, filepath: str):
        """Load model parameters"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            
        self.n_arms = model_data['n_arms']
        self.context_dim = model_data['context_dim']
        self.alpha = model_data['alpha']
        self.lambda_reg = model_data['lambda_reg']
        self.A = [np.array(A) for A in model_data['A']]
        self.b = [np.array(b) for b in model_data['b']]
        self.theta = [np.array(theta) for theta in model_data['theta']]
        self.arm_counts = np.array(model_data['arm_counts'])
        self.rewards_list = model_data['rewards_list']
        self.regret_list = model_data.get('regret_list', [])
        
        print(f"Model loaded from: {filepath}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'total_decisions': len(self.rewards_list),
            'average_reward': np.mean(self.rewards_list) if self.rewards_list else 0,
            'cumulative_reward': np.sum(self.rewards_list) if self.rewards_list else 0,
            'cumulative_regret': np.sum(self.regret_list) if self.regret_list else 0,
            'foods_recommended': np.sum(self.arm_counts > 0),
            'exploration_rate': len(self.rewards_list) / (self.n_arms * 10)  # Simple exploration metric
        }
# %%
