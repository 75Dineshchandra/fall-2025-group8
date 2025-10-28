# ===== model.py =====
# Purpose: LinUCB algorithm for contextual bandits
# This is the machine learning model that learns which items to recommend

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) algorithm.
    Used for contextual multi-armed bandits.
    
    For each item (arm), we maintain:
    - A: A matrix for solving linear equations (d x d)
    - b: A vector for reward tracking (d x 1)
    """
    
    def __init__(self, d, n_arms, alpha=1.0, l2=1.0, seed=42):
        """
        Initialize the LinUCB model.
        
        # d = number of features (18 nutritional dimensions)
        # n_arms = number of meal items (160 items)
        # alpha = exploration parameter (how much to try new things)
        # l2 = regularization (prevects overfitting)
        seed: random seed for reproducibility
        """
        self.d = int(d)
        self.n_arms = int(n_arms)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.random_state = np.random.default_rng(seed)
        
        self.reset()
    
    def reset(self):
        """Initialize or reset all arm parameters"""
        self.A_matrices = [self.l2 * np.eye(self.d, dtype=np.float64) for _ in range(self.n_arms)]
        self.b_vectors = [np.zeros((self.d, 1), dtype=np.float64) for _ in range(self.n_arms)]
        self.total_reward = 0.0
        self.oracle_reward = 0.0
        self.steps_trained = 0
    
    def compute_theta(self, arm_id):
        """
        # Solves: theta = A⁻¹ * b
        # Returns: Weight vector showing feature importance
        # Theta represents the model's "understanding" of each meal's appeal
        """
        theta = np.linalg.solve(self.A_matrices[arm_id], self.b_vectors[arm_id])
        return theta.reshape(-1)
    
    def compute_upper_confidence_bound(self, arm_id, features):
        """
        # estimated_value = theta · features (what we've learned)
        # confidence_width = alpha * √(features · A⁻¹ · features) (uncertainty bonus)
        # return estimated_value + confidence_width
        # why - Balances exploitation (choose what works) vs exploration (try new things)
        """
        theta = self.compute_theta(arm_id)
        estimated_value = float(theta @ features)
        
        # Compute confidence interval width
        A_inv_features = np.linalg.solve(self.A_matrices[arm_id], features)
        confidence_width = float(self.alpha * np.sqrt(max(0.0, features @ A_inv_features)))
        
        ucb_score = estimated_value + confidence_width
        return ucb_score
    
    def select_action(self, available_arms, features_by_arm):
        """
        Select which arm (meal) to recommend.
        Choose arm with highest UCB score.
        
        available_arms: list of arm IDs available at this time slot
        features_by_arm: dict mapping arm_id to feature vector
        """
        best_arm = None
        best_score = -1e18
        
        for arm_id in available_arms:
            features = features_by_arm[arm_id].reshape(-1)
            score = self.compute_upper_confidence_bound(arm_id, features)
            
            if score > best_score:
                best_score = score
                best_arm = arm_id
        
        # If no arm found, pick randomly
        if best_arm is None:
            best_arm = self.random_state.choice(available_arms)
        
        return int(best_arm)
    
    def update_arm(self, arm_id, features, reward):
        """
        Update arm parameters after observing a reward.
        
        A = A + features * features^T
        b = b + reward * features
        """
        features = features.reshape(-1, 1)
        self.A_matrices[arm_id] += features @ features.T
        self.b_vectors[arm_id] += reward * features
    
    def train(self, X, rows_df, rewards, avail_mat, verbose=False):
        """
        Train the model on historical data.
        
        X: feature matrix (samples x features)
        rows_df: metadata with time_slot_id and item_idx
        rewards: reward values (one per sample)
        avail_mat: availability matrix (time_slots x items)
        """
        
        # Validate inputs
        if X.shape[0] != len(rows_df):
            raise ValueError(f"X has {X.shape[0]} rows but rows_df has {len(rows_df)} rows")
        if len(rewards) != len(rows_df):
            raise ValueError(f"rewards has {len(rewards)} rows but rows_df has {len(rows_df)} rows")
        if avail_mat.shape[1] != self.n_arms:
            raise ValueError(f"avail_mat has {avail_mat.shape[1]} arms but model has {self.n_arms}")
        
        # Group samples by time slot
        groups = {}
        for time_slot_id, group_indices in rows_df.groupby('time_slot_id').groups.items():
            groups[int(time_slot_id)] = np.array(list(group_indices), dtype=int)
        
        total_reward = 0.0
        oracle_reward = 0.0
        steps = 0
        
        # Process each time slot in order
        for time_slot_id in sorted(groups.keys()):
            sample_indices = groups[time_slot_id]
            
            # Check if time slot is in availability matrix
            if time_slot_id >= avail_mat.shape[0]:
                continue
            
            # Get available items at this time slot
            available_items = np.where(avail_mat[time_slot_id] == 1)[0].tolist()
            if not available_items:
                continue
            
            # Get features and rewards for available items at this time slot
            features_for_items = {}
            rewards_for_items = {}
            
            for sample_idx in sample_indices:
                arm_id = int(rows_df.iloc[sample_idx]['item_idx'])
                
                if arm_id in available_items:
                    features_for_items[arm_id] = X[sample_idx]
                    rewards_for_items[arm_id] = float(rewards[sample_idx])
            
            # If no valid items, skip
            if not features_for_items:
                continue
            
            # Select best item using UCB
            selected_arm = self.select_action(list(features_for_items.keys()), features_for_items)
            selected_reward = rewards_for_items[selected_arm]
            
            # Update model with selected arm
            self.update_arm(selected_arm, features_for_items[selected_arm], selected_reward)
            
            # Track metrics
            total_reward += selected_reward
            oracle_reward += max(rewards_for_items.values())
            steps += 1
            
            if verbose and steps % 1000 == 0:
                print(f"[t={time_slot_id}] steps={steps} total={total_reward:.1f} oracle={oracle_reward:.1f}")
        
        self.total_reward = total_reward
        self.oracle_reward = oracle_reward
        self.steps_trained = steps
        
        regret = oracle_reward - total_reward
        avg_reward = total_reward / max(1, steps)
        
        return {
            "steps": steps,
            "total_reward": float(total_reward),
            "oracle_reward": float(oracle_reward),
            "regret": float(regret),
            "avg_reward": float(avg_reward)
        }
    
    def get_recommendations(self, features_by_arm, top_k=5):
        """
        Get top K recommendations for a time slot.
        
        features_by_arm: dict mapping arm_id to feature vector
        top_k: number of recommendations to return
        
        Returns list of (arm_id, ucb_score) tuples
        """
        scores = []
        
        for arm_id, features in features_by_arm.items():
            score = self.compute_upper_confidence_bound(arm_id, features.reshape(-1))
            scores.append((arm_id, score))

            # treat this as a time step -> update it.
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    # -------------------------
    # Persistence (canonical)
    # -------------------------
    def save_model(self, file_path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        model_data = {
            "A_matrices": self.A_matrices,
            "b_vectors": self.b_vectors,
            "d": self.d,
            "n_arms": self.n_arms,
            "alpha": self.alpha,
            "l2": self.l2
        }
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """Load model from disk"""
        model_data = joblib.load(file_path)
        
        model = cls(
            d=model_data["d"],
            n_arms=model_data["n_arms"],
            alpha=model_data["alpha"],
            l2=model_data["l2"]
        )
        
        model.A_matrices = model_data["A_matrices"]
        model.b_vectors = model_data["b_vectors"]
        
        print(f"Model loaded from {file_path}")
        return model

    # -------------------------
    # Back-compat aliases
    # -------------------------
    def save(self, file_path: str):
        """Alias for compatibility with scripts calling model.save(.)"""
        return self.save_model(file_path)

    @classmethod
    def load(cls, file_path: str):
        """Alias for compatibility with scripts calling LinUCB.load(.)"""
        return cls.load_model(file_path)