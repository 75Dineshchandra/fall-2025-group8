import numpy as np
import pandas as pd
import json
import pickle

class LinUCB:
    """
    Contextual Multi-Armed Bandit using Linear Upper Confidence Bound algorithm.
    
    Parameters:
    - n_arms (int): Number of arms (food items) in the bandit
    - context_dim (int): Dimensionality of the context feature vector
    - alpha (float): Exploration parameter controlling confidence bounds
    - lambda_reg (float): Regularization parameter for ridge regression
    - random_seed (int): Random seed for reproducibility
    
    Attributes:
    - A: List of ridge regression matrices (one per arm)
    - b: List of reward-context correlation vectors (one per arm)
    - theta: List of learned weight vectors (one per arm)
    - arm_counts: Number of times each arm was pulled
    - rewards_list: History of received rewards
    """
    
    def __init__(self, n_arms, context_dim, alpha=1.0, lambda_reg=1.0, random_seed=None):
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Basic bandit parameters
        self.n_arms = n_arms  # Number of food items
        self.context_dim = context_dim  # Dimension of context features
        self.alpha = alpha  # Exploration parameter
        self.lambda_reg = lambda_reg  # Regularization parameter
        
        # LinUCB specific components - ONE PER ARM
        # A: (context_dim x context_dim) matrix for each arm - ridge regression
        self.A = [np.eye(context_dim) * lambda_reg for _ in range(n_arms)]
        
        # b: (context_dim x 1) vector for each arm - reward-context correlation
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        
        # theta: (context_dim x 1) vector for each arm - learned weights
        self.theta = [np.zeros((context_dim, 1)) for _ in range(n_arms)]
        
        # Tracking and statistics
        self.arm_counts = np.zeros(n_arms)  # How many times each arm was chosen
        self.rewards_list = []  # History of all rewards received
        self.arm_rewards = [[] for _ in range(n_arms)]  # Rewards per arm
        
        # For debugging and analysis
        self.context_history = []  # Store contexts seen
        self.action_history = []  # Store actions taken
        self.reward_history = []  # Store rewards received
        
        print(f" LinUCB initialized with {n_arms} arms, {context_dim} context features")
        print(f"   Exploration (alpha): {alpha}, Regularization (lambda): {lambda_reg}")

    def action(self, context, available_actions=None):
        """
        Select best food using Upper Confidence Bound strategy
        
        Parameters:
        - context: Feature vector (school, day, nutrients, etc.) from your feature_matrix
        - available_actions: Binary mask (0/1) of available foods from your action_matrix
        
        Returns:
        - best_arm: Index of the recommended food item
        """
        # Ensure context is a column vector
        x = context.reshape(-1, 1) if len(context.shape) == 1 else context
        x = x.astype(np.float64)  # Ensure numerical stability
        
        scores = []
        
        for arm in range(self.n_arms):
            # Skip unavailable foods (action masking)
            if available_actions is not None and available_actions[arm] == 0:
                scores.append(-np.inf)
                continue
                
            try:
                # Calculate prediction: θᵀx (expected reward)
                prediction = self.theta[arm].T @ x
                prediction = float(prediction)  # Convert to scalar
                
                # Calculate confidence bound: α√(xᵀA⁻¹x)
                A_inv = np.linalg.inv(self.A[arm])
                confidence = self.alpha * np.sqrt(x.T @ A_inv @ x)
                confidence = float(confidence)
                
                # Total score = prediction + confidence bound
                total_score = prediction + confidence
                scores.append(total_score)
                
            except np.linalg.LinAlgError:
                # Handle singular matrix case (use prediction only)
                prediction = float(self.theta[arm].T @ x)
                scores.append(prediction)
        
        # Choose arm with highest score
        best_arm = np.argmax(scores)
        return best_arm

    def update(self, context, action, reward):
        """
        Update linear model for the chosen arm using ridge regression
        
        Parameters:
        - context: Feature vector that led to the decision
        - action: The food item that was chosen
        - reward: The observed reward (popularity + health)
        """
        x = context.reshape(-1, 1)  # Ensure column vector
        
        # Update matrices for the chosen arm
        self.A[action] += x @ x.T
        self.b[action] += reward * x
        
        # Recompute theta using ridge regression: θ = A⁻¹b
        try:
            self.theta[action] = np.linalg.inv(self.A[action]) @ self.b[action]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.theta[action] = np.linalg.pinv(self.A[action]) @ self.b[action]
        
        # Update tracking statistics
        self.arm_counts[action] += 1
        self.rewards_list.append(reward)
        self.arm_rewards[action].append(reward)
        
        # Store history for debugging
        self.context_history.append(context.copy())
        self.action_history.append(action)
        self.reward_history.append(reward)

    def calculate_reward(self, chosen_food, sales_count, health_score, lambda_param=0.5):
        """
        Compute reward for the chosen action: reward = popularity + λ * healthiness
        
        Parameters:
        - chosen_food: Index of selected food item
        - sales_count: Actual number of times this food was chosen
        - health_score: Pre-computed health score for this food
        - lambda_param: Trade-off between popularity and health
        
        Returns:
        - reward: Combined reward score
        """
        # Normalize sales count (assuming max 100 sales per time slot)
        popularity_norm = min(sales_count / 100.0, 1.0) * 100
        
        # Your health_score should already be in 0-100 range
        health_norm = health_score
        
        # Combined reward
        reward = popularity_norm + lambda_param * health_norm
        return reward

    def train(self, contexts, available_actions_list, sales_data, health_scores, lambda_param=0.5):
        """
        Train the model on observed rewards
        
        Parameters:
        - contexts: List of context vectors for each time slot
        - available_actions_list: List of available actions masks for each time slot
        - sales_data: Sales counts for each food at each time slot
        - health_scores: Pre-computed health scores for each food
        - lambda_param: Balance between popularity and health in reward
        """
        print(f" Starting training with {len(contexts)} time slots...")
        
        for t in range(len(contexts)):
            context = contexts[t]
            available_actions = available_actions_list[t]
            
            # Choose action
            chosen_food = self.action(context, available_actions)
            
            # Get sales count for chosen food
            sales_count = sales_data[t][chosen_food] if hasattr(sales_data[t], '__getitem__') else sales_data[t]
            
            # Calculate reward
            health_score_val = health_scores[chosen_food] if hasattr(health_scores, '__getitem__') else health_scores
            reward = self.calculate_reward(chosen_food, sales_count, health_score_val, lambda_param)
            
            # Learn from experience
            self.update(context, chosen_food, reward)
            
            if (t + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_list[-100:]) if len(self.rewards_list) >= 100 else np.mean(self.rewards_list)
                print(f"   Time slot {t+1}: Avg reward = {avg_reward:.2f}")
        
        print(" Training completed!")
        print(f"   Total rewards: {len(self.rewards_list)}")
        print(f"   Average reward: {np.mean(self.rewards_list):.2f}")

    def recommend(self, context, available_actions, top_k=3):
        """
        Provide meal recommendations based on learned policy
        
        Parameters:
        - context: Current context features
        - available_actions: Binary mask of available foods
        - top_k: Number of recommendations to return
        
        Returns:
        - recommendations: List of (food_index, predicted_score) tuples
        """
        scores = []
        x = context.reshape(-1, 1) if len(context.shape) == 1 else context
        
        for arm in range(self.n_arms):
            if available_actions is None or available_actions[arm] == 1:
                # Use prediction only (no exploration for final recommendations)
                score = float(self.theta[arm].T @ x)
                scores.append((arm, score))
        
        # Return top K recommendations
        recommendations = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return recommendations

    def reset(self):
        """Reset the model to initial state"""
        self.A = [np.eye(self.context_dim) * self.lambda_reg for _ in range(self.n_arms)]
        self.b = [np.zeros((self.context_dim, 1)) for _ in range(self.n_arms)]
        self.theta = [np.zeros((self.context_dim, 1)) for _ in range(self.n_arms)]
        self.arm_counts = np.zeros(self.n_arms)
        self.rewards_list = []
        self.arm_rewards = [[] for _ in range(self.n_arms)]
        self.context_history = []
        self.action_history = []
        self.reward_history = []
        print(" Model reset to initial state")

    def save(self, filepath):
        """Save model parameters to file"""
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
            'arm_rewards': self.arm_rewards
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f" Model saved to {filepath}")

    def load(self, filepath):
        """Load model parameters from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Verify compatibility
        if model_data['n_arms'] != self.n_arms or model_data['context_dim'] != self.context_dim:
            raise ValueError("Loaded model dimensions don't match current model")
        
        # Restore parameters
        self.alpha = model_data['alpha']
        self.lambda_reg = model_data['lambda_reg']
        self.A = [np.array(A) for A in model_data['A']]
        self.b = [np.array(b) for b in model_data['b']]
        self.theta = [np.array(theta) for theta in model_data['theta']]
        self.arm_counts = np.array(model_data['arm_counts'])
        self.rewards_list = model_data['rewards_list']
        self.arm_rewards = model_data['arm_rewards']
        
        print(f" Model loaded from {filepath}")


# Test the complete implementation
if __name__ == "__main__":
    print("Testing LinUCB implementation...")
    
    # Create test instance
    n_arms = 5
    context_dim = 4
    model = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=1.0)
    
    # Test data
    test_context = np.array([1.0, 0.5, -0.2, 0.8])
    test_available = np.array([1, 0, 1, 1, 0])  # Only arms 0, 2, 3 available
    
    print("\n1. Testing action() method...")
    chosen_food = model.action(test_context, test_available)
    print(f"   Context: {test_context}")
    print(f"   Available foods: {test_available}")
    print(f"   Recommended food index: {chosen_food}")
    print(f"   ✓ Expected: 0, 2, or 3 (only available foods)")
    
    print("\n2. Testing update() method...")
    test_reward = 85.5
    model.update(test_context, chosen_food, test_reward)
    print(f"   Updated arm {chosen_food} with reward {test_reward}")
    print(f"   Arm counts: {model.arm_counts}")
    print(f"   Total rewards: {len(model.rewards_list)}")
    
    print("\n3. Testing calculate_reward() method...")
    sales_count = 75
    health_score = 80
    reward = model.calculate_reward(chosen_food, sales_count, health_score, lambda_param=0.3)
    print(f"   Sales: {sales_count}, Health: {health_score}")
    print(f"   Calculated reward: {reward:.2f}")
    
    print("\n4. Testing recommend() method...")
    recommendations = model.recommend(test_context, test_available, top_k=2)
    print(f"   Top 2 recommendations: {recommendations}")
    
    print("\n5. Testing save/load functionality...")
    model.save("test_model.json")
    model.load("test_model.json")
    
    print("\n6. Testing reset() method...")
    model.reset()
    print(f"   Arm counts after reset: {model.arm_counts}")
    
    print("\n All tests completed successfully!")
    print("Your LinUCB model is ready for integration with your school meal data!")