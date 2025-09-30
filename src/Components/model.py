# model.py
import numpy as np
from typing import Dict, List, Tuple

class LinUCB:
    """
    Linear UCB contextual bandit implementation.
    """
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # For each arm: A = dxd identity, b = dx1 zero
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def select_arm(self, x: np.ndarray) -> int:
        """
        Select arm using LinUCB rule.
        Args:
            x: context vector (d,)
        Returns:
            chosen arm index
        """
        x = x.reshape(-1, 1)
        p = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta.T @ x)
            var = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
            p[a] = mean + var

        return int(np.argmax(p))

    def update(self, chosen_arm: int, reward: float, x: np.ndarray):
        """
        Update A and b for chosen arm.
        """
        x = x.reshape(-1, 1)
        self.A[chosen_arm] += x @ x.T
        self.b[chosen_arm] += reward * x

# ===================== Training =====================

def train_linucb(
    X_all: np.ndarray,
    groups: Dict[int, np.ndarray],
    rewards: np.ndarray,
    alpha: float = 1.0
) -> Tuple[LinUCB, List[int], List[float]]:
    """
    Train LinUCB on provided dataset.
    
    Args:
        X_all: feature matrix (n_samples, n_features)
        groups: mapping t -> indices of actions available at time t
        rewards: reward array (n_samples,)
        alpha: exploration parameter
    
    Returns:
        (trained agent, actions taken, rewards received)
    """
    n_features = X_all.shape[1]
    n_arms = len(np.unique([idx for arr in groups.values() for idx in arr]))

    agent = LinUCB(n_arms=n_arms, n_features=n_features, alpha=alpha)
    actions_taken, rewards_received = [], []

    for t, indices in sorted(groups.items()):
        # choose arm from available options
        arm_scores = {}
        for idx in indices:
            x = X_all[idx]
            arm_scores[idx] = agent.select_arm(x)

        # map local choice to dataset index
        chosen_idx = np.random.choice(indices)
        chosen_arm = agent.select_arm(X_all[chosen_idx])

        reward = rewards[chosen_idx]

        # update agent
        agent.update(chosen_arm, reward, X_all[chosen_idx])

        actions_taken.append(chosen_arm)
        rewards_received.append(reward)

    return agent, actions_taken, rewards_received

# ===================== Evaluation =====================

def evaluate(actions: List[int], rewards: List[float]) -> Dict[str, float]:
    """
    Simple evaluation metrics.
    """
    return {
        "total_reward": float(np.sum(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "n_actions": len(actions)
    }
