# ===== model.py =====
# Purpose: LinUCB algorithm for contextual bandits
# Updated to use Variant1 reward calculation (sales_scaled + λ * health_scaled)

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

__all__ = [
    "LinUCB",
    "load_feature_matrix",
    "load_action_matrix",
    "compute_rewards_for_lambda",
    "train_linucb_model",
]

# -------------------------------------------------------------------
# LinUCB model
# -------------------------------------------------------------------
class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) for contextual bandits.

    Args:
      d (int): number of features.
      n_arms (int): number of items (arms).
      alpha (float): exploration weight.
      l2 (float): ridge regularization for numerical stability.
      seed (int): RNG seed.
    """

    def __init__(self, d: int, n_arms: int, alpha: float = 1.0, l2: float = 1.0, seed: int = 42):
        self.d = int(d)
        self.n_arms = int(n_arms)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.random_state = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Initialize or reset all arm parameters and stats."""
        self.A_matrices = [self.l2 * np.eye(self.d, dtype=np.float64) for _ in range(self.n_arms)]
        self.b_vectors = [np.zeros((self.d, 1), dtype=np.float64) for _ in range(self.n_arms)]
        self.total_reward = 0.0
        self.oracle_reward = 0.0
        self.steps_trained = 0
        self.N = np.zeros(self.n_arms, dtype=int)       # selections per arm
        self.R_sum = np.zeros(self.n_arms, dtype=float) # cumulative reward per arm

    # ----- math -----
    def compute_theta(self, arm_id: int) -> np.ndarray:
        """Solve θ = A⁻¹b (stable)."""
        A = self.A_matrices[arm_id]
        b = self.b_vectors[arm_id]
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.solve(A + 1e-8 * np.eye(self.d), b)
        return theta.reshape(-1)

    def compute_upper_confidence_bound(self, arm_id: int, features: np.ndarray) -> float:
        """Compute UCB score = est + conf for given arm and features."""
        x = features.reshape(-1)
        theta = self.compute_theta(arm_id)
        est = float(theta @ x)
        A = self.A_matrices[arm_id]
        try:
            A_inv_x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A_inv_x = np.linalg.solve(A + 1e-8 * np.eye(self.d), x)
        conf = float(self.alpha * np.sqrt(max(0.0, x @ A_inv_x)))
        return est + conf

    # ----- policy -----
    def select_action(self, available_arms: List[int], features_by_arm: Dict[int, np.ndarray]) -> int:
        """Choose the available arm with highest UCB score; fallback random."""
        best_arm, best_score = None, -np.inf
        for arm_id in available_arms:
            score = self.compute_upper_confidence_bound(arm_id, features_by_arm[arm_id])
            if score > best_score:
                best_score, best_arm = score, arm_id
        if best_arm is None:
            best_arm = int(self.random_state.choice(available_arms))
        return int(best_arm)
        
    def update_arm(self, arm_id: int, features: np.ndarray, reward: float) -> None:
        """A ← A + xxᵀ ; b ← b + r x ; update per-arm counters."""
        x = features.reshape(-1, 1)
        self.A_matrices[arm_id] += x @ x.T
        self.b_vectors[arm_id] += float(reward) * x
        self.N[arm_id] += 1
        self.R_sum[arm_id] += float(reward)
        
    # ----- whiteboard-style training (T, A, d) -----
    def train(
        self,
        data: np.ndarray,         # shape: (T, A, d)
        rewards: np.ndarray,      # shape: (T, A)
        mask: np.ndarray,         # shape: (T, A), 1/True if arm available at t
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Train on tensor data:
          sample = data[t]            -> (A, d)
          available = where(mask[t])  -> indices
          choose arm via UCB on available, update with reward, track regret.
        """
        if data.ndim != 3:
            raise ValueError("data must be (T, A, d)")
        if rewards.shape != data.shape[:2]:
            raise ValueError("rewards must be (T, A)")
        if mask.shape != data.shape[:2]:
            raise ValueError("mask must be (T, A)")

        T, A, d = data.shape
        if A != self.n_arms or d != self.d:
            raise ValueError(f"shape mismatch: expected A={self.n_arms}, d={self.d}")

        total_reward = 0.0
        oracle_reward = 0.0
        steps = 0

        for t in range(T):
            sample = data[t]                      # (A, d)
            available = np.where(mask[t])[0]     # (k,)
            if len(available) == 0:
                continue

            feats = {a: sample[a] for a in available}
            rews  = {a: float(rewards[t, a]) for a in available}

            a = self.select_action(list(available), feats)
            r = rews[a]

            self.update_arm(a, feats[a], r)

            total_reward += r
            oracle_reward += max(rews.values())
            steps += 1

            if verbose and steps % 1000 == 0:
                print(f"[t={t}] steps={steps} total={total_reward:.1f} oracle={oracle_reward:.1f}")

        self.total_reward, self.oracle_reward, self.steps_trained = total_reward, oracle_reward, steps
        regret = oracle_reward - total_reward
        avg_reward = total_reward / max(1, steps)
        return {
            "steps": float(steps),
            "total_reward": float(total_reward),
            "oracle_reward": float(oracle_reward),
            "regret": float(regret),
            "avg_reward": float(avg_reward),
        }

    # ----- inference -----
    def get_recommendations(self, features_by_arm: Dict[int, np.ndarray], top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k (arm_id, score) pairs for a slot; no state updates."""
        scores: List[Tuple[int, float]] = []
        for arm_id, feats in features_by_arm.items():
            score = self.compute_upper_confidence_bound(arm_id, feats.reshape(-1))
            scores.append((arm_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ----- persistence -----
    def save_model(self, file_path: str) -> None:
        """Save parameters + simple stats to disk."""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        joblib.dump(
            {
                "A_matrices": self.A_matrices,
                "b_vectors": self.b_vectors,
                "d": self.d,
                "n_arms": self.n_arms,
                "alpha": self.alpha,
                "l2": self.l2,
                "N": self.N,
                "R_sum": self.R_sum,
                "total_reward": self.total_reward,
                "oracle_reward": self.oracle_reward,
                "steps_trained": self.steps_trained,
            },
            file_path,
        )
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path: str) -> "LinUCB":
        """Load parameters + stats from disk."""
        m = joblib.load(file_path)
        model = cls(m["d"], m["n_arms"], m["alpha"], m["l2"])
        model.A_matrices = m["A_matrices"]
        model.b_vectors = m["b_vectors"]
        model.N = m.get("N", np.zeros(model.n_arms, dtype=int))
        model.R_sum = m.get("R_sum", np.zeros(model.n_arms, dtype=float))
        model.total_reward = m.get("total_reward", 0.0)
        model.oracle_reward = m.get("oracle_reward", 0.0)
        model.steps_trained = m.get("steps_trained", 0)
        print(f"Model loaded from {file_path}")
        return model

    # Back-compat aliases
    def save(self, file_path: str) -> None: self.save_model(file_path)
    @classmethod
    def load(cls, file_path: str) -> "LinUCB": return cls.load_model(file_path)

# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
def load_feature_matrix(file_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Return (feature_array, metadata_df, feature_cols). metadata includes time_slot_id,item,item_idx."""
    df = pd.read_csv(file_path, low_memory=False)
    metadata_cols = ["time_slot_id", "item", "item_idx"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    feature_array = df[feature_cols].to_numpy(dtype=np.float64)
    metadata_df = df[metadata_cols].copy()
    return feature_array, metadata_df, feature_cols

def load_action_matrix(file_path: str) -> np.ndarray:
    """Return availability/action matrix with item_* columns as int32 (shape: T * A)."""
    df = pd.read_csv(file_path, low_memory=False)
    item_cols = [c for c in df.columns if c.startswith("item_")]
    return df[item_cols].to_numpy(dtype=np.int32)

def standardize_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].astype(str).str.strip()
    return data

def compute_rewards_for_lambda(
    lambda_value: float,
    feature_matrix_file: str,
    merged_data_file: str,
    time_slot_mapping_file: str,
) -> np.ndarray:
    """
    VARIANT1 Reward Calculation (Best Performance - Lowest Regret)
    
    Formula: reward = sales_scaled + λ * health_scaled
    
    Where:
    - sales_scaled: Sales per time-slot scaled to [0, 10]
    - health_scaled: Health scores scaled to [0, 10] using scale_health_score()
    - λ (lambda): Health weight parameter (0 = only popularity, 1 = only health)
    
    This approach:
    1. Puts both metrics on the same scale (0-10)
    2. Makes lambda interpretation direct (0.5 = 50% weight to each)
    3. Handles varying sales volumes fairly via per-timeslot scaling
    4. No complex z-scores or multiplicative factors
    
    Returns:
        numpy array of rewards for each meal serving
    """
    # Import scale_health_score from utils
    try:
        from Components.utils import scale_health_score
    except ImportError:
        from utils import scale_health_score
    
    print(f"  Computing VARIANT1 rewards (λ={lambda_value})...")
    
    # Load feature matrix metadata
    feature_df = pd.read_csv(feature_matrix_file, low_memory=False)
    rows = feature_df[["time_slot_id", "item", "item_idx"]].copy()
    rows["time_slot_id"] = pd.to_numeric(rows["time_slot_id"], errors="coerce").astype(int)
    rows["item"] = rows["item"].astype(str)

    # Load merged sales data
    merged = pd.read_csv(merged_data_file, low_memory=False)
    for c in ["date", "school_name", "time_of_day", "description"]:
        merged = standardize_text_column(merged, c)

    # Load time slot mapping
    ts = pd.read_csv(time_slot_mapping_file, low_memory=False)
    key2id = dict(
        zip(
            zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
            ts["time_slot_id"].astype(int),
        )
    )

    # Map time slots to merged data
    merged["time_slot_key"] = list(zip(merged["date"], merged["school_name"], merged["time_of_day"]))
    merged["time_slot_id"] = merged["time_slot_key"].map(key2id)
    merged = merged.dropna(subset=["time_slot_id"])
    merged["time_slot_id"] = merged["time_slot_id"].astype(int)

    # Aggregate sales and health by time_slot and item
    agg = merged.groupby(["time_slot_id", "description"], as_index=False).agg(
        total=("total", "sum"),
        health_score=("HealthScore", "median"),
    )

    # Align with feature matrix rows
    aligned = rows.merge(
        agg,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",
        validate="m:1",
    )

    # Extract sales and health scores
    aligned["total"] = pd.to_numeric(aligned["total"], errors="coerce").fillna(0.0)
    total_sales = aligned["total"].to_numpy(dtype=float)
    
    health = aligned["health_score"].to_numpy(dtype=float)
    health = np.where(np.isnan(health), np.nanmedian(health), health)

    # VARIANT1: Scale sales per time-slot to [0, 10]
    print("  Scaling sales per time-slot to [0, 10]...")
    sales_scaled = np.zeros_like(total_sales, dtype=float)
    
    for ts_id in aligned['time_slot_id'].unique():
        # Get indices for this time slot
        mask = (aligned['time_slot_id'] == ts_id).values
        slot_sales = total_sales[mask]
        
        # Scale to [0, 10] range
        vmin = slot_sales.min()
        vmax = slot_sales.max()
        
        if vmax <= vmin:
            # All sales same -> assign middle value
            sales_scaled[mask] = 5.0
        else:
            # Linear scaling to [0, 10]
            sales_scaled[mask] = 10.0 * (slot_sales - vmin) / (vmax - vmin)
    
    # VARIANT1: Re-normalize health scores to full [0, 10] range
    # NOTE: CSV HealthScore values are already in [0, 10] scale but use narrow range (4.01-5.71)
    # We re-normalize them to use FULL [0, 10] range for better differentiation
    # This makes health a meaningful differentiator (improves model performance from 32% to 27% regret)
    print("  Re-normalizing health scores to full [0, 10] range...")
    health_min = np.nanmin(health)
    health_max = np.nanmax(health)
    
    if health_max <= health_min:
        # All health scores same -> assign middle value
        health_scaled = np.full_like(health, 5.0, dtype=float)
    else:
        # Scale to [0, 10] using actual data range (consistent with sales scaling)
        health_scaled = 10.0 * (health - health_min) / (health_max - health_min)
        health_scaled = np.clip(health_scaled, 0.0, 10.0)  # Clamp to [0, 10]
    
    print(f"  Health range: raw={health_min:.2f}-{health_max:.2f}, scaled={health_scaled.min():.2f}-{health_scaled.max():.2f}")
    
    # VARIANT1: Combine with lambda weighting
    # reward = sales_scaled + λ * health_scaled
    rewards = sales_scaled + float(lambda_value) * health_scaled
    
    print(f"  Rewards computed: mean={rewards.mean():.2f}, std={rewards.std():.2f}")
    print(f"  Sales scaled: mean={sales_scaled.mean():.2f}, std={sales_scaled.std():.2f}")
    print(f"  Health scaled: mean={health_scaled.mean():.2f}, std={health_scaled.std():.2f}")
    
    return rewards

# -------------------------------------------------------------------
# Tensor builder + training wrapper
# -------------------------------------------------------------------
def _build_bandit_tensors(
    feature_array: np.ndarray,     # (N, d)
    metadata_df: pd.DataFrame,     # cols: time_slot_id, item_idx
    rewards_vec: np.ndarray,       # (N,)
    avail_mat: np.ndarray,         # (T, A) 0/1 availability
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (data, rewards, mask) tensors to match whiteboard-style train():
      data   -> (T, A, d)
      rewards-> (T, A)
      mask   -> (T, A)
    Fills rewards by summing if multiple rows map to the same (t, arm).
    """
    T, A = avail_mat.shape
    d = feature_array.shape[1]
    data = np.zeros((T, A, d), dtype=np.float64)
    rewards = np.zeros((T, A), dtype=np.float64)
    mask = (avail_mat.astype(bool))

    # Map rows to (t, arm)
    ts = metadata_df["time_slot_id"].to_numpy(dtype=int, copy=False)
    arms = metadata_df["item_idx"].to_numpy(dtype=int, copy=False)
    rvec = np.asarray(rewards_vec, dtype=float)

    for i in range(len(metadata_df)):
        t = ts[i]; a = arms[i]
        if 0 <= t < T and 0 <= a < A:
            data[t, a, :] = feature_array[i]
            rewards[t, a] += rvec[i]   # sum if duplicated entries

    return data, rewards, mask

def train_linucb_model(
    feature_array: np.ndarray,
    action_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    rewards: np.ndarray,
    lambda_value: float,
    verbose: bool = False,
):
    """Initialize LinUCB, build (T, A, d) tensors, train, and return (results, model)."""
    n_samples, n_features = feature_array.shape
    n_arms = action_matrix.shape[1]

    print(f"Training LinUCB with lambda = {lambda_value}...")
    print(f"  Features: {n_features}")
    print(f"  Arms (items): {n_arms}")
    print(f"  Samples: {n_samples}")

    # Build tensors for whiteboard-style train()
    data_3d, rewards_2d, mask_2d = _build_bandit_tensors(
        feature_array, metadata_df, rewards, action_matrix
    )

    model = LinUCB(d=n_features, n_arms=n_arms, alpha=1.0, l2=1.0, seed=42)
    results = model.train(data=data_3d, rewards=rewards_2d, mask=mask_2d, verbose=verbose)
    results["lambda"] = float(lambda_value)
    return results, model

# -------------------------------------------------------------------
# Optional: standalone training runner
# -------------------------------------------------------------------
def main():
    """
    Standalone training with VARIANT1 rewards.
    Tests multiple lambda values to find optimal health-popularity balance.
    """
    # Paths relative to repo root
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    repo_root = src_dir.parent  # Go up from src/Components to src, then to repo root
    data_dir = repo_root / "data"

    feature_matrix_file = data_dir / "feature_matrix.csv"
    action_matrix_file = data_dir / "action_matrix.csv"
    merged_data_file = data_dir / "data_healthscore_mapped.csv"
    time_slot_mapping_file = data_dir / "time_slot_mapping.csv"

    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lambda_values_to_test = [0.2, 0.3,0.4, 0.6, 0.8]

    print("=" * 70)
    print("LINUCB TRAINING WITH VARIANT1 REWARDS")
    print("=" * 70)
    print("reward = sales_scaled + λ * health_scaled")
    print("Both metrics scaled to [0, 10] for fair comparison")
    print()

    print("[1/3] Loading data...")
    feature_array, metadata_df, feature_cols = load_feature_matrix(str(feature_matrix_file))
    action_matrix = load_action_matrix(str(action_matrix_file))
    print(f"Loaded {len(metadata_df)} feature samples")
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Action matrix shape: {action_matrix.shape}")

    all_results = []
    all_models = {}

    print("\n[2/3] Training models with different lambda values...")
    for lam in lambda_values_to_test:
        print(f"\n--- Lambda = {lam} ---")
        rewards = compute_rewards_for_lambda(
            lam, 
            str(feature_matrix_file), 
            str(merged_data_file), 
            str(time_slot_mapping_file)
        )
        results, model = train_linucb_model(
            feature_array, 
            action_matrix, 
            metadata_df, 
            rewards, 
            lam
        )
        all_results.append(results)
        all_models[lam] = model

        model_filename = f"model_lambda_{lam:.2f}.joblib"
        model_filepath = results_dir / model_filename
        model.save(str(model_filepath))
        print(f"Saved model to {model_filepath}")

    print("\n[3/3] Results Summary")
    print("=" * 70)
    print("Lambda     Total Reward    Oracle Reward       Regret      Regret %")
    print("-" * 70)
    
    best_result = min(all_results, key=lambda r: r['regret'])
    
    for result in sorted(all_results, key=lambda r: r['lambda']):
        lam = result['lambda']
        total = result['total_reward']
        oracle = result['oracle_reward']
        regret = result['regret']
        regret_pct = 100 * regret / max(oracle, 1)
        
        marker = "→" if lam == best_result['lambda'] else " "
        print(f"{marker} {lam:<8.2f} {total:>12.2f} {oracle:>15.2f} {regret:>12.2f} {regret_pct:>11.1f}%")
    
    print()
    print(f"Best lambda: {best_result['lambda']:.2f}")
    print(f"Regret: {best_result['regret']:.2f} ({100*best_result['regret']/max(best_result['oracle_reward'],1):.1f}%)")
    print("\nTraining complete! ✓")

if __name__ == "__main__":
    main()