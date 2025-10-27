# ===== model.py =====
# Purpose: LinUCB algorithm for contextual bandits
# Can be run directly (python src/components/model.py) to train and save models.

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


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
        # Per-arm stats
        self.N = np.zeros(self.n_arms, dtype=int)
        self.R_sum = np.zeros(self.n_arms, dtype=float)
        # reward buffers removed on purpose (kept lean)

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
       
        x = features.reshape(-1, 1)
        self.A_matrices[arm_id] += x @ x.T
        self.b_vectors[arm_id] += float(reward) * x
        self.N[arm_id] += 1
        self.R_sum[arm_id] += float(reward)

    # ----- training / inference -----
    def train(
        self,
        X: np.ndarray,
        rows_df: pd.DataFrame,   # must contain ['time_slot_id','item_idx']
        rewards: np.ndarray,     # len == len(rows_df)
        avail_mat: np.ndarray,   # shape (n_timeslots, n_arms), values in {0,1}
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Group by time_slot_id, pick among available arms via UCB, update, and track regret."""
        if X.shape[0] != len(rows_df):
            raise ValueError(f"X has {X.shape[0]} rows but rows_df has {len(rows_df)} rows")
        if len(rewards) != len(rows_df):
            raise ValueError(f"rewards has {len(rewards)} rows but rows_df has {len(rows_df)} rows")
        if avail_mat.shape[1] != self.n_arms:
            raise ValueError(f"avail_mat has {avail_mat.shape[1]} arms but model has {self.n_arms}")

        X = np.asarray(X, dtype=np.float64)

        # group indices by time slot
        groups: Dict[int, np.ndarray] = {}
        for t, idxs in rows_df.groupby("time_slot_id").groups.items():
            groups[int(t)] = np.array(list(idxs), dtype=int)

        total_reward = 0.0
        oracle_reward = 0.0
        steps = 0

        for t in sorted(groups.keys()):
            if t >= avail_mat.shape[0]:
                continue
            available_items = np.where(avail_mat[t] == 1)[0].tolist()
            if not available_items:
                continue

            feats, rews = {}, {}
            for i in groups[t]:
                arm = int(rows_df.iloc[i]["item_idx"])
                if arm in available_items:
                    feats[arm] = X[i]
                    rews[arm] = float(rewards[i])
            if not feats:
                continue

            a = self.select_action(list(feats.keys()), feats)
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
        model_data = {
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
        }
        joblib.dump(model_data, file_path)
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


# ===== Helpers to load data / compute rewards / train (used by main()) =====

def load_feature_matrix(file_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Return (feature_array, metadata_df, feature_cols). metadata includes time_slot_id,item,item_idx."""
    df = pd.read_csv(file_path, low_memory=False)
    metadata_cols = ["time_slot_id", "item", "item_idx"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    feature_array = df[feature_cols].to_numpy(dtype=np.float64)
    metadata_df = df[metadata_cols].copy()
    return feature_array, metadata_df, feature_cols


def load_action_matrix(file_path: str) -> np.ndarray:
    """Return availability/action matrix with item_* columns as int32."""
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
    Reward = total_sales * (1 + λ * z(health)) with a 20% boost to top-40% healthy & popular items.
    """
    feature_df = pd.read_csv(feature_matrix_file, low_memory=False)
    rows = feature_df[["time_slot_id", "item", "item_idx"]].copy()
    rows["time_slot_id"] = pd.to_numeric(rows["time_slot_id"], errors="coerce").astype(int)
    rows["item"] = rows["item"].astype(str)

    merged = pd.read_csv(merged_data_file, low_memory=False)
    for c in ["date", "school_name", "time_of_day", "description"]:
        merged = standardize_text_column(merged, c)

    ts = pd.read_csv(time_slot_mapping_file, low_memory=False)
    key2id = dict(
        zip(
            zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
            ts["time_slot_id"].astype(int),
        )
    )

    merged["time_slot_key"] = list(zip(merged["date"], merged["school_name"], merged["time_of_day"]))
    merged["time_slot_id"] = merged["time_slot_key"].map(key2id)
    merged = merged.dropna(subset=["time_slot_id"])
    merged["time_slot_id"] = merged["time_slot_id"].astype(int)

    agg = merged.groupby(["time_slot_id", "description"], as_index=False).agg(
        total=("total", "sum"),
        health_score=("HealthScore", "median"),
    )

    aligned = rows.merge(
        agg,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",
        validate="m:1",
    )

    aligned["total"] = pd.to_numeric(aligned["total"], errors="coerce").fillna(0.0)

    health = aligned["health_score"].to_numpy(dtype=float)
    health = np.where(np.isnan(health), np.nanmedian(health), health)

    mu = float(np.nanmean(health))
    sd = float(np.nanstd(health))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    health_z = (health - mu) / sd

    sales = aligned["total"].to_numpy(dtype=float)
    rewards = sales * (1.0 + float(lambda_value) * health_z)

    h_thr = np.percentile(health_z, 60)
    p_thr = np.percentile(sales, 60)
    mask = (health_z > h_thr) & (sales > p_thr)
    rewards[mask] *= 1.2

    print(f"  Boosted {mask.sum()} health-popularity 'sweet spot' items")
    return rewards


def train_linucb_model(
    feature_array: np.ndarray,
    action_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    rewards: np.ndarray,
    lambda_value: float,
    verbose: bool = False,
):
    """Initialize LinUCB, train, and return (results, model)."""
    n_samples, n_features = feature_array.shape
    n_arms = action_matrix.shape[1]

    print(f"Training LinUCB with lambda = {lambda_value}...")
    print(f"  Features: {n_features}")
    print(f"  Arms (items): {n_arms}")
    print(f"  Samples: {n_samples}")

    model = LinUCB(d=n_features, n_arms=n_arms, alpha=1.0, l2=1.0, seed=42)
    results = model.train(
        X=feature_array, rows_df=metadata_df, rewards=rewards, avail_mat=action_matrix, verbose=verbose
    )
    results["lambda"] = float(lambda_value)
    return results, model


def main():
    # Paths relative to src/
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    data_dir = src_dir / "data"

    feature_matrix_file = data_dir / "feature_matrix.csv"
    action_matrix_file = data_dir / "action_matrix.csv"
    merged_data_file = data_dir / "data_healthscore_mapped.csv"
    time_slot_mapping_file = data_dir / "time_slot_mapping.csv"

    results_dir = src_dir / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lambda_values_to_test = [0.05, 0.60, 0.70]

    print("[1/3] Loading data.")
    feature_array, metadata_df, feature_cols = load_feature_matrix(str(feature_matrix_file))
    action_matrix = load_action_matrix(str(action_matrix_file))
    print(f"Loaded {len(metadata_df)} feature samples")
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Action matrix shape: {action_matrix.shape}")

    all_results = []
    all_models = {}

    for lam in lambda_values_to_test:
        print(f"\n--- Lambda = {lam} ---")
        rewards = compute_rewards_for_lambda(lam, str(feature_matrix_file), str(merged_data_file), str(time_slot_mapping_file))
        results, model = train_linucb_model(feature_array, action_matrix, metadata_df, rewards, lam)
        all_results.append(results)
        all_models[lam] = model

        model_filename = f"model_lambda_{lam:.2f}.joblib"
        model_filepath = results_dir / model_filename
        model.save(str(model_filepath))
        print(f"Saved model to {model_filepath}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
