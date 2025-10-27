# ===== model.py =====
# Purpose: LinUCB algorithm for contextual bandits
# Can be run directly (python src/Components/model.py) to train and save models.

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) for contextual bandits.

    For each arm i:
      A_i: d x d matrix (design / precision)
      b_i: d x 1 vector (responses)
    """

    def __init__(self, d: int, n_arms: int, alpha: float = 1.0, l2: float = 1.0, seed: int = 42):
        self.d = int(d)
        self.n_arms = int(n_arms)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.random_state = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Initialize or reset all parameters and stats."""
        self.A_matrices = [self.l2 * np.eye(self.d, dtype=np.float64) for _ in range(self.n_arms)]
        self.b_vectors = [np.zeros((self.d, 1), dtype=np.float64) for _ in range(self.n_arms)]
        self.total_reward = 0.0
        self.oracle_reward = 0.0
        self.steps_trained = 0
        # Per-arm statistics (selection counts and cumulative rewards)
        self.N = np.zeros(self.n_arms, dtype=int)
        self.R_sum = np.zeros(self.n_arms, dtype=float)
        self.rewards_list: List[float] = []
        self.rewards_matrix = np.zeros((0, self.n_arms), dtype=float)  # one row per timestep

    # ----- core math -----
    def compute_theta(self, arm_id: int) -> np.ndarray:
        """Solve theta = A^{-1} b for a given arm (stable)."""
        A = self.A_matrices[arm_id]
        b = self.b_vectors[arm_id]
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            A = A + 1e-8 * np.eye(self.d)
            theta = np.linalg.solve(A, b)
        return theta.reshape(-1)

    def compute_upper_confidence_bound(self, arm_id: int, features: np.ndarray) -> float:
        """
        UCB score = theta·x + alpha * sqrt(x^T A^{-1} x)
        """
        x = features.reshape(-1)
        theta = self.compute_theta(arm_id)
        est = float(theta @ x)

        A = self.A_matrices[arm_id]
        try:
            A_inv_x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A = A + 1e-8 * np.eye(self.d)
            A_inv_x = np.linalg.solve(A, x)

        conf = float(self.alpha * np.sqrt(max(0.0, x @ A_inv_x)))
        return est + conf

    # ----- policy -----
    def select_action(self, available_arms: List[int], features_by_arm: Dict[int, np.ndarray]) -> int:
        """Choose the available arm with the highest UCB score."""
        best_arm, best_score = None, -np.inf
        for arm_id in available_arms:
            score = self.compute_upper_confidence_bound(arm_id, features_by_arm[arm_id])
            if score > best_score:
                best_score, best_arm = score, arm_id
        if best_arm is None:
            best_arm = int(self.random_state.choice(available_arms))
        return int(best_arm)

    def update_arm(self, arm_id: int, features: np.ndarray, reward: float) -> None:
        """A_i ← A_i + x x^T ; b_i ← b_i + r x ; update stats."""
        x = features.reshape(-1, 1)
        self.A_matrices[arm_id] += x @ x.T
        self.b_vectors[arm_id] += float(reward) * x

        # stats
        self.N[arm_id] += 1
        self.R_sum[arm_id] += float(reward)

        row = np.zeros(self.n_arms, dtype=float)
        row[arm_id] = float(reward)
        self.rewards_matrix = np.vstack([self.rewards_matrix, row.reshape(1, -1)])
        self.rewards_list.append(float(reward))

    # ----- training / inference -----
    def train(
        self,
        X: np.ndarray,
        rows_df: pd.DataFrame,
        rewards: np.ndarray,
        avail_mat: np.ndarray,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Train on historical data grouped by time_slot_id.

        X: (n_samples, d) features
        rows_df: must contain ['time_slot_id','item_idx']
        rewards: (n_samples,)
        avail_mat: (n_timeslots, n_arms) 0/1 availability mask
        """
        if X.shape[0] != len(rows_df):
            raise ValueError(f"X has {X.shape[0]} rows but rows_df has {len(rows_df)} rows")
        if len(rewards) != len(rows_df):
            raise ValueError(f"rewards has {len(rewards)} rows but rows_df has {len(rows_df)} rows")
        if avail_mat.shape[1] != self.n_arms:
            raise ValueError(f"avail_mat has {avail_mat.shape[1]} arms but model has {self.n_arms}")

        X = np.asarray(X, dtype=np.float64)

        # group by time slot
        groups: Dict[int, np.ndarray] = {}
        for t, idxs in rows_df.groupby("time_slot_id").groups.items():
            groups[int(t)] = np.array(list(idxs), dtype=int)

        total_reward = 0.0
        oracle_reward = 0.0
        steps = 0
        history: List[Dict[str, float]] = []

        for t in sorted(groups.keys()):
            idxs = groups[t]
            if t >= avail_mat.shape[0]:
                continue

            available_items = np.where(avail_mat[t] == 1)[0].tolist()
            if not available_items:
                continue

            feats: Dict[int, np.ndarray] = {}
            rews: Dict[int, float] = {}

            for i in idxs:
                arm = int(rows_df.iloc[i]["item_idx"])
                if arm in available_items:
                    feats[arm] = X[i]
                    rews[arm] = float(rewards[i])

            if not feats:
                continue

            arm_sel = self.select_action(list(feats.keys()), feats)
            r_sel = rews[arm_sel]
            self.update_arm(arm_sel, feats[arm_sel], r_sel)

            total_reward += r_sel
            o = max(rews.values())
            oracle_reward += o
            steps += 1

            if verbose and steps % 1000 == 0:
                print(f"[t={t}] steps={steps} total={total_reward:.1f} oracle={oracle_reward:.1f}")

            history.append({"t": float(t), "arm": float(arm_sel), "reward": float(r_sel), "oracle": float(o)})

        self.total_reward = total_reward
        self.oracle_reward = oracle_reward
        self.steps_trained = steps

        regret = oracle_reward - total_reward
        avg_reward = total_reward / max(1, steps)

        return {
            "steps": float(steps),
            "total_reward": float(total_reward),
            "oracle_reward": float(oracle_reward),
            "regret": float(regret),
            "avg_reward": float(avg_reward),
            "history": history,
        }

    def get_recommendations(self, features_by_arm: Dict[int, np.ndarray], top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top-k (arm, score) pairs for the current slot; no model updates here."""
        scores: List[Tuple[int, float]] = []
        for arm_id, feats in features_by_arm.items():
            score = self.compute_upper_confidence_bound(arm_id, feats.reshape(-1))
            scores.append((arm_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ----- API-compat wrappers (EpsilonGreedy pattern) -----
    def action(self, available_arms: List[int], features_by_arm: Dict[int, np.ndarray]) -> int:
        return self.select_action(available_arms, features_by_arm)

    def update(self, arm_id: int, features: np.ndarray, reward: float) -> None:
        self.update_arm(arm_id, features, reward)

    def recommend(self, features_by_arm: Dict[int, np.ndarray], top_k: int = 5) -> List[Tuple[int, float]]:
        return self.get_recommendations(features_by_arm, top_k=top_k)

    def calculate_reward(self, *_, **__):
        raise NotImplementedError(
            "calculate_reward() is dataset-specific. Implement in your training script or a subclass."
        )

    # ----- Helper for bandit-style evaluation parity -----
    def bandit(self, rewards_array: np.ndarray, arm_id: int, sample_idx: int) -> float:
        """
        If rewards_array is 1D: return rewards_array[sample_idx].
        If 2D: treat row sample_idx as reward vector; return value for arm_id and zero others (for logging).
        """
        arr = np.asarray(rewards_array)
        if arr.ndim == 1:
            r = float(arr[sample_idx]) if sample_idx < arr.shape[0] else 0.0
            row = np.zeros(self.n_arms, dtype=float)
            row[arm_id] = r
            self.rewards_matrix = np.vstack([self.rewards_matrix, row.reshape(1, -1)])
            self.rewards_list.append(r)
            return r

        row_vec = arr[sample_idx:sample_idx + 1][0].astype(float)
        r = float(row_vec[arm_id])
        row_vec[:arm_id] = 0
        row_vec[arm_id + 1:] = 0
        self.rewards_matrix = np.vstack([self.rewards_matrix, row_vec.reshape(1, -1)])
        self.rewards_list.append(r)
        return r

    def create_table(self) -> str:
        """Pretty string table of arms with selection counts and average reward."""
        avg = np.zeros_like(self.R_sum)
        mask = self.N > 0
        avg[mask] = self.R_sum[mask] / self.N[mask]
        table = np.hstack([
            np.arange(self.n_arms).reshape(self.n_arms, 1),
            self.N.reshape(self.n_arms, 1),
            avg.reshape(self.n_arms, 1),
        ]).astype(float)
        df = pd.DataFrame(table, columns=["Arm", "Selections", "AvgReward"])
        return df.to_string(index=False)

    def save_stats(self) -> pd.DataFrame:
        """Return a DataFrame with per-arm counts and average rewards (not persisted)."""
        avg = np.zeros_like(self.R_sum, dtype=float)
        mask = self.N > 0
        avg[mask] = self.R_sum[mask] / self.N[mask]
        return pd.DataFrame({"Arm": np.arange(self.n_arms), "N": self.N, "AvgReward": avg})

    # ----- persistence -----
    def save_model(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        model_data = {
            "A_matrices": self.A_matrices,
            "b_vectors": self.b_vectors,
            "d": self.d,
            "n_arms": self.n_arms,
            "alpha": self.alpha,
            "l2": self.l2,
            # training stats (useful for inspection)
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
        model_data = joblib.load(file_path)
        model = cls(
            d=model_data["d"],
            n_arms=model_data["n_arms"],
            alpha=model_data["alpha"],
            l2=model_data["l2"],
        )
        model.A_matrices = model_data["A_matrices"]
        model.b_vectors = model_data["b_vectors"]
        model.N = model_data.get("N", np.zeros(model.n_arms, dtype=int))
        model.R_sum = model_data.get("R_sum", np.zeros(model.n_arms, dtype=float))
        model.total_reward = model_data.get("total_reward", 0.0)
        model.oracle_reward = model_data.get("oracle_reward", 0.0)
        model.steps_trained = model_data.get("steps_trained", 0)
        print(f"Model loaded from {file_path}")
        return model

    # Back-compat aliases
    def save(self, file_path: str) -> None:
        self.save_model(file_path)

    @classmethod
    def load(cls, file_path: str) -> "LinUCB":
        return cls.load_model(file_path)


# ===== Helpers to load data / compute rewards / train (used by main()) =====

def load_feature_matrix(file_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Return (feature_array, metadata_df, feature_cols). metadata must include time_slot_id,item,item_idx."""
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
    action_array = df[item_cols].to_numpy(dtype=np.int32)
    return action_array


def _standardize_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].astype(str).str.strip()
    return data


def compute_rewards_for_lambda(
    lambda_value: float,
    feature_matrix_file: str,
    merged_data_file: str,
    time_slot_mapping_file: str,
) -> np.ndarray:
    """
    Reward = total_sales * (1 + λ * z(health))
    + optional boost for items in top-40% health and top-40% popularity.
    """
    feature_df = pd.read_csv(feature_matrix_file, low_memory=False)
    rows_metadata = feature_df[["time_slot_id", "item", "item_idx"]].copy()
    rows_metadata["time_slot_id"] = pd.to_numeric(rows_metadata["time_slot_id"], errors="coerce").astype(int)
    rows_metadata["item"] = rows_metadata["item"].astype(str)

    merged_data = pd.read_csv(merged_data_file, low_memory=False)
    for c in ["date", "school_name", "time_of_day", "description"]:
        merged_data = _standardize_text_column(merged_data, c)

    time_slot_df = pd.read_csv(time_slot_mapping_file, low_memory=False)
    time_slot_map = dict(
        zip(
            zip(
                time_slot_df["date"].astype(str),
                time_slot_df["school_name"].astype(str),
                time_slot_df["time_of_day"].astype(str),
            ),
            time_slot_df["time_slot_id"].astype(int),
        )
    )

    merged_data["time_slot_key"] = list(
        zip(merged_data["date"], merged_data["school_name"], merged_data["time_of_day"])
    )
    merged_data["time_slot_id"] = merged_data["time_slot_key"].map(time_slot_map)
    merged_data = merged_data.dropna(subset=["time_slot_id"])
    merged_data["time_slot_id"] = merged_data["time_slot_id"].astype(int)

    aggregated = merged_data.groupby(["time_slot_id", "description"], as_index=False).agg(
        total=("total", "sum"),
        health_score=("HealthScore", "median"),
    )

    aligned = rows_metadata.merge(
        aggregated,
        left_on=["time_slot_id", "item"],
        right_on=["time_slot_id", "description"],
        how="left",
        validate="m:1",
    )

    aligned["total"] = pd.to_numeric(aligned["total"], errors="coerce").fillna(0.0)

    health = aligned["health_score"].to_numpy(dtype=float)
    median_health = np.nanmedian(health)
    health = np.where(np.isnan(health), median_health, health)

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
    rewards[mask] *= 1.2  # boost
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
    n_samples, n_features = feature_array.shape
    n_arms = action_matrix.shape[1]

    print(f"Training LinUCB with lambda = {lambda_value}...")
    print(f"  Features: {n_features}")
    print(f"  Arms (items): {n_arms}")
    print(f"  Samples: {n_samples}")

    model = LinUCB(d=n_features, n_arms=n_arms, alpha=1.0, l2=1.0, seed=42)
    results = model.train(
        X=feature_array,
        rows_df=metadata_df,
        rewards=rewards,
        avail_mat=action_matrix,
        verbose=verbose,
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

    lambda_values_to_test = [0.03, 0.06, 0.08]

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
        rewards = compute_rewards_for_lambda(
            lam, str(feature_matrix_file), str(merged_data_file), str(time_slot_mapping_file)
        )
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