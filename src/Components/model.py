import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class LinUCB:
    """
    Per-arm linear UCB for contextual bandits.
    """

    def __init__(self, d: int, n_arms: int, alpha: float = 1.0, l2: float = 1.0, seed: int = 42):
        self.d = int(d)
        self.n_arms = int(n_arms)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------- internals ----------
    def _theta(self, a: int) -> np.ndarray:
        # stable solve: A theta = b
        return np.linalg.solve(self.A[a], self.b[a]).reshape(-1)

    def _ucb(self, a: int, x: np.ndarray) -> float:
        theta = self._theta(a)
        est = float(theta @ x)
        Ax = np.linalg.solve(self.A[a], x)  # A^{-1} x
        conf = float(self.alpha * np.sqrt(max(0.0, x @ Ax)))
        return est + conf

    # ---------- API ----------
    def reset(self) -> None:
        self.A = [self.l2 * np.eye(self.d, dtype=np.float64) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.d, 1), dtype=np.float64) for _ in range(self.n_arms)]
        self.total_reward = 0.0
        self.oracle_reward = 0.0
        self.steps = 0

    def action(self, available_arms: List[int], x_by_arm: Dict[int, np.ndarray]) -> int:
        best_arm, best_val = None, -1e18
        for a in available_arms:
            x = x_by_arm[a].reshape(-1)
            val = self._ucb(a, x)
            if val > best_val:
                best_val, best_arm = val, a
        if best_arm is None:
            best_arm = self.rng.choice(available_arms)
        return int(best_arm)

    def calculate_reward(self, r: float) -> float:
        return float(r)

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        x = x.reshape(-1, 1)
        self.A[a] += x @ x.T
        self.b[a] += r * x

    def train(self, X: np.ndarray, rows_df: pd.DataFrame, rewards: np.ndarray, avail_mat: np.ndarray,
              verbose: bool = False) -> Dict[str, float]:
        # ---- column name robustness: accept either 'time_slot_id' or 't'
        time_col = "time_slot_id" if "time_slot_id" in rows_df.columns else ("t" if "t" in rows_df.columns else None)
        if time_col is None:
            raise KeyError("rows_df must contain 'time_slot_id' or 't'")

        # ---- basic shape/align checks
        if X.shape[0] != len(rows_df):
            raise ValueError(f"X rows ({X.shape[0]}) != rows_df rows ({len(rows_df)})")
        if rewards.shape[0] != len(rows_df):
            raise ValueError(f"rewards length ({rewards.shape[0]}) != rows_df rows ({len(rows_df)})")
        if avail_mat.shape[1] != self.n_arms:
            raise ValueError(f"avail_mat n_arms ({avail_mat.shape[1]}) != model.n_arms ({self.n_arms})")

        # group indices by time slot
        groups = {int(t): np.array(list(idxs), dtype=int)
                  for t, idxs in rows_df.groupby(time_col).groups.items()}

        total, oracle, steps = 0.0, 0.0, 0
        for t in sorted(groups.keys()):
            ridxs = groups[t]
            if t >= avail_mat.shape[0]:
                # if an unseen t slips in, skip gracefully
                continue
            available = np.where(avail_mat[t] == 1)[0].tolist()
            if not available:
                continue

            x_by_arm, r_by_arm = {}, {}
            for ridx in ridxs:
                a = int(rows_df.iloc[ridx]["item_idx"])
                if a in available:
                    x_by_arm[a] = X[ridx]
                    r_by_arm[a] = float(rewards[ridx])

            if not x_by_arm:
                continue

            a_star = self.action(list(x_by_arm.keys()), x_by_arm)
            r = self.calculate_reward(r_by_arm[a_star])
            self.update(a_star, x_by_arm[a_star], r)

            total  += r
            oracle += max(r_by_arm.values())
            steps  += 1

            if verbose and steps % 1000 == 0:
                print(f"[t={t}] steps={steps} total={total:.1f} oracle={oracle:.1f}")

        self.total_reward, self.oracle_reward, self.steps = total, oracle, steps
        return {
            "steps": steps,
            "total_reward": float(total),
            "oracle_reward": float(oracle),
            "regret": float(oracle - total),
            "avg_reward": float(total / max(1, steps)),
        }

    def recommend(self, x_by_arm: Dict[int, np.ndarray], topk: int = 5) -> List[Tuple[int, float]]:
        scores = [(a, self._ucb(a, x.reshape(-1))) for a, x in x_by_arm.items()]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:topk]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True
        )
        joblib.dump({"A": self.A, "b": self.b, "d": self.d, "n_arms": self.n_arms, "alpha": self.alpha, "l2": self.l2}, path)
        print(f" Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LinUCB":
        obj = joblib.load(path)
        model = cls(d=obj["d"], n_arms=obj["n_arms"], alpha=obj["alpha"], l2=obj["l2"])
        model.A, model.b = obj["A"], obj["b"]
        print(f"Model loaded from {path}")
        return model
