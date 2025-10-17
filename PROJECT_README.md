# Capstone Proposal
## Health-Aware School Meal Recommendations with Contextual Bandits
#### The George Washington University, Washington DC  
#### Data Science Program

# Health-Aware School Meal Recommendations (LinUCB case study)

-- This repository implements an open-source case study that uses Contextual Multi-Armed Bandits (LinUCB) to recommend school meals that balance student popularity (sales) and nutrition (health score). The code trains LinUCB on FCPS historical meal sales and health data, runs a λ ablation (trade-off between popularity and health), and saves trained models and results.

Why this project
- School meal planning must balance student participation (popularity) and nutritional goals. Contextual bandits let us learn recommendations that adapt to context (school, time of day, nutritional features) while balancing exploration and exploitation.

Key ideas implemented
- Meals are arms, nutritional features are contexts, and rewards are a health-aware popularity score:
    - reward = total_sales * (1 + λ * health_score_z)
    - LinUCB learns per-arm linear models (θ_a) and uses an uncertainty bonus to explore.

Repository layout
- `src/` – core code
    - `Components/model.py` – LinUCB implementation (A, b, theta, UCB scoring, train loop)
    - `Components/features.py` – example item feature dictionaries
    - `Components/env.py` – environment helpers (feature matrix building, I/O)
    - `fcps_dataextractor/` – helper scripts for fetching/processing FCPS data
- `src/data/` – processed CSVs used by the code (feature_matrix.csv, action_matrix.csv, fcps_data_with_timestamps.csv, etc.)
- `src/tests/` – scripts used to train, evaluate, and save results
    - `train_eval.py` – main ablation script (compute rewards for different λ; train LinUCB; save results)
    - `main.py` – example inference script using a saved model for predictions
    - `results/` – trained models and `ablation_results.json`


Data
- The project uses historical FCPS sales data (provided in `src/data/`). The important files:
    - `feature_matrix.csv` – rows of meal servings with nutritional feature columns and metadata (time_slot_id, item, item_idx)
    - `action_matrix.csv` – binary availability matrix: rows=time_slots, cols=item_k
    - `fcps_data_with_timestamps.csv` – raw sales with timestamps used to compute aggregated sales and health scores
    - `time_slot_mapping.csv` – maps (date, school_name, time_of_day) → time_slot_id

Quick start — setup
1.  install minimal dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

2. Train the default λ ablation (this will write models to `src/tests/results/`):

```powershell
python .\src\tests\train_eval.py
```

What the training script does (at a glance)
- Loads features and availability
- For each λ in `lambda_values_to_test` it:
    - computes a reward vector using `compute_rewards_for_lambda(λ)`
    - instantiates `LinUCB(d, n_arms, alpha=1.0, l2=1.0)`
    - trains via `model.train(...)`, which simulates sequential decisions per time_slot
    - saves the model as `model_lambda_{λ:.2f}.joblib`
- Writes `ablation_results.json` to `src/tests/results/`

The LinUCB algorithm — math and code mapping
- Implementation: `src/Components/model.py` (class `LinUCB`)
- Per-arm statistics (disjoint LinUCB):
    - A_a ∈ ℝ^{d×d} initialized to l2 * I  — stored in `self.A_matrices[arm_id]`
    - b_a ∈ ℝ^{d×1} initialized to 0      — stored in `self.b_vectors[arm_id]`
    - θ_a = A_a^{-1} b_a                 — computed by `compute_theta(arm_id)` using `np.linalg.solve`
- UCB scoring (in code):
    - estimated_value = θ_a^T x  (code: `theta @ features`)
    - confidence = α * sqrt(x^T A_a^{-1} x)  (code: solve A y = x then sqrt(features @ y))
    - ucb_score = estimated_value + confidence
- Updates executed only for the chosen arm after observing reward r:
    - A_a += x x^T  (code: `self.A_matrices[arm_id] += features @ features.T`)
    - b_a += r x    (code: `self.b_vectors[arm_id] += reward * features`)

Reward design (what we optimize)
- The project uses a health-aware popularity reward:
    - compute initial aggregated `total` (total sold) and `health_score` per (time_slot, item)
    - health_z = (health_score − mean) / std
    - reward = total * (1 + λ * health_z)
    - boost factor: if item in top 40% by both health_z and total → reward *= 1.2
- Significance & cautions:
    - Multiplicative form retains sales scale — health is a relative nudge unless λ is large.
    - Negative rewards are possible if (1 + λ * health_z) < 0; the code does not clip by default.
    - Consider alternative reward forms (additive normalized, log-sales) if you need interpretability or robustness.

Evaluation metrics
- `total_reward`: sum of rewards obtained by the trained policy
- `oracle_reward`: sum of the maximum available reward at each time slot (clairvoyant)
- `regret` = oracle_reward − total_reward (lower is better)
- The `train_eval.py` script prints a summary table across λ values and saves `ablation_results.json`.



Where outputs are saved
- Models: `src/tests/results/model_lambda_*.joblib`
- Ablation summary: `src/tests/results/ablation_results.json`

Developer notes (for maintainers)
- The code simulates online learning by grouping rows by `time_slot_id` (mapping from date-school-time). For each time slot it only updates the chosen arm to mimic partial feedback. This is critical: updating all arms with historical rewards would be an offline supervised fit, not a bandit simulation.
- Key files to inspect:
    - `src/Components/model.py` — LinUCB implementation
    - `src/tests/train_eval.py` — reward construction + ablation loop
    - `src/tests/main.py` — example inference using saved model

Contributing and contact
- If you want to contribute: fork the repo, add a feature branch, and open a PR. Add tests under `src/tests/` for any new behavior.
- Contacts: Ganesh Kumar Boini, Dinesh Chandra Gaddam, Sirisha Ginnu.


