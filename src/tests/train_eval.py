# train_eval.py
# Purpose: Train LinUCB model with different lambda values and compare results
# Lambda (λ) = Health weight parameter (0 = only popularity, 1 = only health)

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path so we can import our custom model + helpers
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import functions from Components.model
from Components.model import (
    load_feature_matrix,
    load_action_matrix,
    compute_rewards_for_lambda,
    train_linucb_model,
)


# ===== FILE PATHS =====
# Go up from src/tests to repo root, then into data/
repo_root = src_dir.parent
data_dir = repo_root / "data"
feature_matrix_file = data_dir / "feature_matrix.csv"
action_matrix_file = data_dir / "action_matrix.csv"
merged_data_file = data_dir / "data_healthscore_mapped.csv"
time_slot_mapping_file = data_dir / "time_slot_mapping.csv"

# Directory to save trained models and results
results_dir = repo_root / "data" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# ===== CONFIGURATION =====
lambda_values_to_test = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]  # lower = popularity, higher = health

# ===== MAIN =====
def main():
    print("=" * 70)
    print("LINUCB TRAINING WITH LAMBDA ABLATION")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/3] Loading data...")
    feature_array, metadata_df, feature_cols = load_feature_matrix(str(feature_matrix_file))
    action_matrix = load_action_matrix(str(action_matrix_file))
    print(f"Loaded {len(metadata_df)} feature samples")
    print(f"Feature array shape: {feature_array.shape}")
    print(f"Action matrix shape: {action_matrix.shape}\n")

    # Step 2: Train models
    print("[2/3] Training models with different λ values...")
    all_results = []
    for lam in lambda_values_to_test:
        print(f"\n--- Lambda = {lam} ---")
        rewards = compute_rewards_for_lambda(
            lam, str(feature_matrix_file), str(merged_data_file), str(time_slot_mapping_file)
        )
        results, model = train_linucb_model(feature_array, action_matrix, metadata_df, rewards, lam)
        all_results.append(results)

        model_path = results_dir / f"model_lambda_{lam:.2f}.joblib"
        model.save(str(model_path))
        print(f"Saved model → {model_path.name}")

    # Step 3: Print results summary table
    print("\n" + "=" * 70)
    print("HEALTH-POPULARITY TRADE-OFF ANALYSIS")
    print("=" * 70)
    print()
    print("LAMBDA VALUE GUIDE:")
    print("0.05-0.15: Popularity Focused (student preferences dominate)")
    print("0.20-0.30: Balanced Approach (mix of popularity and health)")
    print("0.40-0.50+: Health Focused (nutritional goals prioritized)")
    print()
    print("SUMMARY: LAMBDA ABLATION RESULTS")
    print("Lambda = Health Weight | Lower values favor popularity, Higher values favor health")
    print("Regret = How much worse than perfect choices | Lower is better")
    print("─" * 80)
    print(f"{'Lambda':<10}{'Total Reward':>15}{'Oracle Reward':>18}{'Regret':>15}{'Regret %':>14}")
    print("─" * 80)

    best = min(all_results, key=lambda r: r["regret"])
    for r in sorted(all_results, key=lambda x: x["lambda"]):
        lam, total, oracle, regret = r["lambda"], r["total_reward"], r["oracle_reward"], r["regret"]
        pct = 100 * regret / max(oracle, 1)
        mark = "-" if lam == best["lambda"] else " "
        print(f"{mark} {lam:<8.2f}{total:>15.2f}{oracle:>18.2f}{regret:>15.2f}{pct:>13.1f}%")

    print("\nANALYSIS")
    print(f" Best performing lambda: λ = {best['lambda']:.2f}")
    print(f"  Regret: {best['regret']:.2f} "
          f"({100 * best['regret'] / max(best['oracle_reward'],1):.1f}% of optimal)")

    # Save results as JSON
    results_json = results_dir / "ablation_results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_json}")
    print("\nTRAINING COMPLETE ")


if __name__ == "__main__":
    
    main()
