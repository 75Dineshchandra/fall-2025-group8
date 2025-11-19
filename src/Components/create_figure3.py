"""
Create Figure 3: Continuous Regret Calculation
Rolling average regret percentage over time for LinUCB (λ = 0.3)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get project paths
HERE = Path(__file__).resolve().parent
SRC = HERE if HERE.name == "src" else HERE.parent
sys.path.insert(0, str(SRC))

# Import model and helper functions
from Components.model import LinUCB, load_feature_matrix, load_action_matrix, compute_rewards_for_lambda

# File paths
DATA_DIR = SRC / "data"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"
ACTION_MATRIX = DATA_DIR / "action_matrix.csv"
MERGED_DATA = DATA_DIR / "data_healthscore_mapped.csv"
TIMESLOTS = DATA_DIR / "time_slot_mapping.csv"

RESULTS_DIR = SRC / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
LAMBDA = 0.30

def rolling_mean(values, window_size):
    """Calculate expanding-window rolling average"""
    series = pd.Series(values)
    return series.expanding().mean().to_numpy()

def create_figure3():
    """Create Figure 3: Rolling Average Regret Percentage Over Time"""
    
    print("=" * 72)
    print("CREATING FIGURE 3: Continuous Regret Calculation")
    print("=" * 72)
    
    # Load data
    print("\n[1/3] Loading data...")
    X, meta, _ = load_feature_matrix(str(FEATURE_MATRIX))
    A = load_action_matrix(str(ACTION_MATRIX))
    print(f"Feature matrix: {X.shape} | Action matrix: {A.shape}")
    
    # Compute rewards
    print("\n[2/3] Computing rewards...")
    rewards_vec = compute_rewards_for_lambda(
        lambda_value=LAMBDA,
        feature_matrix_file=str(FEATURE_MATRIX),
        merged_data_file=str(MERGED_DATA),
        time_slot_mapping_file=str(TIMESLOTS)
    )
    
    # Build tensors
    T = A.shape[0]
    A_arms = A.shape[1]
    d = X.shape[1]
    
    data_TAD = np.zeros((T, A_arms, d), dtype=np.float32)
    rewards_TA = np.zeros((T, A_arms), dtype=np.float32)
    mask_TA = A.astype(bool)
    
    # Fill tensors
    for idx, row in meta.iterrows():
        t = int(row['time_slot_id'])
        a = int(row['item_idx'])
        if t < T and a < A_arms:
            if isinstance(meta.index, pd.RangeIndex):
                array_idx = idx
            else:
                array_idx = meta.index.get_loc(idx)
            data_TAD[t, a] = X[array_idx]
            rewards_TA[t, a] = float(rewards_vec[array_idx])
    
    # Train LinUCB and track regret
    print("\n[3/3] Training LinUCB and tracking regret...")
    bandit = LinUCB(d=d, n_arms=A_arms, alpha=1.0, l2=1.0, seed=42)
    
    rows = []
    for t in range(T):
        avail = np.where(mask_TA[t])[0]
        if avail.size == 0:
            continue
        
        feats = {a: data_TAD[t, a] for a in avail}
        a = bandit.select_action(list(avail), feats)
        
        r = float(rewards_TA[t, a])
        oracle = float(np.max(rewards_TA[t, avail]))
        regret = oracle - r
        regret_pct = (regret / oracle * 100) if oracle > 0 else 0.0
        
        rows.append({
            "t": t,
            "reward": r,
            "oracle": oracle,
            "regret": regret,
            "regret_pct": regret_pct,
        })
        
        bandit.update_arm(a, feats[a], r)
        
        if (t+1) % 5000 == 0:
            print(f"  Processed {t+1}/{T} time steps...")
    
    df = pd.DataFrame(rows)
    
    # Calculate rolling average regret percentage
    df["roll_regret_pct"] = rolling_mean(df["regret_pct"].to_numpy(), 50)
    
    # Create figure
    print("\n[4/4] Creating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot rolling average regret percentage
    ax.plot(df["t"], df["roll_regret_pct"], lw=2.5, color="tab:blue", label="LinUCB (λ=0.3)")
    
    # Add horizontal reference line at final value
    final_regret = df["roll_regret_pct"].iloc[-1]
    ax.axhline(final_regret, ls="--", lw=1.5, color="tab:red", alpha=0.7, 
               label=f"Final: {final_regret:.1f}%")
    
    # Styling
    ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rolling Average Regret (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: Continuous Regret Calculation\nRolling Average Regret Percentage Over Time (LinUCB, λ=0.3)", 
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    
    # Set y-axis limits to show full range
    ax.set_ylim([0, max(100, df["roll_regret_pct"].max() * 1.1)])
    
    # Add text annotation for initial and final values
    initial_regret = df["roll_regret_pct"].iloc[0]
    ax.text(0.02, 0.98, 
            f"Initial: {initial_regret:.1f}%\nFinal: {final_regret:.1f}%\nReduction: {initial_regret - final_regret:.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    out_png = RESULTS_DIR / "figure3_continuous_regret_calculation.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Figure 3: {out_png}")
    
    # Print summary statistics
    print("\n" + "=" * 72)
    print("FIGURE 3 SUMMARY STATISTICS")
    print("=" * 72)
    print(f"Initial Regret: {initial_regret:.2f}%")
    print(f"Final Regret: {final_regret:.2f}%")
    print(f"Total Reduction: {initial_regret - final_regret:.2f} percentage points")
    print(f"Relative Improvement: {((initial_regret - final_regret) / initial_regret * 100):.1f}%")
    print(f"Total Time Steps: {len(df)}")
    print("=" * 72)
    
    plt.show()
    
    return df

if __name__ == "__main__":
    df = create_figure3()

