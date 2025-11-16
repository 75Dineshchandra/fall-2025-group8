## Instruction for full report

# My Reinforcement Learning Study Report
*Personal notes and highlights from DATS 6450*

# week 1 and 2

##  The Big Picture - Why RL Matters

**Key Insight**: RL isn't just another ML technique - it's about discovering solutions that go beyond human thinking.

**Memorable Quote**: *"It's not a human move. I've never seen a human play this move."* - Commentary on AlphaGo's Move 37

**What RL Really Is**: "Learning Optimal Sequential Decision-Making Under Uncertainty"

**RL vs Traditional ML**:
- Traditional ML: "Here are examples, learn patterns" (instructive feedback)
- RL: "Here's an environment, figure it out" (evaluative feedback)

**Real-world impact I found interesting**:
- AlphaTensor discovered faster matrix multiplication algorithms than humans
- ChatGPT uses RL for human feedback (RLHF)
- Netflix/YouTube recommendations adapt in real-time

---

##  Math Foundations - Building Blocks

**Critical realization**: These aren't just abstract concepts - they're the language RL speaks.

### Set Theory Highlights
- Sets are collections where membership is binary (in or out)
- Cartesian products create high-dimensional spaces (GPT-4 has trillion+ dimensions!)
- Set-builder notation: {x ∈ ℝ | x > 0} for complex sets

### Probability Theory Core
**Kolmogorov's Three Axioms** (memorize these):
1. P(A) ≥ 0 (nothing negative)
2. P(Ω) = 1 (something must happen)  
3. P(A ∪ B) = P(A) + P(B) if disjoint (addition for exclusive events)

**Conditional Probability Formula** (use constantly):
P(A|B) = P(A ∩ B)/P(B)

**Product Rule**: P(A ∩ B) = P(A|B)P(B)

**Total Probability**: P(A) = ΣP(A|Bi)P(Bi)

### Random Variables
**Discrete**: PMF maps values to probabilities
**Continuous**: PDF describes probability density

**Expected Value**:
- Discrete: E[X] = Σx·P(X=x)
- Continuous: E[X] = ∫x·f(x)dx

**Key Distributions**:
- Bernoulli: Binary outcomes (coin flip)
- Gaussian: Bell curve (central limit theorem)
- Beta: Bounded [0,1], good for probabilities




# Week 3 – Nutritional Scoring & Health Index

This week focuses on computing a **health score** for foods using nutritional information, inspired by NRF9.3 scoring methodology. The goal is to calculate a normalized score (0–10) for each food item based on its nutrient content relative to daily recommended values (DV) for different school age groups.

---

## Key Concepts

- **Good Nutrients** (encouraged): Protein, Dietary Fiber, Vitamin D, Calcium, Iron, Potassium, Vitamin A, Vitamin C  
- **Bad Nutrients** (to limit): Added Sugars, Saturated Fat, Sodium  
- **Daily Values (DV)** vary by school group:
  - Elementary  
  - Middle  
  - High School  

- **NRF9.3**: Nutrient Rich Foods index.  
  - Formula: sum of %DV for 9 qualifying nutrients minus sum of %DV for 3 limiting nutrients.  
  - Source: [ScienceDirect NRF9.3](https://www.sciencedirect.com/science/article/pii/S0022316622068420?via%3Dihub)

- **Normalization**: Raw health scores are scaled to **0–10** using default min/max ranges for comparability.

---

## Code

### Load Data
```python
import pandas as pd

def load_data(file_path):
    """Load CSV data as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def get_actions(data):
    """Return unique food items."""
    return data['sales_item'].unique()

def get_features(data):
    """Return feature matrix (all columns except 'sales_item')."""
    return data

def health_score(row, min_score=-300, max_score=800):
    DV = {
        "elementary": {"Calories":1600,"Protein":19,"Total Carbohydrate":130,
                       "Dietary Fiber":25,"Added Sugars":25,"Total Fat":40,
                       "Saturated Fat":20,"Sodium":1500,"Vitamin D":20,
                       "Calcium":1000,"Iron":10,"Potassium":4700,
                       "Vitamin A":900,"Vitamin C":90},
        "middle": {"Calories":2200,"Protein":34,"Total Carbohydrate":130,
                   "Dietary Fiber":31,"Added Sugars":50,"Total Fat":77,
                   "Saturated Fat":20,"Sodium":2300,"Vitamin D":20,
                   "Calcium":1300,"Iron":18,"Potassium":4700,
                   "Vitamin A":900,"Vitamin C":90},
        "high": {"Calories":2600,"Protein":46,"Total Carbohydrate":130,
                 "Dietary Fiber":38,"Added Sugars":50,"Total Fat":91,
                 "Saturated Fat":20,"Sodium":2300,"Vitamin D":20,
                 "Calcium":1300,"Iron":18,"Potassium":4700,
                 "Vitamin A":900,"Vitamin C":90}
    }

    GOOD = ["Protein","Dietary Fiber","Vitamin D","Calcium",
            "Iron","Potassium","Vitamin A","Vitamin C"]
    BAD = ["Added Sugars","Saturated Fat","Sodium"]

    group = str(row.get("school_group","high")).lower()
    dv = DV.get(group, DV["high"])

    good_score = sum(min(100, (row.get(n,0)/dv[n])*100) for n in GOOD)
    bad_score  = sum(min(100, (row.get(n,0)/dv[n])*100) for n in BAD)

    raw_score = good_score - bad_score

    
    return raw_score

data = load_data("data/sales.csv")
data["HealthScore"] = data.apply(health_score, axis=1)
data.to_csv("scored_data.csv", index=False)
print("✅ Health scores calculated and saved to scored_data.csv")

```
# Week 4 - Initial LinUCB version with others

---

##  Multi-Armed Bandits - First Real RL

**The Core Problem**: Exploration vs Exploitation tradeoff
- Explore: Try new things to learn
- Exploit: Use what you know works

**Real-life analogy that clicked**: Restaurant selection
- Arepas (known good) vs Chipotle (okay) vs Falafel (unknown)
- How do you maximize satisfaction while staying open to discovery?

### Algorithm 1: ε-Greedy
**Strategy**: Random exploration with probability ε

```python
def epsilon_greedy(Q, epsilon, k):
    """
    ε-Greedy action selection
    
    Args:
        Q: Action values (numpy array of size k)
        epsilon: Exploration probability
        k: Number of actions
    
    Returns:
        Selected action index
    """
    if random.random() < epsilon:
        # Explore: choose random action
        return random.randint(0, k-1)
    else:
        # Exploit: choose action with highest value
        return np.argmax(Q)

def update_action_value(Q, N, action, reward):
    """
    Incremental update of action values
    
    Args:
        Q: Action values
        N: Action counts
        action: Selected action
        reward: Received reward
    """
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]
    
# Example usage
k = 3  # Number of actions
Q = np.zeros(k)  # Action values
N = np.zeros(k)  # Action counts
epsilon = 0.1

# Simulate bandit problem
for t in range(1000):
    action = epsilon_greedy(Q, epsilon, k)
    reward = np.random.normal(0, 1)  # Simulated reward
    update_action_value(Q, N, action, reward)
```

**Limitation**: Wastes time on obviously bad choices

### Algorithm 2: Upper Confidence Bound (UCB)
**Strategy**: Be optimistic about uncertainty

**Running trail analogy**:
- Trail A: Known to be good
- Trail B: Modest but unexplored (high upside potential)
- Trail C: Completely unknown

```python
import numpy as np
import math

def ucb_action_selection(Q, N, t, c=2):
    """
    Upper Confidence Bound action selection
    
    Args:
        Q: Action values
        N: Action counts
        t: Current time step
        c: Confidence parameter
    
    Returns:
        Selected action index
    """
    k = len(Q)
    ucb_values = np.zeros(k)
    
    for a in range(k):
        if N[a] == 0:
            # If action never tried, give it infinite value
            return a
        else:
            confidence = c * math.sqrt(math.log(t) / N[a])
            ucb_values[a] = Q[a] + confidence
    
    return np.argmax(ucb_values)

# Example usage
k = 3  # Number of actions
Q = np.zeros(k)  # Action values
N = np.zeros(k)  # Action counts
c = 2  # Confidence parameter

# Simulate UCB bandit
for t in range(1, 1001):  # Start from t=1
    action = ucb_action_selection(Q, N, t, c)
    reward = np.random.normal(0, 1)  # Simulated reward
    update_action_value(Q, N, action, reward)
```

**Key insight**: Explores systematically based on uncertainty, not randomly

### Algorithm 3: Thompson Sampling
**Strategy**: Sample from your beliefs

**TV show analogy**:
- Seinfeld: Reliable comfort show
- The Office: Few episodes, uncertain potential  
- House: New show, wide uncertainty

```python
import numpy as np
from scipy import stats

def thompson_sampling(alpha, beta):
    """
    Thompson Sampling for Bernoulli bandits
    
    Args:
        alpha: Success counts + 1 (Beta prior parameter)
        beta: Failure counts + 1 (Beta prior parameter)
    
    Returns:
        Selected action index
    """
    k = len(alpha)
    theta_samples = np.zeros(k)
    
    # Sample from Beta distribution for each action
    for a in range(k):
        theta_samples[a] = np.random.beta(alpha[a], beta[a])
    
    return np.argmax(theta_samples)

def thompson_sampling_gaussian(mu, sigma, n):
    """
    Thompson Sampling for Gaussian bandits
    
    Args:
        mu: Mean estimates
        sigma: Standard deviation
        n: Number of samples per action
    
    Returns:
        Selected action index
    """
    k = len(mu)
    theta_samples = np.zeros(k)
    
    # Sample from posterior Normal distribution
    for a in range(k):
        if n[a] == 0:
            theta_samples[a] = np.random.normal(0, 1)  # Prior
        else:
            posterior_std = sigma / np.sqrt(n[a])
            theta_samples[a] = np.random.normal(mu[a], posterior_std)
    
    return np.argmax(theta_samples)

# Example usage (Bernoulli case)
k = 3  # Number of actions
alpha = np.ones(k)  # Prior successes + 1
beta = np.ones(k)   # Prior failures + 1

# Simulate Thompson Sampling
for t in range(1000):
    action = thompson_sampling(alpha, beta)
    reward = np.random.binomial(1, 0.5)  # Bernoulli reward
    
    if reward == 1:
        alpha[action] += 1
    else:
        beta[action] += 1
```

**Beautiful aspect**: Natural exploration-exploitation balance through Bayesian reasoning

### Algorithm 4: LinUCB (Contextual Bandits)
**Extension**: What if context matters?

**Real example**: Yahoo! news recommendations (user features + article features)

```python
import numpy as np

class LinUCB:
    """
    Linear Upper Confidence Bound for Contextual Bandits
    """
    
    def __init__(self, k, d, alpha=1.0):
        """
        Initialize LinUCB
        
        Args:
            k: Number of actions
            d: Dimension of context features
            alpha: Confidence parameter
        """
        self.k = k
        self.d = d
        self.alpha = alpha
        
        # Initialize for each action a
        self.A = [np.identity(d) for _ in range(k)]  # A_a matrices
        self.b = [np.zeros(d) for _ in range(k)]     # b_a vectors
        
    def select_action(self, x_t):
        """
        Select action using LinUCB algorithm
        
        Args:
            x_t: Context feature vector for all actions (k x d matrix)
        
        Returns:
            Selected action index
        """
        p = np.zeros(self.k)
        
        for a in range(self.k):
            # Compute theta_a = A_a^(-1) * b_a
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            
            # Compute confidence bound
            x_a = x_t[a]  # Context for action a
            confidence = self.alpha * np.sqrt(x_a.T.dot(A_inv).dot(x_a))
            
            # Upper confidence bound
            p[a] = theta_a.T.dot(x_a) + confidence
            
        return np.argmax(p)
    
    def update(self, action, x_t, reward):
        """
        Update model parameters
        
        Args:
            action: Selected action
            x_t: Context features
            reward: Received reward
        """
        x_a = x_t[action]
        
        # Update A_a and b_a
        self.A[action] += np.outer(x_a, x_a)
        self.b[action] += reward * x_a

# Example usage
k = 3  # Number of actions
d = 5  # Feature dimension
linucb = LinUCB(k, d, alpha=1.0)

# Simulate contextual bandit
for t in range(1000):
    # Generate context features for each action
    x_t = np.random.randn(k, d)
    
    # Select action
    action = linucb.select_action(x_t)
    
    # Simulate reward (linear model + noise)
    true_theta = np.random.randn(d)
    reward = x_t[action].dot(true_theta) + np.random.normal(0, 0.1)
    
    # Update model
    linucb.update(action, x_t, reward)
```

---

## My Key Takeaways So Far

**Algorithm Progression**:
1. ε-Greedy: Simple but wasteful
2. UCB: Systematic uncertainty-based exploration
3. Thompson Sampling: Principled Bayesian approach
4. LinUCB: Context-aware decisions

**When to Use What**:
- Few actions, simple: ε-Greedy
- Need systematic exploration: UCB
- Want principled uncertainty: Thompson Sampling
- Have contextual info: LinUCB

**Common Patterns I Notice**:
```python
# Action value updates always follow this pattern
Q[action] += (reward - Q[action]) / N[action]

# Confidence bounds appear everywhere
confidence = c * sqrt(log(t) / N[action])

# Bayesian updates for Thompson Sampling
alpha[action] += reward
beta[action] += (1 - reward)
```

# Week 5 - Normalising Healthscore function to the 8 good and 3 bad nutrients

### Load Data
```python
import pandas as pd

def load_data(file_path):
    """Load CSV data as a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def get_actions(data):
    """Return unique food items."""
    return data['sales_item'].unique()

def get_features(data):
    """Return feature matrix (all columns except 'sales_item')."""
    return data

def health_score(row, min_score=-300, max_score=800):
    DV = {
        "elementary": {"Calories":1600,"Protein":19,"Total Carbohydrate":130,
                       "Dietary Fiber":25,"Added Sugars":25,"Total Fat":40,
                       "Saturated Fat":20,"Sodium":1500,"Vitamin D":20,
                       "Calcium":1000,"Iron":10,"Potassium":4700,
                       "Vitamin A":900,"Vitamin C":90},
        "middle": {"Calories":2200,"Protein":34,"Total Carbohydrate":130,
                   "Dietary Fiber":31,"Added Sugars":50,"Total Fat":77,
                   "Saturated Fat":20,"Sodium":2300,"Vitamin D":20,
                   "Calcium":1300,"Iron":18,"Potassium":4700,
                   "Vitamin A":900,"Vitamin C":90},
        "high": {"Calories":2600,"Protein":46,"Total Carbohydrate":130,
                 "Dietary Fiber":38,"Added Sugars":50,"Total Fat":91,
                 "Saturated Fat":20,"Sodium":2300,"Vitamin D":20,
                 "Calcium":1300,"Iron":18,"Potassium":4700,
                 "Vitamin A":900,"Vitamin C":90}
    }

    GOOD = ["Protein","Dietary Fiber","Vitamin D","Calcium",
            "Iron","Potassium","Vitamin A","Vitamin C"]
    BAD = ["Added Sugars","Saturated Fat","Sodium"]

    group = str(row.get("school_group","high")).lower()
    dv = DV.get(group, DV["high"])

    good_score = sum(min(100, (row.get(n,0)/dv[n])*100) for n in GOOD)
    bad_score  = sum(min(100, (row.get(n,0)/dv[n])*100) for n in BAD)

    raw_score = good_score - bad_score
    
    min_score=-300
    max_score=800

    # Normalize to 0-10
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    return round(max(0, min(10, scaled_score)), 1)

data = load_data("data/sales.csv")
data["HealthScore"] = data.apply(health_score, axis=1)
data.to_csv("scored_data.csv", index=False)
print("✅ Health scores calculated and saved to scored_data.csv")

```


# Week 6 Presentation Preparation and Initial results

Debugged the code and Assisted in creating Presentation

# Week 7 finalised Healthscore code adjusting it to schools

```python
def infer_school_group(school_name):
    """Infer school group (elementary, middle, high) from school name"""
    school_lower = str(school_name).lower()
    if 'elementary' in school_lower:
        return 'elementary'
    elif 'middle' in school_lower:
        return 'middle'
    else:
        return 'high'

def health_score(row: pd.Series) -> float:
    """
    Calculate health score for a meal item based on nutritional content
    Returns a score from 0-10 where higher is healthier
    """
    DV = {
        "elementary": {
            "Calories": 2000, "Protein": 19, "Total Carbohydrate": 130, 
            "Dietary Fiber": 28, "Added Sugars": 50, "Total Fat": 78, 
            "Saturated Fat": 22, "Sodium": 1500, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 400, "Vitamin C": 25
        },
        "middle": {
            "Calories": 2600, "Protein": 34, "Total Carbohydrate": 130, 
            "Dietary Fiber": 36.4, "Added Sugars": 65, "Total Fat": 101, 
            "Saturated Fat": 29, "Sodium": 1800, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 600, "Vitamin C": 45
        },
        "high": {
            "Calories": 3200, "Protein": 52, "Total Carbohydrate": 130, 
            "Dietary Fiber": 44.8, "Added Sugars": 80, "Total Fat": 124, 
            "Saturated Fat": 36, "Sodium": 2300, "Vitamin D": 20, 
            "Calcium": 1300, "Iron": 18, "Potassium": 4700, 
            "Vitamin A": 900, "Vitamin C": 75
        }
    }
    
    GOOD = ["Protein", "Dietary Fiber", "Vitamin D", "Calcium", "Iron", 
            "Potassium", "Vitamin A", "Vitamin C"]
    BAD = ["Added Sugars", "Saturated Fat", "Sodium"]

    school_group = str(row.get("school_group", "high")).lower()
    dv = DV["elementary"] if "elementary" in school_group else (
        DV["middle"] if "middle" in school_group else DV["high"]
    )

    good_score, bad_score = 0.0, 0.0
    for n in GOOD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        good_score += min(100, (val/ref)*100)
    for n in BAD:
        val = row.get(n, 0) or 0
        ref = dv.get(n, 1)
        bad_score += min(100, (val/ref)*100)

    raw_score = good_score - bad_score
    return raw_score  # Return raw unscaled score

def scale_health_score(raw_score: float) -> float:
    """
    Scale a raw health score from its natural range (-300 to 800) to 0-10.
    
    Args:
        raw_score: Raw health score from health_score() function
        
    Returns:
        float: Score scaled to 0-10 range with 2 decimal precision
    """
    min_score = -300  # worst-case negative score
    max_score = 800   # best-case score
    
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    scaled_score = max(0, min(10, scaled_score))  # clamp to [0,10]
    return np.round(scaled_score, 2)
```

# Week 8 Finetuning with multiple variants of code 
```python

v1= sales_scaled + lambda_value * health_scaled
v2= a * total_sales + lambda_value * health_z where a= 1- lambda_value 
v3= total_sales + lambda_value * health_scores
────────────────────────────────────────────────────────────────────────────────
Lambda     Total Reward       Oracle Reward      Regret          Regret %
────────────────────────────────────────────────────────────────────────────────
V1- 0.05        150210.63        218389.73        68179.11         31.2%
V2- 0.05       1576056.20       3204577.30      1628521.10         50.8%
V3- 0.05       1663291.65       3377472.54      1714180.89         50.8%
  0.20        156920.20        231677.35        74757.15         32.3%
  0.20       1326647.24       2699165.20      1372517.97         50.8%
  0.20       1673954.61       3390746.16      1716791.55         50.6%
  0.30        166869.95        240547.61        73677.67         30.6%
  0.30       1163077.74       2362226.04      1199148.30         50.8%
  0.30       1684371.08       3399595.25      1715224.16         50.5%
  #0.40        174615.93        249429.22        74813.29         30.0%
  0.40        996252.49       2025297.77      1029045.28         50.8%
  0.40       1697399.90       3408444.33      1711044.43         50.2%
  #0.50        180732.92        258322.84        77589.91         30.0%
  0.50        830359.31       1688388.81       858029.51         50.8%
  0.50       1706269.41       3417293.68      1711024.27         50.1%
  0.80        188588.54        285059.25        96470.71         33.8%
  0.80        188588.54        285059.25        96470.71         33.8%
  0.80        336404.22        678055.76       341651.54         50.4%
  0.80       1733364.48       3443852.86      1710488.38         49.7%

"""the delighted are the best models and the code is available in branch Dinesh"""

```
# Week 9 plot correction and debugginng

```python
# src/plots/linucb_learning_performance_full4.py
"""
LinUCB Learning Performance (VARIANT1, λ=0.3) — 4 panels:
1) Rolling Avg Reward
2) Rolling Avg Regret
3) Rolling Avg Sales (raw)
4) Rolling Avg Health (raw)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- repo path ---
HERE = Path(__file__).resolve().parent
SRC  = HERE if HERE.name == "src" else HERE.parent
sys.path.insert(0, str(SRC))

# --- project imports ---
try:
    from Components.model import (
        LinUCB, load_feature_matrix, load_action_matrix,
        compute_rewards_for_lambda, _build_bandit_tensors
    )
except Exception:
    from model import (
        LinUCB, load_feature_matrix, load_action_matrix,
        compute_rewards_for_lambda, _build_bandit_tensors
    )

DATA_DIR = SRC / "data"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"
ACTION_MATRIX  = DATA_DIR / "action_matrix.csv"
MERGED_DATA    = DATA_DIR / "data_healthscore_mapped.csv"
TIMESLOTS      = DATA_DIR / "time_slot_mapping.csv"

RESULTS_DIR = SRC / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.80
ROLL_W = 50

# ---------- helpers ----------
def rolling_mean(x, w):
    s = pd.Series(x)
    return s.expanding().mean().to_numpy()



def build_time_item_lookup(merged_csv, timeslots_csv):
    """(time_slot_id, item_name) -> (raw_sales_sum, raw_health_median)"""
    ts = pd.read_csv(timeslots_csv, low_memory=False)
    key2id = dict(zip(
        zip(ts["date"].astype(str), ts["school_name"].astype(str), ts["time_of_day"].astype(str)),
        ts["time_slot_id"].astype(int)
    ))
    m = pd.read_csv(merged_csv, low_memory=False)
    for c in ["date","school_name","time_of_day","description"]:
        m[c] = m[c].astype(str).str.strip()
    m["time_slot_key"] = list(zip(m["date"], m["school_name"], m["time_of_day"]))
    m["time_slot_id"]  = m["time_slot_key"].map(key2id)
    m = m.dropna(subset=["time_slot_id"]).copy()
    m["time_slot_id"] = m["time_slot_id"].astype(int)

    agg = m.groupby(["time_slot_id","description"], as_index=False).agg(
        sales=("total","sum"),
        health=("HealthScore","median"),
    )
    return {
        (int(r.time_slot_id), str(r.description)): (float(r.sales), float(r.health))
        for r in agg.itertuples(index=False)
    }

def make_meta_item_map(metadata_df):
    """Map (t, arm) -> item_name for fast lookup."""
    return {
        (int(r.time_slot_id), int(r.item_idx)): str(r.item)
        for r in metadata_df.itertuples(index=False)
    }

# ---------- training + tracking ----------
def train_linucb_and_track(
    feature_array, action_matrix, metadata_df, rewards, alpha=1.0, ridge=1.0
):
    data_TAD, rewards_TA, mask_TA = _build_bandit_tensors(
        feature_array, metadata_df, rewards, action_matrix
    )
    T, A, d = data_TAD.shape
    bandit = LinUCB(d=d, n_arms=A, alpha=alpha, l2=ridge, seed=42)

    # lookups for raw sales/health
    meta_item = make_meta_item_map(metadata_df)
    time_item_lookup = build_time_item_lookup(MERGED_DATA, TIMESLOTS)

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

        # raw sales/health for the actually chosen (t, item_name)
        item_name = meta_item.get((t, a))
        if item_name is not None and (t, item_name) in time_item_lookup:
            sales_raw, health_raw = time_item_lookup[(t, item_name)]
        else:
            sales_raw, health_raw = 0.0, 5.0

        rows.append({
            "t": t,
            "reward": r,
            "oracle": oracle,
            "regret": regret,
            "sales": sales_raw,
            "health": health_raw,
        })

        bandit.update_arm(a, feats[a], r)

        if (t+1) % 2000 == 0:
            print(f"[t={t+1}/{T}] avg reward last2k = {np.mean([rr['reward'] for rr in rows[-2000:]]):.3f}")

    df = pd.DataFrame(rows)

    # Rolling avgs (short-term)
    df["roll_reward"] = rolling_mean(df["reward"].to_numpy(), ROLL_W)
    df["roll_regret"] = rolling_mean(df["regret"].to_numpy(), ROLL_W)
    df["roll_sales"]  = rolling_mean(df["sales"].to_numpy(),  ROLL_W)
    df["roll_health"] = rolling_mean(df["health"].to_numpy(), ROLL_W)
    return df

# ---------- plotting (8 panels) ----------

def plot_rolling4(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LinUCB Learning Performance (λ={LAMBDA})", fontsize=16, fontweight="bold", y=0.995)

    # 1) Rolling Avg Reward
    ax = axes[0,0]
    ax.plot(df["t"], df["roll_reward"], lw=2.2, color="tab:blue")
    ax.set_title(f"Rolling Avg Reward (window={ROLL_W})"); ax.set_ylabel("Avg Reward"); ax.grid(True, alpha=0.3)

    # 2) Rolling Avg Regret
    ax = axes[0,1]
    ax.plot(df["t"], df["roll_regret"], lw=2.2, color="tab:red")
    ax.axhline(0, ls="--", lw=1.2, color="tab:green", alpha=0.6)
    ax.set_title(f"Rolling Avg Regret (window={ROLL_W})"); ax.set_ylabel("Avg Regret/Step"); ax.grid(True, alpha=0.3)

    # 3) Rolling Avg Sales (raw)
    ax = axes[1,0]
    ax.plot(df["t"], df["roll_sales"], lw=2.2, color="tab:purple")
    ax.set_title(f"Rolling Avg Sales (raw, window={ROLL_W})"); ax.set_ylabel("Avg Sales"); ax.grid(True, alpha=0.3)

    # 4) Rolling Avg Health (raw)
    ax = axes[1,1]
    ax.plot(df["t"], df["roll_health"], lw=2.2, color="tab:orange")
    ax.set_title(f"Rolling Avg Health (raw, window={ROLL_W})"); ax.set_ylabel("Avg Health"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / f"linucb_learning_performance_lambda_{LAMBDA}_rolling4.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"✓ Saved rolling 4-panel plot: {out_png}")
    plt.show()


def main():
    print("="*72)
    print("LINUCB LEARNING PERFORMANCE — ROLLING 4 PANELS")
    print("="*72)

    X, meta, _ = load_feature_matrix(str(FEATURE_MATRIX))
    A = load_action_matrix(str(ACTION_MATRIX))
    print(f"Feature matrix: {X.shape} | Action matrix: {A.shape}")

    rewards_vec = compute_rewards_for_lambda(
        lambda_value=LAMBDA,
        feature_matrix_file=str(FEATURE_MATRIX),
        merged_data_file=str(MERGED_DATA),
        time_slot_mapping_file=str(TIMESLOTS),
    )

    df = train_linucb_and_track(X, A, meta, rewards_vec, alpha=1.0, ridge=1.0)

    # save data + plot
    out_csv = RESULTS_DIR / f"linucb_performance_lambda_{LAMBDA}_rolling4.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved performance CSV: {out_csv}")

    plot_rolling4(df)

if __name__ == "__main__":
    main()
```


# Week 10 introductory Paper

Overleaf link : https://www.overleaf.com/read/psmsbpntymkx#befa8d

