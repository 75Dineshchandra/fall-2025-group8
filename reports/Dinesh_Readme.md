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

    # Normalize to 0-10
    scaled_score = 10 * (raw_score - min_score) / (max_score - min_score)
    return round(max(0, min(10, scaled_score)), 1)

data = load_data("data/sales.csv")
data["HealthScore"] = data.apply(health_score, axis=1)
data.to_csv("scored_data.csv", index=False)
print("✅ Health scores calculated and saved to scored_data.csv")
