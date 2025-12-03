# Ganesh Kumar Boini – Personal Weekly Report

**Project:** Health-Aware Meal Recommendation using Contextual Multi-Armed Bandits (CMAB)  
**Duration:** September 10 – December 3, 2025  

---

## Date: Week 1 – September 10 – September 17, 2025

### Topics of Discussion
- Understood the project problem
- Studied Contextual Multi-Armed Bandits and LinUCB
- Reviewed course rubric
- Explored FCPS dataset

### Action Items
- Reviewed problem statement  
- Studied LinUCB  
- Reviewed grading rubric  
- Explored FCPS data structure  

---

## Date: Week 2 – September 17 – September 24, 2025

### Topics of Discussion
- Created pipeline design
- Added dummy `env.py`
- Pushed code to repository
- Continued RL study

### Code – Initial Environment Stub

```python
# env.py (Week 2 stub)
def main():
    print("Initializing FCPS environment...")
    print("Placeholder for data loading, mapping, and matrix creation.")
```

### Action Items
- Added `env.py`
- Committed code
- Validated structure

---

## Date: Week 3 – September 24 – October 1, 2025  
### Nutrition Ingestion

### Code – API Request Layer

```python
def get_menu_data(building_id, building_name, start_date, end_date):
    params = {
        "buildingId": building_id,
        "districtId": DISTRICT_ID,
        "startDate": start_date,
        "endDate": end_date
    }
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        print(f"HTTP {response.status_code} for {building_name}")
    except requests.exceptions.Timeout:
        print(f"Timeout for {building_name}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error for {building_name}")
    except requests.exceptions.RequestException as e:
        print(f"Request error for {building_name}: {e}")
    return None
```

### Code – JSON Flattening

```python
def extract_nutrition_data(menu_data, building_id, building_name, date_range_info):
    records = []
    for session in menu_data.get('FamilyMenuSessions', []):
        for plan in session.get('MenuPlans', []):
            for day in plan.get('Days', []):
                for meal in day.get('MenuMeals', []):
                    for category in meal.get('RecipeCategories', []):
                        for recipe in category.get('Recipes', []):
                            records.append(create_nutrition_record(
                                recipe, building_id, building_name, date_range_info
                            ))
    return records
```

### Action Items
- Integrated LINQ API
- Parsed JSON
- Created nutrition dataset

---

## Date: Week 4 – October 1 – October 8, 2025  
### Action & Feature Matrices

### Code – Action Matrix

```python
def build_action_matrix(merged, item_col="description", strict=False, mapping_json="item_to_idx.json"):
    if "time_slot_id" not in merged.columns:
        raise KeyError("Required column 'time_slot_id' not found.")

    try:
        with open(mapping_json, "r") as f:
            item_to_idx = json.load(f)
    except FileNotFoundError:
        item_to_idx = {}

    for item in merged[item_col].astype(str).unique():
        if item not in item_to_idx:
            item_to_idx[item] = len(item_to_idx)

    all_items = [None] * len(item_to_idx)
    for item, idx in item_to_idx.items():
        all_items[idx] = item

    action_matrix = np.zeros((merged["time_slot_id"].max() + 1, len(all_items)), dtype=int)
    grouped = merged.groupby("time_slot_id")[item_col].unique()
    for t, items in grouped.items():
        for it in items:
            action_matrix[int(t), item_to_idx[it]] = 1

    return action_matrix, all_items, item_to_idx
```

### Code – Feature Matrix

```python
def build_feature_matrix(df):
    df["t"] = df["time_slot_id"]

    nutrient_cols = ["Calories", "Protein", "Total Sugars", "Sodium"]
    X = []

    for col in nutrient_cols:
        v = pd.to_numeric(df[col], errors="coerce")
        v = v.fillna(v.median())
        z = (v - v.mean()) / (v.std() + 1e-8)
        X.append(z)

    return np.vstack(X).T
```

### Action Items
- Built matrices
- Normalized features
- Added grouping

---

## Date: Week 5 – October 8 – October 15, 2025  
### Health Scoring & Utils

### Code – Health Score

```python
def compute_health_score(row):
    positive = ["Protein", "Dietary Fiber", "Vitamin A"]
    negative = ["Total Sugars", "Sodium"]

    pos = sum(float(row.get(p, 0) or 0) for p in positive)
    neg = sum(float(row.get(n, 0) or 0) for n in negative)

    return max(pos - neg, 0)
```

### Action Items
- Health calculations
- Utility cleanup

---

## Date: Week 6 – October 15 – October 22, 2025  
### Random Baseline

### Code

```python
def run_random_baseline(metadata_df, rewards, seed=42):
    rng = np.random.default_rng(seed)
    rows_by_slot = metadata_df.groupby("time_slot_id").groups

    total_reward, oracle_reward = 0, 0

    for slot in rows_by_slot.values():
        slot_rewards = rewards[slot]
        total_reward += rewards[rng.choice(slot)]
        oracle_reward += np.max(slot_rewards)

    return {
        "reward": total_reward,
        "oracle": oracle_reward,
        "regret": oracle_reward - total_reward
    }
```

### Action Items
- Random policy created

---

## Date: Week 7 – October 22 – October 29, 2025  
### Health-First Baseline

### Code

```python
def run_health_first_eval(metadata_df, health_scores, rewards):
    rows_by_slot = metadata_df.groupby("time_slot_id").groups

    total, oracle = 0, 0
    for slot in rows_by_slot.values():
        idx = np.array(slot)
        order = np.lexsort((-rewards[idx], -health_scores[idx]))
        best = idx[order[0]]
        total += rewards[best]
        oracle += np.max(rewards[idx])

    return {"reward": total, "oracle": oracle, "regret": oracle-total}
```

### Action Items
- Implemented health-first rule

---

## Date: Week 8 – October 29 – November 5, 2025  
### Midterm

- Slides prepared
- Architecture presented

---

## Date: Week 9 – November 5 – November 12, 2025  
### Ablation

### Code

```python
fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
for frac in fractions:
    sub_df = metadata_df.sample(frac=frac, random_state=42)
    results = train_linucb(sub_df)
    print(frac, results)
```

---

## Date: Week 10 – November 12 – November 19, 2025

- Health-First tie-break improvements

---

## Date: Week 11 – November 19 – November 26, 2025

- Updated research paper

---

## Date: Week 12 – November 26 – December 3, 2025

- Final presentation
- Final video
- Final paper

---

## Final Summary

| Week | Contribution |
|------|-------------|
| 1 | Theory |
| 2 | Env |
| 3 | Nutrition |
| 4 | Matrices |
| 5 | Health scoring |
| 6 | Random |
| 7 | Health-first |
| 8 | Midterm |
| 9 | Ablation |
|10 | Refinement |
|11 | Paper |
|12 | Final |

---
