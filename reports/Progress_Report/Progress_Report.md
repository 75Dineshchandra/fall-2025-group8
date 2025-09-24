Note: Use Markdown Cheat Sheet download in the directory as needed.

- Useful links
  - https://github.com/im-luka/markdown-cheatsheet

---
## Date: Week 1 - Sep 10, 2025 
- Topics of discussion  
    - Studied Reinforcement Learning framework with emphasis on Contextual Multi-Armed Bandits (CMAB).  
    - Compared exploration–exploitation algorithms: ε-Greedy, LinUCB, Thompson Sampling.  
    - Defined reward structure combining popularity and health score with weighting λ.  
    - Outlined cradle-to-launch roadmap for data ingestion → feature engineering → bandit modeling → evaluation.  
    - Reviewed Fairfax County Public Schools (FCPS) dataset schema and structure. 

- Action Items:  

* [x] Created repository (`fall-2025-group8`) and set up folder structure.  
* [x] Added `amir-capstone-jafari` as collaborator.  
* [x] Created GitHub Project Board with Kanban workflow.  
* [ ] Prepare Week 2 EDA plan.  
* [ ] Draft health score formula baseline.  
* [ ] Implemented load_data() function (utils/env.py)  

## Key Learnings This Week 

 - Reward Definition 

 [reward = sales_norm+ lambda {health_score_norm} ]

. sales_norm = normalized participation (per school/time to avoid scale bias)  
. health_score_norm = 0–1 score combining fiber, protein, sugar, sat fat, sodium  
. λ = health–popularity tradeoff hyperparameter (swept in benchmarking) 
 

- Algorithms Reviewed 

ε-Greedy: baseline, simple, uniform exploration; ε decay schedules studied 
LinUCB: linear payoff model with uncertainty bonus (α as tunable hyperparameter) 
Thompson Sampling: Bayesian posterior sampling, strong benchmark candidate 
 
 
-Roadmap Week-by-Week (Condensed) 

W1: RL/CMAB theory, reward definition, roadmap drafting  
W2–3: Data ingestion, preprocessing, env.py setup 
W4–6: Implement ε-Greedy & LinUCB, offline replay 
W7: Offline Policy Evaluation (IPS, DR) 
W8: Benchmark λ, ε, α across policies 
W9–12: Research paper + tool development 
 

- References 

FCPS Nutrition Services. (2025). School Meals Data (internal Box link). 

1. Introduction—“Why Should I Learn Reinforcement Learning?” from DATS 6450 — Reinforcement Learning course by Tyler Wallett.  

2. FCPS Nutrition Services. (2025). School Meals Data 

3. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. In Proceedings of the 19th International Conference on World Wide Web (WWW), ACM. 

 

 
---

## Date: Week 2 - Sep 11 – Sep 17, 2025  
- Topics of discussion  
    - Implemented `load_data()` in `utils/env.py` to load preprocessed FCPS dataset.  
    - Built EDA workflows using a Jupyter notebook (`src/data/eda.ipynb`) and Python script (`src/data/eda.py`).  
    - Added mapping utilities: `src/data/mapping.py` (sales → nutrition) and `src/data/4.invert_mapping.py` (nutrition → sales).  
    - Explored data distributions: top meals, participation trends by weekday/time, variability across schools.  
    - Identified missing values, categorical encoding requirements, normalization strategies.  

- Action Items:  

* [x] Implemented `load_data()` function (`utils/env.py`).  
* [x] Created EDA notebook (`src/data/eda.ipynb`) and script (`src/data/eda.py`).  
* [x] Added mapping scripts (`src/data/mapping.py`, `src/data/4.invert_mapping.py`).  
* [x] Generated plots for top-selling meals and participation trends.  
* [x] Finalize preprocessing: missing values, categorical encoding.  
* [ ] Save cleaned dataset as `src/data/fcps_sales_clean.csv`.  

---


- **_Add Equation_**  
  

- **_Add Python Code_**  

```python
# utils/env.py
import pandas as pd

def load_data(filepath: str = "src/data/fcps_sales_clean.csv") -> pd.DataFrame:
    """
    Load preprocessed FCPS sales CSV.
    Returns DataFrame with columns:
    ['date', 'school_id', 'time_of_day', 'item_id', 'meals_sold']
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df
# src/data/4.invert_mapping.py
import pandas as pd, numpy as np, re
from pathlib import Path

sales_p = "src/data/sales.csv"
nutr_p  = "src/data/nutrition_items.csv"
outdir  = Path("src/data/")

sales = pd.read_csv(sales_p)
nutr  = pd.read_csv(nutr_p)
nutr.columns = nutr.columns.str.strip()
nutr.drop(columns=[c for c in nutr.columns if c.endswith("_Unit")],
          inplace=True, errors="ignore")
sales["description"] = sales["description"].astype(str).str.strip()

def norm(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    return re.sub("\s+", " ", s).strip()

CATEGORY_RULES = {}


Add flow chart
graph TD;
    A[Sales.csv + Nutrition.csv] --> B[load_data()]
    B --> C[get_actions()]
    B --> D[get_features]
    C --> E[Bandit Model: ε-Greedy, LinUCB]
    D --> E
    E --> F[Reward Function r_t]
---
## Date: Week 3 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 4 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 5 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 6 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 7 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
----
## Date: Week 8 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---

## Date: Week 9 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 10 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 11 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 12 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 13 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
## Date: Week 14 - Month Day Year 
- Topics of discussion
    - Item1
    - Item2
    - Item3

- Action Items:

* [ ] Action Item 1
* [ ] Action Item 2
* [ ] Action Item 3
* [ ] Action Item 4
* [ ] Action Item 5
---
- **_Add Images and Diagrams Using Excalidraw_**
  - Just draw it and then copy as png paster in the editor
![img_2.png](img_2.png)


- **_Add Equation_**
  - $e^{\pi i} + 1 = 0$



- **_Add Pyhton Code_**

```
import numpy as np
a = np.array()
```

- **_Add Tables as needed._**

| Checkbox Experiments | checked header | crossed header |
| ---------------------|:--------------:|:--------------:|
| checkbox             |  &check; Row   |  &cross; row   |


- **_Add Tables as needed._**


|checked|unchecked|crossed|
|---|---|---|
|&check;|_|&cross;|
|&#x2611;|&#x2610;|&#x2612;|


- **_Add Tables as needed._**

| Selection |        |
| --------- | ------ |
| &#x2610;  |

| Selection |        |
| --------- | ------ |
| &#x2611; |

- **_Create Links as needed_**
  - [link text](full url minus the en-us locale)

- **_Add Geo Json_**

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [-90,35],
              [-90,30],
              [-85,30],
              [-85,35],
              [-90,35]
          ]
        ]
      }
    }
  ]
}
```

- **_Add flow chart_**


```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```