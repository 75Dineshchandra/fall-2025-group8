# Health-Aware School Meal Recommendations with Contextual Bandits

A data-driven recommendation system that uses Contextual Multi-Armed Bandit (CMAB) algorithms to optimize school meal offerings by balancing student preference (popularity) and nutritional value. This project is developed in collaboration with Fairfax County Public Schools (FCPS) and provides an open-source tool for school nutritionists and researchers.

##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts and Outputs](#scripts-and-outputs)
- [Usage Examples](#usage-examples)
- [Data Files](#data-files)
- [Results](#results)
- [Contributors](#contributors)

##  Overview

This project implements a **LinUCB (Linear Upper Confidence Bound)** contextual bandit algorithm to recommend school meals that:
- Maximize student participation (sales/popularity)
- Maintain nutritional quality (health scores)
- Adapt to contextual features (school, time of day, day of week)

The reward function balances these objectives:
```
reward = sales_scaled + λ * health_scaled
```
where `λ` (lambda) controls the trade-off between popularity and healthiness.

##  Project Structure

```
fall-2025-group8/
├── src/                          # Source code
│   ├── main.py                  # Main recommendation script
│   ├── Components/              # Core components
│   │   ├── env.py              # Environment and data processing
│   │   ├── model.py            # LinUCB model implementation
│   │   ├── utils.py            # Utility functions
│   │   ├── plot2.py            # Visualization scripts
│   │   └── create_figure3.py   # Figure generation
│   ├── tests/                   # Test scripts
│   │   └── train_eval.py       # Training and evaluation
│   └── fcps_dataextractor/     # Data extraction tools
├── data/                        # Data files
│   ├── *.csv                   # Processed data files
│   ├── *.pdf                   # Reference documents
│   └── results/                # Model outputs and results
│       ├── model_lambda_*.joblib  # Trained models
│       ├── *.png               # Visualization figures
│       └── *.csv               # Performance metrics
├── reports/                     # Project reports
├── research_paper/             # Research paper files
├── demo/                       # Demo materials
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/75Dineshchandra/fall-2025-group8.git
cd fall-2025-group8
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

This will install all required packages:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `joblib` - Model serialization
- `matplotlib` - Plotting and visualization
- `requests` - HTTP requests for data extraction

4. **Verify installation:**
```bash
python3 -c "import pandas, numpy, sklearn, joblib, matplotlib, requests; print('All packages installed successfully!')"
```

5. **Ensure data files are in the `data/` directory** (see [Data Files](#data-files) section)

##  Quick Start

### 1. Build Data Matrices
```bash
python3 src/Components/env.py
```

### 2. Train Models
```bash
python3 src/Components/model.py
```

### 3. Generate Recommendations
```bash
python3 src/main.py
```

##  Scripts and Outputs

### 1. `src/Components/env.py` - Environment Builder

**Purpose**: Transforms raw FCPS sales data into structured matrices for machine learning.

**What it does**:
- Loads raw sales data from `data/data_healthscore_mapped.csv`
- Builds item mapping (meal names to indices)
- Creates time slot mapping (time periods to IDs)
- Generates action matrix (which items were available at each time step)
- Builds feature matrix (nutritional features for each item)

**Output**:
```
======================================================================
BUILDING FCPS DATA MATRICES
======================================================================
[1/5] Loading raw sales data.
Loaded 224536 rows of raw sales data

[2/5] Building item mapping.
Item mapping already exists, skipping rebuild

[3/5] Building time slot mapping.
Time slot mapping already exists, skipping rebuild

[4/5] Building action matrix.
Action matrix already exists, skipping rebuild

[5/5] Building feature matrix.
Feature matrix already exists, skipping rebuild

Added time_slot_id to 224536/224536 rows -> saved to data/with_timeslot.csv
```

**Generated Files**:
- `data/item_mapping.csv` - Maps meal names to numeric indices
- `data/time_slot_mapping.csv` - Maps time periods to slot IDs
- `data/action_matrix.csv` - Binary matrix of available items per time step
- `data/feature_matrix.csv` - Nutritional features (18 dimensions) for each item
- `data/with_timeslot.csv` - Sales data with time slot IDs added

**Configuration**:
- Set `overwrite_existing = True` in `env.py` to rebuild all matrices from scratch

---

### 2. `src/Components/model.py` - Model Trainer

**Purpose**: Trains LinUCB contextual bandit models with different lambda (λ) values to find optimal health-popularity balance.

**What it does**:
- Loads feature and action matrices
- Computes rewards using Variant1 formula: `reward = sales_scaled + λ * health_scaled`
- Trains LinUCB models for multiple lambda values (0.2, 0.3, 0.4, 0.6, 0.8)
- Evaluates performance (total reward, oracle reward, regret)
- Saves trained models to disk

**Output**:
```
======================================================================
LINUCB TRAINING WITH VARIANT1 REWARDS
======================================================================
reward = sales_scaled + λ * health_scaled
Both metrics scaled to [0, 10] for fair comparison

[1/3] Loading data...
Loaded 224536 feature samples
Feature array shape: (224536, 18)
Action matrix shape: (21656, 160)

[2/3] Training models with different lambda values...

--- Lambda = 0.2 ---
  Computing VARIANT1 rewards (λ=0.2)...
  Scaling sales per time-slot to [0, 10]...
  Re-normalizing health scores to full [0, 10] range...
  Rewards computed: mean=3.51, std=3.33
Training LinUCB with lambda = 0.2...
Model saved to data/results/model_lambda_0.20.joblib

[... similar output for other lambda values ...]

[3/3] Results Summary
======================================================================
Lambda     Total Reward    Oracle Reward       Regret      Regret %
----------------------------------------------------------------------
  0.20        161891.02       236917.25     75026.23        31.7%
  0.30        180809.28       248626.74     67817.46        27.3%
  0.40        190248.71       260521.28     70272.58        27.0%
  0.60        194638.84       284887.49     90248.65        31.7%
  0.80         222064.53       310090.76     88026.23        28.4%

Best lambda: 0.30
Regret: 67817.46 (27.3%)

Training complete! 
```

**Generated Files**:
- `data/results/model_lambda_0.20.joblib` - Trained model with λ=0.2
- `data/results/model_lambda_0.30.joblib` - Trained model with λ=0.3 (optimal)
- `data/results/model_lambda_0.40.joblib` - Trained model with λ=0.4
- `data/results/model_lambda_0.60.joblib` - Trained model with λ=0.6
- `data/results/model_lambda_0.80.joblib` - Trained model with λ=0.8

**Key Metrics**:
- **Total Reward**: Cumulative reward achieved by the model
- **Oracle Reward**: Maximum possible reward (best item selected at each step)
- **Regret**: Difference between oracle and model performance
- **Regret %**: Percentage regret (lower is better)

---

### 3. `src/main.py` - Recommendation Generator

**Purpose**: Generates meal recommendations for a specific date, school, and meal time using a trained model.

**What it does**:
- Loads environment data (merged data, feature matrix, item mapping)
- Loads a trained LinUCB model
- Generates top-k recommendations based on contextual features
- Displays recommendations with sales, health scores, and model confidence

**Output**:
```
Loading merged data from: data/data_healthscore_mapped.csv
Loading feature matrix from: data/feature_matrix.csv
Loading item mapping from: data/item_mapping.csv
Environment data loaded into env module globals.
Getting SMART recommendations using trained model...
Date: 2025-11-25, School: HERNDON_HIGH, Meal: lunch

Model loaded from data/results/model_lambda_0.30.joblib
Found 50 typically available items

TOP RECOMMENDATIONS - Using Trained Model (Optimal Balance)
================================================================================
1. COOKIE
   Sales: 97 (Moderate )
   Health: 5.3
   Model Score: 8.21 (confidence)

2. CHICKEN TENDERS SECONDARY
   Sales: 252 (VERY POPULAR )
   Health: 4.4
   Model Score: 8.08 (confidence)

3. CHEESE PIZZA SECONDARY
   Sales: 64 (Moderate )
   Health: 4.9
   Model Score: 7.95 (confidence)

4. PBJ POWER PACK SECONDARY
   Sales: 94 (Moderate )
   Health: 4.9
   Model Score: 7.14 (confidence)

5. SPICY CHICKEN ON BUN SECONDARY
   Sales: 224 (VERY POPULAR )
   Health: 4.3
   Model Score: 7.11 (confidence)
```

**Configuration**:
Edit `src/main.py` to change:
- `target_date`: Date for recommendations (format: 'YYYY-MM-DD')
- `school`: School name (e.g., "HERNDON_HIGH")
- `meal_time`: Meal period ("breakfast", "lunch", etc.)
- `model_path`: Path to trained model file
- `top_k`: Number of recommendations to generate


##  Data Files

### Required Input Files (in `data/` directory):
- `data_healthscore_mapped.csv` - Raw sales data with health scores
- `sales.csv` - Historical sales data
- `data_sales_nutrition.csv` - Nutrition information for items
- `rewards.csv` - Precomputed reward values (optional)

### Generated Data Files:
- `item_mapping.csv` - Meal name to index mapping
- `time_slot_mapping.csv` - Time period to slot ID mapping
- `action_matrix.csv` - Available items matrix
- `feature_matrix.csv` - Nutritional feature matrix (18 features)
- `with_timeslot.csv` - Sales data with time slot IDs

### Reference Documents:
- `CME_FDA-AAP_KeyNutrients&YourHealth_March2023.pdf`
- `Dietary_Guidelines_for_Americans_2020-2025.pdf`
- `InteractiveNFL_TotalFat_October2021.pdf`

---

##  Results

### Model Performance

The optimal lambda value (λ=0.30) achieves:
- **27.3% regret** - Close to oracle performance
- **Balanced approach** - Maintains both popularity and health
- **85.7% higher reward** than health-first baseline
- **163% higher sales** than health-first baseline

### Output Files Location

All results are saved in `data/results/`:
- **Trained Models**: `model_lambda_*.joblib`
- **Performance Metrics**: `*_performance_*.csv`
- **Visualizations**: `*.png` figures
- **Ablation Results**: `ablation_results.json`

---

##  Additional Scripts

### `src/tests/train_eval.py`
Training and evaluation script with detailed metrics and ablation studies.

### `src/Components/plot2_simplified.py` & `src/Components/plot2.py`
Generate visualization figures comparing model performance across different lambda values.

### `src/Components/create_figure3.py`
Creates Figure 3: Continuous Regret Calculation visualization.

---

##  Contributors

- **Ganesh Kumar Boini** - [g.boini@gwu.edu](mailto:g.boini@gwu.edu)
- **Dinesh Chandra Gaddam** - [dineshchandra.gaddam@gwmail.gwu.edu](mailto:dineshchandra.gaddam@gwmail.gwu.edu)
- **Sirisha Ginnu** - [sirisha.ginnu@gwmail.gwu.edu](mailto:sirisha.ginnu@gwmail.gwu.edu)

**Advisor**: Amir Jafari  , Tyler Wallett
**Institution**: The George Washington University, Washington DC  
**Program**: Data Science Program



##  Related Resources

- Research Paper: See `research_paper/` directory
- Presentations: See `presentation/` directory
- Progress Reports: See `reports/` directory

---

##  Notes

- All scripts should be run from the repository root directory
- Data files must be present in `data/` directory before running scripts
- Models are saved in `data/results/` directory
- The system uses 18 nutritional features and supports 160 meal items
- Contextual features include: school, time of day, day of week

---

