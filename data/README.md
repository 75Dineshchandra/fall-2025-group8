# Health-Aware School Meal Recommendations with Contextual Bandits

**Capstone Project — George Washington University, Fall 2025**  
Proposed by: Tyler Wallett  
Advisor: Amir Jafari  

---

## Objective

The goal of this project is to develop a **free and open-source analysis and recommendation tool** that helps **school nutritionists, cafeteria staff, and researchers** optimize school meal offerings for both **student preference** and **healthiness**.

We leverage **Contextual Multi-Armed Bandit (CMAB)** algorithms to recommend meals that balance **popularity** and **nutritional value**.  

The project is affiliated with **Fairfax County Public Schools (FCPS)**, which provides the historical meal sales dataset used here.

---

##  Dataset Overview

### Files
| **Filename**  | **Description**                                                                 |
| ------------- | ------------------------------------------------------------------------------- |
| `sales.csv`   | Core dataset of school meal transactions, with breakdowns by pricing and program |
| (future) `meals.csv` | Optional metadata on meal nutrition (calories, sodium, protein, etc.)     |
| (future) `schools.csv` | School-level metadata (location, type, demographics)                   |

---

##  `sales.csv` — Data Dictionary

### Column Definitions

| Column                | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `time_of_day`         | Meal program (e.g., *breakfast*, *lunch*)                     |
| `school_code`         | Unique school identifier (numeric)                            |
| `school_name`         | Human-readable school name                                    |
| `date`                | Date of service (MM/DD/YYYY format)                           |
| `item`                | Unique item/meal identifier (numeric code)                    |
| `description`         | Human-readable meal/food description                          |
| `total`               | Total number of meals served for this item/date/school        |
| `free_meals`          | Meals served under the free lunch program                     |
| `reduced_price_meals` | Meals served under reduced-price program                      |
| `full_price_meals`    | Meals served under full-price program                         |
| `adults`              | Meals sold to adults                                          |
| `alac_student`        | À la carte meals sold to students                             |
| `alac_adult`          | À la carte meals sold to adults                               |
| `earned_student`      | Revenue from student meals                                    |
| `earned_adult`        | Revenue from adult meals                                      |
| `earned_alac_student` | Revenue from student à la carte items                         |
| `earned_alac_adult`   | Revenue from adult à la carte items                           |
| `adj_alac`            | Adjustments made to à la carte counts (post-sales correction) |
| `adj_meal`            | Adjustments made to reimbursable meal counts                  |

---

### Sample (First 5 Rows)

```text
time_of_day school_code school_name            date       item  description           total free_meals reduced_price_meals full_price_meals adults alac_student alac_adult earned_student earned_adult earned_alac_student earned_alac_adult adj_alac adj_meal
breakfast   17          COLVIN_RUN_ELEMENTARY  03/03/2025 1146  CEREAL MEAL           19    3          0                   16               0      0            0          0              0            0                   0                  0        0
breakfast   17          COLVIN_RUN_ELEMENTARY  03/03/2025 1164  BAGEL W/CREAM CHEESE  6     0          0                   6                0      0            0          0              0            0                   0                  0        0
breakfast   17          COLVIN_RUN_ELEMENTARY  03/03/2025 1310  ALC BREAKFAST ENTREE  11    0          0                   0                0      11           0          0              0            0                   0                  0        0
breakfast   17          COLVIN_RUN_ELEMENTARY  03/03/2025 151   CEREAL/ NO MILK       13    0          0                   0                0      13           0          0              0            0                   0                  0        0
breakfast   17          COLVIN_RUN_ELEMENTARY  03/03/2025 188   $1.00 WATER 16.9oz    1     0          0                   0                0      1            0          0              0            0                   0                  0        0
