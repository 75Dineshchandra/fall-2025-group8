

#%%

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ---------- PATHS ----------
script_dir = Path(__file__).parent
sales_path = script_dir / "sales.csv"
nutr_path = script_dir / "nutrition items.csv"

# ---------- LOAD WITH PROPER COLUMN HANDLING ----------
# Load sales data with dtype specification to handle mixed types
sales = pd.read_csv(sales_path, low_memory=False)

# Check what column contains the item descriptions
# Common column names for food items:
possible_desc_columns = ['description', 'Description', 'DESCRIPTION', 'item', 'Item', 'ITEM', 
                        'product', 'Product', 'PRODUCT', 'name', 'Name', 'NAME', 'menu_item', 'Menu_Item']

# Find which column exists
desc_column = None
for col in possible_desc_columns:
    if col in sales.columns:
        desc_column = col
        break

if desc_column is None:
    print(f"Available columns in sales data: {list(sales.columns)}")
    raise KeyError("Could not find description column. Please specify which column contains food item names.")

print(f"Using column '{desc_column}' for food item descriptions")

# Load nutrition data
nutr = pd.read_csv(nutr_path)

# ---------- CLEAN NUTRITION HEADERS / DROP UNIT COLUMNS ----------
nutr.columns = nutr.columns.str.strip()
nutr = nutr.drop(columns=[c for c in nutr.columns if c.endswith("_Unit")], errors="ignore")

# ---------- NORMALIZATION ----------
def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\bw\/\b", " with ", s)
    s = re.sub(r"\/", " ", s)
    s = re.sub(r"[^a-z0-9\s%+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Use the correct column name for descriptions
sales[desc_column] = sales[desc_column].astype(str).fillna("").str.strip()

# ---------- CATEGORY RULES ----------
CATEGORY_RULES = {
    "cereal": ["cereal","cheerios","chex","toast crunch","kix","krispies","corn flakes"],
    "bagel": ["bagel"],
    "cream_cheese": ["cream cheese"],
    "parfait": ["parfait"],
    "milk": ["milk","1%","one percent","fat free"],
    "juice": ["juice"],
    "fruit": ["apple","banana","orange","mandarin","peach","pineapple","grape","pear"],
    "biscuit": ["biscuit"],
    "muffin": ["muffin"],
    "egg": ["egg"],
    "cheese": ["cheese","american cheese","string cheese"],
    "yogurt": ["yogurt"],
    "bread": ["bread","muffin top"],
    "pancake": ["pancake"],
    "sausage": ["sausage"],
    "sandwich": ["sandwich","english muffin"],
}

def tag_categories(text: str) -> set:
    text_n = norm(text)
    cats = set()
    for cat, kws in CATEGORY_RULES.items():
        if any(kw in text_n for kw in kws):
            cats.add(cat)
    return cats or {"uncategorized"}

# ---------- PREP NUTRITION TABLE ----------
must_have = ["RecipeName", "Calories", "Protein"]
missing = [c for c in must_have if c not in nutr.columns]
if missing:
    print(f"Nutrition CSV missing columns: {missing}")
    print(f"Available nutrition columns: {list(nutr.columns)}")
    raise ValueError(f"Nutrition CSV missing required columns: {missing}")

nutr = nutr.copy()
nutr["name_norm"] = nutr["RecipeName"].apply(norm)
nutr["categories"] = nutr["RecipeName"].apply(tag_categories)
nutr["token_set"] = nutr["name_norm"].str.split().apply(set)

# numeric nutrient columns
exclude_cols = {"RecipeID","RecipeName","ServingSize","ItemID","name_norm","categories","token_set",
                "SchoolID","SchoolName","DistrictID","DistrictName","Month","MonthNumber","Year",
                "StartDate","EndDate","Date","MealTime","MenuPlan","MealCategory","FoodCategory",
                "HasNutrients","Allergens","DietaryRestrictions","ReligiousRestrictions"}
num_cols = [c for c in nutr.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(nutr[c])]

preferred_order = [
    "GramsPerServing",
    "Calories","Protein",
    "Total Carbohydrate","Dietary Fiber","Total Sugars","Added Sugars",
    "Total Fat","Saturated Fat","Trans Fat",
    "Cholesterol","Sodium",
    "Vitamin D (D2 + D3)","Calcium","Iron","Potassium","Vitamin A","Vitamin C",
]
ordered_cols = [c for c in preferred_order if c in num_cols] + [c for c in num_cols if c not in preferred_order]

# ---------- SCORING ----------
def candidates_for_sales_item(sales_item: str) -> pd.DataFrame:
    cats = tag_categories(sales_item)
    if cats == {"uncategorized"}:
        pool = nutr
    else:
        pool = nutr[nutr["categories"].apply(lambda c: len(c & cats) > 0)]
        if len(pool) < 3:
            pool = nutr

    s_norm = norm(sales_item)
    stoks = set(s_norm.split())

    overlap = pool["token_set"].apply(lambda ts: len(ts & stoks))

    long_tokens = [t for t in stoks if len(t) > 3]
    if long_tokens:
        regex = "|".join(map(re.escape, long_tokens))
        substr_bonus = pool["name_norm"].str.contains(regex, regex=True)
    else:
        substr_bonus = pd.Series(False, index=pool.index)

    cat_align = pool["categories"].apply(lambda c: len(c & cats))

    score = overlap + (2 * substr_bonus.astype(int)) + 0.5 * cat_align
    return pool.assign(score=score).sort_values("score", ascending=False)

# ---------- COMPOSITES ----------
def split_components(sales_item: str) -> list[str]:
    s = norm(sales_item)
    parts = re.split(r"\b(with|and|\+)\b", s)
    parts = [p.strip() for p in parts if p.strip() and p.strip() not in {"with","and","+"}]
    return parts if len(parts) > 1 else []

def aggregate_numeric(df: pd.DataFrame, method="median") -> pd.Series:
    if df.empty:
        return pd.Series({c: np.nan for c in ordered_cols})
    if method == "median":
        ser = df[ordered_cols].median(numeric_only=True)
    elif method == "mean":
        ser = df[ordered_cols].mean(numeric_only=True)
    else:
        ser = df[ordered_cols].median(numeric_only=True)
    return ser.reindex(ordered_cols)

def sum_numeric(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series({c: np.nan for c in ordered_cols})
    return df[ordered_cols].sum(numeric_only=True).reindex(ordered_cols)

# ---------- UNIQUE SALES ITEMS ----------
# Find which column contains the quantity/total
possible_total_columns = ['total', 'Total', 'TOTAL', 'quantity', 'Quantity', 'QUANTITY', 
                         'count', 'Count', 'COUNT', 'qty', 'Qty', 'QTY']

total_column = None
for col in possible_total_columns:
    if col in sales.columns:
        total_column = col
        break

if total_column is None:
    print(f"Available columns for quantity: {list(sales.columns)}")
    # If no quantity column found, assume each row is 1
    sales['quantity'] = 1
    total_column = 'quantity'
    print("No quantity column found. Using default quantity=1 for all items.")

print(f"Using column '{total_column}' for quantities")

unique_sales_items = (
    sales.groupby(desc_column, as_index=False)[total_column].sum()
         .rename(columns={desc_column: "sales_item", total_column: "sales_volume"})
)

# ---------- MAIN LOOP ----------
records = []
for _, row in unique_sales_items.iterrows():
    sales_item = row["sales_item"]
    volume = row["sales_volume"]

    comps = split_components(sales_item)
    if comps:
        comp_best, comp_names = [], []
        for comp in comps:
            cand = candidates_for_sales_item(comp).head(3)
            if len(cand):
                best = cand.head(1)
                comp_best.append(best)
                comp_names.append(best["RecipeName"].iloc[0])
        if comp_best:
            comp_df = pd.concat(comp_best, ignore_index=True)
            agg = sum_numeric(comp_df)
            rec = {
                "sales_item": sales_item,
                "mapped_items": " + ".join(comp_names),
                "method": "composite_sum",
                "n_candidates": len(comp_names),
                "sales_volume": volume,
            }
            rec.update({c: (float(agg[c]) if pd.notna(agg[c]) else np.nan) for c in ordered_cols})
            records.append(rec)
            continue

    cand = candidates_for_sales_item(sales_item).head(5)
    if "cereal" in tag_categories(sales_item):
        cand = cand[cand["categories"].apply(lambda s: "cereal" in s)].head(3)
    agg = aggregate_numeric(cand, method="median")

    rec = {
        "sales_item": sales_item,
        "mapped_items": ", ".join(cand["RecipeName"].tolist()[:3]),
        "method": "median_over_topK",
        "n_candidates": int(len(cand)),
        "sales_volume": volume,
    }
    rec.update({c: (float(agg[c]) if pd.notna(agg[c]) else np.nan) for c in ordered_cols})
    records.append(rec)

full_map = pd.DataFrame(records)

# ---------- OUTPUT ----------
assert set(full_map["sales_item"]) == set(unique_sales_items["sales_item"]), "Some sales items were missed."

full_map = full_map[["sales_item","mapped_items"] + ordered_cols + ["method","n_candidates","sales_volume"]]\
                 .sort_values("sales_volume", ascending=False)

out_dir = script_dir
csv_full = out_dir / "sales_to_nutrition_mapping_FULL.csv"
xlsx_full = out_dir / "sales_to_nutrition_mapping_FULL.xlsx"
csv_slim = out_dir / "sales_to_nutrition_mapping_SLIM.csv"

full_map.to_csv(csv_full, index=False)
try:
    full_map.to_excel(xlsx_full, index=False)
except Exception as e:
    print("Excel export skipped:", e)

slim = full_map[["sales_item","mapped_items","Calories","Protein","Total Sugars"]]
slim.to_csv(csv_slim, index=False)

print("Done.")
print("Full mapping ->", csv_full)
print("Slim mapping ->", csv_slim)
print(full_map[["sales_item","mapped_items","Calories","Protein","Total Sugars","Total Fat","Saturated Fat","Sodium"]].head(20).to_string(index=False))
# %%
