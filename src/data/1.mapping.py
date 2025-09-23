#%%

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ---------- INPUTS ----------
sales_path = "/Users/sirishag/Documents/fall-2025-group8/src/data/sales.csv"
nutr_path  = "/Users/sirishag/Documents/fall-2025-group8/src/data/nutrition items.csv"

# ---------- LOAD ----------
sales = pd.read_csv(sales_path)
nutr  = pd.read_csv(nutr_path)

# ---------- CLEAN NUTRITION HEADERS / DROP UNIT COLUMNS ----------
# strip accidental whitespace (handles "Added Sugars   " etc.)
nutr.columns = nutr.columns.str.strip()
# drop all "*_Unit" columns (we only aggregate numeric values)
nutr = nutr.drop(columns=[c for c in nutr.columns if c.endswith("_Unit")], errors="ignore")

# ---------- NORMALIZATION ----------
def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\bw\/\b", " with ", s)   # "w/" -> "with"
    s = re.sub(r"\/", " ", s)             # slashes -> space
    s = re.sub(r"[^a-z0-9\s%+]", " ", s)  # drop punctuation but keep %, +
    s = re.sub(r"\s+", " ", s).strip()
    return s

sales["description"] = sales["description"].astype(str).fillna("").str.strip()

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
# sanity: minimal required columns
must_have = ["RecipeName", "Calories", "Protein"]
missing = [c for c in must_have if c not in nutr.columns]
if missing:
    raise ValueError(f"Nutrition CSV missing columns: {missing}")

nutr = nutr.copy()
nutr["name_norm"]  = nutr["RecipeName"].apply(norm)
nutr["categories"] = nutr["RecipeName"].apply(tag_categories)
nutr["token_set"]  = nutr["name_norm"].str.split().apply(set)

# numeric nutrient columns = all numeric except these:
exclude_cols = {"RecipeID","RecipeName","ServingSize","ItemID","name_norm","categories","token_set",
                "SchoolID","SchoolName","DistrictID","DistrictName","Month","MonthNumber","Year",
                "StartDate","EndDate","Date","MealTime","MenuPlan","MealCategory","FoodCategory",
                "HasNutrients","Allergens","DietaryRestrictions","ReligiousRestrictions"}
num_cols = [c for c in nutr.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(nutr[c])]
if not num_cols:
    raise ValueError("No numeric nutrient columns detected. Check your CSV headers and types.")

# preferred output order (only those that exist), then any extra numeric cols
preferred_order = [
    "GramsPerServing",
    "Calories","Protein",
    "Total Carbohydrate","Dietary Fiber","Total Sugars","Added Sugars",
    "Total Fat","Saturated Fat","Trans Fat",
    "Cholesterol","Sodium",
    "Vitamin D (D2 + D3)","Calcium","Iron","Potassium","Vitamin A","Vitamin C",
]
ordered_cols = [c for c in preferred_order if c in num_cols] + [c for c in num_cols if c not in preferred_order]

# ---------- SCORING (VECTORIZED) ----------
def candidates_for_sales_item(sales_item: str) -> pd.DataFrame:
    cats = tag_categories(sales_item)
    # category block
    if cats == {"uncategorized"}:
        pool = nutr
    else:
        pool = nutr[nutr["categories"].apply(lambda c: len(c & cats) > 0)]
        if len(pool) < 3:
            pool = nutr  # fallback if block too small

    s_norm = norm(sales_item)
    stoks = set(s_norm.split())

    # token overlap
    overlap = pool["token_set"].apply(lambda ts: len(ts & stoks))

    # substring bonus for longer tokens
    long_tokens = [t for t in stoks if len(t) > 3]
    if long_tokens:
        regex = "|".join(map(re.escape, long_tokens))
        substr_bonus = pool["name_norm"].str.contains(regex, regex=True)
    else:
        substr_bonus = pd.Series(False, index=pool.index)

    # category alignment
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
unique_sales_items = (
    sales.groupby("description", as_index=False)["total"].sum()
         .rename(columns={"description":"sales_item","total":"sales_volume"})
)

# ---------- MAIN LOOP ----------
records = []
for _, row in unique_sales_items.iterrows():
    sales_item = row["sales_item"]
    volume     = row["sales_volume"]

    # composites first (e.g., "bagel with cream cheese")
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

    # non-composite: median over top-K
    cand = candidates_for_sales_item(sales_item).head(5)
    # cereal bucket: restrict to cereals & top 3
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

# ---------- COVERAGE GUARANTEE ----------
assert set(full_map["sales_item"]) == set(unique_sales_items["sales_item"]), "Some sales items were missed."


full_map = full_map[["sales_item","mapped_items"] + ordered_cols + ["method","n_candidates","sales_volume"]]\
                 .sort_values("sales_volume", ascending=False)

out_dir = Path("/Users/sirishag/Documents/fall-2025-group8/src/data")
csv_full = out_dir / "sales_to_nutrition_mapping_FULL.csv"
xlsx_full = out_dir / "sales_to_nutrition_mapping_FULL.xlsx"
csv_slim = out_dir / "sales_to_nutrition_mapping_SLIM.csv"

full_map.to_csv(csv_full, index=False)
try:
    full_map.to_excel(xlsx_full, index=False)
except Exception as e:
    print("Excel export skipped:", e)

# optional slim, like your example
slim = full_map[["sales_item","mapped_items","Calories","Protein","Total Sugars"]]
slim.to_csv(csv_slim, index=False)

print("Done.")
print("Full mapping ->", csv_full)
print("Slim mapping ->", csv_slim)
print(full_map[["sales_item","mapped_items","Calories","Protein","Total Sugars","Total Fat","Saturated Fat","Sodium"]].head(20).to_string(index=False))



# %%
