#%%

# Inverse map: nutrition -> best sales (no averaging), with uncategorized fallback.
import pandas as pd, numpy as np, re
from pathlib import Path

# -------- paths --------
sales_p = "/Users/sirishag/Documents/fall-2025-group8/src/data/sales.csv"
nutr_p  = "/Users/sirishag/Documents/fall-2025-group8/src/data/nutrition items.csv"
outdir  = Path("/Users/sirishag/Documents/fall-2025-group8/src/data")

# -------- load & tidy --------
sales = pd.read_csv(sales_p)
nutr  = pd.read_csv(nutr_p)
nutr.columns = nutr.columns.str.strip()
nutr.drop(columns=[c for c in nutr.columns if c.endswith("_Unit")], inplace=True, errors="ignore")
sales["description"] = sales["description"].astype(str).str.strip()

def norm(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace("&"," and ")
    s = re.sub(r"\bw\/\b"," with ",s); s = re.sub(r"\/"," ",s)
    s = re.sub(r"[^a-z0-9\s%+]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

# -------- taxonomy --------
CATEGORY_RULES = {
    "fruit_strict": ["apple","banana","orange","grape","pear","peach","pineapple","berries","strawberry","blueberry","melon","fruit"],
    "veg_strict":   ["vegetable","broccoli","carrot","corn","peas","beans","salad","lettuce","spinach","tomato","potato"],
    "water":["water"], "milk":["milk"], "juice":["juice"],
    "entree":["chicken","pizza","burger","sandwich","tenders","meatball","taco","pasta","spaghetti","burrito"],
    "grain":["bread","bun","roll","muffin","bagel","tortilla","rice"],
    "cereal":["cereal","cheerios","chex","toast crunch","krispies"],
    "pancake":["pancake","waffle","french toast"],
    "yogurt":["yogurt"], "cheese":["cheese","string cheese","cream cheese"],
    "dessert":["cookie","brownie","pudding"], "parfait":["parfait"],
}
def tag_cats(text):
    t=norm(text); cats=set()
    for cat,kws in CATEGORY_RULES.items():
        if any(re.search(rf"\b{re.escape(kw)}\b",t) for kw in kws): cats.add(cat)
    return cats or {"uncategorized"}

# -------- prep tables --------
sales_u = sales.groupby("description",as_index=False)["total"].sum()\
               .rename(columns={"description":"sales_item","total":"sales_volume"})
sales_u["name_norm"]=sales_u["sales_item"].apply(norm)
sales_u["token_set"]=sales_u["name_norm"].str.split().apply(set)
sales_u["categories"]=sales_u["sales_item"].apply(tag_cats)

nutr_u = nutr.drop_duplicates(subset=["RecipeID"] if "RecipeID" in nutr.columns else ["RecipeName"]).copy()
nutr_u["name_norm"]=nutr_u["RecipeName"].apply(norm)
nutr_u["token_set"]=nutr_u["name_norm"].str.split().apply(set)
nutr_u["categories"]=nutr_u["RecipeName"].apply(tag_cats)

# -------- fallback string similarity --------
def simple_similarity(a,b):
    # normalized longest common substring length / shorter string length
    min_len=min(len(a),len(b))
    if min_len==0: return 0
    longest=0
    for i in range(len(a)):
        for j in range(len(b)):
            k=0
            while i+k<len(a) and j+k<len(b) and a[i+k]==b[j+k]:
                k+=1; longest=max(longest,k)
    return longest/min_len

# -------- scorer with fallback --------
def best_sales_for_recipe(r):
    nc, rtoks, rname = r["categories"], r["token_set"], r["name_norm"]

    # 1) category filter if not uncategorized
    if "uncategorized" in nc:
        block = sales_u.copy()
    else:
        block = sales_u[sales_u["categories"].apply(lambda sc: len(nc & sc)>0)]
        if block.empty: block = sales_u

    # 2) token overlap scoring
    overlap = block["token_set"].apply(lambda ts: len(ts & rtoks))
    long_tokens=[t for t in rname.split() if len(t)>2]
    if long_tokens:
        regex=r"\b("+"|".join(map(re.escape,long_tokens))+r")\b"
        bonus=block["name_norm"].str.contains(regex,regex=True)
    else:
        bonus=pd.Series(False,index=block.index)
    score=(4*overlap)+(2*bonus.astype(int))
    scored=block.assign(match_score=score,overlap=overlap).sort_values(
        ["overlap","match_score","sales_volume"],ascending=[False,False,False]
    )

    # 3) fallback if uncategorized & no overlap
    if "uncategorized" in nc and (scored.empty or scored.iloc[0]["overlap"]==0):
        sims=[simple_similarity(rname, x) for x in sales_u["name_norm"]]
        j=int(np.argmax(sims)); best=sales_u.iloc[j]
        return best, sims[j]
    return scored.iloc[0], None

# -------- map all --------
rows=[]
for _, r in nutr_u.iterrows():
    m, sim = best_sales_for_recipe(r)
    rec=dict(r)
    rec.update({
        "sales_item":m["sales_item"],
        "sales_volume":m["sales_volume"],
        "match_score":float(m.get("match_score",0)),
        "needs_review":bool(("uncategorized" in r["categories"]) and (sim is not None) and (sim<0.3))
    })
    rows.append(rec)
mapped=pd.DataFrame(rows)

# -------- neat output --------
NEAT=["RecipeID","RecipeName","ServingSize","GramsPerServing","Calories","Protein","Total Fat","Saturated Fat",
      "Trans Fat","Total Carbohydrate","Dietary Fiber","Total Sugars","Added Sugars","Sodium","Cholesterol",
      "sales_item","match_score","needs_review"]
NEAT=[c for c in NEAT if c in mapped.columns]
for c in NEAT:
    if c not in {"RecipeID","RecipeName","ServingSize","sales_item","needs_review"} and pd.api.types.is_numeric_dtype(mapped[c]):
        mapped[c]=mapped[c].round(1)
neat=mapped[NEAT].sort_values(["needs_review","sales_item","RecipeName"])
out_csv=outdir/"4nutrition_to_sales_MAPPING_NEAT.csv"
neat.to_csv(out_csv,index=False)

print("✅ Saved:",out_csv,"| Rows:",len(neat),"| flagged:",neat['needs_review'].sum())
print(neat.head(15).to_string(index=False))

# %%
