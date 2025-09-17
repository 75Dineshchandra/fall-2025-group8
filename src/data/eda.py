#%%
# eda.py (or notebooks/eda.ipynb if you want interactive notebook style)

import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
file_path = "/Users/sirishag/Documents/CapStone8/fall-2025-group8/src/data/sales.csv"
df = pd.read_csv(file_path)

# --- Basic Info ---
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing Values:\n", df.isna().sum())
print("\nSample Rows:\n", df.head())

# --- Preprocessing ---
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month

# --- Top 10 Meals ---
top_meals = df.groupby("description")["total"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
top_meals.plot(kind="barh")
plt.title("Top 10 Meals by Total Servings")
plt.xlabel("Total Servings")
plt.gca().invert_yaxis()
plt.show()

# --- Popularity by Time of Day ---
tod_meals = df.groupby("time_of_day")["total"].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
tod_meals.plot(kind="bar")
plt.title("Meal Popularity by Time of Day")
plt.ylabel("Total Servings")
plt.show()

# --- Seasonal Trends ---
monthly = df.groupby("month")["total"].mean()
plt.figure(figsize=(8,5))
monthly.plot(kind="line", marker="o")
plt.title("Average Meals Served per Month")
plt.xlabel("Month")
plt.ylabel("Avg Total Servings")
plt.grid(True)
plt.show()

# --- Free vs Paid Distribution ---
meal_type = df[["free_meals","reduced_price_meals","full_price_meals"]].sum()
plt.figure(figsize=(7,5))
meal_type.plot(kind="pie", autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Free vs Reduced vs Full-Price Meals")
plt.ylabel("")
plt.show()

# %%
