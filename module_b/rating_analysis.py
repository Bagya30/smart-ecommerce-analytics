# ============================================================
# FILE: module_b/rating_analysis.py
# PURPOSE: Analyse rating trends by month and category
# SYLLABUS: Unit IV — Data Exploration and Analysis
#           Unit V  — Visualizing Data with Pandas & Matplotlib
# CONCEPTS USED:
#   - groupby(), merge(), pivot_table()
#   - Line plot, Bar chart, Heatmap, Box plot
#   - Time series analysis of ratings
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# Create output folder for plots
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Connect to PostgreSQL
engine = create_engine(SQLALCHEMY_URL)
print("Connected to PostgreSQL for Rating Analysis!\n")

# ============================================================
# STEP 1: Load required tables
# ============================================================
print("Loading tables...")
orders   = pd.read_sql("SELECT * FROM orders",       engine)
reviews  = pd.read_sql("SELECT * FROM order_reviews", engine)
items    = pd.read_sql("SELECT * FROM order_items",   engine)
products = pd.read_sql("SELECT * FROM products",      engine)
translation = pd.read_sql(
    "SELECT * FROM product_category_translation", engine
)
print("Tables loaded!\n")

# ============================================================
# STEP 2: Prepare data
# Merge reviews with orders to get purchase date
# Merge with items and products to get category
# ============================================================
print("=" * 55)
print("STEP 1: PREPARING RATING DATA")
print("=" * 55)

# Convert timestamps to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)
reviews['review_creation_date'] = pd.to_datetime(
    reviews['review_creation_date'], errors='coerce'
)

# Extract year-month from order purchase date
orders['year_month'] = orders[
    'order_purchase_timestamp'
].dt.to_period('M')

# Merge reviews with orders to get order date
reviews_orders = pd.merge(
    reviews[['order_id', 'review_score', 'review_creation_date']],
    orders[['order_id', 'year_month', 'order_purchase_timestamp']],
    on='order_id', how='inner'
)
print(f"Reviews merged with orders: {len(reviews_orders):,}")

# Merge order_items with products to get category
items_products = pd.merge(
    items[['order_id', 'product_id']],
    products[['product_id', 'product_category_name']],
    on='product_id', how='left'
)

# Add English translation
items_products = pd.merge(
    items_products,
    translation,
    on='product_category_name', how='left'
)

# Use English name if available, otherwise use Portuguese name
items_products['category'] = items_products[
    'product_category_name_english'
].fillna(items_products['product_category_name']).fillna('unknown')

# Merge reviews with category data
reviews_full = pd.merge(
    reviews_orders,
    items_products[['order_id', 'category']].drop_duplicates('order_id'),
    on='order_id', how='left'
)
reviews_full['category'] = reviews_full['category'].fillna('unknown')
print(f"Reviews with category data: {len(reviews_full):,}")

# ============================================================
# STEP 3: Monthly Rating Trend Analysis
# SYLLABUS: Unit IV — Time-Related Data Analysis
# groupby() on year_month gives average rating per month
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: MONTHLY RATING TREND ANALYSIS")
print("=" * 55)

monthly_ratings = reviews_full.groupby('year_month').agg(
    avg_rating    = ('review_score', 'mean'),
    total_reviews = ('review_score', 'count'),
    rating_std    = ('review_score', 'std'),   # Standard deviation
    pct_5star     = ('review_score',
                     lambda x: (x == 5).sum() / len(x) * 100),
    pct_1star     = ('review_score',
                     lambda x: (x == 1).sum() / len(x) * 100)
).reset_index()

monthly_ratings['month_str'] = monthly_ratings['year_month'].astype(str)
monthly_ratings = monthly_ratings.sort_values('year_month')

print("\nMonthly Rating Summary:")
print(monthly_ratings[['month_str', 'avg_rating',
                        'total_reviews', 'pct_5star',
                        'pct_1star']].to_string(index=False))

print(f"\nOverall Average Rating : {reviews_full['review_score'].mean():.3f}")
print(f"Best Month             : "
      f"{monthly_ratings.loc[monthly_ratings['avg_rating'].idxmax(), 'month_str']}"
      f" ({monthly_ratings['avg_rating'].max():.3f})")
print(f"Worst Month            : "
      f"{monthly_ratings.loc[monthly_ratings['avg_rating'].idxmin(), 'month_str']}"
      f" ({monthly_ratings['avg_rating'].min():.3f})")

# ============================================================
# STEP 4: Category-wise Rating Analysis
# groupby() on category gives average rating per category
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: CATEGORY-WISE RATING ANALYSIS")
print("=" * 55)

category_ratings = reviews_full.groupby('category').agg(
    avg_rating    = ('review_score', 'mean'),
    total_reviews = ('review_score', 'count'),
    pct_5star     = ('review_score',
                     lambda x: (x == 5).sum() / len(x) * 100),
    pct_1star     = ('review_score',
                     lambda x: (x == 1).sum() / len(x) * 100)
).reset_index()

# Filter categories with at least 100 reviews for reliability
category_ratings = category_ratings[
    category_ratings['total_reviews'] >= 100
].sort_values('avg_rating', ascending=False)

print("\nTop 10 Highest Rated Categories:")
print(category_ratings.head(10)[
    ['category', 'avg_rating', 'total_reviews', 'pct_5star']
].to_string(index=False))

print("\nBottom 10 Lowest Rated Categories:")
print(category_ratings.tail(10)[
    ['category', 'avg_rating', 'total_reviews', 'pct_1star']
].to_string(index=False))

# ============================================================
# STEP 5: Pivot Table — Rating by Year and Score
# SYLLABUS: Unit IV — pivot_table()
# Rows = year, Columns = review score, Values = count
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: PIVOT TABLE — RATING BY YEAR AND SCORE")
print("=" * 55)

reviews_full['year'] = reviews_full[
    'order_purchase_timestamp'
].dt.year

pivot_ratings = pd.pivot_table(
    reviews_full,
    values  = 'order_id',
    index   = 'year',
    columns = 'review_score',
    aggfunc = 'count',
    fill_value = 0
)
pivot_ratings.columns = [f'Score_{c}' for c in pivot_ratings.columns]
print("\nReview Score Distribution by Year:")
print(pivot_ratings)

# ============================================================
# STEP 6: Visualizations
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: GENERATING RATING VISUALIZATIONS")
print("=" * 55)

# --- PLOT 1: Monthly Average Rating Trend (Line Plot) ---
fig, ax1 = plt.subplots(figsize=(14, 5))

# Line for average rating
ax1.plot(
    monthly_ratings['month_str'],
    monthly_ratings['avg_rating'],
    marker='o', linewidth=2,
    color='steelblue', label='Avg Rating'
)
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Rating', color='steelblue')
ax1.set_ylim(1, 5)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Second y-axis for review count
ax2 = ax1.twinx()
ax2.bar(
    monthly_ratings['month_str'],
    monthly_ratings['total_reviews'],
    alpha=0.3, color='orange', label='Review Count'
)
ax2.set_ylabel('Number of Reviews', color='orange')

plt.title('Monthly Rating Trend — Average Score and Review Volume',
          fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rating_monthly_trend.png'), dpi=150)
plt.close()
print("  Saved: rating_monthly_trend.png")

# --- PLOT 2: Top and Bottom 10 Categories (Bar Chart) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 10
top10 = category_ratings.head(10)
ax1.barh(top10['category'], top10['avg_rating'],
         color='steelblue', edgecolor='white')
ax1.set_title('Top 10 Highest Rated Categories',
              fontsize=11, fontweight='bold')
ax1.set_xlabel('Average Rating')
ax1.set_xlim(3, 5)
ax1.invert_yaxis()
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Bottom 10
bottom10 = category_ratings.tail(10)
ax2.barh(bottom10['category'], bottom10['avg_rating'],
         color='lightcoral', edgecolor='white')
ax2.set_title('Bottom 10 Lowest Rated Categories',
              fontsize=11, fontweight='bold')
ax2.set_xlabel('Average Rating')
ax2.set_xlim(1, 5)
ax2.invert_yaxis()
ax2.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rating_by_category.png'), dpi=150)
plt.close()
print("  Saved: rating_by_category.png")

# --- PLOT 3: Rating Heatmap by Month and Score ---
# Pivot: rows = months, columns = scores 1-5
# month_str column is added here before groupby
reviews_full['month_str'] = reviews_full['year_month'].astype(str)

heatmap_pivot = reviews_full.groupby(
    ['month_str', 'review_score']
)['order_id'].count().unstack(fill_value=0)
heatmap_pivot.columns = [f'Score {c}' for c in heatmap_pivot.columns]
heatmap_pivot = heatmap_pivot.sort_index()

plt.figure(figsize=(10, 10))
sns.heatmap(
    heatmap_pivot,
    cmap       = 'YlOrRd',
    annot      = True,
    fmt        = 'd',
    linewidths = 0.3,
    cbar_kws   = {'label': 'Number of Reviews'}
)
plt.title('Rating Heatmap — Score Distribution by Month',
          fontsize=12, fontweight='bold')
plt.xlabel('Review Score')
plt.ylabel('Month')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rating_heatmap.png'), dpi=150)
plt.close()
print("  Saved: rating_heatmap.png")

# --- PLOT 4: Box Plot — Rating Distribution by Year ---
# Box plots show median, IQR, and outliers per year
plt.figure(figsize=(8, 5))
reviews_full['year'] = reviews_full['year'].astype(str)
reviews_full.boxplot(
    column='review_score',
    by='year',
    patch_artist=True,
    grid=False
)
plt.suptitle('')
plt.title('Review Score Distribution by Year (Box Plot)',
          fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Review Score')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rating_boxplot_year.png'), dpi=150)
plt.close()
print("  Saved: rating_boxplot_year.png")

# --- PLOT 5: 5-Star vs 1-Star % Trend (Line Plot) ---
plt.figure(figsize=(14, 5))
plt.plot(
    monthly_ratings['month_str'],
    monthly_ratings['pct_5star'],
    marker='o', linewidth=2,
    color='green', label='5-Star %'
)
plt.plot(
    monthly_ratings['month_str'],
    monthly_ratings['pct_1star'],
    marker='s', linewidth=2,
    color='red', label='1-Star %',
    linestyle='--'
)
plt.title('Monthly 5-Star vs 1-Star Review Percentage',
          fontsize=13, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Percentage of Reviews (%)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rating_5star_vs_1star.png'), dpi=150)
plt.close()
print("  Saved: rating_5star_vs_1star.png")

print("\n" + "=" * 55)
print("RATING ANALYSIS COMPLETE!")
print(f"Overall avg rating     : "
      f"{reviews_full['review_score'].mean():.3f} / 5.0")
print(f"Total reviews analysed : {len(reviews_full):,}")
print(f"Categories analysed    : {len(category_ratings):,}")
print("Module B — Rating Analysis complete.")
print("=" * 55)
