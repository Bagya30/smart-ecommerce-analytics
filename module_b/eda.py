# ============================================================
# FILE: module_b/eda.py
# PURPOSE: Full Exploratory Data Analysis using pandas
# SYLLABUS: Unit IV — Data Exploration and Analysis
#           Unit V  — Visualizing Data with Pandas & Matplotlib
# CONCEPTS USED:
#   - describe(), info(), shape, dtypes
#   - corr(), groupby(), pivot_table(), merge()
#   - Line plot, Heatmap, Bar chart, Box plot
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
print("Connected to PostgreSQL for EDA!\n")

# ============================================================
# STEP 1: Load all required tables from PostgreSQL
# We use pandas read_sql() to load cleaned data directly
# ============================================================
print("Loading tables from PostgreSQL...")
orders    = pd.read_sql("SELECT * FROM orders",         engine)
items     = pd.read_sql("SELECT * FROM order_items",    engine)
payments  = pd.read_sql("SELECT * FROM order_payments", engine)
reviews   = pd.read_sql("SELECT * FROM order_reviews",  engine)
customers = pd.read_sql("SELECT * FROM customers",      engine)
products  = pd.read_sql("SELECT * FROM products",       engine)
sellers   = pd.read_sql("SELECT * FROM sellers",        engine)
print("All tables loaded!\n")

# ============================================================
# STEP 2: Basic DataFrame Exploration
# SYLLABUS: Unit IV — Exploring Table Functions
# describe() gives count, mean, std, min, max for numeric cols
# info() gives column names, data types, non-null counts
# ============================================================
print("=" * 55)
print("SECTION 1: BASIC DATA EXPLORATION")
print("=" * 55)

print("\n--- Orders Table Shape ---")
print(f"Rows: {orders.shape[0]:,} | Columns: {orders.shape[1]}")

print("\n--- Orders Table Info ---")
print(orders.info())

print("\n--- Orders Statistical Summary (describe()) ---")
print(orders.describe())

print("\n--- Order Items Statistical Summary ---")
print(items[['price', 'freight_value']].describe())

print("\n--- Payments Statistical Summary ---")
print(payments[['payment_value', 'payment_installments']].describe())

# ============================================================
# STEP 3: Merge DataFrames
# SYLLABUS: Unit IV — Joining Numerous Datasets
# pd.merge() combines two DataFrames on a common column
# how='inner' keeps only rows that exist in both tables
# ============================================================
print("\n" + "=" * 55)
print("SECTION 2: MERGING DATASETS")
print("=" * 55)

# Merge orders with customers to get customer state
orders_customers = pd.merge(
    orders,
    customers[['customer_id', 'customer_state', 'customer_city']],
    on='customer_id',
    how='inner'
)
print(f"Orders + Customers merged: {orders_customers.shape}")

# Merge order_items with payments to get revenue per order
items_payments = pd.merge(
    items[['order_id', 'price', 'freight_value']],
    payments[['order_id', 'payment_value', 'payment_type']],
    on='order_id',
    how='inner'
)
print(f"Items + Payments merged: {items_payments.shape}")

# ============================================================
# STEP 4: GroupBy Analysis
# SYLLABUS: Unit IV — Creating Groupings
# groupby() groups rows by a column and applies aggregation
# ============================================================
print("\n" + "=" * 55)
print("SECTION 3: GROUPBY ANALYSIS")
print("=" * 55)

# Total revenue and orders per state
state_analysis = orders_customers.merge(
    items[['order_id', 'price', 'freight_value']], on='order_id', how='left'
)
state_grouped = state_analysis.groupby('customer_state').agg(
    total_orders  = ('order_id',      'count'),
    total_revenue = ('price',         'sum'),
    avg_price     = ('price',         'mean'),
    avg_freight   = ('freight_value', 'mean')
).reset_index()
state_grouped = state_grouped.sort_values('total_revenue', ascending=False)
print("\nTop 10 States by Revenue:")
print(state_grouped.head(10).to_string(index=False))

# Payment type distribution
payment_groups = payments.groupby('payment_type').agg(
    total_transactions = ('order_id',      'count'),
    total_value        = ('payment_value', 'sum'),
    avg_value          = ('payment_value', 'mean')
).reset_index().sort_values('total_transactions', ascending=False)
print("\nPayment Type Distribution:")
print(payment_groups.to_string(index=False))

# ============================================================
# STEP 5: Pivot Table Analysis
# SYLLABUS: Unit IV — Separating and Focusing the Data
# pivot_table() reshapes data — rows become one category,
# columns become another, values are aggregated
# ============================================================
print("\n" + "=" * 55)
print("SECTION 4: PIVOT TABLE ANALYSIS")
print("=" * 55)

# Convert purchase timestamp to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)
orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
orders['order_year']  = orders['order_purchase_timestamp'].dt.year

# Merge with items for revenue
orders_items = pd.merge(
    orders[['order_id', 'order_month', 'order_year', 'order_status']],
    items[['order_id', 'price']],
    on='order_id',
    how='left'
)

# Pivot table: Year vs Order Status — count of orders
pivot_status = pd.pivot_table(
    orders_items,
    values  = 'order_id',
    index   = 'order_year',
    columns = 'order_status',
    aggfunc = 'count',
    fill_value = 0          # Fill missing combinations with 0
)
print("\nPivot Table — Orders by Year and Status:")
print(pivot_status)

# ============================================================
# STEP 6: Correlation Analysis
# SYLLABUS: Unit IV — Identifying Correlations
# corr() computes pairwise correlation between numeric columns
# Values range from -1 (inverse) to +1 (direct correlation)
# ============================================================
print("\n" + "=" * 55)
print("SECTION 5: CORRELATION ANALYSIS")
print("=" * 55)

# Build a combined DataFrame for correlation
corr_data = pd.merge(
    items[['order_id', 'price', 'freight_value']],
    payments[['order_id', 'payment_value', 'payment_installments']],
    on='order_id', how='inner'
)
corr_data = pd.merge(
    corr_data,
    reviews[['order_id', 'review_score']],
    on='order_id', how='left'
)

# Calculate correlation matrix
corr_matrix = corr_data[['price', 'freight_value',
                          'payment_value', 'payment_installments',
                          'review_score']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

print("\nKey Insights from Correlation:")
print(f"  Price vs Payment Value correlation     : {corr_matrix.loc['price','payment_value']:.3f}")
print(f"  Price vs Review Score correlation      : {corr_matrix.loc['price','review_score']:.3f}")
print(f"  Freight vs Review Score correlation    : {corr_matrix.loc['freight_value','review_score']:.3f}")

# ============================================================
# STEP 7: Value Counts Analysis
# SYLLABUS: Unit IV — Exploring Table Functions
# value_counts() counts occurrences of each unique value
# ============================================================
print("\n" + "=" * 55)
print("SECTION 6: VALUE COUNTS ANALYSIS")
print("=" * 55)

print("\nOrder Status Distribution:")
print(orders['order_status'].value_counts())

print("\nTop 10 Product Categories:")
print(products['product_category_name'].value_counts().head(10))

print("\nReview Score Distribution:")
print(reviews['review_score'].value_counts().sort_index())

# ============================================================
# STEP 8: Visualizations
# SYLLABUS: Unit V — Visualizing Data with Pandas & Matplotlib
# ============================================================
print("\n" + "=" * 55)
print("SECTION 7: GENERATING VISUALIZATIONS")
print("=" * 55)

# --- PLOT 1: Monthly Order Trend (Line Plot) ---
# Line plots are used to show evolution over time (Unit V)
monthly_orders = orders.groupby('order_month')['order_id'].count().reset_index()
monthly_orders.columns = ['month', 'order_count']
monthly_orders['month_str'] = monthly_orders['month'].astype(str)

plt.figure(figsize=(14, 5))
plt.plot(
    monthly_orders['month_str'],
    monthly_orders['order_count'],
    marker='o', linewidth=2, color='steelblue', markersize=4
)
plt.title('Monthly Order Volume Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_monthly_orders.png'), dpi=150)
plt.close()
print("  Saved: eda_monthly_orders.png")

# --- PLOT 2: Correlation Heatmap ---
# Heatmap visualizes correlation matrix using color intensity
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot     = True,      # Show correlation values in cells
    fmt       = '.2f',     # Format to 2 decimal places
    cmap      = 'coolwarm',# Red = positive, Blue = negative
    center    = 0,         # Center color scale at 0
    linewidths= 0.5
)
plt.title('Correlation Heatmap — Price, Freight, Payment, Review', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_correlation_heatmap.png'), dpi=150)
plt.close()
print("  Saved: eda_correlation_heatmap.png")

# --- PLOT 3: Revenue by State (Bar Chart) ---
top10_states = state_grouped.head(10)
plt.figure(figsize=(12, 5))
plt.bar(
    top10_states['customer_state'],
    top10_states['total_revenue'],
    color='steelblue', edgecolor='white'
)
plt.title('Top 10 States by Total Revenue', fontsize=14, fontweight='bold')
plt.xlabel('State')
plt.ylabel('Total Revenue (BRL)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_revenue_by_state.png'), dpi=150)
plt.close()
print("  Saved: eda_revenue_by_state.png")

# --- PLOT 4: Price Distribution (Box Plot) ---
# Box plots show distribution, median, and outliers (Unit V)
plt.figure(figsize=(10, 5))
plt.boxplot(
    items['price'].dropna(),
    vert      = False,
    patch_artist= True,
    boxprops  = dict(facecolor='steelblue', alpha=0.6)
)
plt.title('Price Distribution (After IQR Cleaning)', fontsize=14)
plt.xlabel('Price (BRL)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_price_boxplot.png'), dpi=150)
plt.close()
print("  Saved: eda_price_boxplot.png")

# --- PLOT 5: Payment Type Distribution (Bar Chart) ---
plt.figure(figsize=(8, 5))
plt.bar(
    payment_groups['payment_type'],
    payment_groups['total_transactions'],
    color=['steelblue','orange','green','red'][:len(payment_groups)]
)
plt.title('Payment Type Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Payment Type')
plt.ylabel('Number of Transactions')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_payment_types.png'), dpi=150)
plt.close()
print("  Saved: eda_payment_types.png")

# --- PLOT 6: Review Score Distribution (Bar Chart) ---
review_counts = reviews['review_score'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(
    review_counts.index.astype(str),
    review_counts.values,
    color=['red','orange','yellow','lightgreen','green']
)
plt.title('Review Score Distribution (1-5)', fontsize=14, fontweight='bold')
plt.xlabel('Review Score')
plt.ylabel('Number of Reviews')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_review_scores.png'), dpi=150)
plt.close()
print("  Saved: eda_review_scores.png")

# --- PLOT 7: Lag Plot (Unit V — Pandas Plotting Subpackage) ---
# Lag plot checks if time series data is random
# If points form a pattern, the data has autocorrelation
from pandas.plotting import lag_plot
plt.figure(figsize=(6, 6))
lag_plot(monthly_orders['order_count'])
plt.title('Lag Plot — Monthly Orders (Unit V)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_lag_plot.png'), dpi=150)
plt.close()
print("  Saved: eda_lag_plot.png")

print("\n" + "=" * 55)
print("EDA COMPLETE!")
print(f"All plots saved to: outputs/plots/")
print("Module B — EDA complete.")
print("=" * 55)
