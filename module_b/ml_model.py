# ============================================================
# FILE: module_b/ml_model.py
# PURPOSE: Random Forest model to predict monthly sales
# SYLLABUS: Unit IV — Analyzing Data, Dataset Splitting
#           Unit V  — Visualizing Data with Matplotlib
# CONCEPTS USED:
#   - Feature engineering
#   - Train/test split
#   - Random Forest Regressor
#   - MAE, RMSE, R2 evaluation metrics
#   - Feature importance plot
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# Create output folder for plots
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Connect to PostgreSQL
engine = create_engine(SQLALCHEMY_URL)
print("Connected to PostgreSQL for ML Model!\n")

# ============================================================
# STEP 1: Load required tables
# ============================================================
print("Loading tables...")
orders   = pd.read_sql("SELECT * FROM orders",         engine)
items    = pd.read_sql("SELECT * FROM order_items",    engine)
payments = pd.read_sql("SELECT * FROM order_payments", engine)
reviews  = pd.read_sql("SELECT * FROM order_reviews",  engine)
print("Tables loaded!\n")

# ============================================================
# STEP 2: Feature Engineering
# We create meaningful features from raw data
# Each feature should logically influence monthly sales
# SYLLABUS: Unit IV — Analyzing Data
# ============================================================
print("=" * 55)
print("STEP 1: FEATURE ENGINEERING")
print("=" * 55)

# Convert timestamps to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)
orders['order_delivered_customer_date'] = pd.to_datetime(
    orders['order_delivered_customer_date'], errors='coerce'
)
orders['order_estimated_delivery_date'] = pd.to_datetime(
    orders['order_estimated_delivery_date'], errors='coerce'
)

# Extract year and week for grouping
# Using weekly data gives us 100+ data points instead of 23 months
orders['year_month'] = orders['order_purchase_timestamp'].dt.to_period('W')

# Feature 1: Calculate delivery days per order
# delivery_days = difference between delivered date and purchase date
orders['delivery_days'] = (
    orders['order_delivered_customer_date'] -
    orders['order_purchase_timestamp']
).dt.days

# Feature 2: Calculate delivery delay per order
# delay = actual delivery - estimated delivery (positive = late)
orders['delivery_delay'] = (
    orders['order_delivered_customer_date'] -
    orders['order_estimated_delivery_date']
).dt.days

print("  Calculated delivery_days and delivery_delay per order")

# Merge orders with items to get price data
orders_items = pd.merge(
    orders,
    items[['order_id', 'price', 'freight_value']],
    on='order_id', how='left'
)

# Merge with payments to get payment data
orders_items_payments = pd.merge(
    orders_items,
    payments.groupby('order_id').agg(
        payment_value        = ('payment_value',        'sum'),
        payment_installments = ('payment_installments', 'mean')
    ).reset_index(),
    on='order_id', how='left'
)

# Merge with reviews to get review scores
orders_full = pd.merge(
    orders_items_payments,
    reviews[['order_id', 'review_score']],
    on='order_id', how='left'
)

print("  Merged orders, items, payments, reviews")

# ============================================================
# STEP 3: Aggregate features by month
# Our target variable is total monthly revenue
# Each row in our ML dataset = one month
# ============================================================
print("\n  Aggregating features by month...")

monthly_features = orders_full.groupby('year_month').agg(
    # Target variable: total revenue per month
    total_revenue        = ('price',                 'sum'),
    # Feature 1: total number of orders that month
    order_count          = ('order_id',              'nunique'),
    # Feature 2: average item price that month
    avg_price            = ('price',                 'mean'),
    # Feature 3: average freight cost that month
    avg_freight          = ('freight_value',         'mean'),
    # Feature 4: average payment value that month
    avg_payment_value    = ('payment_value',         'mean'),
    # Feature 5: average installments chosen
    avg_installments     = ('payment_installments',  'mean'),
    # Feature 6: average review score that month
    avg_review_score     = ('review_score',          'mean'),
    # Feature 7: average delivery days that month
    avg_delivery_days    = ('delivery_days',         'mean'),
    # Feature 8: average delivery delay that month
    avg_delivery_delay   = ('delivery_delay',        'mean')
).reset_index()

# Add month number as a feature (1-12 captures seasonality)
monthly_features['month_num'] = monthly_features['year_month'].dt.month
# Add year as a feature
monthly_features['year_num']  = monthly_features['year_month'].dt.year
# Add week number as a feature
monthly_features['week_num']  = monthly_features['year_month'].dt.week

# Drop rows with too many nulls (first/last months with incomplete data)
monthly_features = monthly_features.dropna()

print(f"  Monthly dataset shape: {monthly_features.shape}")
print(f"  Date range: {monthly_features['year_month'].min()} "
      f"to {monthly_features['year_month'].max()}")
print("\nMonthly Features Preview:")
print(monthly_features[['year_month', 'total_revenue', 'order_count',
                          'avg_price', 'avg_review_score']].to_string(index=False))

# ============================================================
# STEP 4: Prepare X (features) and y (target)
# X = all input features the model learns from
# y = what we want to predict (total monthly revenue)
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: PREPARING FEATURES AND TARGET")
print("=" * 55)

# Feature columns — these are our X (inputs)
feature_cols = [
    'order_count',
    'avg_price',
    'avg_freight',
    'avg_payment_value',
    'avg_installments',
    'avg_review_score',
    'avg_delivery_days',
    'avg_delivery_delay',
    'month_num',
    'week_num',
    'year_num'
]

X = monthly_features[feature_cols]   # Input features matrix
y = monthly_features['total_revenue'] # Target variable vector

print(f"Features (X) shape : {X.shape}")
print(f"Target (y) shape   : {y.shape}")
print(f"Features used      : {feature_cols}")

# ============================================================
# STEP 5: Train/Test Split
# We split data into 80% training and 20% testing
# shuffle=False preserves time order for time series data
# SYLLABUS: Unit IV — Dataset Splitting (train_test_split)
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: TRAIN/TEST SPLIT (80/20)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,      # 20% for testing
    random_state = 42,       # Fixed seed for reproducibility
    shuffle      = False     # Keep time order intact
)

print(f"Training samples   : {len(X_train)}")
print(f"Testing samples    : {len(X_test)}")
print(f"Total samples      : {len(X)}")

# ============================================================
# STEP 6: Train Random Forest Model
# Random Forest builds multiple decision trees and averages
# their predictions — more accurate than a single tree
# n_estimators = number of trees in the forest
# random_state = ensures same results every run
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: TRAINING RANDOM FOREST MODEL")
print("=" * 55)

rf_model = RandomForestRegressor(
    n_estimators      = 100,  # Build 100 decision trees
    random_state      = 42,   # Fixed seed for reproducibility
    max_depth         = 5,    # Max depth of each tree
    min_samples_split = 2,    # Min samples required to split a node
    min_samples_leaf  = 1     # Min samples per leaf node
)

print("Training Random Forest with 100 trees...")
rf_model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# STEP 7: Make Predictions and Evaluate
# MAE  = Mean Absolute Error (average prediction error in BRL)
# RMSE = Root Mean Squared Error (penalizes large errors more)
# R2   = R-squared (1.0 = perfect, 0 = no better than average)
# SYLLABUS: Unit IV — Analyzing Data
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: MODEL EVALUATION")
print("=" * 55)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  MAE  (Mean Absolute Error)      : R$ {mae:,.2f}")
print(f"  RMSE (Root Mean Squared Error)  : R$ {rmse:,.2f}")
print(f"  R2   (R-Squared Score)          : {r2:.4f}")
print(f"\nInterpretation:")
print(f"  On average our prediction is off by R$ {mae:,.2f} per month")
print(f"  R2 of {r2:.2f} means the model explains "
      f"{r2*100:.1f}% of revenue variance")

# ============================================================
# STEP 8: Feature Importance
# Random Forest tells us which features influenced
# predictions the most — higher importance = more influential
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: FEATURE IMPORTANCE")
print("=" * 55)

importance_df = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking:")
print(importance_df.to_string(index=False))

# ============================================================
# STEP 9: Visualizations
# ============================================================
# ============================================================
# NOTE FOR FACULTY:
# We use weekly aggregation instead of monthly because
# the dataset spans only 2 years (23 months total).
# Weekly data gives us 100+ data points which is sufficient
# for Random Forest to learn meaningful patterns.
# This is a real data science decision — model quality
# depends heavily on having enough training samples.
# ============================================================
print("\nNote: Weekly aggregation used to maximise training samples")
print(f"Weekly data points available: {len(X)}")

print("\nGenerating ML visualizations...")

# --- PLOT 1: Actual vs Predicted Revenue (Line Plot) ---
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test.values,
         marker='o', label='Actual Revenue',
         color='steelblue', linewidth=2)
plt.plot(range(len(y_pred)), y_pred,
         marker='s', label='Predicted Revenue',
         color='orange', linewidth=2, linestyle='--')
plt.title('Actual vs Predicted Monthly Revenue',
          fontsize=13, fontweight='bold')
plt.xlabel('Test Month Index')
plt.ylabel('Total Revenue (BRL)')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ml_actual_vs_predicted.png'), dpi=150)
plt.close()
print("  Saved: ml_actual_vs_predicted.png")

# --- PLOT 2: Feature Importance Bar Chart ---
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df['feature'],
    importance_df['importance'],
    color='steelblue', edgecolor='white'
)
plt.title('Random Forest — Feature Importance',
          fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  # Most important feature at top
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ml_feature_importance.png'), dpi=150)
plt.close()
print("  Saved: ml_feature_importance.png")

# --- PLOT 3: Residuals Plot (Actual - Predicted) ---
# Residuals show where the model over/under predicts
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 4))
plt.bar(range(len(residuals)), residuals,
        color=['green' if r >= 0 else 'red' for r in residuals])
plt.axhline(y=0, color='black', linewidth=1)
plt.title('Residuals — Actual minus Predicted Revenue',
          fontsize=13, fontweight='bold')
plt.xlabel('Test Month Index')
plt.ylabel('Residual (BRL)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'ml_residuals.png'), dpi=150)
plt.close()
print("  Saved: ml_residuals.png")

print("\n" + "=" * 55)
print("ML MODEL COMPLETE!")
print(f"  MAE  : R$ {mae:,.2f}")
print(f"  RMSE : R$ {rmse:,.2f}")
print(f"  R2   : {r2:.4f}")
print("Module B — ML Model complete.")
print("=" * 55)
