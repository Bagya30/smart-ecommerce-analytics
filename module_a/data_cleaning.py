# ============================================================
# FILE: module_a/data_cleaning.py
# PURPOSE: Clean all loaded data in PostgreSQL
#          - Handle null values
#          - Remove duplicates
#          - Detect and remove outliers using IQR method
#          - Standardize date formats
# SYLLABUS: Unit I  — Data Wrangling
#           Unit III — Data Cleanup
# ============================================================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import os

# Import database config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# Create database engine
engine = create_engine(SQLALCHEMY_URL)
print("Connected to PostgreSQL for data cleaning!\n")

# ============================================================
# HELPER FUNCTION: Save cleaned DataFrame back to PostgreSQL
# We use this after cleaning each table
# ============================================================
def save_to_db(df, table_name):
    df.to_sql(
        name      = table_name,
        con       = engine,
        if_exists = 'replace',   # Replace existing table with cleaned version
        index     = False,
        chunksize = 1000
    )
    print(f"  Saved cleaned data back to PostgreSQL table '{table_name}'")

# ============================================================
# HELPER FUNCTION: Print null summary for a DataFrame
# Shows how many nulls exist in each column before cleaning
# ============================================================
def print_null_summary(df, table_name):
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]  # Only show columns that have nulls
    if len(nulls) == 0:
        print(f"  No null values found in '{table_name}'")
    else:
        print(f"  Null values in '{table_name}':")
        for col, count in nulls.items():
            pct = (count / len(df)) * 100
            print(f"    {col}: {count:,} nulls ({pct:.1f}%)")

# ============================================================
# CLEANING FUNCTION 1: Clean customers table
# - Remove duplicate customer_id rows
# - Fill missing city/state with 'Unknown'
# ============================================================
print("=" * 55)
print("CLEANING TABLE 1: customers")
print("=" * 55)

customers = pd.read_sql("SELECT * FROM customers", engine)
print(f"  Original shape: {customers.shape}")
print_null_summary(customers, "customers")

# Remove duplicate rows based on customer_id
before = len(customers)
customers = customers.drop_duplicates(subset=['customer_id'])
after = len(customers)
print(f"  Duplicates removed: {before - after}")

# Fill missing city and state with 'Unknown'
customers['customer_city']  = customers['customer_city'].fillna('Unknown')
customers['customer_state'] = customers['customer_state'].fillna('Unknown')

print(f"  Final shape: {customers.shape}")
save_to_db(customers, 'customers')
print()

# ============================================================
# CLEANING FUNCTION 2: Clean orders table
# - Standardize all timestamp columns to datetime format
# - Fill missing delivery dates with NaT (Not a Time)
# - Remove duplicate order_id rows
# SYLLABUS: Unit III — Date format standardization
# ============================================================
print("=" * 55)
print("CLEANING TABLE 2: orders")
print("=" * 55)

orders = pd.read_sql("SELECT * FROM orders", engine)
print(f"  Original shape: {orders.shape}")
print_null_summary(orders, "orders")

# List of all timestamp columns in orders table
date_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

# Convert each date column to proper datetime format
# errors='coerce' turns invalid dates into NaT instead of crashing
for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')
    print(f"  Converted '{col}' to datetime")

# Remove duplicate orders
before = len(orders)
orders = orders.drop_duplicates(subset=['order_id'])
print(f"  Duplicates removed: {before - len(orders)}")

# Keep only orders with valid purchase timestamp
orders = orders.dropna(subset=['order_purchase_timestamp'])
print(f"  Final shape: {orders.shape}")
save_to_db(orders, 'orders')
print()

# ============================================================
# CLEANING FUNCTION 3: Clean order_items table
# - Remove duplicates
# - Use IQR method to detect and remove price outliers
# - Use IQR method to detect and remove freight outliers
# SYLLABUS: Unit III — Finding Outliers using IQR method
# ============================================================
print("=" * 55)
print("CLEANING TABLE 3: order_items")
print("=" * 55)

order_items = pd.read_sql("SELECT * FROM order_items", engine)
print(f"  Original shape: {order_items.shape}")
print_null_summary(order_items, "order_items")

# Remove duplicates
before = len(order_items)
order_items = order_items.drop_duplicates()
print(f"  Duplicates removed: {before - len(order_items)}")

# --- IQR Outlier Detection for 'price' column ---
# IQR = Q3 - Q1 (the middle 50% of data)
# Upper fence = Q3 + 1.5 * IQR
# Lower fence = Q1 - 1.5 * IQR
# Anything outside these fences is an outlier
print("\n  Applying IQR outlier detection on 'price':")
Q1_price = order_items['price'].quantile(0.25)   # 25th percentile
Q3_price = order_items['price'].quantile(0.75)   # 75th percentile
IQR_price = Q3_price - Q1_price                  # Interquartile range
lower_price = Q1_price - 1.5 * IQR_price         # Lower fence
upper_price = Q3_price + 1.5 * IQR_price         # Upper fence

print(f"    Q1={Q1_price:.2f}, Q3={Q3_price:.2f}, IQR={IQR_price:.2f}")
print(f"    Lower fence={lower_price:.2f}, Upper fence={upper_price:.2f}")

before = len(order_items)
# Keep only rows where price is within the fences
order_items = order_items[
    (order_items['price'] >= lower_price) &
    (order_items['price'] <= upper_price)
]
print(f"    Price outliers removed: {before - len(order_items)}")

# --- IQR Outlier Detection for 'freight_value' column ---
print("\n  Applying IQR outlier detection on 'freight_value':")
Q1_freight = order_items['freight_value'].quantile(0.25)
Q3_freight = order_items['freight_value'].quantile(0.75)
IQR_freight = Q3_freight - Q1_freight
lower_freight = Q1_freight - 1.5 * IQR_freight
upper_freight = Q3_freight + 1.5 * IQR_freight

print(f"    Q1={Q1_freight:.2f}, Q3={Q3_freight:.2f}, IQR={IQR_freight:.2f}")
print(f"    Lower fence={lower_freight:.2f}, Upper fence={upper_freight:.2f}")

before = len(order_items)
order_items = order_items[
    (order_items['freight_value'] >= lower_freight) &
    (order_items['freight_value'] <= upper_freight)
]
print(f"    Freight outliers removed: {before - len(order_items)}")

print(f"\n  Final shape: {order_items.shape}")
save_to_db(order_items, 'order_items')
print()

# ============================================================
# CLEANING FUNCTION 4: Clean order_reviews table
# - Fill missing review comments with empty string
# - Remove duplicate review entries
# - Standardize date columns
# ============================================================
print("=" * 55)
print("CLEANING TABLE 4: order_reviews")
print("=" * 55)

reviews = pd.read_sql("SELECT * FROM order_reviews", engine)
print(f"  Original shape: {reviews.shape}")
print_null_summary(reviews, "order_reviews")

# Fill missing comment title and message with empty string
reviews['review_comment_title']   = reviews['review_comment_title'].fillna('')
reviews['review_comment_message'] = reviews['review_comment_message'].fillna('')

# Convert date columns to datetime
reviews['review_creation_date']    = pd.to_datetime(reviews['review_creation_date'],   errors='coerce')
reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'],errors='coerce')

# Remove duplicates based on review_id and order_id together
before = len(reviews)
reviews = reviews.drop_duplicates(subset=['review_id', 'order_id'])
print(f"  Duplicates removed: {before - len(reviews)}")

print(f"  Final shape: {reviews.shape}")
save_to_db(reviews, 'order_reviews')
print()

# ============================================================
# CLEANING FUNCTION 5: Clean products table
# - Fill missing category names with 'unknown'
# - Fill missing numeric dimensions with median values
# - Remove duplicates
# SYLLABUS: Unit III — Handling missing values with median
# ============================================================
print("=" * 55)
print("CLEANING TABLE 5: products")
print("=" * 55)

products = pd.read_sql("SELECT * FROM products", engine)
print(f"  Original shape: {products.shape}")
print_null_summary(products, "products")

# Fill missing category name with 'unknown'
products['product_category_name'] = products['product_category_name'].fillna('unknown')

# Fill missing numeric columns with their median value
# Median is better than mean for skewed data — less affected by outliers
numeric_cols = [
    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty',
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm'
]

for col in numeric_cols:
    median_val = products[col].median()
    missing = products[col].isnull().sum()
    if missing > 0:
        products[col] = products[col].fillna(median_val)
        print(f"  Filled {missing} nulls in '{col}' with median={median_val:.1f}")

# Remove duplicates
before = len(products)
products = products.drop_duplicates(subset=['product_id'])
print(f"  Duplicates removed: {before - len(products)}")

print(f"  Final shape: {products.shape}")
save_to_db(products, 'products')
print()

# ============================================================
# CLEANING FUNCTION 6: Clean order_payments table
# - Remove rows with zero payment value
# - Remove duplicates
# ============================================================
print("=" * 55)
print("CLEANING TABLE 6: order_payments")
print("=" * 55)

payments = pd.read_sql("SELECT * FROM order_payments", engine)
print(f"  Original shape: {payments.shape}")
print_null_summary(payments, "order_payments")

# Remove rows where payment value is 0 or negative
before = len(payments)
payments = payments[payments['payment_value'] > 0]
print(f"  Zero/negative payment rows removed: {before - len(payments)}")

# Remove duplicates
before = len(payments)
payments = payments.drop_duplicates()
print(f"  Duplicates removed: {before - len(payments)}")

print(f"  Final shape: {payments.shape}")
save_to_db(payments, 'order_payments')
print()

# ============================================================
# FINAL SUMMARY
# Print a summary of all cleaned tables
# ============================================================
print("=" * 55)
print("CLEANING COMPLETE — FINAL SUMMARY")
print("=" * 55)

tables = ['customers', 'orders', 'order_items',
          'order_reviews', 'products', 'order_payments']

for table in tables:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    nulls = df.isnull().sum().sum()
    print(f"  {table:30s} | Rows: {len(df):>7,} | Remaining nulls: {nulls:>5,}")

print("\nAll tables cleaned and saved back to PostgreSQL!")
print("Module A — Data Cleaning complete.")
