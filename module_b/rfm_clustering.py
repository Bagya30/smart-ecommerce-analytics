# ============================================================
# FILE: module_b/rfm_clustering.py
# PURPOSE: RFM Analysis + K-Means Clustering
# SYLLABUS: Unit IV — Data Exploration and Analysis
#           Unit V  — Visualizing Data with Pandas & Matplotlib
# CONCEPTS USED:
#   - RFM: Recency, Frequency, Monetary analysis
#   - RFM Scoring (1-5 scale)
#   - K-Means clustering with elbow method
#   - Scatter plot, Bar chart
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# Create output folder for plots
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Connect to PostgreSQL
engine = create_engine(SQLALCHEMY_URL)
print("Connected to PostgreSQL for RFM Analysis!\n")

# ============================================================
# STEP 1: Load required tables
# ============================================================
print("Loading tables...")
orders   = pd.read_sql("SELECT * FROM orders",         engine)
items    = pd.read_sql("SELECT * FROM order_items",    engine)
payments = pd.read_sql("SELECT * FROM order_payments", engine)
customers= pd.read_sql("SELECT * FROM customers",      engine)
print("Tables loaded!\n")

# ============================================================
# STEP 2: Prepare data for RFM
# We need: customer_unique_id, order_date, order_value
# ============================================================
print("=" * 55)
print("STEP 1: PREPARING RFM DATA")
print("=" * 55)

# Convert purchase timestamp to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)

# Merge orders with customers to get customer_unique_id
# customer_unique_id tracks the same person across multiple orders
orders_customers = pd.merge(
    orders[['order_id', 'customer_id', 'order_purchase_timestamp',
            'order_status']],
    customers[['customer_id', 'customer_unique_id']],
    on='customer_id', how='inner'
)

# Keep only delivered orders for accurate RFM
orders_customers = orders_customers[
    orders_customers['order_status'] == 'delivered'
]
print(f"Delivered orders for RFM: {len(orders_customers):,}")

# Merge with payments to get order value
orders_payments = pd.merge(
    orders_customers,
    payments.groupby('order_id')['payment_value'].sum().reset_index(),
    on='order_id', how='inner'
)
print(f"Orders with payment data: {len(orders_payments):,}")

# ============================================================
# STEP 3: Calculate RFM Values
# RECENCY  = How many days since the customer's last purchase
#            Lower recency = more recent = better customer
# FREQUENCY = How many orders the customer placed
#             Higher frequency = more loyal customer
# MONETARY  = Total amount the customer spent
#             Higher monetary = more valuable customer
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: CALCULATING RFM VALUES")
print("=" * 55)

# Reference date = 1 day after the last order in dataset
# This is used to calculate recency
reference_date = orders_payments['order_purchase_timestamp'].max() + \
                 pd.Timedelta(days=1)
print(f"Reference date for recency: {reference_date.date()}")

# Calculate RFM for each unique customer
rfm = orders_payments.groupby('customer_unique_id').agg(
    recency   = ('order_purchase_timestamp',
                 lambda x: (reference_date - x.max()).days),
    frequency = ('order_id',       'count'),
    monetary  = ('payment_value',  'sum')
).reset_index()

print(f"\nTotal unique customers: {len(rfm):,}")
print("\nRFM Summary Statistics:")
print(rfm[['recency', 'frequency', 'monetary']].describe().round(2))

# ============================================================
# STEP 4: RFM Scoring (1-5 scale)
# We divide each RFM metric into 5 equal groups (quintiles)
# For Recency: lower days = better = score 5
# For Frequency and Monetary: higher = better = score 5
# pd.qcut() divides data into equal-sized buckets
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: CALCULATING RFM SCORES (1-5)")
print("=" * 55)

# Recency score: lower recency days = better = higher score
# So we reverse the labels (5 for lowest recency)
rfm['r_score'] = pd.qcut(
    rfm['recency'],
    q      = 5,
    labels = [5, 4, 3, 2, 1],  # Reversed: recent customers get 5
    duplicates='drop'
)

# Frequency score: higher frequency = better = higher score
# Using custom bins because 96% of customers bought only once
# so qcut cannot create 5 equal groups from this data
rfm['f_score'] = pd.cut(
    rfm['frequency'],
    bins   = [0, 1, 2, 3, 5, float('inf')],
    labels = [1, 2, 3, 4, 5]
)

# Monetary score: higher spend = better = higher score
rfm['m_score'] = pd.qcut(
    rfm['monetary'],
    q      = 5,
    labels = [1, 2, 3, 4, 5],
    duplicates='drop'
)

# Combined RFM score = sum of all three scores
rfm['rfm_score'] = (
    rfm['r_score'].astype(int) +
    rfm['f_score'].astype(int) +
    rfm['m_score'].astype(int)
)

print("RFM Score Distribution:")
print(rfm['rfm_score'].value_counts().sort_index())
print(f"\nRFM Score Range: {rfm['rfm_score'].min()} to {rfm['rfm_score'].max()}")

# ============================================================
# STEP 5: K-Means Clustering with Elbow Method
# Before clustering we standardize RFM values using
# StandardScaler — this ensures no single metric dominates
# because they have very different scales
# (recency in days vs monetary in BRL)
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: K-MEANS CLUSTERING")
print("=" * 55)

# Select features for clustering
features = rfm[['recency', 'frequency', 'monetary']].copy()

# Standardize features: mean=0, std=1
# This is normalization/standardization from Unit III syllabus
scaler          = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("Features standardized (mean=0, std=1)")

# --- Elbow Method ---
# We try K from 1 to 10 and plot inertia (within-cluster sum of squares)
# The "elbow" point where inertia stops dropping fast = best K
print("\nRunning Elbow Method (K=1 to 10)...")
inertia_values = []
k_range        = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia_values.append(kmeans.inertia_)
    print(f"  K={k} | Inertia={kmeans.inertia_:.2f}")

# --- Plot Elbow Curve ---
plt.figure(figsize=(9, 5))
plt.plot(k_range, inertia_values, marker='o', linewidth=2,
         color='steelblue', markersize=7)
plt.axvline(x=3, color='red', linestyle='--',
            label='Chosen K=3')          # Mark our chosen K
plt.title('Elbow Method — Finding Optimal K for K-Means',
          fontsize=13, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rfm_elbow_curve.png'), dpi=150)
plt.close()
print("\nSaved: rfm_elbow_curve.png")

# ============================================================
# STEP 6: Apply K-Means with K=3
# K=3 gives us 3 meaningful customer segments:
# High Value, Medium Value, Low Value customers
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: APPLYING K-MEANS WITH K=3")
print("=" * 55)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['cluster'] = kmeans_final.fit_predict(features_scaled)
print("K-Means clustering done!")

# ============================================================
# STEP 7: Label the Clusters
# We look at the mean RFM values per cluster to decide labels
# Low recency + high frequency + high monetary = High Value
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: CLUSTER ANALYSIS AND LABELLING")
print("=" * 55)

cluster_summary = rfm.groupby('cluster').agg(
    customer_count = ('customer_unique_id', 'count'),
    avg_recency    = ('recency',   'mean'),
    avg_frequency  = ('frequency', 'mean'),
    avg_monetary   = ('monetary',  'mean'),
    avg_rfm_score  = ('rfm_score', 'mean')
).reset_index().round(2)

print("\nCluster Summary:")
print(cluster_summary.to_string(index=False))

# Assign meaningful labels based on avg_monetary rank
# Highest monetary = High Value, Lowest = Low Value
cluster_summary = cluster_summary.sort_values('avg_monetary',
                                               ascending=False)
cluster_summary['label'] = ['High Value', 'Medium Value', 'Low Value']

# Map labels back to main RFM dataframe
label_map = dict(zip(cluster_summary['cluster'],
                     cluster_summary['label']))
rfm['segment'] = rfm['cluster'].map(label_map)

print("\nCustomer Segments:")
segment_counts = rfm['segment'].value_counts()
print(segment_counts)

# ============================================================
# STEP 8: Save RFM results to PostgreSQL
# So dashboard.py can use it directly
# ============================================================
rfm.to_sql('rfm_segments', engine, if_exists='replace', index=False)
print("\nRFM segments saved to PostgreSQL table 'rfm_segments'")

# ============================================================
# STEP 9: Visualizations
# ============================================================
print("\nGenerating RFM visualizations...")

# --- PLOT 1: Customer Segments Bar Chart ---
plt.figure(figsize=(8, 5))
colors = ['gold', 'steelblue', 'lightcoral']
segment_counts.plot(kind='bar', color=colors, edgecolor='white')
plt.title('Customer Segments — K-Means Clustering (K=3)',
          fontsize=13, fontweight='bold')
plt.xlabel('Customer Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rfm_segments_bar.png'), dpi=150)
plt.close()
print("  Saved: rfm_segments_bar.png")

# --- PLOT 2: RFM Scatter Plot (Recency vs Monetary) ---
plt.figure(figsize=(10, 6))
colors_map = {'High Value': 'gold',
              'Medium Value': 'steelblue',
              'Low Value': 'lightcoral'}

for segment, color in colors_map.items():
    subset = rfm[rfm['segment'] == segment]
    plt.scatter(
        subset['recency'],
        subset['monetary'],
        c      = color,
        label  = segment,
        alpha  = 0.4,
        s      = 15
    )

plt.title('RFM Scatter — Recency vs Monetary by Segment',
          fontsize=13, fontweight='bold')
plt.xlabel('Recency (Days Since Last Purchase)')
plt.ylabel('Monetary (Total Spend in BRL)')
plt.legend()
plt.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rfm_scatter.png'), dpi=150)
plt.close()
print("  Saved: rfm_scatter.png")

# --- PLOT 3: RFM Heatmap (Avg values per segment) ---
heatmap_data = rfm.groupby('segment')[
    ['recency', 'frequency', 'monetary']
].mean().round(2)

# Normalize each column for better color visualization
heatmap_norm = (heatmap_data - heatmap_data.min()) / \
               (heatmap_data.max() - heatmap_data.min())

plt.figure(figsize=(8, 4))
sns.heatmap(
    heatmap_norm,
    annot     = heatmap_data,  # Show actual values
    fmt       = '.1f',
    cmap      = 'YlOrRd',
    linewidths= 0.5
)
plt.title('RFM Heatmap — Average Values per Segment',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rfm_heatmap.png'), dpi=150)
plt.close()
print("  Saved: rfm_heatmap.png")

print("\n" + "=" * 55)
print("RFM CLUSTERING COMPLETE!")
print(f"Total customers analysed : {len(rfm):,}")
print(f"High Value customers     : {(rfm['segment']=='High Value').sum():,}")
print(f"Medium Value customers   : {(rfm['segment']=='Medium Value').sum():,}")
print(f"Low Value customers      : {(rfm['segment']=='Low Value').sum():,}")
print("Module B — RFM Clustering complete.")
print("=" * 55)
