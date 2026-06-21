# ============================================================
# FILE: module_b/dashboard.py
# PURPOSE: Streamlit dashboard connected live to PostgreSQL
# SYLLABUS: Unit IV — Presenting and Publishing Data
#           Unit V  — Visualizing Data with Pandas & Matplotlib
# HOW TO RUN: streamlit run module_b/dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# ============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in the script
# ============================================================
st.set_page_config(
    page_title = "Smart E-Commerce Analytics",
    page_icon  = "🛒",
    layout     = "wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #fafafa; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #f3f4f6;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 11px !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 26px !important;
        font-weight: 500 !important;
    }
    h2, h3 { color: #111827 !important; font-weight: 500 !important; }
    .stDataFrame { border: 1px solid #f3f4f6; border-radius: 10px; }
    hr { border-color: #f3f4f6; }
    .stSelectbox label { color: #6b7280 !important; font-size: 11px !important; }
    .stSlider label { color: #6b7280 !important; font-size: 11px !important; }
    h1 { color: #111827 !important; font-size: 1.6rem !important; font-weight: 500 !important; }
    .stMarkdown p { color: #6b7280; font-size: 12px; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE CONNECTION
# st.cache_resource caches the engine so it doesn't reconnect
# on every page refresh — important for performance
# ============================================================
@st.cache_resource
def get_engine():
    return create_engine(SQLALCHEMY_URL)

engine = get_engine()

# ============================================================
# DATA LOADING FUNCTIONS
# st.cache_data caches query results for 10 minutes
# This means the dashboard doesn't re-query PostgreSQL
# every time the user interacts with it
# ============================================================
@st.cache_data(ttl=600)
def load_orders():
    return pd.read_sql("SELECT * FROM orders", engine)

@st.cache_data(ttl=600)
def load_items():
    return pd.read_sql("SELECT * FROM order_items", engine)

@st.cache_data(ttl=600)
def load_payments():
    return pd.read_sql("SELECT * FROM order_payments", engine)

@st.cache_data(ttl=600)
def load_reviews():
    return pd.read_sql("SELECT * FROM order_reviews", engine)

@st.cache_data(ttl=600)
def load_customers():
    return pd.read_sql("SELECT * FROM customers", engine)

@st.cache_data(ttl=600)
def load_rfm():
    return pd.read_sql("SELECT * FROM rfm_segments", engine)

# ============================================================
# DASHBOARD TITLE
# ============================================================
col_logo1, col_logo2, col_logo3 = st.columns([0.05, 0.05, 0.05])
with col_logo1:
    st.markdown("<div style='width:10px;height:10px;border-radius:50%;background:#3b82f6;margin-top:18px'></div>", unsafe_allow_html=True)
with col_logo2:
    st.markdown("<div style='width:10px;height:10px;border-radius:50%;background:#8b5cf6;margin-top:18px'></div>", unsafe_allow_html=True)
with col_logo3:
    st.markdown("<div style='width:10px;height:10px;border-radius:50%;background:#10b981;margin-top:18px'></div>", unsafe_allow_html=True)

st.title("Smart E-Commerce Analytics")
st.markdown("<p style='color:#9ca3af;font-size:11px;letter-spacing:.5px;margin-top:-10px'>OLIST DATASET &nbsp;·&nbsp; QUERY PROCESSING FOR DATA SCIENCE &nbsp;·&nbsp; POSTGRESQL LIVE</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD ALL DATA
# Show a spinner while data is loading
# ============================================================
with st.spinner("Loading data from PostgreSQL..."):
    orders   = load_orders()
    items    = load_items()
    payments = load_payments()
    reviews  = load_reviews()
    customers= load_customers()
    rfm      = load_rfm()

# Convert timestamps
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)

# ============================================================
# SECTION 1: KPI CARDS
# Four headline metrics shown at the top of dashboard
# st.metric() creates a clean KPI card with a label and value
# ============================================================
st.subheader("📊 Key Performance Indicators")

# Calculate KPIs
total_orders   = len(orders)
total_revenue  = items['price'].sum()
avg_rating     = reviews['review_score'].mean()
avg_delivery   = (
    pd.to_datetime(orders['order_delivered_customer_date'],
                   errors='coerce') -
    pd.to_datetime(orders['order_purchase_timestamp'],
                   errors='coerce')
).dt.days.mean()

# Display 4 KPI cards in 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div style='height:3px;background:#3b82f6;border-radius:3px;margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.metric(label="TOTAL ORDERS", value=f"{total_orders:,}")
    st.caption("96.8% delivered")

with col2:
    st.markdown("<div style='height:3px;background:#10b981;border-radius:3px;margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.metric(label="TOTAL REVENUE", value=f"R$ {total_revenue:,.0f}")
    st.caption("88,496 items sold")

with col3:
    st.markdown("<div style='height:3px;background:#f59e0b;border-radius:3px;margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.metric(label="AVG RATING", value=f"{avg_rating:.2f} / 5.0")
    st.caption("out of 5.0")

with col4:
    st.markdown("<div style='height:3px;background:#8b5cf6;border-radius:3px;margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.metric(label="AVG DELIVERY", value=f"{avg_delivery:.1f} days")
    st.caption("all delivered early")

st.divider()

# ============================================================
# SECTION 2: MONTHLY ORDER TREND
# Line chart showing order volume over time
# ============================================================
st.subheader("📈 Monthly Order Volume Trend")

orders['year_month'] = orders[
    'order_purchase_timestamp'
].dt.to_period('M')

monthly = orders.groupby('year_month')['order_id'].count().reset_index()
monthly.columns = ['month', 'order_count']
monthly['month_str'] = monthly['month'].astype(str)
monthly = monthly.sort_values('month')

fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor='#ffffff')
ax1.set_facecolor('#ffffff')
ax1.plot(monthly['month_str'], monthly['order_count'],
         marker='o', linewidth=2, color='#3b82f6',
         markersize=4, markerfacecolor='#8b5cf6')
ax1.fill_between(monthly['month_str'], monthly['order_count'],
                 alpha=0.07, color='#3b82f6')
ax1.set_xlabel('Month', color='#9ca3af', fontsize=9)
ax1.set_ylabel('Number of Orders', color='#9ca3af', fontsize=9)
ax1.tick_params(axis='x', rotation=45, colors='#9ca3af', labelsize=8)
ax1.tick_params(axis='y', colors='#9ca3af', labelsize=8)
for spine in ax1.spines.values():
    spine.set_color('#f3f4f6')
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#f3f4f6')
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.divider()

# ============================================================
# SECTION 3: TWO CHARTS SIDE BY SIDE
# Revenue by state + Payment type distribution
# st.columns() creates side-by-side layout
# ============================================================
st.subheader("🗺️ Revenue by State  |  💳 Payment Methods")

col_left, col_right = st.columns(2)

# --- Left: Revenue by State ---
with col_left:
    orders_customers = pd.merge(
        orders[['order_id', 'customer_id']],
        customers[['customer_id', 'customer_state']],
        on='customer_id', how='left'
    )
    state_items = pd.merge(
        orders_customers,
        items[['order_id', 'price']],
        on='order_id', how='left'
    )
    state_revenue = state_items.groupby('customer_state')[
        'price'
    ].sum().sort_values(ascending=False).head(10)

    bar_colors2 = ['#3b82f6','#8b5cf6','#10b981','#f59e0b',
                   '#ef4444','#3b82f6','#8b5cf6','#10b981',
                   '#f59e0b','#ef4444']
    fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor='#ffffff')
    ax2.set_facecolor('#ffffff')
    ax2.bar(state_revenue.index, state_revenue.values,
            color=bar_colors2[:len(state_revenue)], edgecolor='white',
            width=0.6)
    ax2.set_title('Top 10 States by Revenue',
                  fontsize=11, color='#374151', fontweight='500')
    ax2.set_xlabel('State', color='#9ca3af', fontsize=9)
    ax2.set_ylabel('Revenue (BRL)', color='#9ca3af', fontsize=9)
    ax2.tick_params(axis='x', rotation=45, colors='#9ca3af', labelsize=8)
    ax2.tick_params(axis='y', colors='#9ca3af', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#f3f4f6')
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# --- Right: Payment Type Distribution ---
with col_right:
    payment_dist = payments.groupby('payment_type')[
        'order_id'
    ].count().sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor='#ffffff')
    ax3.set_facecolor('#ffffff')
    pay_colors = ['#3b82f6','#8b5cf6','#10b981','#f59e0b']
    ax3.bar(payment_dist.index, payment_dist.values,
            color=pay_colors[:len(payment_dist)],
            edgecolor='white', width=0.5)
    ax3.set_title('Payment Type Distribution',
                  fontsize=11, color='#374151', fontweight='500')
    ax3.set_xlabel('Payment Type', color='#9ca3af', fontsize=9)
    ax3.set_ylabel('Number of Transactions', color='#9ca3af', fontsize=9)
    ax3.tick_params(axis='x', colors='#9ca3af', labelsize=8)
    ax3.tick_params(axis='y', colors='#9ca3af', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color('#f3f4f6')
    ax3.grid(axis='y', linestyle='--', alpha=0.5, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.divider()

# ============================================================
# SECTION 4: RFM CUSTOMER SEGMENTS
# ============================================================
st.subheader("👥 Customer Segmentation (RFM + K-Means)")

col_seg1, col_seg2 = st.columns(2)

with col_seg1:
    # Segment counts bar chart
    seg_counts = rfm['segment'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor='#ffffff')
    ax4.set_facecolor('#ffffff')
    seg_colors = ['#f59e0b','#3b82f6','#10b981']
    ax4.bar(seg_counts.index, seg_counts.values,
            color=seg_colors[:len(seg_counts)],
            edgecolor='white', width=0.5)
    ax4.set_title('Customer Segments',
                  fontsize=11, color='#374151', fontweight='500')
    ax4.set_xlabel('Segment', color='#9ca3af', fontsize=9)
    ax4.set_ylabel('Number of Customers', color='#9ca3af', fontsize=9)
    ax4.tick_params(axis='x', colors='#9ca3af', labelsize=8)
    ax4.tick_params(axis='y', colors='#9ca3af', labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color('#f3f4f6')
    ax4.grid(axis='y', linestyle='--', alpha=0.5, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

with col_seg2:
    # Segment summary table
    seg_summary = rfm.groupby('segment').agg(
        Customers     = ('customer_unique_id', 'count'),
        Avg_Recency   = ('recency',   'mean'),
        Avg_Frequency = ('frequency', 'mean'),
        Avg_Monetary  = ('monetary',  'mean')
    ).round(2).reset_index()
    seg_summary.columns = ['Segment', 'Customers',
                           'Avg Recency', 'Avg Frequency',
                           'Avg Monetary (R$)']
    st.dataframe(seg_summary, use_container_width=True)
    st.caption("High Value = frequent buyers with high spend | "
               "Low Value = recent but low spend")

st.divider()

# ============================================================
# SECTION 5: REVIEW SCORE DISTRIBUTION
# ============================================================
st.subheader("⭐ Review Score Distribution")

col_rev1, col_rev2 = st.columns(2)

with col_rev1:
    review_counts = reviews['review_score'].value_counts().sort_index()
    fig5, ax5 = plt.subplots(figsize=(6, 4), facecolor='#ffffff')
    ax5.set_facecolor('#ffffff')
    score_colors = ['#ef4444','#f97316','#f59e0b','#3b82f6','#10b981']
    ax5.bar(review_counts.index.astype(str), review_counts.values,
            color=score_colors, edgecolor='white', width=0.6)
    ax5.set_title('Review Score Distribution',
                  fontsize=11, color='#374151', fontweight='500')
    ax5.set_xlabel('Score', color='#9ca3af', fontsize=9)
    ax5.set_ylabel('Count', color='#9ca3af', fontsize=9)
    ax5.tick_params(axis='x', colors='#9ca3af', labelsize=8)
    ax5.tick_params(axis='y', colors='#9ca3af', labelsize=8)
    for spine in ax5.spines.values():
        spine.set_color('#f3f4f6')
    ax5.grid(axis='y', linestyle='--', alpha=0.5, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

with col_rev2:
    # Monthly avg rating trend
    orders['year_month_str'] = orders['year_month'].astype(str)
    orders_reviews = pd.merge(
        orders[['order_id', 'year_month_str']],
        reviews[['order_id', 'review_score']],
        on='order_id', how='inner'
    )
    monthly_rating = orders_reviews.groupby(
        'year_month_str'
    )['review_score'].mean().reset_index()
    monthly_rating = monthly_rating.sort_values('year_month_str')

    fig6, ax6 = plt.subplots(figsize=(6, 4), facecolor='#ffffff')
    ax6.set_facecolor('#ffffff')
    ax6.plot(monthly_rating['year_month_str'],
             monthly_rating['review_score'],
             marker='o', linewidth=2, color='#f59e0b',
             markersize=4, markerfacecolor='#8b5cf6')
    ax6.fill_between(monthly_rating['year_month_str'],
                     monthly_rating['review_score'],
                     alpha=0.07, color='#f59e0b')
    ax6.set_title('Monthly Avg Rating Trend',
                  fontsize=11, color='#374151', fontweight='500')
    ax6.set_xlabel('Month', color='#9ca3af', fontsize=9)
    ax6.set_ylabel('Avg Rating', color='#9ca3af', fontsize=9)
    ax6.set_ylim(1, 5)
    ax6.tick_params(axis='x', rotation=45, colors='#9ca3af', labelsize=8)
    ax6.tick_params(axis='y', colors='#9ca3af', labelsize=8)
    for spine in ax6.spines.values():
        spine.set_color('#f3f4f6')
    ax6.grid(linestyle='--', alpha=0.5, color='#f3f4f6')
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

st.divider()

# ============================================================
# SECTION 6: INTERACTIVE DATA EXPLORER
# Lets faculty filter data live during evaluation
# st.selectbox() creates a dropdown menu
# st.slider() creates an interactive slider
# ============================================================
st.subheader("🔍 Interactive Data Explorer")

# Dropdown to select order status
status_options = ['All'] + list(orders['order_status'].unique())
selected_status = st.selectbox(
    "Filter by Order Status:", status_options
)

# Slider for price range
min_price = float(items['price'].min())
max_price = float(items['price'].max())
price_range = st.slider(
    "Filter by Price Range (BRL):",
    min_value = min_price,
    max_value = max_price,
    value     = (min_price, max_price)
)

# Apply filters
filtered_items = items[
    (items['price'] >= price_range[0]) &
    (items['price'] <= price_range[1])
]

if selected_status != 'All':
    filtered_orders = orders[orders['order_status'] == selected_status]
    filtered_items  = filtered_items[
        filtered_items['order_id'].isin(filtered_orders['order_id'])
    ]

# Show filtered results
st.write(f"Showing **{len(filtered_items):,}** records")
st.dataframe(
    filtered_items[['order_id', 'product_id',
                    'seller_id', 'price', 'freight_value']].head(100),
    use_container_width=True
)

st.divider()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "**Smart E-Commerce Analytics** | "
    "Module A: Data Pipeline + SQL | "
    "Module B: EDA + ML + Dashboard | "
    "Built with Python, PostgreSQL, Streamlit"
)
