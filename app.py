# ============================================================
# FILE: app.py
# PURPOSE: Flask backend - serves the website, runs live SQL
#          queries, trains/serves the ML model, and powers
#          the AI assistant - all connected to PostgreSQL
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from config.db_config import SQLALCHEMY_URL

app = Flask(__name__)
engine = create_engine(SQLALCHEMY_URL)

# ============================================================
# GLOBAL: Trained model cache
# We train once on startup and reuse for every /api/predict call
# This avoids retraining on every request (slow) while still
# being a genuinely live model trained from PostgreSQL data
# ============================================================
MODEL_CACHE = {"model": None, "mae": None, "r2": None, "feature_cols": None}

def train_model_from_db():
    """Trains the Random Forest model using the same weekly
    aggregation approach as module_b/ml_model.py, but reading
    live from PostgreSQL so it stays consistent with the data."""
    orders = pd.read_sql("SELECT * FROM orders", engine)
    items = pd.read_sql("SELECT * FROM order_items", engine)
    payments = pd.read_sql("SELECT * FROM order_payments", engine)
    reviews = pd.read_sql("SELECT * FROM order_reviews", engine)

    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'], errors='coerce')
    orders['year_week'] = orders['order_purchase_timestamp'].dt.to_period('W')

    orders['delivery_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    orders['delivery_delay'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days

    df = pd.merge(orders, items[['order_id', 'price', 'freight_value']], on='order_id', how='left')
    df = pd.merge(df, payments.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'mean')
    ).reset_index(), on='order_id', how='left')
    df = pd.merge(df, reviews[['order_id', 'review_score']], on='order_id', how='left')

    weekly = df.groupby('year_week').agg(
        total_revenue=('price', 'sum'),
        order_count=('order_id', 'nunique'),
        avg_price=('price', 'mean'),
        avg_freight=('freight_value', 'mean'),
        avg_payment_value=('payment_value', 'mean'),
        avg_installments=('payment_installments', 'mean'),
        avg_review_score=('review_score', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_delivery_delay=('delivery_delay', 'mean')
    ).reset_index().dropna()

    feature_cols = ['order_count', 'avg_price', 'avg_freight', 'avg_payment_value',
                     'avg_installments', 'avg_review_score', 'avg_delivery_days', 'avg_delivery_delay']

    X = weekly[feature_cols]
    y = weekly['total_revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    MODEL_CACHE["model"] = model
    MODEL_CACHE["mae"] = round(float(mae), 2)
    MODEL_CACHE["r2"] = round(float(r2), 4)
    MODEL_CACHE["feature_cols"] = feature_cols
    print(f"Model trained! MAE={mae:.2f}  R2={r2:.4f}")

# Train the model once when the Flask app starts
train_model_from_db()

# ============================================================
# PAGE ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html', active='home')

@app.route('/queries')
def queries_page():
    return render_template('queries.html', active='queries')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', active='predict')

# ============================================================
# API: KPI summary data for home page
# ============================================================
@app.route('/api/kpis')
def get_kpis():
    with engine.connect() as conn:
        total_orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        total_revenue = conn.execute(text("SELECT SUM(price) FROM order_items")).scalar()
        avg_rating = conn.execute(text("SELECT AVG(review_score) FROM order_reviews")).scalar()
        total_customers = conn.execute(text("SELECT COUNT(DISTINCT customer_id) FROM customers")).scalar()

    return jsonify({
        "total_orders": int(total_orders),
        "total_revenue": round(float(total_revenue), 2),
        "avg_rating": round(float(avg_rating), 2),
        "total_customers": int(total_customers)
    })

# ============================================================
# API: Revenue by state (for chart)
# ============================================================
@app.route('/api/revenue_by_state')
def revenue_by_state():
    query = """
        SELECT c.customer_state AS state, SUM(oi.price) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.customer_state
        ORDER BY revenue DESC
        LIMIT 10
    """
    df = pd.read_sql(query, engine)
    return jsonify({
        "labels": df['state'].tolist(),
        "values": [round(v, 2) for v in df['revenue'].tolist()]
    })

# ============================================================
# API: Customer segments (RFM)
# ============================================================
@app.route('/api/segments')
def get_segments():
    df = pd.read_sql("SELECT segment, COUNT(*) as count FROM rfm_segments GROUP BY segment", engine)
    return jsonify({
        "labels": df['segment'].tolist(),
        "values": [int(v) for v in df['count'].tolist()]
    })

# ============================================================
# API: Run live SQL queries for the SQL Insights page
# Each of the 5 queries from module_a/sql_queries.py is
# exposed here as its own endpoint, timed and returned as JSON
# ============================================================
def run_and_time(query):
    start = time.time()
    df = pd.read_sql(query, engine)
    exec_time = round((time.time() - start) * 1000, 1)
    df = df.round(2)
    df = df.replace({np.nan: None})
    return df.to_dict(orient='records'), exec_time

@app.route('/api/run_query/top_sellers')
def run_top_sellers():
    rows, t = run_and_time("""
        WITH seller_revenue AS (
            SELECT seller_id, SUM(price) AS total_revenue
            FROM order_items GROUP BY seller_id
        )
        SELECT seller_id, total_revenue,
               RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
        FROM seller_revenue ORDER BY total_revenue DESC LIMIT 10
    """)
    return jsonify({"rows": rows, "exec_time": t})

@app.route('/api/run_query/monthly_growth')
def run_monthly_growth():
    rows, t = run_and_time("""
        WITH monthly_orders AS (
            SELECT TO_CHAR(order_purchase_timestamp, 'YYYY-MM') AS month,
                   COUNT(order_id) AS total_orders
            FROM orders GROUP BY 1
        )
        SELECT month, total_orders,
               LAG(total_orders) OVER (ORDER BY month) AS prev_month
        FROM monthly_orders ORDER BY month
    """)
    return jsonify({"rows": rows, "exec_time": t})

@app.route('/api/run_query/delivery_delay')
def run_delivery_delay():
    rows, t = run_and_time("""
        WITH delivery_data AS (
            SELECT c.customer_state,
                   EXTRACT(DAY FROM (order_delivered_customer_date - order_estimated_delivery_date)) AS delay_days
            FROM orders o JOIN customers c ON o.customer_id = c.customer_id
            WHERE order_delivered_customer_date IS NOT NULL
        )
        SELECT customer_state, ROUND(AVG(delay_days),2) AS avg_delay
        FROM delivery_data GROUP BY customer_state
        ORDER BY avg_delay DESC LIMIT 10
    """)
    return jsonify({"rows": rows, "exec_time": t})

@app.route('/api/run_query/repeat_rate')
def run_repeat_rate():
    rows, t = run_and_time("""
        WITH customer_orders AS (
            SELECT customer_unique_id, COUNT(o.order_id) AS order_count
            FROM customers c JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY customer_unique_id
        )
        SELECT
            CASE WHEN order_count = 1 THEN 'One-Time Buyer'
                 WHEN order_count = 2 THEN 'Repeat Buyer (2x)'
                 ELSE 'Loyal Buyer (3x+)' END AS buyer_type,
            COUNT(*) AS customer_count
        FROM customer_orders GROUP BY buyer_type
    """)
    return jsonify({"rows": rows, "exec_time": t})

@app.route('/api/run_query/cancellation_rate')
def run_cancellation_rate():
    rows, t = run_and_time("""
        WITH category_orders AS (
            SELECT p.product_category_name,
                   COUNT(o.order_id) AS total_orders,
                   SUM(CASE WHEN o.order_status='canceled' THEN 1 ELSE 0 END) AS cancelled
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            GROUP BY p.product_category_name
            HAVING COUNT(o.order_id) >= 50
        )
        SELECT product_category_name, total_orders, cancelled,
               ROUND(cancelled * 100.0 / total_orders, 2) AS cancellation_rate_pct
        FROM category_orders
        ORDER BY cancellation_rate_pct DESC LIMIT 10
    """)
    return jsonify({"rows": rows, "exec_time": t})

# ============================================================
# API: ML Prediction endpoint
# Takes feature values from the form and returns a live
# prediction from the cached Random Forest model
# ============================================================
@app.route('/api/predict', methods=['POST'])
def predict_revenue():
    data = request.json
    feature_cols = MODEL_CACHE["feature_cols"]

    input_row = pd.DataFrame([[data[col] for col in feature_cols]], columns=feature_cols)
    prediction = MODEL_CACHE["model"].predict(input_row)[0]

    return jsonify({
        "prediction": round(float(prediction), 2),
        "mae": MODEL_CACHE["mae"],
        "r2": MODEL_CACHE["r2"]
    })

# ============================================================
# AI ASSISTANT HELPER: Find a specific state's order count
# ============================================================
BRAZIL_STATES = ['SP','RJ','MG','RS','PR','SC','BA','DF','ES','GO',
                  'PE','CE','PA','MT','MA','MS','PB','PI','RN','AL',
                  'SE','TO','RO','AM','AC','AP','RR']

def find_state_in_question(question):
    upper_q = question.upper()
    for state in BRAZIL_STATES:
        if f" {state} " in f" {upper_q} " or upper_q.endswith(state):
            return state
    return None

# ============================================================
# API: AI Assistant - answers questions about the dataset
# ============================================================
@app.route('/api/ask', methods=['POST'])
def ask_assistant():
    question = request.json.get('question', '').lower()

    with engine.connect() as conn:

        mentioned_state = find_state_in_question(question)
        if mentioned_state and ('order' in question or 'revenue' in question or 'sales' in question):
            result = conn.execute(text("""
                SELECT COUNT(o.order_id) as cnt, COALESCE(SUM(oi.price),0) as rev
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                LEFT JOIN order_items oi ON o.order_id = oi.order_id
                WHERE c.customer_state = :state
            """), {"state": mentioned_state}).fetchone()
            answer = f"{mentioned_state} has {result[0]:,} orders generating R$ {result[1]:,.2f} in revenue."

        elif 'top' in question and 'seller' in question:
            result = conn.execute(text("""
                SELECT seller_id, SUM(price) as revenue, COUNT(*) as orders
                FROM order_items GROUP BY seller_id
                ORDER BY revenue DESC LIMIT 1
            """)).fetchone()
            answer = f"The top seller is {result[0][:12]}... with total revenue of R$ {result[1]:,.2f} across {result[2]:,} order items."

        elif 'seller' in question and ('how many' in question or 'total' in question or 'number of' in question):
            result = conn.execute(text("SELECT COUNT(*) FROM sellers")).scalar()
            answer = f"There are {result:,} unique sellers on the platform."

        elif 'state' in question and ('most' in question or 'top' in question or 'highest' in question):
            result = conn.execute(text("""
                SELECT c.customer_state, COUNT(*) as cnt
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                GROUP BY c.customer_state ORDER BY cnt DESC LIMIT 1
            """)).fetchone()
            answer = f"{result[0]} has the most orders with {result[1]:,} orders — it's the leading state in this dataset."

        elif 'total revenue' in question or 'how much revenue' in question or ('revenue' in question and 'total' in question):
            result = conn.execute(text("SELECT SUM(price) FROM order_items")).scalar()
            answer = f"Total revenue across all orders is R$ {result:,.2f}."

        elif 'average order' in question or 'avg order value' in question:
            result = conn.execute(text("SELECT AVG(price) FROM order_items")).scalar()
            answer = f"The average order item value is R$ {result:,.2f}."

        elif 'rating' in question or 'review' in question:
            result = conn.execute(text("SELECT AVG(review_score) FROM order_reviews")).scalar()
            dist = conn.execute(text("""
                SELECT review_score, COUNT(*) as cnt FROM order_reviews
                GROUP BY review_score ORDER BY review_score DESC
            """)).fetchall()
            top = dist[0]
            answer = f"The average review score is {result:.2f} out of 5.0. {top[1]:,} customers gave a {top[0]}-star rating, the most common score."

        elif ('order' in question and 'how many' in question) or 'total orders' in question or 'number of orders' in question:
            result = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
            answer = f"There are {result:,} total orders in the dataset."

        elif 'repeat' in question and 'customer' in question:
            result = conn.execute(text("""
                WITH order_counts AS (
                    SELECT c.customer_unique_id, COUNT(o.order_id) as cnt
                    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
                    GROUP BY c.customer_unique_id
                )
                SELECT
                    COUNT(*) FILTER (WHERE cnt = 1) as one_time,
                    COUNT(*) FILTER (WHERE cnt > 1) as repeat_buyers,
                    COUNT(*) as total
                FROM order_counts
            """)).fetchone()
            pct = (result[1] / result[2]) * 100
            answer = f"Out of {result[2]:,} unique customers, only {result[1]:,} ({pct:.1f}%) are repeat buyers. The remaining {result[0]:,} bought only once."

        elif 'customer' in question and ('how many' in question or 'total' in question or 'number of' in question):
            result = conn.execute(text("SELECT COUNT(DISTINCT customer_id) FROM customers")).scalar()
            answer = f"There are {result:,} unique customers in the dataset."

        elif 'delivery' in question or 'delay' in question or 'shipping time' in question:
            result = conn.execute(text("""
                SELECT AVG(EXTRACT(DAY FROM (order_delivered_customer_date - order_purchase_timestamp)))
                FROM orders WHERE order_delivered_customer_date IS NOT NULL
            """)).scalar()
            answer = f"The average delivery time is {result:.1f} days from purchase to delivery."

        elif 'cancel' in question:
            result = conn.execute(text("""
                SELECT COUNT(*) FILTER (WHERE order_status = 'canceled') as cancelled,
                       COUNT(*) as total
                FROM orders
            """)).fetchone()
            pct = (result[0] / result[1]) * 100
            answer = f"{result[0]:,} out of {result[1]:,} orders were cancelled — a cancellation rate of {pct:.2f}%."

        elif 'payment' in question:
            result = conn.execute(text("""
                SELECT payment_type, COUNT(*) as cnt FROM order_payments
                GROUP BY payment_type ORDER BY cnt DESC LIMIT 1
            """)).fetchone()
            total = conn.execute(text("SELECT COUNT(*) FROM order_payments")).scalar()
            pct = (result[1] / total) * 100
            answer = f"The most used payment method is {result[0]} with {result[1]:,} transactions ({pct:.1f}% of all payments)."

        elif 'category' in question and ('best' in question or 'top' in question or 'highest rated' in question):
            result = conn.execute(text("""
                SELECT p.product_category_name, AVG(r.review_score) as avg_score, COUNT(*) as cnt
                FROM order_reviews r
                JOIN order_items oi ON r.order_id = oi.order_id
                JOIN products p ON oi.product_id = p.product_id
                GROUP BY p.product_category_name
                HAVING COUNT(*) >= 50
                ORDER BY avg_score DESC LIMIT 1
            """)).fetchone()
            answer = f"The best rated category is {result[0]} with an average rating of {result[1]:.2f} out of {result[2]:,} reviews."

        elif 'category' in question and ('worst' in question or 'lowest' in question):
            result = conn.execute(text("""
                SELECT p.product_category_name, AVG(r.review_score) as avg_score, COUNT(*) as cnt
                FROM order_reviews r
                JOIN order_items oi ON r.order_id = oi.order_id
                JOIN products p ON oi.product_id = p.product_id
                GROUP BY p.product_category_name
                HAVING COUNT(*) >= 50
                ORDER BY avg_score ASC LIMIT 1
            """)).fetchone()
            answer = f"The lowest rated category is {result[0]} with an average rating of {result[1]:.2f} out of {result[2]:,} reviews."

        elif 'category' in question and ('how many' in question or 'number of' in question):
            result = conn.execute(text("SELECT COUNT(DISTINCT product_category_name) FROM products")).scalar()
            answer = f"There are {result:,} distinct product categories in this dataset."

        elif 'high value' in question or 'segment' in question or 'rfm' in question:
            result = conn.execute(text("SELECT segment, COUNT(*) as cnt FROM rfm_segments GROUP BY segment ORDER BY cnt DESC")).fetchall()
            parts = [f"{r[0]}: {r[1]:,} customers" for r in result]
            answer = ("Using RFM analysis and K-Means clustering, customers are segmented into 3 groups — " +
                      ", ".join(parts) + ". High Value customers buy more frequently and spend more.")

        elif 'product' in question and ('how many' in question or 'total' in question or 'number of' in question):
            result = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
            answer = f"There are {result:,} unique products listed in the dataset."

        elif 'csv' in question or 'dataset' in question or 'how many files' in question or 'about this project' in question or 'about the project' in question:
            answer = ("This project uses the Olist Brazilian E-Commerce dataset from Kaggle, "
                      "containing 9 CSV files — customers, orders, order_items, order_payments, "
                      "order_reviews, products, sellers, geolocation, and product_category_translation. "
                      "Together they contain over 100,000 real orders and 1.5 million+ rows of data, "
                      "loaded into PostgreSQL and analysed using SQL, pandas, and scikit-learn.")

        elif question.strip() in ['hi', 'hello', 'hey', 'hi there', 'hello there']:
            answer = "Hello! Ask me anything about orders, revenue, customers, ratings, sellers, or the dataset itself."

        else:
            answer = ("I can answer questions about: total orders, revenue, average rating, top sellers, "
                      "states (e.g. 'orders in SP'), delivery times, cancellation rate, repeat customers, "
                      "payment methods, product categories, and customer segments. Try rephrasing your question "
                      "or pick one of the suggestions below.")

    return jsonify({"answer": answer})

# ============================================================
# Run the Flask app
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
