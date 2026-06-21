# ============================================================
# FILE: module_a/sql_queries.py
# PURPOSE: Execute 5 advanced SQL queries using psycopg2
#          Each query answers a real business question
# SYLLABUS: Unit II  — Python SQL Libraries (psycopg2)
#           Unit IV  — Data Exploration and Analysis
# CONCEPTS USED:
#   - CTE (Common Table Expressions) using WITH clause
#   - Window functions: RANK(), LAG(), LEAD(), PARTITION BY
#   - GROUP BY, ORDER BY, HAVING, CASE WHEN
# ============================================================

import psycopg2                    # Direct PostgreSQL connection library
import pandas as pd                # For displaying results as tables
from tabulate import tabulate      # For pretty printing query results
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import PSYCOPG2_CONFIG

# ============================================================
# Connect to PostgreSQL using psycopg2
# psycopg2 is different from SQLAlchemy — it gives us direct
# cursor-based control over SQL execution
# ============================================================
print("Connecting to PostgreSQL using psycopg2...")
try:
    conn   = psycopg2.connect(**PSYCOPG2_CONFIG)  # ** unpacks dictionary as arguments
    cursor = conn.cursor()                         # Cursor executes SQL statements
    print("Connected successfully!\n")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

# ============================================================
# HELPER FUNCTION: Run a query and return results as DataFrame
# cursor.execute() sends SQL to PostgreSQL
# cursor.fetchall() retrieves all result rows
# cursor.description gives us the column names
# ============================================================
def run_query(query, title):
    print("=" * 65)
    print(f"QUERY: {title}")
    print("=" * 65)
    try:
        cursor.execute(query)
        rows    = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]  # Extract column names
        df      = pd.DataFrame(rows, columns=columns)        # Convert to DataFrame
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print(f"\nTotal rows returned: {len(df)}")
        return df
    except Exception as e:
        print(f"Query failed: {e}")
        conn.rollback()  # Reset connection after error
        return None
    print()

# ============================================================
# QUERY 1: Top 10 Sellers by Revenue with Their Rank
# CONCEPTS: CTE, RANK() window function, PARTITION BY
# BUSINESS QUESTION: Which sellers are generating the most
#                    revenue and what is their rank?
# ============================================================
query1 = """
-- CTE 1: Calculate total revenue per seller
WITH seller_revenue AS (
    SELECT
        oi.seller_id,
        s.seller_city,
        s.seller_state,
        ROUND(CAST(SUM(oi.price + oi.freight_value) AS NUMERIC), 2) AS total_revenue,
        COUNT(DISTINCT oi.order_id) AS total_orders
    FROM order_items oi
    JOIN sellers s ON oi.seller_id = s.seller_id
    GROUP BY oi.seller_id, s.seller_city, s.seller_state
),

-- CTE 2: Rank sellers by revenue using RANK() window function
-- RANK() assigns a rank to each row based on ORDER BY
-- PARTITION BY is not used here so all sellers are ranked together
ranked_sellers AS (
    SELECT
        seller_id,
        seller_city,
        seller_state,
        total_revenue,
        total_orders,
        RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
    FROM seller_revenue
)

-- Final SELECT: Show only top 10 ranked sellers
SELECT
    revenue_rank,
    seller_id,
    seller_city,
    seller_state,
    total_revenue,
    total_orders
FROM ranked_sellers
WHERE revenue_rank <= 10
ORDER BY revenue_rank;
"""

df1 = run_query(query1, "Top 10 Sellers by Revenue (RANK + CTE)")
print()

# ============================================================
# QUERY 2: Month-over-Month Order Growth Rate
# CONCEPTS: CTE, LAG() window function
# BUSINESS QUESTION: How did order volume grow or shrink
#                    compared to the previous month?
# LAG() looks at the previous row's value in the result set
# ============================================================
query2 = """
-- CTE: Count total orders per month
WITH monthly_orders AS (
    SELECT
        TO_CHAR(order_purchase_timestamp, 'YYYY-MM') AS order_month,
        COUNT(order_id) AS total_orders
    FROM orders
    WHERE order_purchase_timestamp IS NOT NULL
    GROUP BY TO_CHAR(order_purchase_timestamp, 'YYYY-MM')
    ORDER BY order_month
),

-- Add previous month's order count using LAG()
-- LAG(column, 1) gets the value from 1 row before current row
growth_calc AS (
    SELECT
        order_month,
        total_orders,
        LAG(total_orders, 1) OVER (ORDER BY order_month) AS prev_month_orders
    FROM monthly_orders
)

-- Calculate growth rate as percentage
-- ROUND to 2 decimal places for clean output
SELECT
    order_month,
    total_orders,
    prev_month_orders,
    CASE
        WHEN prev_month_orders IS NULL THEN NULL
        ELSE ROUND(
            CAST(
                ((total_orders - prev_month_orders) * 100.0 / prev_month_orders)
            AS NUMERIC), 2)
    END AS growth_rate_pct
FROM growth_calc
ORDER BY order_month;
"""

df2 = run_query(query2, "Month-over-Month Order Growth Rate (LAG + CTE)")
print()

# ============================================================
# QUERY 3: Average Delivery Delay per State
# CONCEPTS: CTE, CASE WHEN, date arithmetic
# BUSINESS QUESTION: Which states have the worst delivery
#                    delays compared to estimated date?
# ============================================================
query3 = """
-- CTE: Calculate delivery delay for each delivered order
-- Delay = actual delivery date minus estimated delivery date
-- Positive delay = late delivery, Negative = early delivery
WITH delivery_data AS (
    SELECT
        c.customer_state,
        o.order_id,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        EXTRACT(DAY FROM (
            o.order_delivered_customer_date - o.order_estimated_delivery_date
        )) AS delay_days
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
),

-- CTE: Aggregate delay by state
state_delays AS (
    SELECT
        customer_state,
        COUNT(order_id)                              AS total_orders,
        ROUND(CAST(AVG(delay_days) AS NUMERIC), 2)  AS avg_delay_days,
        ROUND(CAST(MIN(delay_days) AS NUMERIC), 2)  AS min_delay_days,
        ROUND(CAST(MAX(delay_days) AS NUMERIC), 2)  AS max_delay_days,
        -- Classify each state as On Time or Delayed
        CASE
            WHEN AVG(delay_days) > 0 THEN 'Delayed'
            ELSE 'On Time'
        END AS delivery_status
    FROM delivery_data
    GROUP BY customer_state
)

SELECT *
FROM state_delays
ORDER BY avg_delay_days DESC;
"""

df3 = run_query(query3, "Average Delivery Delay per State (CTE + CASE WHEN)")
print()

# ============================================================
# QUERY 4: Customer Repeat Purchase Rate
# CONCEPTS: CTE, COUNT, CASE WHEN, aggregation
# BUSINESS QUESTION: How many customers made more than
#                    one purchase? What is the repeat rate?
# ============================================================
query4 = """
-- CTE: Count orders per unique customer
WITH customer_orders AS (
    SELECT
        customer_unique_id,
        COUNT(o.order_id) AS order_count
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY customer_unique_id
),

-- CTE: Classify customers as one-time or repeat buyers
customer_segments AS (
    SELECT
        customer_unique_id,
        order_count,
        CASE
            WHEN order_count = 1 THEN 'One-Time Buyer'
            WHEN order_count = 2 THEN 'Repeat Buyer (2x)'
            ELSE 'Loyal Buyer (3x+)'
        END AS buyer_type
    FROM customer_orders
)

-- Final: Count customers in each segment with percentage
SELECT
    buyer_type,
    COUNT(customer_unique_id)                           AS customer_count,
    ROUND(
        CAST(COUNT(customer_unique_id) * 100.0 /
        SUM(COUNT(customer_unique_id)) OVER ()
        AS NUMERIC), 2)                                  AS percentage
FROM customer_segments
GROUP BY buyer_type
ORDER BY customer_count DESC;
"""

df4 = run_query(query4, "Customer Repeat Purchase Rate (CTE + CASE WHEN)")
print()

# ============================================================
# QUERY 5: Product Categories with Highest Cancellation Rate
# CONCEPTS: CTE, LEAD() window function, CASE WHEN
# BUSINESS QUESTION: Which product categories have the most
#                    cancellations and what is their rate?
# LEAD() looks at the next row's value in the result set
# ============================================================
query5 = """
-- CTE: Count total and cancelled orders per category
WITH category_orders AS (
    SELECT
        COALESCE(pct.product_category_name_english,
                 p.product_category_name,
                 'unknown')          AS category,
        COUNT(o.order_id)            AS total_orders,
        SUM(CASE WHEN o.order_status = 'canceled'
                 THEN 1 ELSE 0 END)  AS cancelled_orders
    FROM orders o
    JOIN order_items oi  ON o.order_id   = oi.order_id
    JOIN products p      ON oi.product_id = p.product_id
    LEFT JOIN product_category_translation pct
           ON p.product_category_name = pct.product_category_name
    GROUP BY COALESCE(pct.product_category_name_english,
                      p.product_category_name, 'unknown')
    HAVING COUNT(o.order_id) >= 50   -- Only categories with enough orders
),

-- CTE: Calculate cancellation rate and add LEAD() for next category comparison
cancellation_rates AS (
    SELECT
        category,
        total_orders,
        cancelled_orders,
        ROUND(
            CAST(cancelled_orders * 100.0 / total_orders AS NUMERIC), 2
        ) AS cancellation_rate_pct,
        -- LEAD() shows the next category's cancellation rate for comparison
        LEAD(
            ROUND(CAST(cancelled_orders * 100.0 / total_orders AS NUMERIC), 2)
        ) OVER (ORDER BY cancelled_orders * 100.0 / total_orders DESC)
            AS next_category_rate
    FROM category_orders
)

SELECT
    category,
    total_orders,
    cancelled_orders,
    cancellation_rate_pct,
    next_category_rate
FROM cancellation_rates
ORDER BY cancellation_rate_pct DESC
LIMIT 15;
"""

df5 = run_query(query5, "Top 15 Categories by Cancellation Rate (LEAD + CTE)")
print()

# ============================================================
# Close the psycopg2 connection
# Always close cursor and connection after use
# ============================================================
cursor.close()
conn.close()
print("=" * 65)
print("All 5 SQL queries executed successfully!")
print("psycopg2 connection closed.")
print("Module A — SQL Queries complete.")
print("=" * 65)
