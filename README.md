# Smart E-Commerce Analytics and Prediction Engine

A full data science capstone project built on the Olist Brazilian E-Commerce dataset (100k+ orders, 9 CSV files). This project demonstrates end-to-end data pipeline development, advanced SQL querying, exploratory data analysis, machine learning, and interactive visualization.

**Subject:** DSA05 — Query Processing for Data Science
**Dataset:** Brazilian E-Commerce Public Dataset by Olist (Kaggle)
**Tech Stack:** Python, PostgreSQL, psycopg2, SQLAlchemy, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit

---

## Data Flow

9 CSV Files → data_loader.py → PostgreSQL Database → data_cleaning.py → sql_queries.py + eda.py → rfm_clustering.py → ml_model.py → rating_analysis.py → dashboard.py

---

## Project Structure

smart_ecommerce_analytics/
├── data/raw/                         # 9 Olist CSV files
├── module_a/
│   ├── data_loader.py                # Load CSVs into PostgreSQL
│   ├── data_cleaning.py              # Clean nulls, duplicates, outliers
│   └── sql_queries.py                # 5 advanced SQL queries
├── module_b/
│   ├── eda.py                        # Exploratory Data Analysis
│   ├── rfm_clustering.py             # RFM Analysis + K-Means
│   ├── ml_model.py                   # Random Forest sales prediction
│   ├── rating_analysis.py            # Rating trend analysis
│   └── dashboard.py                  # Streamlit live dashboard
├── config/
│   └── db_config.py                  # Database connection settings
├── outputs/
│   ├── plots/                        # All saved chart images
│   └── reports/                      # Exported reports
├── main.py                           # Run entire pipeline in one command
├── requirements.txt                  # All required libraries
└── README.md                         # This file

---

## Syllabus Coverage

Unit I   - Data Wrangling              - data_loader.py
Unit II  - Python SQL Libraries        - data_loader.py, sql_queries.py
Unit III - Data Cleanup                - data_cleaning.py
Unit IV  - Data Exploration            - eda.py, rfm_clustering.py, ml_model.py
Unit V   - Visualization               - eda.py, rating_analysis.py, dashboard.py
Unit VI  - Capstone Project            - All files + dashboard

---

## Setup Instructions

### Step 1 - Prerequisites
- Python 3.11 or higher
- PostgreSQL 16
- pgAdmin 4

### Step 2 - Download Dataset
1. Go to kaggle.com
2. Search: Brazilian E-Commerce Public Dataset by Olist
3. Download and extract all 9 CSV files into data/raw/

### Step 3 - Create PostgreSQL Database
1. Open pgAdmin 4
2. Right click Databases → Create → Database
3. Name it: olist_db
4. Click Save

### Step 4 - Configure Database Password
Open config/db_config.py and update DB_PASSWORD to your PostgreSQL password

### Step 5 - Install Libraries
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### Step 6 - Run the Full Pipeline
python main.py

This runs all modules in order and launches the dashboard automatically at http://localhost:8501

---

## Running Individual Modules

python module_a/data_loader.py
python module_a/data_cleaning.py
python module_a/sql_queries.py
python module_b/eda.py
python module_b/rfm_clustering.py
python module_b/ml_model.py
python module_b/rating_analysis.py
streamlit run module_b/dashboard.py

---

## Module A - Data Pipeline and SQL Engine

data_loader.py
- Reads all 9 CSV files using pandas read_csv()
- Connects to PostgreSQL using SQLAlchemy create_engine()
- Writes each DataFrame to PostgreSQL using to_sql()
- Verifies all 9 tables were created successfully

data_cleaning.py
- Handles null values using fillna() with median
- Removes duplicate rows using drop_duplicates()
- Detects and removes outliers using IQR method
- Standardizes date columns using pd.to_datetime()
- Saves cleaned data back to PostgreSQL

sql_queries.py
- Connects to PostgreSQL using psycopg2
- Query 1: Top 10 sellers by revenue using RANK() and CTE
- Query 2: Month-over-month order growth using LAG() and CTE
- Query 3: Average delivery delay per state using CTE and CASE WHEN
- Query 4: Customer repeat purchase rate using CTE and COUNT
- Query 5: Product categories by cancellation rate using LEAD() and CTE

---

## Module B - EDA, ML and Visualization

eda.py
- describe(), info(), shape for basic exploration
- merge() to join multiple tables
- groupby() for revenue and payment analysis
- pivot_table() for orders by year and status
- corr() for correlation matrix
- 7 visualizations including lag plot

rfm_clustering.py
- Calculates Recency, Frequency, Monetary values per customer
- Scores each metric on 1-5 scale using pd.qcut()
- Applies StandardScaler for feature normalization
- Uses elbow method to find optimal K
- K-Means clustering with K=3
- Labels: High Value, Medium Value, Low Value

ml_model.py
- Feature engineering: delivery days, delay, review score
- Weekly aggregation for 90 data points
- train_test_split() with 80/20 split
- Random Forest Regressor with 100 trees
- Evaluation metrics: MAE, RMSE, R2=0.94
- Feature importance plot saved to outputs/plots/

rating_analysis.py
- Monthly average rating trend analysis
- Category-wise rating analysis across 48 categories
- Pivot table showing rating distribution by year
- 5 visualizations including dual-axis chart and heatmap

dashboard.py
- Live connection to PostgreSQL via SQLAlchemy
- 4 KPI cards: Total Orders, Revenue, Rating, Delivery Days
- Monthly order trend line chart
- Revenue by state bar chart
- Payment method distribution
- RFM customer segment visualization
- Review score distribution
- Interactive filter by order status and price range

---

## Key Results

Total Orders              : 99,441
Total Revenue             : R$ 6.5M+
Average Rating            : 4.09 / 5.0
Delivery Success Rate     : 96.8%
ML Model R2 Score         : 0.94
Customer Segments         : 3 (High, Medium, Low Value)
High Value Customers      : 2,807 (3% of total)
Top Seller Revenue        : R$ 184,551
Best Rated Category       : Books (4.47 / 5.0)
Worst Rated Category      : Office Furniture (3.68 / 5.0)
Repeat Purchase Rate      : 3.12% of customers

---

## Faculty Evaluation Notes

Module A is demonstrated by:
- Running data_loader.py and showing all 9 tables in pgAdmin
- Running data_cleaning.py and showing before and after null counts
- Running sql_queries.py and explaining each CTE and window function line by line

Module B is demonstrated by:
- Running eda.py and explaining each pandas function used
- Showing elbow curve and explaining why K=3 was chosen
- Explaining R2=0.94 and what feature importance means
- Opening dashboard at http://localhost:8501 and interacting live with filters

---

## Authors

Module A - Data Pipeline and SQL Engine (J Bagyalakshmi)
Module B - EDA, ML and Visualization (Madhumitha Sathish)

Institution : SIMATS Engineering
Subject     : DSA0513 - Query Processing for Data Science
