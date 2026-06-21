# ============================================================
# FILE: module_a/data_loader.py
# PURPOSE: Load all 9 Olist CSV files into PostgreSQL
# SYLLABUS: Unit I — Data Wrangling (loading raw data)
#           Unit II — Python SQL Libraries (SQLAlchemy)
# HOW IT WORKS:
#   1. Read each CSV file using pandas
#   2. Connect to PostgreSQL using SQLAlchemy
#   3. Create tables automatically and insert all rows
# ============================================================

import pandas as pd                          # For reading CSV files
from sqlalchemy import create_engine, text   # For connecting to PostgreSQL
import sys                                   # For stopping script on error
import os                                    # For building file paths

# Import our database config from config folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.db_config import SQLALCHEMY_URL

# ============================================================
# STEP 1: Create SQLAlchemy engine
# The engine is the connection object between Python and PostgreSQL
# create_engine() takes our connection URL and returns an engine object
# ============================================================
print("Connecting to PostgreSQL...")
try:
    engine = create_engine(SQLALCHEMY_URL)
    # Test the connection by running a simple query
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Connected to PostgreSQL successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    print("Check your password in config/db_config.py")
    sys.exit(1)  # Stop the script if connection fails

# ============================================================
# STEP 2: Define CSV file paths and their target table names
# Each CSV file maps to one table in PostgreSQL
# os.path.join builds the correct path on any operating system
# ============================================================
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Dictionary: key = PostgreSQL table name, value = CSV file name
CSV_FILES = {
    "customers"                  : "olist_customers_dataset.csv",
    "geolocation"                : "olist_geolocation_dataset.csv",
    "orders"                     : "olist_orders_dataset.csv",
    "order_items"                : "olist_order_items_dataset.csv",
    "order_payments"             : "olist_order_payments_dataset.csv",
    "order_reviews"              : "olist_order_reviews_dataset.csv",
    "products"                   : "olist_products_dataset.csv",
    "sellers"                    : "olist_sellers_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv"
}

# ============================================================
# STEP 3: Load each CSV into PostgreSQL
# pandas read_csv() reads the CSV into a DataFrame
# to_sql() writes the DataFrame into a PostgreSQL table
# if_exists='replace' drops and recreates the table each time
# index=False means we don't write the pandas row numbers
# ============================================================
print("\nLoading CSV files into PostgreSQL...")
print("-" * 50)

for table_name, csv_file in CSV_FILES.items():
    try:
        # Build full path to CSV file
        file_path = os.path.join(BASE_PATH, csv_file)

        # Read CSV into pandas DataFrame
        print(f"Reading {csv_file}...")
        df = pd.read_csv(file_path)

        # Show basic info about what we loaded
        print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

        # Write DataFrame to PostgreSQL table
        # chunksize=1000 means insert 1000 rows at a time (memory efficient)
        print(f"  Writing to table '{table_name}'...")
        df.to_sql(
            name      = table_name,      # Table name in PostgreSQL
            con       = engine,          # Our SQLAlchemy engine
            if_exists = 'replace',       # Drop table if it exists and recreate
            index     = False,           # Don't write pandas index as a column
            chunksize = 1000             # Insert 1000 rows at a time
        )
        print(f"  Done! Table '{table_name}' created successfully.")

    except FileNotFoundError:
        print(f"  ERROR: File not found — {csv_file}")
        print(f"  Make sure all CSV files are in data/raw/")
    except Exception as e:
        print(f"  ERROR loading {csv_file}: {e}")

    print()

# ============================================================
# STEP 4: Verify all tables were created
# We query PostgreSQL to list all tables in our database
# information_schema.tables is a built-in PostgreSQL system table
# ============================================================
print("-" * 50)
print("Verifying tables in PostgreSQL...")

with engine.connect() as conn:
    # Query to list all tables we created
    result = conn.execute(text("""
        SELECT table_name, 
               (SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY table_name
    """))
    tables = result.fetchall()

print(f"\nTotal tables created: {len(tables)}")
for table in tables:
    print(f"  ✓ {table[0]} ({table[1]} columns)")

print("\nData loading complete! All 9 CSV files loaded into PostgreSQL.")
print("You can verify the tables in pgAdmin under olist_db > Schemas > Tables")
