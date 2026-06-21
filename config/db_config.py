# ============================================================
# FILE: config/db_config.py
# PURPOSE: Central database connection settings
# All other scripts import from this file
# SYLLABUS: Unit II — Python SQL Libraries (PostgreSQL)
# ============================================================

import os

# --- Use cloud database (Render) if DATABASE_URL env var is set ---
# Otherwise fall back to local PostgreSQL for development
CLOUD_DATABASE_URL = "postgresql://olist_db_sswk_user:rzR6OVn4VNqcj8fjBudu0FqCZ1mCL3jK@dpg-d8s1mre8bjmc73bldejg-a.singapore-postgres.render.com/olist_db_sswk"

USE_CLOUD = True

if USE_CLOUD:
    SQLALCHEMY_URL = CLOUD_DATABASE_URL
    from urllib.parse import urlparse
    parsed = urlparse(CLOUD_DATABASE_URL)
    PSYCOPG2_CONFIG = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip('/'),
        "user": parsed.username,
        "password": parsed.password
    }
else:
    DB_HOST     = "localhost"
    DB_PORT     = "5432"
    DB_NAME     = "olist_db"
    DB_USER     = "postgres"
    DB_PASSWORD = "admin123"

    SQLALCHEMY_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    PSYCOPG2_CONFIG = {
        "host"    : DB_HOST,
        "port"    : DB_PORT,
        "dbname"  : DB_NAME,
        "user"    : DB_USER,
        "password": DB_PASSWORD
    }
