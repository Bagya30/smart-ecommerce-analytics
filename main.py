# ============================================================
# FILE: main.py
# PURPOSE: Run the entire project pipeline in one command
# USAGE: python main.py
# SYLLABUS: Unit VI — Capstone Project
# HOW IT WORKS:
#   Runs all modules in correct order:
#   1. Load CSVs into PostgreSQL
#   2. Clean the data
#   3. Run SQL queries
#   4. Run EDA
#   5. Run RFM clustering
#   6. Run ML model
#   7. Run rating analysis
#   Then launches the Streamlit dashboard
# ============================================================

import subprocess   # For running other Python scripts
import sys          # For getting current Python interpreter path
import os           # For building file paths

# ============================================================
# HELPER FUNCTION: Run a Python script and show output
# subprocess.run() executes a script as a separate process
# sys.executable ensures we use the same Python/venv
# ============================================================
def run_script(script_path, description):
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print(f"Script : {script_path}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],  # Use current Python
        capture_output = False,          # Show output in terminal
        text           = True            # Output as text not bytes
    )

    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully!")
    else:
        print(f"\n✗ {description} failed!")
        print("Please fix the error above before continuing.")
        sys.exit(1)  # Stop pipeline if any script fails

# ============================================================
# PIPELINE EXECUTION
# Scripts run in strict order — each depends on the previous
# ============================================================
print("=" * 60)
print("SMART E-COMMERCE ANALYTICS — FULL PIPELINE")
print("=" * 60)
print("This will run all modules in order.")
print("Estimated time: 5-8 minutes")
print("=" * 60)

# Step 1: Load data
run_script("module_a/data_loader.py",
           "Step 1: Loading CSVs into PostgreSQL")

# Step 2: Clean data
run_script("module_a/data_cleaning.py",
           "Step 2: Data Cleaning")

# Step 3: SQL queries
run_script("module_a/sql_queries.py",
           "Step 3: Advanced SQL Queries")

# Step 4: EDA
run_script("module_b/eda.py",
           "Step 4: Exploratory Data Analysis")

# Step 5: RFM clustering
run_script("module_b/rfm_clustering.py",
           "Step 5: RFM Analysis + K-Means Clustering")

# Step 6: ML model
run_script("module_b/ml_model.py",
           "Step 6: Random Forest ML Model")

# Step 7: Rating analysis
run_script("module_b/rating_analysis.py",
           "Step 7: Rating Trend Analysis")

# ============================================================
# PIPELINE COMPLETE
# ============================================================
print("\n" + "=" * 60)
print("ALL MODULES COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nOutputs saved to:")
print("  outputs/plots/  — All chart images")
print("  PostgreSQL      — All cleaned tables + rfm_segments")
print("\nLaunching Streamlit Dashboard...")
print("Dashboard will open at: http://localhost:8501")
print("Press Ctrl+C to stop the dashboard when done.")
print("=" * 60)

# Launch Streamlit dashboard
# This keeps running until user presses Ctrl+C
subprocess.run([
    sys.executable, "-m", "streamlit",
    "run", "module_b/dashboard.py"
])
