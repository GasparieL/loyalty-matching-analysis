# How to Run the Python Scripts

This guide provides step-by-step instructions for running the loyalty matching analysis Python scripts.

## Prerequisites

### 1. Python Installation
Ensure you have Python 3.7 or higher installed:
```bash
python3 --version
# Should show: Python 3.7.x or higher
```

### 2. Required Python Packages
Install the required packages:
```bash
pip install pandas numpy pyodbc matplotlib seaborn tqdm
```

Or if you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Data Files Required
Both scripts expect data files in the `./analysis_data/` directory:
- `facts_data_2025_combined_customer_id.parquet` (or `.pkl`)
- `bank_data_combined.parquet` (or `.pkl`)

**Important**: Make sure these files exist before running the scripts!

## Running the Scripts

### Option 1: Run Loyalty Bank Card Matching Analysis

This script matches loyalty customers with their bank cards.

```bash
# Navigate to the project directory
cd /Users/lana/Downloads/loyalty_matching

# Run the script
python3 loyalty_1to1_min_spending_fixed.py
```

**Expected Output:**
- Console output showing analysis progress (STEP 0 through STEP 6)
- CSV files created in `./data/` directory:
  - `customer_card_pairs.csv` - Primary output
  - `final_clean_customers_summary.csv` - Summary statistics
  - `customer_card_combinations_detailed.csv` - Detailed information
  - `analysis_summary.csv` - Overall metrics

**Runtime:** Depends on data size (typically 5-15 minutes for large datasets)

### Option 2: Run Non-Loyalty Bank Clients Analysis

This script identifies customers and bank cards with no connections.

```bash
# Navigate to the project directory
cd /Users/lana/Downloads/loyalty_matching

# Run the script
python3 noloyalty_bankclients.py
```

**Expected Output:**
- Console output showing analysis progress (PART 1 and PART 2)
- CSV files created in `./data/` directory:
  - `customers_no_bank_connections_details.csv`
  - `customer_ids_no_bank_connections.csv`
  - `bank_cards_no_loyalty_details.csv`
  - `bank_card_ids_no_loyalty.csv`
  - `cash_loyalty_analysis_summary.csv`

**Runtime:** Depends on data size (typically 10-20 minutes for large datasets)

### Option 3: Run Both Scripts in Sequence

```bash
cd /Users/lana/Downloads/loyalty_matching

# Run loyalty matching first
python3 loyalty_1to1_min_spending_fixed.py

# Then run non-loyalty analysis
python3 noloyalty_bankclients.py
```

## Scheduling the Scripts

### Using Cron (macOS/Linux)

Edit your crontab:
```bash
crontab -e
```

Add these lines to run daily at 2 AM:
```bash
# Run loyalty matching analysis daily at 2:00 AM
0 2 * * * cd /Users/lana/Downloads/loyalty_matching && /usr/bin/python3 loyalty_1to1_min_spending_fixed.py >> /tmp/loyalty_matching.log 2>&1

# Run non-loyalty analysis daily at 3:00 AM
0 3 * * * cd /Users/lana/Downloads/loyalty_matching && /usr/bin/python3 noloyalty_bankclients.py >> /tmp/noloyalty_analysis.log 2>&1
```

**Cron Schedule Examples:**
```bash
# Every day at 2 AM
0 2 * * *

# Every Monday at 6 AM
0 6 * * 1

# Every hour
0 * * * *

# Every 6 hours
0 */6 * * *

# First day of every month at midnight
0 0 1 * *
```

### Using Windows Task Scheduler

1. **Open Task Scheduler**:
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Basic Task**:
   - Click "Create Basic Task" in the right panel
   - Name: "Loyalty Matching Analysis"
   - Description: "Run loyalty bank card matching analysis"

3. **Set Trigger**:
   - Choose when to run (Daily, Weekly, etc.)
   - Set time (e.g., 2:00 AM)

4. **Set Action**:
   - Action: "Start a program"
   - Program/script: `C:\Python39\python.exe` (adjust to your Python path)
   - Add arguments: `loyalty_1to1_min_spending_fixed.py`
   - Start in: `C:\Users\lana\Downloads\loyalty_matching` (your project path)

5. **Finish and Test**:
   - Review settings and click "Finish"
   - Right-click the task and select "Run" to test

### Using Airflow (Advanced)

Create a DAG file `loyalty_analysis_dag.py`:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'loyalty_matching_analysis',
    default_args=default_args,
    description='Run loyalty matching and non-loyalty analysis',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
)

run_loyalty_matching = BashOperator(
    task_id='run_loyalty_matching',
    bash_command='cd /path/to/loyalty_matching && python3 loyalty_1to1_min_spending_fixed.py',
    dag=dag,
)

run_noloyalty_analysis = BashOperator(
    task_id='run_noloyalty_analysis',
    bash_command='cd /path/to/loyalty_matching && python3 noloyalty_bankclients.py',
    dag=dag,
)

run_loyalty_matching >> run_noloyalty_analysis  # Run in sequence
```

## Troubleshooting

### Error: "No module named 'pandas'"
**Solution:** Install pandas
```bash
pip install pandas
```

### Error: "Could not find saved data"
**Solution:** Ensure data files exist in `./analysis_data/` directory
```bash
ls -la analysis_data/
# Should show:
# facts_data_2025_combined_customer_id.parquet
# bank_data_combined.parquet
```

### Error: "Permission denied"
**Solution:** Make scripts executable
```bash
chmod +x loyalty_1to1_min_spending_fixed.py
chmod +x noloyalty_bankclients.py
```

### Error: Database connection issues
**Solution:** Check SQL Server connection settings in the script
- Verify server IP
- Verify database name
- Ensure you have network access
- Check Windows authentication settings

### Script runs but produces no output
**Solution:** Check the `./data/` directory exists
```bash
mkdir -p data
```

## Checking Output Files

After running the scripts, verify the output:

```bash
# List all output files
ls -lh data/*.csv

# Preview a file
head -n 20 data/customer_card_pairs.csv

# Count rows in output
wc -l data/customer_card_pairs.csv
```

## Performance Tips

### For Large Datasets

1. **Increase chunk size** in `noloyalty_bankclients.py`:
```python
results = analyze_cash_vs_loyalty_patterns(facts_df, bank_df, chunk_size=2000000)
```

2. **Monitor memory usage**:
```bash
# While script is running, in another terminal:
top -p $(pgrep -f loyalty_1to1)
```

3. **Use parquet files** instead of pickle for faster loading

### For Faster Execution

1. **Ensure data files are in parquet format** (faster than pickle)
2. **Run on a machine with sufficient RAM** (16GB+ recommended for large datasets)
3. **Consider splitting data** if datasets are extremely large

## Logging

### Add Logging to Files

Redirect output to log files:

```bash
# Run with logging
python3 loyalty_1to1_min_spending_fixed.py > logs/loyalty_$(date +%Y%m%d).log 2>&1

# View log in real-time
tail -f logs/loyalty_20251027.log
```

### Create logs directory

```bash
mkdir -p logs
```

## Testing Before Scheduling

Before setting up automated scheduling, test the scripts manually:

```bash
# Test 1: Check Python version
python3 --version

# Test 2: Check data files exist
ls -la analysis_data/

# Test 3: Test import of required packages
python3 -c "import pandas, numpy, pyodbc, matplotlib, seaborn, tqdm; print('All packages installed')"

# Test 4: Run the scripts
python3 loyalty_1to1_min_spending_fixed.py
python3 noloyalty_bankclients.py

# Test 5: Verify output files were created
ls -la data/*.csv
```

## Getting Help

If you encounter issues:

1. **Check the error message** - it usually indicates what's wrong
2. **Verify data files** are in the correct location
3. **Check Python version** is 3.7+
4. **Ensure all packages** are installed
5. **Review the verification report** - `VERIFICATION_REPORT.md`

## Quick Reference

```bash
# Run loyalty matching
python3 loyalty_1to1_min_spending_fixed.py

# Run non-loyalty analysis
python3 noloyalty_bankclients.py

# Run with logging
python3 loyalty_1to1_min_spending_fixed.py > logs/output.log 2>&1

# Check data files
ls analysis_data/

# Check output files
ls data/

# Install packages
pip install pandas numpy pyodbc matplotlib seaborn tqdm
```

## Summary

✅ **To run once**: Navigate to project directory and run `python3 <script_name>.py`
✅ **To schedule**: Use cron (macOS/Linux) or Task Scheduler (Windows)
✅ **Output location**: `./data/` directory
✅ **Data requirement**: Files in `./analysis_data/` directory
✅ **Runtime**: 5-20 minutes depending on data size
