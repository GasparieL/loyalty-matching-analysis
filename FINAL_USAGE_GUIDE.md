# FINAL USAGE GUIDE - PORTABLE ANALYSIS SCRIPTS

## ✅ VERIFICATION STATUS

**Both scripts are now FULLY INDEPENDENT and PORTABLE!**

✓ No local file paths required
✓ Automatically finds or creates data directories
✓ Fetches data from database if not found
✓ Works on any computer with network access to the database
✓ Syntax validated - both files compile successfully
✓ Pandas aggregation fix applied for compatibility (tested on pandas 2.0.3)
✓ Import tests passed - both scripts load without errors
✓ Aggregation pattern tested and verified working

---

## 📁 FINAL FILES TO USE

### 1. **loyalty_1to1_min_spending_fixed.py**
**Purpose**: Match loyalty customers with their bank cards (1-2 cards per customer)

**How to Run**:
```bash
# Simple - just run it!
python3 loyalty_1to1_min_spending_fixed.py

# Or from any directory
python3 /path/to/loyalty_1to1_min_spending_fixed.py
```

**What It Does**:
1. Looks for existing data files in multiple locations
2. If not found, connects to database and fetches fresh data
3. Saves data locally for future runs (faster)
4. Runs analysis on loyalty-bank card matching
5. Outputs results to `./data/` directory

**Output Files** (saved to `./data/`):
- `customer_card_pairs.csv` ⭐ **PRIMARY OUTPUT**
  - Simple table: customer_id, bank_card_id, bank_name
  - Each customer has 1-2 cards
  - If 2 cards, they're from different banks (BOG/TBC)

- `final_clean_customers_summary.csv`
  - One row per customer
  - Aggregated spending, transaction counts, card lists

- `customer_card_combinations_detailed.csv`
  - Full details for each customer-card combination
  - Includes spending, cheque count, reference numbers

- `analysis_summary.csv`
  - Overall metrics and success rates

---

### 2. **noloyalty_bankclients.py**
**Purpose**: Find customers/cards with NO connections between loyalty and bank systems

**How to Run**:
```bash
# Simple - just run it!
python3 noloyalty_bankclients.py

# Or from any directory
python3 /path/to/noloyalty_bankclients.py
```

**What It Does**:
1. Looks for existing data files (same as script #1)
2. If not found, fetches from database
3. Analyzes disconnections between loyalty and bank data
4. Identifies cash-only customers and non-loyalty bank cards

**Output Files** (saved to `./data/`):
- `customers_no_bank_connections_details.csv`
  - Loyalty customers who never use bank cards
  - Includes spending patterns, transaction counts

- `customer_ids_no_bank_connections.csv`
  - Simple list of customer IDs without bank connections

- `bank_cards_no_loyalty_details.csv`
  - Bank cards never used by loyalty customers
  - Transaction stats and spending patterns

- `bank_card_ids_no_loyalty.csv`
  - Simple list of bank card IDs without loyalty

- `cash_loyalty_analysis_summary.csv` ⭐ **KEY SUMMARY**
  - Overall statistics on connections/disconnections
  - Percentages and totals

---

## 🚀 USAGE EXAMPLES

### Run on Current Computer
```bash
cd /Users/lana/Downloads/loyalty_matching
python3 loyalty_1to1_min_spending_fixed.py
python3 noloyalty_bankclients.py
```

### Copy to Another Computer and Run
```bash
# On new computer
scp user@server:/path/to/*.py /local/path/
cd /local/path
python3 loyalty_1to1_min_spending_fixed.py
```

### Force Fresh Data from Database
```python
# In Python interpreter
from loyalty_1to1_min_spending_fixed import run_full_analysis
results = run_full_analysis(force_reload=True)
```

### Use Different Database Server
```python
from loyalty_1to1_min_spending_fixed import run_full_analysis
results = run_full_analysis(
    server_ip='YOUR_SERVER_IP',
    database='YOUR_DATABASE_NAME'
)
```

---

## 📊 DATA FLOW

```
First Run:
1. Script looks for data files → NOT FOUND
2. Connects to database (192.168.20.9/ORINABIJI_DWH)
3. Executes SQL queries to fetch:
   - Facts data (loyalty transactions with customer_id)
   - Bank data (bank transactions with client/card info)
4. Saves data to ./analysis_data/ as .parquet files
5. Runs analysis
6. Saves results to ./data/ as .csv files

Subsequent Runs:
1. Script looks for data files → FOUND
2. Loads from local files (much faster!)
3. Runs analysis
4. Saves results to ./data/ as .csv files
```

---

## 🔧 REQUIREMENTS

### Python Packages
```bash
pip install pandas numpy pyodbc matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Database Access
- Network connectivity to `192.168.20.9`
- SQL Server access with Windows authentication
- Read access to:
  - `dbo.facts` table
  - `loyalty.cards` table
  - `bank_data.bank_transactions` table

---

## 📂 DIRECTORY STRUCTURE

After running, you'll have:
```
loyalty_matching/
├── loyalty_1to1_min_spending_fixed.py  ← Script 1
├── noloyalty_bankclients.py            ← Script 2
├── analysis_data/                      ← Auto-created data cache
│   ├── facts_data_2025_combined_customer_id.parquet
│   └── bank_data_combined.parquet
└── data/                               ← Output results
    ├── customer_card_pairs.csv         ⭐ Main output #1
    ├── final_clean_customers_summary.csv
    ├── customer_card_combinations_detailed.csv
    ├── analysis_summary.csv
    ├── customers_no_bank_connections_details.csv
    ├── customer_ids_no_bank_connections.csv
    ├── bank_cards_no_loyalty_details.csv
    ├── bank_card_ids_no_loyalty.csv
    └── cash_loyalty_analysis_summary.csv  ⭐ Main output #2
```

---

## ⚡ KEY FEATURES

### ✅ Fully Portable
- No hardcoded paths
- Works from any directory on any computer
- Automatically finds or creates directories

### ✅ Smart Data Management
- Checks multiple locations for existing data
- Falls back to database if files not found
- Caches data locally for faster subsequent runs

### ✅ Database Integration
- SQL queries from original notebook
- Fetches facts + bank data automatically
- Saves fetched data for reuse

### ✅ Clear Output
- Progress messages show what's happening
- Error messages explain what to check
- Results saved to predictable locations

---

## 🎯 MAIN OUTPUTS SUMMARY

| Script | Primary Output | Purpose |
|--------|---------------|---------|
| **loyalty_1to1_min_spending_fixed.py** | `customer_card_pairs.csv` | Customer-to-bank-card mappings (1-2 cards each) |
| **noloyalty_bankclients.py** | `cash_loyalty_analysis_summary.csv` | Connection analysis summary stats |

Both scripts can run independently and will produce consistent results!

---

## 🔍 TROUBLESHOOTING

### "No module named 'pyodbc'"
```bash
pip install pyodbc
```

### "Cannot connect to database"
- Check network connection to 192.168.20.9
- Verify VPN if required
- Confirm Windows authentication is enabled

### "Permission denied"
```bash
chmod +x loyalty_1to1_min_spending_fixed.py
chmod +x noloyalty_bankclients.py
```

### Want fresh data from database?
```python
# Force reload instead of using cached files
python3 -c "from loyalty_1to1_min_spending_fixed import run_full_analysis; run_full_analysis(force_reload=True)"
```

---

## ✨ YOU'RE DONE!

Both scripts are ready to run anywhere, anytime. No local files needed - they'll fetch everything from the database automatically on first run, then use cached data for speed on subsequent runs.
