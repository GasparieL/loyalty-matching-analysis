"""
Loyalty 1-to-1 Bank Card Matching Analysis

This script analyzes loyalty customer transactions and matches them with bank card data.
It identifies customers with 1 or 2 valid bank cards from BOG/TBC banks.

Key constraints:
- Each customer has <=2 bank cards from <=2 different banks
- If 2 cards, they must be from 2 different banks
- Only cards with >=3 transactions OR >$35 spending
- Shared cards (used by multiple customers) are removed

Output files:
- customer_card_pairs.csv: Primary output with customer-card mappings
- final_clean_customers_summary.csv: Customer-level summary
- customer_card_combinations_detailed.csv: Detailed card usage data
"""

import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration - Portable data directory setup
def get_data_directory():
    """
    Find data directory in multiple possible locations:
    1. ./analysis_data (relative to script location)
    2. ./analysis_data (relative to current working directory)
    3. ../analysis_data (one level up from script)
    4. Any subdirectory containing .parquet or .pkl files
    """
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    cwd = Path.cwd()

    # Possible data directory locations (in order of preference)
    possible_dirs = [
        script_dir / "analysis_data",      # Same directory as script
        cwd / "analysis_data",              # Current working directory
        script_dir.parent / "analysis_data", # One level up
        script_dir / "data",                # Alternative name
        cwd / "data",                       # Alternative name in cwd
    ]

    # Check for existing data directory
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            # Check if it contains expected data files
            has_data = any(dir_path.glob("*.parquet")) or any(dir_path.glob("*.pkl"))
            if has_data:
                print(f"Using data directory: {dir_path}")
                return dir_path

    # If no existing directory found, create in script directory
    default_dir = script_dir / "analysis_data"
    default_dir.mkdir(exist_ok=True)
    print(f"Created new data directory: {default_dir}")
    return default_dir

DATA_DIR = get_data_directory()
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


class CheckpointManager:
    """Manages checkpoint files for resumable data processing"""

    def __init__(self, process_name, parameters=None):
        self.process_name = process_name
        self.parameters = parameters or {}
        param_hash = hash(str(sorted(self.parameters.items())))
        self.checkpoint_file = CHECKPOINT_DIR / f"{process_name}_{param_hash}_progress.json"
        self.progress = self.load_progress()

    def load_progress(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                if data.get("parameters") == self.parameters:
                    return data
                else:
                    print(f"Parameters changed for {self.process_name}, starting fresh")
                    return self._create_new_progress()
        return self._create_new_progress()

    def _create_new_progress(self):
        return {
            "parameters": self.parameters,
            "completed_chunks": [],
            "status": "not_started",
            "last_chunk": -1
        }

    def save_progress(self, chunk_info):
        self.progress["completed_chunks"].append(chunk_info)
        self.progress["last_chunk"] = len(self.progress["completed_chunks"]) - 1
        self.progress["status"] = "in_progress"
        self.progress["parameters"] = self.parameters
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def mark_completed(self):
        self.progress["status"] = "completed"
        self.progress["parameters"] = self.parameters
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def is_chunk_completed(self, chunk_id):
        return any(chunk["chunk_id"] == chunk_id for chunk in self.progress["completed_chunks"])

    def get_completed_chunks(self):
        return [chunk["filename"] for chunk in self.progress["completed_chunks"]]


def save_dataframe(df, filename_base):
    """Save dataframe with best available format"""
    try:
        parquet_file = DATA_DIR / f"{filename_base}.parquet"
        df.to_parquet(parquet_file, index=False)
        return parquet_file
    except ImportError:
        pickle_file = DATA_DIR / f"{filename_base}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(df, f)
        return pickle_file


def load_dataframe(filename_base):
    """Load dataframe from best available format"""
    for ext, loader in [('.parquet', pd.read_parquet), ('.pkl', lambda x: pickle.load(open(x, 'rb')))]:
        file_path = DATA_DIR / f"{filename_base}{ext}"
        if file_path.exists():
            return loader(file_path)
    return None


def connect_server(ip, database):
    """Connect to SQL Server database"""
    conn_str = (
        "Driver={SQL Server};"
        "Server=" + ip + ";" +
        "Database=" + database + ";" +
        "Trusted_Connection=yes;"
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    return conn, cursor


def sql_to_pandas(conn, cursor, sql_string):
    """Execute SQL query and return results as pandas DataFrame"""
    cursor.execute(sql_string)
    rows = cursor.fetchall()
    if rows:
        return pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
    return pd.DataFrame()


def close_conn(conn, cursor):
    """Close database connection"""
    cursor.close()
    conn.close()


def analyze_loyalty_bank_cards_corrected(facts_df, bank_df):
    """
    Analyze loyalty customer bank card usage

    Logic:
    1. Aggregate facts by cheque_id to get cheque totals
    2. Join with bank_df using reference_number to get client_id (bank card)
    3. Validate each customer_id + client_id combination (>=3 cheques OR >$35 total)
    4. Keep customers with <=2 different client_ids from <=2 different banks
    5. If customer has 2 client_ids, they MUST be from 2 different banks
    6. Remove shared client_ids (used by multiple customers)
    """
    print("\n=== LOYALTY BANK CARD ANALYSIS ===")

    if facts_df is None or bank_df is None or facts_df.empty or bank_df.empty:
        print("Invalid input data")
        return None, None

    # Determine the correct column name for bank card identifier
    print("\n--- CHECKING AVAILABLE COLUMNS ---")
    print(f"Facts columns: {list(facts_df.columns)}")
    print(f"Bank columns: {list(bank_df.columns)}")

    bank_card_column = None
    if 'client_id' in bank_df.columns:
        bank_card_column = 'client_id'
    elif 'client' in bank_df.columns:
        bank_card_column = 'client'
    else:
        print("\nERROR: Cannot find bank card identifier column!")
        return None, None

    print(f"\nUsing '{bank_card_column}' as bank card identifier")

    # STEP 0: AGGREGATE FACTS DATA BY CHEQUE_ID
    print("\n--- STEP 0: AGGREGATING FACTS DATA BY CHEQUE ---")

    print(f"Raw transaction lines: {len(facts_df):,}")
    print(f"Unique cheques: {facts_df['cheque_id'].nunique():,}")
    print(f"Unique customers: {facts_df['customer_id'].nunique():,}")
    print(f"Unique reference numbers: {facts_df['reference_number'].nunique():,}")

    cheque_totals = facts_df.groupby(['cheque_id', 'customer_id', 'reference_number']).agg({
        'total_price': 'sum'
    }).reset_index()

    cheque_totals.columns = ['cheque_id', 'customer_id', 'reference_number', 'cheque_total']

    print(f"Cheque-level data: {len(cheque_totals):,} records")
    print(f"Average spending per cheque: ${cheque_totals['cheque_total'].mean():.2f}")

    # STEP 1: JOIN WITH BANK DATA
    print(f"\n--- STEP 1: JOINING WITH BANK DATA ---")

    cheque_totals['reference_number'] = cheque_totals['reference_number'].astype(str)
    bank_df['reference_number'] = bank_df['reference_number'].astype(str)

    bank_columns = ['reference_number', bank_card_column, 'bank_name']
    if 'client' in bank_df.columns and bank_card_column != 'client':
        bank_columns.append('client')

    merged_df = cheque_totals.merge(
        bank_df[bank_columns],
        on='reference_number',
        how='inner'
    )

    merged_df = merged_df.rename(columns={bank_card_column: 'bank_card_id'})

    print(f"After join:")
    print(f"  Successfully merged cheque records: {len(merged_df):,}")
    print(f"  Unique customers with bank data: {merged_df['customer_id'].nunique():,}")
    print(f"  Unique bank cards: {merged_df['bank_card_id'].nunique():,}")

    if len(merged_df) == 0:
        print("No matching data found!")
        return None, None

    # STEP 2: VALIDATE CUSTOMER-CARD COMBINATIONS
    print("\n--- STEP 2: VALIDATING CUSTOMER-CARD COMBINATIONS ---")
    print("Validation: customer_id must have >=3 cheques OR >$35 total spending per bank_card_id")

    agg_dict = {
        'cheque_id': 'nunique',
        'cheque_total': 'sum',
        'bank_name': 'first',
        'reference_number': lambda x: list(x.unique())
    }

    if 'client' in merged_df.columns:
        agg_dict['client'] = 'first'

    customer_card_usage = merged_df.groupby(['customer_id', 'bank_card_id']).agg(agg_dict).reset_index()

    col_names = ['customer_id', 'bank_card_id', 'cheque_count',
                 'total_spending', 'bank_name', 'reference_numbers']
    if 'client' in merged_df.columns:
        col_names.append('client')

    customer_card_usage.columns = col_names

    print(f"Customer-card combinations: {len(customer_card_usage):,}")

    valid_customer_cards = customer_card_usage[
        (customer_card_usage['cheque_count'] >= 3) |
        (customer_card_usage['total_spending'] > 35)
    ].copy()

    invalid_customer_cards = customer_card_usage[
        (customer_card_usage['cheque_count'] < 3) &
        (customer_card_usage['total_spending'] <= 35)
    ].copy()

    print(f"Valid customer-card combinations: {len(valid_customer_cards):,}")
    print(f"Invalid customer-card combinations: {len(invalid_customer_cards):,}")

    # STEP 3: FILTER TO BOG AND TBC BANKS
    print("\n--- STEP 3: FILTERING TO BOG AND TBC BANKS ---")

    target_banks = ['BOG', 'TBC']
    bog_tbc_valid_cards = valid_customer_cards[
        valid_customer_cards['bank_name'].isin(target_banks)
    ].copy()

    print(f"Valid combinations with BOG/TBC banks: {len(bog_tbc_valid_cards):,}")

    if len(bog_tbc_valid_cards) == 0:
        print("No valid BOG/TBC combinations found!")
        return None, None

    # STEP 4: FILTER CUSTOMERS BY CARD AND BANK LIMITS
    print("\n--- STEP 4: FILTERING CUSTOMERS BY CARD AND BANK LIMITS ---")

    customer_summary = bog_tbc_valid_cards.groupby('customer_id').agg({
        'bank_card_id': ['nunique', lambda x: list(x.unique())],
        'bank_name': ['nunique', lambda x: list(x.unique())],
        'cheque_count': 'sum',
        'total_spending': 'sum'
    }).reset_index()

    customer_summary.columns = ['customer_id', 'valid_cards_count', 'valid_cards_list',
                                'different_banks_count', 'different_banks_list',
                                'total_cheques', 'total_spending']

    eligible_customers = customer_summary[
        (customer_summary['valid_cards_count'] <= 2) &
        (customer_summary['different_banks_count'] <= 2) &
        (
            (customer_summary['valid_cards_count'] == 1) |
            ((customer_summary['valid_cards_count'] == 2) & (customer_summary['different_banks_count'] == 2))
        )
    ].copy()

    print(f"Eligible customers: {len(eligible_customers):,}")

    if len(eligible_customers) == 0:
        print("No customers meet the criteria!")
        return None, None

    # STEP 5: REMOVE SHARED BANK_CARD_IDS
    print("\n--- STEP 5: REMOVING SHARED BANK_CARD_IDS ---")

    eligible_customer_ids = set(eligible_customers['customer_id'])
    eligible_customer_cards = bog_tbc_valid_cards[
        bog_tbc_valid_cards['customer_id'].isin(eligible_customer_ids)
    ].copy()

    card_usage_analysis = eligible_customer_cards.groupby('bank_card_id').agg({
        'customer_id': ['nunique', lambda x: list(x.unique())],
        'bank_name': 'first',
        'cheque_count': 'sum',
        'total_spending': 'sum'
    }).reset_index()

    card_usage_analysis.columns = ['bank_card_id', 'customer_count', 'customer_list',
                                   'bank_name', 'total_cheques', 'total_spending']

    shared_cards = card_usage_analysis[
        card_usage_analysis['customer_count'] > 1
    ]['bank_card_id'].tolist()

    clean_customer_cards = eligible_customer_cards[
        ~eligible_customer_cards['bank_card_id'].isin(shared_cards)
    ].copy()

    print(f"Customer-card combinations after removing shared cards: {len(clean_customer_cards):,}")

    # STEP 6: CREATE FINAL CLEAN DATASET
    print("\n--- STEP 6: CREATING FINAL CLEAN DATASET ---")

    final_agg_dict = {
        'bank_card_id': ['nunique', lambda x: list(x.unique())],
        'bank_name': ['nunique', lambda x: list(x.unique())],
        'cheque_count': 'sum',
        'total_spending': 'sum',
        'reference_numbers': lambda x: list(set([ref for refs in x for ref in refs]))
    }

    if 'client' in clean_customer_cards.columns:
        final_agg_dict['client'] = lambda x: list(x.unique())

    final_customer_summary = clean_customer_cards.groupby('customer_id').agg(final_agg_dict).reset_index()

    col_names = ['customer_id', 'final_cards_count', 'final_cards_list',
                'different_banks_count', 'different_banks_list',
                'total_cheques', 'total_spending', 'reference_numbers']

    if 'client' in clean_customer_cards.columns:
        col_names.append('clients_list')

    final_customer_summary.columns = col_names

    final_customer_summary = final_customer_summary[
        final_customer_summary['final_cards_count'] > 0
    ].copy()

    print(f"Final clean customers: {len(final_customer_summary):,}")

    # Verify constraints
    constraint_violations = final_customer_summary[
        (final_customer_summary['final_cards_count'] == 2) &
        (final_customer_summary['different_banks_count'] != 2)
    ]

    if len(constraint_violations) > 0:
        print(f"WARNING: {len(constraint_violations)} customers have 2 cards from same bank")
        final_customer_summary = final_customer_summary[
            ~final_customer_summary['customer_id'].isin(constraint_violations['customer_id'])
        ].copy()
        print(f"Removed violations, final customers: {len(final_customer_summary):,}")

    # CREATE CUSTOMER-CARD PAIRS OUTPUT
    print("\n--- CREATING CUSTOMER-CARD PAIRS ---")

    customer_card_pairs = clean_customer_cards[['customer_id', 'bank_card_id', 'bank_name']].drop_duplicates()
    print(f"Total customer-card pairs: {len(customer_card_pairs):,}")

    results = {
        'raw_transaction_lines': len(facts_df),
        'total_cheques': len(cheque_totals),
        'total_loyalty_customers': cheque_totals['customer_id'].nunique(),
        'customers_with_bank_data': merged_df['customer_id'].nunique(),
        'total_customer_card_combinations': len(customer_card_usage),
        'valid_customer_card_combinations': len(valid_customer_cards),
        'bog_tbc_valid_combinations': len(bog_tbc_valid_cards),
        'eligible_customers_before_shared_removal': len(eligible_customers),
        'shared_cards_count': len(shared_cards),
        'final_clean_customers': len(final_customer_summary),
        'final_clean_combinations': len(clean_customer_cards),
        'bank_card_column_used': bank_card_column,

        'cheque_totals': cheque_totals,
        'customer_card_usage': customer_card_usage,
        'valid_customer_cards': valid_customer_cards,
        'invalid_customer_cards': invalid_customer_cards,
        'bog_tbc_valid_cards': bog_tbc_valid_cards,
        'eligible_customers': eligible_customers,
        'card_usage_analysis': card_usage_analysis,
        'shared_cards_info': card_usage_analysis[card_usage_analysis['customer_count'] > 1],
        'shared_cards_list': shared_cards,
        'final_clean_cards': clean_customer_cards,
        'final_customer_summary': final_customer_summary,
        'customer_card_pairs': customer_card_pairs,
        'merged_data': merged_df,
    }

    return results, final_customer_summary


def save_corrected_results(results, output_dir="./data"):
    """Save all results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    saved_files = []

    try:
        # PRIMARY OUTPUT: Customer-Card Pairs
        if 'customer_card_pairs' in results and not results['customer_card_pairs'].empty:
            file_path = output_path / "customer_card_pairs.csv"
            results['customer_card_pairs'].to_csv(file_path, index=False)
            saved_files.append(str(file_path))
            print(f"\nPRIMARY OUTPUT SAVED: {file_path}")

        # Customer summary
        if 'final_customer_summary' in results and not results['final_customer_summary'].empty:
            summary_csv = results['final_customer_summary'].copy()

            summary_csv['final_cards_list_str'] = summary_csv['final_cards_list'].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))
            summary_csv['different_banks_list_str'] = summary_csv['different_banks_list'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x))

            if 'clients_list' in summary_csv.columns:
                summary_csv['clients_list_str'] = summary_csv['clients_list'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x))

            summary_csv['reference_numbers_str'] = summary_csv['reference_numbers'].apply(
                lambda x: ', '.join(map(str, x[:20])) + (f' (+{len(x)-20} more)' if len(x) > 20 else '')
                if isinstance(x, list) else str(x))

            csv_columns = ['customer_id', 'final_cards_count', 'different_banks_count', 'total_cheques',
                          'total_spending', 'final_cards_list_str', 'different_banks_list_str', 'reference_numbers_str']

            if 'clients_list_str' in summary_csv.columns:
                csv_columns.insert(-1, 'clients_list_str')

            file_path = output_path / "final_clean_customers_summary.csv"
            summary_csv[csv_columns].to_csv(file_path, index=False)
            saved_files.append(str(file_path))

        # Detailed combinations
        if 'final_clean_cards' in results and not results['final_clean_cards'].empty:
            clean_cards_csv = results['final_clean_cards'].copy()
            clean_cards_csv['reference_numbers_str'] = clean_cards_csv['reference_numbers'].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

            output_cols = ['customer_id', 'bank_card_id', 'bank_name', 'cheque_count',
                          'total_spending', 'reference_numbers_str']

            if 'client' in clean_cards_csv.columns:
                output_cols.insert(3, 'client')

            file_path = output_path / "customer_card_combinations_detailed.csv"
            clean_cards_csv[output_cols].to_csv(file_path, index=False)
            saved_files.append(str(file_path))

        # Analysis summary
        analysis_summary = {
            'total_loyalty_customers': results['total_loyalty_customers'],
            'final_clean_customers': results['final_clean_customers'],
            'success_rate_percent': results['final_clean_customers']/results['total_loyalty_customers']*100,
            'shared_cards_removed': results['shared_cards_count'],
            'validation_rate_percent': results['valid_customer_card_combinations']/results['total_customer_card_combinations']*100,
            'bank_card_column_in_original_data': results.get('bank_card_column_used', 'client')
        }

        file_path = output_path / "analysis_summary.csv"
        pd.DataFrame([analysis_summary]).to_csv(file_path, index=False)
        saved_files.append(str(file_path))

        print(f"\nSAVED {len(saved_files)} FILES:")
        for file_path in saved_files:
            print(f"   {file_path}")

    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def load_facts_from_database(server_ip='192.168.20.9', database='ORINABIJI_DWH',
                            start_date='2025-01-01', end_date='2025-12-31'):
    """Load facts data from database with customer_id"""
    print(f"\n=== FETCHING FACTS DATA FROM DATABASE ===")
    print(f"Server: {server_ip}, Database: {database}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        conn, cursor = connect_server(server_ip, database)
        print("✓ Connected successfully")

        sql_facts = f"""
        SELECT
            a.cheque_id,
            CAST(a.is_loyalty AS INT) as is_loyalty,
            a.discount_card_no,
            c.customer_id,
            CAST(a.reference_number AS NVARCHAR(50)) as reference_number,
            SUM(CAST(a.price AS DECIMAL(15,2))) AS total_price,
            COUNT(*) AS item_count
        FROM dbo.facts a
        LEFT JOIN loyalty.cards c ON a.discount_card_no = c.card_no
        WHERE YEAR(a.date) = 2025
            AND a.date >= '{start_date}'
            AND a.date <= '{end_date}'
            AND a.reference_number IS NOT NULL
            AND a.reference_number != ''
            AND c.customer_id IS NOT NULL
        GROUP BY a.cheque_id, a.is_loyalty, a.discount_card_no, c.customer_id, a.reference_number
        """

        facts_df = sql_to_pandas(conn, cursor, sql_facts)
        print(f"✓ Fetched {len(facts_df):,} records")

        if not facts_df.empty:
            facts_df['reference_number'] = facts_df['reference_number'].astype(str).str.strip()
            facts_df['customer_id'] = facts_df['customer_id'].astype(str).str.strip()
            facts_df = facts_df[facts_df['reference_number'] != 'nan']
            facts_df = facts_df[facts_df['reference_number'] != '']
            facts_df = facts_df[facts_df['customer_id'] != 'nan']
            facts_df = facts_df[facts_df['customer_id'] != '']

            save_dataframe(facts_df, "facts_data_2025_combined_customer_id")
            print(f"✓ Saved to {DATA_DIR / 'facts_data_2025_combined_customer_id.parquet'}")

        close_conn(conn, cursor)
        return facts_df

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_bank_from_database(server_ip='192.168.20.9', database='ORINABIJI_DWH'):
    """Load bank data from database"""
    print(f"\n=== FETCHING BANK DATA FROM DATABASE ===")
    print(f"Server: {server_ip}, Database: {database}")

    try:
        conn, cursor = connect_server(server_ip, database)
        print("✓ Connected successfully")

        sql_bank = """
        SELECT DISTINCT
            CAST(reference_number AS NVARCHAR(50)) as reference_number,
            CAST(transaction_amount AS DECIMAL(15,2)) as transaction_amount,
            CAST(bank_name AS NVARCHAR(100)) as bank_name,
            CAST(client AS NVARCHAR(100)) as client
        FROM bank_data.bank_transactions
        WHERE reference_number IS NOT NULL
            AND reference_number != ''
            AND client IS NOT NULL
            AND client != ''
            AND bank_name IS NOT NULL
            AND bank_name != ''
        """

        bank_df = sql_to_pandas(conn, cursor, sql_bank)
        print(f"✓ Fetched {len(bank_df):,} records")

        if not bank_df.empty:
            bank_df['reference_number'] = bank_df['reference_number'].astype(str).str.strip()
            bank_df['client'] = bank_df['client'].astype(str).str.strip()
            bank_df['bank_name'] = bank_df['bank_name'].astype(str).str.strip()
            bank_df = bank_df[bank_df['reference_number'] != 'nan']
            bank_df = bank_df[bank_df['client'] != 'nan']
            bank_df = bank_df[bank_df['bank_name'] != 'nan']

            save_dataframe(bank_df, "bank_data_combined")
            print(f"✓ Saved to {DATA_DIR / 'bank_data_combined.parquet'}")

        close_conn(conn, cursor)
        return bank_df

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_analysis(server_ip='192.168.20.9', database='ORINABIJI_DWH', force_reload=False):
    """
    Run complete analysis - loads from saved files or fetches from database

    Parameters:
    - server_ip: SQL Server IP (default: '192.168.20.9')
    - database: Database name (default: 'ORINABIJI_DWH')
    - force_reload: Force reload from database even if files exist (default: False)
    """
    print("RUNNING COMPLETE ANALYSIS - LOYALTY BANK CARD MATCHING")
    print("="*60)

    if force_reload:
        print("Force reload requested - fetching fresh data from database...")
        facts_df = None
        bank_df = None
    else:
        print("Looking for presaved data files...")
        print(f"Data directory: {DATA_DIR.resolve()}")
        facts_df = load_dataframe("facts_data_2025_combined_customer_id")
        bank_df = load_dataframe("bank_data_combined")

    if facts_df is None or bank_df is None:
        print("\nNo presaved data found - fetching from database...")

        if facts_df is None:
            facts_df = load_facts_from_database(server_ip, database)

        if bank_df is None:
            bank_df = load_bank_from_database(server_ip, database)

        if facts_df is None or bank_df is None:
            print("\n✗ ERROR: Failed to load data!")
            print("\nPlease check:")
            print("  1. Database connection settings")
            print("  2. Network connectivity")
            print(f"  3. Access to {server_ip}")
            return None

    print("\nSUCCESS: Found presaved data!")
    print(f"  Facts data: {len(facts_df):,} records")
    print(f"  Bank data: {len(bank_df):,} records")

    results, summary = analyze_loyalty_bank_cards_corrected(facts_df, bank_df)

    if results is None:
        print("Analysis failed")
        return None

    save_corrected_results(results)

    return results, summary


def main():
    """Main execution function"""
    print(__doc__)
    results, summary = run_full_analysis()

    if results:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nFinal customers: {results['final_clean_customers']:,}")
        print(f"Success rate: {results['final_clean_customers']/results['total_loyalty_customers']*100:.1f}%")
    else:
        print("\nAnalysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
