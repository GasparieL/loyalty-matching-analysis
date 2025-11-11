"""
Non-Loyalty Bank Clients Analysis

This script analyzes the connection between loyalty customers and bank card transactions.
It identifies:
1. Bank cards with NO connections to loyalty customers
2. Loyalty customers with NO connections to bank transactions

Connection is established through reference_number appearing in both facts and bank tables.

Output files:
- customers_no_bank_connections_details.csv: Detailed stats for customers without bank connections
- customer_ids_no_bank_connections.csv: Simple list of customer IDs
- bank_cards_no_loyalty_details.csv: Detailed stats for bank cards without loyalty
- bank_card_ids_no_loyalty.csv: Simple list of bank card IDs
- cash_loyalty_analysis_summary.csv: Overall summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
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

    # If no existing directory found, use script directory
    default_dir = script_dir / "analysis_data"
    print(f"Will use data directory: {default_dir}")
    return default_dir

DATA_DIR = get_data_directory()


def load_dataframe(filename_base):
    """Load dataframe from best available format"""
    for ext, loader in [('.parquet', pd.read_parquet), ('.pkl', lambda x: pd.read_pickle(x))]:
        file_path = DATA_DIR / f"{filename_base}{ext}"
        if file_path.exists():
            return loader(file_path)
    return None


def analyze_cash_vs_loyalty_patterns(facts_df, bank_df, chunk_size=1000000):
    """
    Analyze customer-bank connections

    KEY CONCEPTS:
    - client (in bank_df) = Bank card number (the actual card identifier)
    - reference_number = Transaction reference that links facts and bank tables
    - customer_id (in facts_df) = Loyalty program customer

    ANALYSIS:
    1. Bank cards (clients) with NO connections to loyalty customers
    2. Loyalty customers with NO connections to bank transactions

    Connection is established through reference_number appearing in both tables
    """
    print("\n=== CUSTOMER-BANK CONNECTION ANALYSIS ===")
    print("="*70)

    if facts_df is None or bank_df is None or facts_df.empty or bank_df.empty:
        print("Invalid input data")
        return None

    # Ensure string types for reference numbers
    facts_df['reference_number'] = facts_df['reference_number'].astype(str)
    bank_df['reference_number'] = bank_df['reference_number'].astype(str)

    print(f"\nDATA OVERVIEW:")
    print(f"  Loyalty transactions in facts: {len(facts_df):,}")
    print(f"  Unique loyalty customers (customer_id): {facts_df['customer_id'].nunique():,}")
    print(f"  Bank transactions: {len(bank_df):,}")
    print(f"  Unique bank cards (client): {bank_df['client'].nunique():,}")

    # Get all reference numbers from facts data
    print("\nExtracting reference numbers from loyalty data...")
    all_facts_refs = set(facts_df['reference_number'].unique())
    print(f"  Unique reference numbers in loyalty data: {len(all_facts_refs):,}")

    # ========================================================================
    # PART 1: ANALYZE BANK CARDS
    # ========================================================================

    print(f"\n--- PART 1: ANALYZING BANK CARDS ---")
    print(f"Processing bank data in chunks (chunk size: {chunk_size:,})")

    # Track all bank cards and their reference numbers
    bank_card_to_refs = {}  # client -> set of reference_numbers

    total_bank_transactions = 0
    num_chunks = (len(bank_df) // chunk_size) + 1

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(bank_df))

        if start_idx >= len(bank_df):
            break

        chunk = bank_df.iloc[start_idx:end_idx].copy()
        print(f"  Processing chunk {i+1}/{num_chunks}: rows {start_idx:,} to {end_idx:,}")

        total_bank_transactions += len(chunk)

        # Build mapping: each bank card (client) -> its reference numbers
        for _, row in chunk.iterrows():
            bank_card = row['client']
            ref_num = row['reference_number']

            if bank_card not in bank_card_to_refs:
                bank_card_to_refs[bank_card] = set()
            bank_card_to_refs[bank_card].add(ref_num)

        del chunk

    print(f"\nBank card processing completed!")
    print(f"  Total unique bank cards: {len(bank_card_to_refs):,}")

    # Categorize each bank card
    print(f"\nCategorizing bank cards by loyalty connection...")

    bank_cards_with_loyalty = set()
    bank_cards_no_loyalty = set()

    for bank_card, card_refs in bank_card_to_refs.items():
        # Check if ANY of this card's reference numbers appear in loyalty data
        has_loyalty_connection = bool(card_refs & all_facts_refs)

        if has_loyalty_connection:
            bank_cards_with_loyalty.add(bank_card)
        else:
            bank_cards_no_loyalty.add(bank_card)

    # Results
    total_bank_cards = len(bank_card_to_refs)
    bank_with_loyalty_count = len(bank_cards_with_loyalty)
    bank_no_loyalty_count = len(bank_cards_no_loyalty)

    print(f"\nBANK CARD ANALYSIS RESULTS:")
    print(f"  Total unique bank cards: {total_bank_cards:,}")
    print(f"  Bank cards with loyalty connections: {bank_with_loyalty_count:,} ({bank_with_loyalty_count/total_bank_cards*100:.1f}%)")
    print(f"  Bank cards with NO loyalty connections: {bank_no_loyalty_count:,} ({bank_no_loyalty_count/total_bank_cards*100:.1f}%)")

    # Get transaction-level details
    print(f"\nCollecting transaction details...")
    bank_no_loyalty_transactions = bank_df[bank_df['client'].isin(bank_cards_no_loyalty)]
    bank_with_loyalty_transactions = bank_df[bank_df['client'].isin(bank_cards_with_loyalty)]

    print(f"  Transactions from cards with NO loyalty: {len(bank_no_loyalty_transactions):,}")
    print(f"  Transactions from cards WITH loyalty: {len(bank_with_loyalty_transactions):,}")

    # ========================================================================
    # PART 2: ANALYZE LOYALTY CUSTOMERS
    # ========================================================================

    print(f"\n--- PART 2: ANALYZING LOYALTY CUSTOMERS ---")

    # Find which loyalty reference numbers appear in bank data
    print("Checking which loyalty references appear in bank data...")

    all_bank_refs = set(bank_df['reference_number'].unique())
    print(f"  Unique reference numbers in bank data: {len(all_bank_refs):,}")

    loyalty_refs_in_bank = all_facts_refs & all_bank_refs
    loyalty_refs_not_in_bank = all_facts_refs - all_bank_refs

    print(f"  Loyalty references that appear in bank: {len(loyalty_refs_in_bank):,}")
    print(f"  Loyalty references NOT in bank: {len(loyalty_refs_not_in_bank):,}")

    # Categorize each loyalty customer
    print(f"\nCategorizing loyalty customers by bank connection...")

    customer_to_refs = facts_df.groupby('customer_id')['reference_number'].apply(
        lambda x: set(x.unique())
    ).to_dict()

    customers_with_bank = set()
    customers_no_bank = set()

    for customer_id, customer_refs in customer_to_refs.items():
        # Check if ANY of this customer's references appear in bank data
        has_bank_connection = bool(customer_refs & all_bank_refs)

        if has_bank_connection:
            customers_with_bank.add(customer_id)
        else:
            customers_no_bank.add(customer_id)

    # Results
    total_customers = len(customer_to_refs)
    customers_with_bank_count = len(customers_with_bank)
    customers_no_bank_count = len(customers_no_bank)

    print(f"\nLOYALTY CUSTOMER ANALYSIS RESULTS:")
    print(f"  Total loyalty customers: {total_customers:,}")
    print(f"  Customers with bank connections: {customers_with_bank_count:,} ({customers_with_bank_count/total_customers*100:.1f}%)")
    print(f"  Customers with NO bank connections: {customers_no_bank_count:,} ({customers_no_bank_count/total_customers*100:.1f}%)")

    # Get detailed stats for customers without bank connections
    print(f"\nCollecting customer details...")

    customers_no_bank_df = facts_df[facts_df['customer_id'].isin(customers_no_bank)]

    customer_details = customers_no_bank_df.groupby('customer_id').agg({
        'cheque_id': 'nunique',
        'total_price': 'sum',
        'discount_card_no': lambda x: list(x.unique())
    }).reset_index()

    customer_details.columns = ['customer_id', 'transaction_count', 'total_spending', 'discount_cards']
    customer_details['avg_transaction'] = customer_details['total_spending'] / customer_details['transaction_count']
    customer_details['cards_count'] = customer_details['discount_cards'].apply(len)

    print(f"  Customers without bank connections: {len(customer_details):,}")

    # Get detailed stats for bank cards without loyalty
    print(f"\nCreating bank card summaries...")

    # Use groupby with separate operations for compatibility across pandas versions
    grouped = bank_no_loyalty_transactions.groupby('client')

    # Transaction statistics
    transaction_count = grouped['transaction_amount'].count()
    total_amount = grouped['transaction_amount'].sum()
    avg_amount = grouped['transaction_amount'].mean()

    # Bank names and reference numbers as lists
    banks_used = grouped['bank_name'].apply(lambda x: list(x.unique()))
    reference_numbers = grouped['reference_number'].apply(lambda x: list(x.unique()))

    # Combine into single dataframe
    bank_no_loyalty_stats = pd.DataFrame({
        'client': transaction_count.index,
        'transaction_count': transaction_count.values,
        'total_amount': total_amount.values,
        'avg_amount': avg_amount.values,
        'banks_used': banks_used.values,
        'reference_numbers': reference_numbers.values
    })

    bank_no_loyalty_stats['banks_count'] = bank_no_loyalty_stats['banks_used'].apply(len)
    bank_no_loyalty_stats['references_count'] = bank_no_loyalty_stats['reference_numbers'].apply(len)

    print(f"  Bank cards without loyalty connections: {len(bank_no_loyalty_stats):,}")

    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================

    results = {
        # Bank card analysis
        'total_bank_cards': total_bank_cards,
        'bank_cards_with_loyalty': bank_with_loyalty_count,
        'bank_cards_no_loyalty': bank_no_loyalty_count,
        'bank_cards_with_loyalty_pct': bank_with_loyalty_count/total_bank_cards*100,
        'bank_cards_no_loyalty_pct': bank_no_loyalty_count/total_bank_cards*100,

        # Loyalty customer analysis
        'total_loyalty_customers': total_customers,
        'customers_with_bank': customers_with_bank_count,
        'customers_no_bank': customers_no_bank_count,
        'customers_with_bank_pct': customers_with_bank_count/total_customers*100,
        'customers_no_bank_pct': customers_no_bank_count/total_customers*100,

        # Detailed data
        'customer_details_no_bank': customer_details,
        'bank_card_stats_no_loyalty': bank_no_loyalty_stats,

        # Lists for export
        'customer_ids_no_bank': list(customers_no_bank),
        'customer_ids_with_bank': list(customers_with_bank),
        'bank_card_ids_no_loyalty': list(bank_cards_no_loyalty),
        'bank_card_ids_with_loyalty': list(bank_cards_with_loyalty),
    }

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")

    return results


def save_cash_loyalty_results(results, output_dir="./data"):
    """Save analysis results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    saved_files = []

    try:
        print(f"\nSAVING RESULTS TO CSV FILES...")

        # 1. Customer details (no bank connections)
        if not results['customer_details_no_bank'].empty:
            file_path = output_path / "customers_no_bank_connections_details.csv"
            export_df = results['customer_details_no_bank'].copy()
            export_df['discount_cards'] = export_df['discount_cards'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            export_df.to_csv(file_path, index=False)
            saved_files.append(str(file_path))
            print(f"  Saved: {file_path.name}")

        # 2. Simple customer ID list (no bank)
        file_path = output_path / "customer_ids_no_bank_connections.csv"
        pd.DataFrame({'customer_id': results['customer_ids_no_bank']}).to_csv(file_path, index=False)
        saved_files.append(str(file_path))
        print(f"  Saved: {file_path.name}")

        # 3. Bank card details (no loyalty)
        if not results['bank_card_stats_no_loyalty'].empty:
            file_path = output_path / "bank_cards_no_loyalty_details.csv"
            export_df = results['bank_card_stats_no_loyalty'].copy()
            export_df['banks_used'] = export_df['banks_used'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            export_df['reference_numbers'] = export_df['reference_numbers'].apply(
                lambda x: ', '.join(map(str, x[:10])) + (f' (+{len(x)-10} more)' if len(x) > 10 else '')
                if isinstance(x, list) else str(x))
            export_df.to_csv(file_path, index=False)
            saved_files.append(str(file_path))
            print(f"  Saved: {file_path.name}")

        # 4. Simple bank card ID list (no loyalty)
        file_path = output_path / "bank_card_ids_no_loyalty.csv"
        pd.DataFrame({'client': results['bank_card_ids_no_loyalty']}).to_csv(file_path, index=False)
        saved_files.append(str(file_path))
        print(f"  Saved: {file_path.name}")

        # 5. Summary statistics
        summary_data = {
            'total_bank_cards': results['total_bank_cards'],
            'bank_cards_with_loyalty': results['bank_cards_with_loyalty'],
            'bank_cards_no_loyalty': results['bank_cards_no_loyalty'],
            'bank_cards_with_loyalty_pct': results['bank_cards_with_loyalty_pct'],
            'bank_cards_no_loyalty_pct': results['bank_cards_no_loyalty_pct'],
            'total_loyalty_customers': results['total_loyalty_customers'],
            'customers_with_bank': results['customers_with_bank'],
            'customers_no_bank': results['customers_no_bank'],
            'customers_with_bank_pct': results['customers_with_bank_pct'],
            'customers_no_bank_pct': results['customers_no_bank_pct']
        }

        file_path = output_path / "cash_loyalty_analysis_summary.csv"
        pd.DataFrame([summary_data]).to_csv(file_path, index=False)
        saved_files.append(str(file_path))
        print(f"  Saved: {file_path.name}")

        print(f"\nTOTAL FILES SAVED: {len(saved_files)}")

    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def print_cash_loyalty_summary(results):
    """Print comprehensive summary of results"""
    print("\n" + "="*70)
    print("CUSTOMER-BANK CONNECTION ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nBANK CARDS:")
    print(f"  Total bank cards: {results['total_bank_cards']:,}")
    print(f"  Cards with loyalty connections: {results['bank_cards_with_loyalty']:,} ({results['bank_cards_with_loyalty_pct']:.1f}%)")
    print(f"  Cards with NO loyalty connections: {results['bank_cards_no_loyalty']:,} ({results['bank_cards_no_loyalty_pct']:.1f}%)")

    print(f"\nLOYALTY CUSTOMERS:")
    print(f"  Total customers: {results['total_loyalty_customers']:,}")
    print(f"  Customers with bank connections: {results['customers_with_bank']:,} ({results['customers_with_bank_pct']:.1f}%)")
    print(f"  Customers with NO bank connections: {results['customers_no_bank']:,} ({results['customers_no_bank_pct']:.1f}%)")

    if not results['customer_details_no_bank'].empty:
        details = results['customer_details_no_bank']
        print(f"\nCUSTOMERS WITHOUT BANK CONNECTIONS - BEHAVIOR:")
        print(f"  Average spending: ${details['total_spending'].mean():.2f}")
        print(f"  Median spending: ${details['total_spending'].median():.2f}")
        print(f"  Average transactions: {details['transaction_count'].mean():.1f}")

    if not results['bank_card_stats_no_loyalty'].empty:
        stats = results['bank_card_stats_no_loyalty']
        print(f"\nBANK CARDS WITHOUT LOYALTY - BEHAVIOR:")
        print(f"  Average spending: ${stats['total_amount'].mean():.2f}")
        print(f"  Median spending: ${stats['total_amount'].median():.2f}")
        print(f"  Average transactions: {stats['transaction_count'].mean():.1f}")

    print(f"\n{'='*70}")


def load_facts_from_database(server_ip='192.168.20.9', database='ORINABIJI_DWH',
                            start_date='2025-01-01', end_date='2025-12-31'):
    """Load facts data from database with customer_id"""
    print(f"\n=== FETCHING FACTS DATA FROM DATABASE ===")
    print(f"Server: {server_ip}, Database: {database}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        from pathlib import Path
        import pyodbc

        def connect_server(ip, database):
            conn_str = (
                "Driver={SQL Server};"
                "Server=" + ip + ";"
                "Database=" + database + ";"
                "Trusted_Connection=yes;"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            return conn, cursor

        def sql_to_pandas(conn, cursor, sql_string):
            import pandas as pd
            cursor.execute(sql_string)
            rows = cursor.fetchall()
            if rows:
                return pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
            return pd.DataFrame()

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

            # Save using the module's save function
            try:
                facts_df.to_parquet(DATA_DIR / "facts_data_2025_combined_customer_id.parquet", index=False)
                print(f"✓ Saved to {DATA_DIR / 'facts_data_2025_combined_customer_id.parquet'}")
            except:
                import pickle
                with open(DATA_DIR / "facts_data_2025_combined_customer_id.pkl", 'wb') as f:
                    pickle.dump(facts_df, f)
                print(f"✓ Saved to {DATA_DIR / 'facts_data_2025_combined_customer_id.pkl'}")

        cursor.close()
        conn.close()
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
        import pyodbc

        def connect_server(ip, database):
            conn_str = (
                "Driver={SQL Server};"
                "Server=" + ip + ";"
                "Database=" + database + ";"
                "Trusted_Connection=yes;"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            return conn, cursor

        def sql_to_pandas(conn, cursor, sql_string):
            import pandas as pd
            cursor.execute(sql_string)
            rows = cursor.fetchall()
            if rows:
                return pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
            return pd.DataFrame()

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

            try:
                bank_df.to_parquet(DATA_DIR / "bank_data_combined.parquet", index=False)
                print(f"✓ Saved to {DATA_DIR / 'bank_data_combined.parquet'}")
            except:
                import pickle
                with open(DATA_DIR / "bank_data_combined.pkl", 'wb') as f:
                    pickle.dump(bank_df, f)
                print(f"✓ Saved to {DATA_DIR / 'bank_data_combined.pkl'}")

        cursor.close()
        conn.close()
        return bank_df

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_cash_loyalty_analysis(server_ip='192.168.20.9', database='ORINABIJI_DWH', force_reload=False):
    """
    Run complete analysis - loads from saved files or fetches from database

    Parameters:
    - server_ip: SQL Server IP (default: '192.168.20.9')
    - database: Database name (default: 'ORINABIJI_DWH')
    - force_reload: Force reload from database even if files exist (default: False)
    """
    print("RUNNING CUSTOMER-BANK CONNECTION ANALYSIS")
    print("="*60)

    if force_reload:
        print("Force reload requested - fetching fresh data from database...")
        facts_df = None
        bank_df = None
    else:
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

    print(f"Loaded data successfully:")
    print(f"  Facts: {len(facts_df):,} records")
    print(f"  Bank: {len(bank_df):,} records")

    # Run analysis
    results = analyze_cash_vs_loyalty_patterns(facts_df, bank_df)

    if results is None:
        print("Analysis failed")
        return None

    # Print summary
    print_cash_loyalty_summary(results)

    # Save results
    save_cash_loyalty_results(results)

    return results


def main():
    """Main execution function"""
    print(__doc__)
    results = run_cash_loyalty_analysis()

    if results:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("\nAnalysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
