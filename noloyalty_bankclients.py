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

# Configuration - use same data directory
DATA_DIR = Path("analysis_data")


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

    bank_no_loyalty_stats = bank_no_loyalty_transactions.groupby('client').agg({
        'transaction_amount': ['count', 'sum', 'mean'],
        'bank_name': lambda x: list(x.unique()),
        'reference_number': lambda x: list(x.unique())
    }).reset_index()

    bank_no_loyalty_stats.columns = ['client', 'transaction_count', 'total_amount',
                                      'avg_amount', 'banks_used', 'reference_numbers']
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


def run_cash_loyalty_analysis():
    """Run complete analysis"""
    print("RUNNING CUSTOMER-BANK CONNECTION ANALYSIS")
    print("="*60)

    # Load data
    facts_df = load_dataframe("facts_data_2025_combined_customer_id")
    bank_df = load_dataframe("bank_data_combined")

    if facts_df is None or bank_df is None:
        print("ERROR: Could not find saved data")
        print("Expected files in './analysis_data/' directory:")
        print("  - facts_data_2025_combined_customer_id.parquet (or .pkl)")
        print("  - bank_data_combined.parquet (or .pkl)")
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
