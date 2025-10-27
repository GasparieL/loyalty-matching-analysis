# Loyalty Matching Analysis

This repository contains Python scripts for analyzing loyalty customer transactions and their connections to bank card data.

## Scripts

### 1. loyalty_1to1_min_spending_fixed.py
Analyzes loyalty customer transactions and matches them with bank card data.

**Purpose**: Identify customers with 1 or 2 valid bank cards from BOG/TBC banks.

**Key Features**:
- Validates customer-card combinations (>=3 transactions OR >$35 spending)
- Filters customers with <=2 different bank cards from <=2 different banks
- Ensures customers with 2 cards have them from 2 different banks
- Removes shared cards (used by multiple customers)

**Output Files**:
- `customer_card_pairs.csv`: Primary output with customer-card mappings
- `final_clean_customers_summary.csv`: Customer-level summary statistics
- `customer_card_combinations_detailed.csv`: Detailed card usage data
- `analysis_summary.csv`: Overall analysis metrics

### 2. noloyalty_bankclients.py
Analyzes the connection between loyalty customers and bank card transactions.

**Purpose**: Identify disconnected entities between loyalty and bank systems.

**Key Features**:
- Identifies bank cards with NO connections to loyalty customers
- Identifies loyalty customers with NO connections to bank transactions
- Connection established through reference_number matching

**Output Files**:
- `customers_no_bank_connections_details.csv`: Detailed stats for customers without bank connections
- `customer_ids_no_bank_connections.csv`: Simple list of customer IDs
- `bank_cards_no_loyalty_details.csv`: Detailed stats for bank cards without loyalty
- `bank_card_ids_no_loyalty.csv`: Simple list of bank card IDs
- `cash_loyalty_analysis_summary.csv`: Overall summary statistics

## Requirements

```
pandas
numpy
pyodbc
matplotlib
seaborn
tqdm
```

## Data Requirements

Both scripts expect the following data files in the `./analysis_data/` directory:
- `facts_data_2025_combined_customer_id.parquet` (or `.pkl`)
- `bank_data_combined.parquet` (or `.pkl`)

## Usage

### Run Loyalty Card Matching Analysis
```bash
python loyalty_1to1_min_spending_fixed.py
```

### Run No-Loyalty Bank Clients Analysis
```bash
python noloyalty_bankclients.py
```

## Key Concepts

- **client** (in bank data): Bank card number (the actual card identifier)
- **reference_number**: Transaction reference that links facts and bank tables
- **customer_id** (in facts data): Loyalty program customer identifier

## Output

All output files are saved to the `./data/` directory, which is created automatically if it doesn't exist.
