#!/usr/bin/env python3
"""
ANSUR II Dataset Preprocessor
Extracts relevant columns for crutch fitting analysis.

Usage:
    python scripts/preprocess_ansur.py
    python scripts/preprocess_ansur.py --auto  # Skip confirmation prompts
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Expected file locations
RAW_DATA_DIR = 'dataset/raw_ansur'
OUTPUT_FILE = 'dataset/ansur_simplified.csv'

# Column mapping (ANSUR column name -> our standardized name)
COLUMN_MAPPING = {
    'stature': 'stature_mm',
    'weightkg': 'weight_kg',
    'Gender': 'sex',
    'wristheight': 'wrist_height_mm',
    'axillaheight': 'axilla_height_mm',
    'subjectid': 'original_id'
}

# Anthropometric ratios for estimation (validated from ANSUR II studies)
WRIST_HEIGHT_RATIO = 0.485  # wrist_height / stature
AXILLA_HEIGHT_RATIO = 0.815  # axilla_height / stature


def find_ansur_files(data_dir: str) -> tuple:
    """Find ANSUR male and female CSV files."""
    male_file = None
    female_file = None

    if not os.path.exists(data_dir):
        return None, None

    for filename in os.listdir(data_dir):
        lower = filename.lower()
        if 'male' in lower and 'female' not in lower and filename.endswith('.csv'):
            male_file = os.path.join(data_dir, filename)
        elif 'female' in lower and filename.endswith('.csv'):
            female_file = os.path.join(data_dir, filename)

    return male_file, female_file


def print_available_columns(filepath: str, label: str) -> list:
    """Print available columns from a CSV file."""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, nrows=0, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        columns = df.columns.tolist()

        print(f"\n📋 {label} - {len(columns)} columns found")
        print("-" * 50)

        # Find and highlight relevant columns
        relevant_keywords = ['stature', 'height', 'weight', 'wrist', 'axilla',
                           'gender', 'sex', 'radiale', 'acromion']

        relevant = []
        for col in columns:
            for keyword in relevant_keywords:
                if keyword in col.lower():
                    relevant.append(col)
                    break

        if relevant:
            print("Relevant columns found:")
            for col in relevant:
                print(f"  - {col}")

        return columns

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []


def load_and_process_file(filepath: str, sex_label: str) -> pd.DataFrame:
    """Load ANSUR file and extract relevant columns."""
    print(f"\n📂 Loading: {filepath}")

    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"   Encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        print(f"   ❌ Could not read file with any encoding")
        return pd.DataFrame()

    print(f"   Rows: {len(df)}")

    # Check which columns exist
    available_cols = set(df.columns)
    cols_to_use = {}

    for ansur_col, our_col in COLUMN_MAPPING.items():
        if ansur_col in available_cols:
            cols_to_use[ansur_col] = our_col
        else:
            print(f"   ⚠️  Column '{ansur_col}' not found")

    # Extract columns
    result = pd.DataFrame()

    # Subject ID
    if 'subjectid' in cols_to_use:
        result['original_id'] = df['subjectid']
    else:
        result['original_id'] = range(1, len(df) + 1)

    # Stature (height) - required
    if 'stature' in cols_to_use:
        result['height_cm'] = df['stature'] / 10.0  # mm to cm
    else:
        print("   ❌ Critical: 'stature' column not found!")
        return pd.DataFrame()

    # Weight - ANSUR 'weightkg' column is actually in decagrams (needs /10)
    if 'weightkg' in cols_to_use:
        # Check if values seem to be in decagrams (> 200 likely means decagrams)
        if df['weightkg'].median() > 200:
            result['weight_kg'] = df['weightkg'] / 10.0
            print("   ℹ️  Weight column appears to be in decagrams, converting to kg")
        else:
            result['weight_kg'] = df['weightkg']
    elif 'Weightlbs' in available_cols:
        # Use pounds and convert
        result['weight_kg'] = df['Weightlbs'] * 0.453592
        print("   ℹ️  Using Weightlbs, converted to kg")
    else:
        result['weight_kg'] = np.nan

    # Sex
    if 'Gender' in cols_to_use:
        # Standardize to M/F
        result['sex'] = df['Gender'].apply(lambda x: 'M' if str(x).upper().startswith('M') else 'F')
    else:
        result['sex'] = sex_label

    # Wrist height
    if 'wristheight' in cols_to_use:
        result['wrist_height_cm'] = df['wristheight'] / 10.0
        result['wrist_estimated'] = False
    else:
        # Estimate from height
        result['wrist_height_cm'] = result['height_cm'] * WRIST_HEIGHT_RATIO
        result['wrist_estimated'] = True
        print(f"   ℹ️  Wrist height estimated using ratio {WRIST_HEIGHT_RATIO}")

    # Axilla height
    if 'axillaheight' in cols_to_use:
        result['axilla_height_cm'] = df['axillaheight'] / 10.0
        result['axilla_estimated'] = False
    else:
        # Estimate from height
        result['axilla_height_cm'] = result['height_cm'] * AXILLA_HEIGHT_RATIO
        result['axilla_estimated'] = True
        print(f"   ℹ️  Axilla height estimated using ratio {AXILLA_HEIGHT_RATIO}")

    return result


def generate_user_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Generate standardized user IDs."""
    df = df.copy()
    df['user_id'] = [f"U{i:05d}" for i in range(1, len(df) + 1)]
    return df


def compute_ideal_crutch_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for ideal crutch settings based on body measurements."""
    df = df.copy()

    # Ideal overall crutch height: 2 inches (5cm) below axilla
    df['ideal_overall_cm'] = df['axilla_height_cm'] - 5.0

    # Ideal grip height: at wrist level
    df['ideal_grip_cm'] = df['wrist_height_cm']

    return df


def main():
    parser = argparse.ArgumentParser(description='Preprocess ANSUR II dataset for crutch fitting')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-detect columns without confirmation')
    parser.add_argument('--input-dir', '-i', type=str, default=RAW_DATA_DIR,
                       help=f'Input directory (default: {RAW_DATA_DIR})')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_FILE,
                       help=f'Output file (default: {OUTPUT_FILE})')

    args = parser.parse_args()

    print("=" * 60)
    print("ANSUR II DATASET PREPROCESSOR")
    print("=" * 60)

    # Find data files
    male_file, female_file = find_ansur_files(args.input_dir)

    if not male_file and not female_file:
        print(f"\n❌ No ANSUR files found in {args.input_dir}/")
        print("Expected files like:")
        print("  - ANSUR II MALE Public.csv")
        print("  - ANSUR II FEMALE Public.csv")
        sys.exit(1)

    print(f"\n📁 Found files:")
    if male_file:
        print(f"   Male: {male_file}")
    if female_file:
        print(f"   Female: {female_file}")

    # Show available columns
    if not args.auto:
        if male_file:
            print_available_columns(male_file, "Male dataset")
        if female_file:
            print_available_columns(female_file, "Female dataset")

        print("\n" + "=" * 60)
        print("COLUMNS TO EXTRACT:")
        print("=" * 60)
        for ansur_col, our_col in COLUMN_MAPPING.items():
            print(f"  {ansur_col:20} -> {our_col}")

        confirm = input("\nProceed with these columns? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            sys.exit(0)

    # Process files
    dataframes = []

    if male_file:
        df_male = load_and_process_file(male_file, 'M')
        if not df_male.empty:
            dataframes.append(df_male)

    if female_file:
        df_female = load_and_process_file(female_file, 'F')
        if not df_female.empty:
            dataframes.append(df_female)

    if not dataframes:
        print("\n❌ No data processed successfully")
        sys.exit(1)

    # Combine datasets
    print("\n📊 Combining datasets...")
    combined = pd.concat(dataframes, ignore_index=True)
    initial_count = len(combined)

    # Remove rows with missing height or weight
    combined = combined.dropna(subset=['height_cm', 'weight_kg'])
    dropped_count = initial_count - len(combined)

    if dropped_count > 0:
        print(f"   Dropped {dropped_count} rows with missing height/weight")

    # Generate user IDs
    combined = generate_user_ids(combined)

    # Compute ideal crutch settings
    combined = compute_ideal_crutch_settings(combined)

    # Sort by height
    combined = combined.sort_values('height_cm').reset_index(drop=True)

    # Select and order final columns
    output_columns = [
        'user_id',
        'height_cm',
        'weight_kg',
        'sex',
        'wrist_height_cm',
        'axilla_height_cm',
        'ideal_overall_cm',
        'ideal_grip_cm',
        'wrist_estimated',
        'axilla_estimated'
    ]

    final_df = combined[output_columns].copy()

    # Round numeric columns
    numeric_cols = ['height_cm', 'weight_kg', 'wrist_height_cm', 'axilla_height_cm',
                   'ideal_overall_cm', 'ideal_grip_cm']
    for col in numeric_cols:
        final_df.loc[:, col] = final_df[col].round(1)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    final_df.to_csv(args.output, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(f"\n📊 Statistics:")
    print(f"   Total subjects: {len(final_df):,}")
    print(f"   Male: {len(final_df[final_df['sex'] == 'M']):,}")
    print(f"   Female: {len(final_df[final_df['sex'] == 'F']):,}")
    print(f"   Rows dropped: {dropped_count}")

    print(f"\n📏 Height range:")
    print(f"   Min: {final_df['height_cm'].min():.1f} cm ({final_df['height_cm'].min()/2.54:.1f}\")")
    print(f"   Max: {final_df['height_cm'].max():.1f} cm ({final_df['height_cm'].max()/2.54:.1f}\")")
    print(f"   Mean: {final_df['height_cm'].mean():.1f} cm ({final_df['height_cm'].mean()/2.54:.1f}\")")

    print(f"\n⚖️  Weight range:")
    print(f"   Min: {final_df['weight_kg'].min():.1f} kg ({final_df['weight_kg'].min()*2.205:.1f} lbs)")
    print(f"   Max: {final_df['weight_kg'].max():.1f} kg ({final_df['weight_kg'].max()*2.205:.1f} lbs)")
    print(f"   Mean: {final_df['weight_kg'].mean():.1f} kg ({final_df['weight_kg'].mean()*2.205:.1f} lbs)")

    # Check estimation status
    wrist_estimated_pct = final_df['wrist_estimated'].mean() * 100
    axilla_estimated_pct = final_df['axilla_estimated'].mean() * 100

    if wrist_estimated_pct > 0:
        print(f"\n⚠️  Wrist height estimated for {wrist_estimated_pct:.0f}% of subjects")
    if axilla_estimated_pct > 0:
        print(f"⚠️  Axilla height estimated for {axilla_estimated_pct:.0f}% of subjects")

    print(f"\n✅ Saved to: {args.output}")

    # Show sample rows
    print(f"\n📋 Sample data (first 5 rows):")
    print(final_df.head().to_string(index=False))


if __name__ == '__main__':
    main()
