"""
Preprocessing script for Motor Vehicle Insurance Policyholder Lapse Prediction.
Designed to run as a SageMaker Processing job.

Input: Raw CSV from S3 (semicolon-delimited)
Output: Train, validation, and test CSV files written to S3
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments passed by SageMaker Processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, 
                        default='/opt/ml/processing/input')
    parser.add_argument('--output-train', type=str, 
                        default='/opt/ml/processing/output/train')
    parser.add_argument('--output-validation', type=str, 
                        default='/opt/ml/processing/output/validation')
    parser.add_argument('--output-test', type=str, 
                        default='/opt/ml/processing/output/test')
    return parser.parse_args()


def load_and_clean(input_path):
    """Load raw data and perform initial cleaning."""
    # Find the CSV file in the input directory
    input_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    if not input_files:
        raise FileNotFoundError(f"No CSV files found in {input_path}")

    filepath = os.path.join(input_path, input_files[0])
    logger.info(f"Loading data from {filepath}")

    # Load with semicolon delimiter (European format)
    df = pd.read_csv(filepath, sep=';')
    logger.info(f"Raw data shape: {df.shape}")

    # Create binary target variable
    df['Lapsed'] = (df['Lapse'] > 0).astype(int)
    logger.info(f"Lapse rate: {df['Lapsed'].mean():.2%}")

    # Drop columns we won't use for modeling
    # Date_lapse leaks the target (it's only filled when Lapse > 0)
    # Lapse is replaced by our binary Lapsed column
    # ID will be used for splitting but dropped before training
    drop_cols = ['Date_lapse', 'Lapse']
    df = df.drop(columns=drop_cols)
    logger.info(f"Dropped columns: {drop_cols}")

    # Parse date columns to extract useful features
    date_cols = ['Date_start_contract', 'Date_last_renewal', 
                 'Date_next_renewal', 'Date_birth', 'Date_driving_licence']

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    # Engineer age and tenure from dates
    reference_date = df['Date_next_renewal'].max()

    if 'Date_birth' in df.columns:
        df['Age'] = ((reference_date - df['Date_birth']).dt.days / 365.25).round(1)

    if 'Date_driving_licence' in df.columns:
        df['Years_driving'] = ((reference_date - df['Date_driving_licence']).dt.days / 365.25).round(1)

    if 'Date_start_contract' in df.columns:
        df['Customer_tenure_days'] = (df['Date_next_renewal'] - df['Date_start_contract']).dt.days

    # Drop the raw date columns (model can't use dates directly)
    df = df.drop(columns=date_cols, errors='ignore')
    logger.info(f"Engineered features: Age, Years_driving, Customer_tenure_days")

    return df


def handle_missing_and_encode(df):
    """Handle missing values and encode categorical features."""
    logger.info("Handling missing values...")
    
    # Numerical columns: fill missing with median
    # Median is more robust than mean for skewed distributions
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {col} missing values with median: {median_val}")
    
    # Categorical columns: fill missing with mode (most frequent value)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Filled {col} missing values with mode: {mode_val}")
    
    # Encode categorical columns
    # Label encoding for low-cardinality categoricals
    logger.info("Encoding categorical features...")
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        logger.info(f"  Encoded {col}: {len(le.classes_)} categories")
    
    # Log final missing value check
    remaining_missing = df.isnull().sum().sum()
    logger.info(f"Remaining missing values: {remaining_missing}")
    
    return df


def split_data(df):
    """Split data into train/validation/test sets by customer ID.
    
    Splits by customer ID (not randomly by row) to prevent data leakage.
    If the same customer appeared in both train and test, the model could
    memorize individual customers rather than learning general patterns.
    """
    logger.info("Splitting data by customer ID...")
    
    # First split: 80% train+val, 20% test
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(splitter1.split(df, groups=df['ID']))
    
    train_val = df.iloc[train_val_idx]
    test = df.iloc[test_idx]
    
    # Second split: from the 80%, take 75% train and 25% validation
    # This gives us an overall 60/20/20 split
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(splitter2.split(train_val, groups=train_val['ID']))
    
    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]
    
    # Now drop ID — the model shouldn't use it as a feature
    train = train.drop(columns=['ID'])
    val = val.drop(columns=['ID'])
    test = test.drop(columns=['ID'])
    
    logger.info(f"Train set: {train.shape[0]:,} rows ({train['Lapsed'].mean():.2%} lapse rate)")
    logger.info(f"Validation set: {val.shape[0]:,} rows ({val['Lapsed'].mean():.2%} lapse rate)")
    logger.info(f"Test set: {test.shape[0]:,} rows ({test['Lapsed'].mean():.2%} lapse rate)")
    
    return train, val, test


def main():
    """Main preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    args = parse_args()
    
    # Step 1: Load and clean
    df = load_and_clean(args.input_data)
    
    # Step 2: Handle missing values and encode categoricals
    df = handle_missing_and_encode(df)
    
    # Step 3: Split into train/validation/test
    train, val, test = split_data(df)
    
    # Step 4: Save to output directories
    # Create output directories if they don't exist
    for output_dir in [args.output_train, args.output_validation, args.output_test]:
        os.makedirs(output_dir, exist_ok=True)
    
    train.to_csv(os.path.join(args.output_train, 'train.csv'), index=False)
    val.to_csv(os.path.join(args.output_validation, 'validation.csv'), index=False)
    test.to_csv(os.path.join(args.output_test, 'test.csv'), index=False)
    
    logger.info(f"Saved train data to {args.output_train}")
    logger.info(f"Saved validation data to {args.output_validation}")
    logger.info(f"Saved test data to {args.output_test}")
    
    # Log feature summary
    logger.info(f"\nFinal feature count: {train.shape[1] - 1} features + 1 target")
    logger.info(f"Features: {[col for col in train.columns if col != 'Lapsed']}")
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()