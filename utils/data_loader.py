import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_mimic_files(missing_percentage=0.25):
    """
    Find the appropriate MIMIC files based on the missing percentage.

    Parameters:
    -----------
    missing_percentage : float
        Percentage of missing values in the files

    Returns:
    --------
    train_file, val_file, test_file : str
        Paths to the train, validation, and test files
    """
    # Format the missing percentage for filename matching
    if missing_percentage == 0.25:
        missing_str = "0.25"
    elif missing_percentage == 0.5:
        missing_str = "0.5"
    elif missing_percentage == 0:
        missing_str = "0"
    else:
        raise ValueError(f"Missing percentage {missing_percentage} not supported")

    # Check different possible data directories
    data_dirs = [
        "data/MIMIC_subset_mcar",
        "DATA/MIMIC_subset_mcar",
        "MIMIC_subset_mcar"
    ]

    data_dir = None
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break

    if data_dir is None:
        raise FileNotFoundError("Could not find MIMIC data directory")

    # Find files matching the pattern with the specified missing percentage
    all_files = os.listdir(data_dir)

    # Look for train file
    train_pattern = f"train_missing_{missing_str}_test_missing_{missing_str}.csv"
    train_files = [f for f in all_files if train_pattern in f and "train_0" in f]

    # If no files found with exact pattern, try a more flexible approach
    if not train_files:
        train_files = [f for f in all_files if f"train_missing_{missing_str}" in f and "train_0" in f]

    # Look for test file
    test_pattern = f"test_missing_{missing_str}.csv"
    test_files = [f for f in all_files if test_pattern in f]

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Could not find MIMIC files with missing percentage {missing_str} in {data_dir}"
        )

    # Select the first matching file for simplicity
    train_file = os.path.join(data_dir, train_files[0])

    # For validation, we'll use the same file as training but split it later
    val_file = train_file
    test_file = os.path.join(data_dir, test_files[0])

    return train_file, val_file, test_file


def load_mimic_dataset(missing_percentage=0.25, random_state=42, train_size=0.8):
    """
    Load the MIMIC dataset with the specified missing percentage.

    Parameters:
    -----------
    missing_percentage : float
        Percentage of missing values in the files
    random_state : int
        Random seed for reproducibility
    train_size : float
        Proportion of training data to use (the rest will be used for validation)

    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : pandas DataFrames/Series
        Split data with missing values
    """
    # Find the appropriate files
    train_file, val_file, test_file = get_mimic_files(missing_percentage)

    print(f"Loading train data from: {train_file}")
    print(f"Loading test data from: {test_file}")

    # Load the data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Extract features and target
    outcome_col = 'outcome'
    X_full = train_data.drop(columns=[outcome_col])
    y_full = train_data[outcome_col]

    X_test = test_data.drop(columns=[outcome_col])
    y_test = test_data[outcome_col]

    # Split the training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, train_size=train_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test