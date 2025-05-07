from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our custom modules
from data_loader import load_mimic_dataset
from evaluation import evaluate_classifier
from visualization import plot_missing_values, compare_imputation_methods


def evaluate_imputation_quality(X_true, X_imputed, mask, dataset_name, imputation_method,
                                train_miss_rate, test_miss_rate, holdout_set):
    """
    Comprehensive evaluation of imputation quality using all three classes of metrics.

    Parameters:
    -----------
    X_true : array-like
        Original complete data
    X_imputed : array-like
        Imputed data
    mask : array-like
        Boolean mask indicating missing values
    dataset_name : str
        Name of the dataset
    imputation_method : str
        Name of the imputation method
    train_miss_rate : float
        Missingness rate in the training data
    test_miss_rate : float
        Missingness rate in the test data
    holdout_set : int
        Index of the holdout set

    Returns:
    --------
    dict
        Dictionary with all imputation quality metrics
    """
    # Class A: Sample-wise metrics
    sample_metrics = compute_sample_wise_discrepancy(X_true, X_imputed, mask)

    # Class B: Feature-wise metrics
    feature_metrics = compute_feature_wise_discrepancy(X_true, X_imputed, mask)

    # Class C: Sliced Wasserstein metrics
    sliced_metrics, baseline_dists, imputed_dists, distance_ratios = compute_sliced_wasserstein(
        X_true, X_imputed, n_directions=50, n_partitions=10
    )

    # Combine all metrics
    all_metrics = {**sample_metrics, **feature_metrics, **sliced_metrics}

    # Add metadata
    all_metrics['dataset'] = dataset_name
    all_metrics['imputation_method'] = imputation_method
    all_metrics['train_miss_rate'] = train_miss_rate
    all_metrics['test_miss_rate'] = test_miss_rate
    all_metrics['holdout_set'] = holdout_set

    # Save raw distributions for further analysis
    all_metrics['baseline_dists'] = baseline_dists
    all_metrics['imputed_dists'] = imputed_dists
    all_metrics['distance_ratios'] = distance_ratios

    return all_metrics


def evaluate_classifier_performance(X_train, y_train, X_test, y_test, classifier_name, classifier_params):
    """
    Train and evaluate a classifier on the imputed data.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    classifier_name : str
        Name of the classifier
    classifier_params : dict
        Classifier hyperparameters

    Returns:
    --------
    dict
        Dictionary with classifier performance metrics
    """
    # Initialize the classifier
    if classifier_name == 'LogisticRegression':
        classifier = LogisticRegression(**classifier_params)
    elif classifier_name == 'RandomForest':
        classifier = RandomForestClassifier(**classifier_params)
    elif classifier_name == 'XGBoost':
        classifier = xgb.XGBClassifier(**classifier_params)
    elif classifier_name == 'NGBoost':
        # NGBoost implementation (you may need to install the ngboost package)
        from ngboost import NGBClassifier
        classifier = NGBClassifier(**classifier_params)
    elif classifier_name == 'NeuralNetwork':
        # Simple MLP classifier from scikit-learn
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(**classifier_params)
    else:
        raise ValueError(f"Classifier {classifier_name} not supported")

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Get predictions
    if hasattr(classifier, 'predict_proba'):
        y_prob = classifier.predict_proba(X_test)[:, 1]
    else:
        y_prob = classifier.predict(X_test)

    # Evaluate performance
    metrics = evaluate_classifier(y_test, y_prob)

    # Add metadata
    metrics['classifier'] = classifier_name

    return metrics, classifier


def run_comprehensive_evaluation(dataset_name, train_miss_rates, test_miss_rates,
                                 imputation_methods, classifiers, n_holdouts=3, random_state=42):
    """
    Run comprehensive evaluation pipeline on a dataset.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    train_miss_rates : list
        List of missingness rates for training data
    test_miss_rates : list
        List of missingness rates for test data
    imputation_methods : dict
        Dictionary mapping method names to imputation functions
    classifiers : dict
        Dictionary mapping classifier names to parameter dictionaries
    n_holdouts : int
        Number of holdout sets to use
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with all evaluation results
    """
    # Dictionary to store results
    all_results = {
        'imputation_quality': [],
        'classifier_performance': []
    }

    # Iterate over all combinations
    combinations = list(itertools.product(train_miss_rates, test_miss_rates))

    for train_miss, test_miss in tqdm(combinations, desc="Missingness combinations"):
        for holdout_idx in range(n_holdouts):
            # Load or generate dataset with specified missingness
            if dataset_name == 'MIMIC':
                # Use your existing function
                X_train, X_val, X_test, y_train, y_val, y_test = load_mimic_dataset(train_miss, random_state)

                # Combine train and validation for simplicity
                X_train_full = np.vstack([X_train, X_val])
                y_train_full = np.hstack([y_train, y_val])

                # For MIMIC, we need to handle the case where we don't have the ground truth
                # We'll use the imputed data from MissForest as a proxy for true data for evaluation
                if 'MissForest' in imputation_methods:
                    X_train_true = imputation_methods['MissForest'](X_train_full, None)[0]
                    X_test_true = imputation_methods['MissForest'](X_test, None)[0]
                else:
                    # Fall back to mean imputation if MissForest is not available
                    X_train_true = SimpleImputer().fit_transform(X_train_full)
                    X_test_true = SimpleImputer().fit_transform(X_test)

                # Create masks of missing values
                train_mask = np.isnan(X_train_full)
                test_mask = np.isnan(X_test)

            elif dataset_name in ['Simulated_N', 'Simulated_NC']:
                # For synthetic data, we can generate complete data and then introduce missingness
                if dataset_name == 'Simulated_N':
                    from sklearn.datasets import make_classification
                    X, y = make_classification(
                        n_samples=1000, n_features=25, n_informative=25,
                        random_state=random_state + holdout_idx
                    )
                else:  # Simulated_NC
                    from your_custom_module import load_simulated_nc_data
                    X, y = load_simulated_nc_data(random_state=random_state + holdout_idx)

                # Split into train/test
                from sklearn.model_selection import train_test_split
                X_train_full, X_test, y_train_full, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=random_state + holdout_idx
                )

                # Store the true values before introducing missingness
                X_train_true = X_train_full.copy()
                X_test_true = X_test.copy()

                # Introduce missingness
                train_mask = np.random.rand(*X_train_full.shape) < train_miss
                test_mask = np.random.rand(*X_test.shape) < test_miss

                X_train_full_missing = X_train_full.copy()
                X_test_missing = X_test.copy()

                X_train_full_missing[train_mask] = np.nan
                X_test_missing[test_mask] = np.nan

                # Update variables to use the missing data versions
                X_train_full = X_train_full_missing
                X_test = X_test_missing

            else:
                raise ValueError(f"Dataset {dataset_name} not supported")

            # Apply each imputation method
            for imp_name, imp_func in imputation_methods.items():
                # Apply imputation
                X_train_imp, X_test_imp = imp_func(X_train_full, X_test)

                # Evaluate imputation quality
                train_quality = evaluate_imputation_quality(
                    X_train_true, X_train_imp, train_mask, dataset_name, imp_name,
                    train_miss, test_miss, holdout_idx
                )

                test_quality = evaluate_imputation_quality(
                    X_test_true, X_test_imp, test_mask, dataset_name, imp_name,
                    train_miss, test_miss, holdout_idx
                )

                all_results['imputation_quality'].append(train_quality)
                all_results['imputation_quality'].append(test_quality)

                # Apply each classifier
                for clf_name, clf_params in classifiers.items():
                    # Evaluate classifier performance
                    performance, clf_model = evaluate_classifier_performance(
                        X_train_imp, y_train_full, X_test_imp, y_test, clf_name, clf_params
                    )

                    # Add metadata
                    performance['dataset'] = dataset_name
                    performance['imputation_method'] = imp_name
                    performance['train_miss_rate'] = train_miss
                    performance['test_miss_rate'] = test_miss
                    performance['holdout_set'] = holdout_idx

                    all_results['classifier_performance'].append(performance)

    # Convert lists to DataFrames for easier analysis
    all_results['imputation_quality'] = pd.DataFrame(all_results['imputation_quality'])
    all_results['classifier_performance'] = pd.DataFrame(all_results['classifier_performance'])

    return all_results