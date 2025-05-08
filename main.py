"""
Main script for running the imputation and classification pipeline.

This script:
1. Imports the necessary modules
2. Sets up configuration parameters
3. Loads data with different missing percentages
4. Applies various imputation methods
5. Trains different classifiers on the imputed data
6. Evaluates performance and generates visualization
7. Saves results to files
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
import argparse

# Setup path for imports
import sys
# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the project root to the Python path
sys.path.append(project_root)
# Also add the utils directory directly to handle relative imports within utility modules
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

# Import our custom modules
from imputation.mean import MeanImputer
from imputation.mice import MICEImputer
from imputation.forest import CustomMissForestImputer
from imputation.gain import GAIN
from imputation.miwae import MIWAE

from classification.logistic import LogisticRegressionClassifier
from classification.random_forest import RandomForestClassifier
from classification.neural_net import NeuralNetClassifier
from classification.xgboots_model import XGBoostClassifier
from classification.ngboost_model import NGBoostClassifier

# Import utility functions directly
from utils.data_loader import load_mimic_dataset

# Suppress warnings
warnings.filterwarnings('ignore')


def evaluate_classifier(y_true, y_pred_proba):
    """
    Evaluate a classifier using various metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities

    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score, brier_score_loss
    )
    import numpy as np

    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_proba),
        "sensitivity": recall_score(y_true, y_pred),  # Same as recall for binary classification
        "specificity": recall_score(1 - y_true, 1 - y_pred),  # Recall for the negative class
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba)
    }

    return metrics


def run_pipeline(missing_percentage, random_state=42):
    """
    Run the imputation and classification pipeline for a specific missing percentage.

    Parameters:
    -----------
    missing_percentage : float
        Percentage of missing values in the dataset
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with results
    """
    print(f"\n=== Running pipeline with missing percentage: {missing_percentage} ===")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Step 1: Load the data
    print("\nLoading data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_mimic_dataset(
            missing_percentage=missing_percentage,
            random_state=random_state
        )
        print(f"Data loaded successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # For demonstration, we'll create synthetic data if the real data can't be loaded
        print("Creating synthetic data for demonstration...")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            random_state=random_state
        )
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=random_state
        )

        # Convert to pandas DataFrame
        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])

        # Introduce missing values
        mask_train = np.random.rand(*X_train.shape) < missing_percentage
        mask_val = np.random.rand(*X_val.shape) < missing_percentage
        mask_test = np.random.rand(*X_test.shape) < missing_percentage

        X_train[mask_train] = np.nan
        X_val[mask_val] = np.nan
        X_test[mask_test] = np.nan

    # Step 2: Define imputation methods
    imputation_methods = {
        "Mean": MeanImputer(random_state=random_state),
        "MICE": MICEImputer(max_iter=10, random_state=random_state),
        "MissForest": CustomMissForestImputer(max_iter=10, n_estimators=100, random_state=random_state)
    }

    # Add more complex imputation methods for smaller datasets or if resources allow
    if X_train.shape[0] < 1000 or missing_percentage <= 0.25:
        imputation_methods.update({
            "GAIN": GAIN(n_epochs=100, batch_size=128, random_state=random_state),
            # Temporarily disable MIWAE due to tensor type issues
            "MIWAE": MIWAE(n_epochs=30, batch_size=32, random_state=random_state)
        })

    # Step 3: Define classifiers with tuned hyperparameters from the paper
    classifiers = {
        "Logistic": LogisticRegressionClassifier(max_iter=200, random_state=random_state),

        "RandomForest": RandomForestClassifier(
            n_estimators=144,  # Number of trees (8 * 18 = 144)
            max_depth=2,       # Maximum depth of trees
            min_samples_split=3,  # Minimum samples needed for splits
            min_samples_leaf=3,   # Minimum number of samples in each leaf
            random_state=random_state
        ),

        "XGBoost": XGBoostClassifier(
            n_estimators=144,  # Number of trees (8 * 18 = 144)
            max_depth=3,       # Maximum depth of trees
            learning_rate=0.1,
            subsample=0.6,     # Subsample of training instances per iteration
            random_state=random_state
        ),

        "NGBoost": NGBoostClassifier(
            n_estimators=192,  # Number of estimators (8 * 24 = 192)
            learning_rate=0.04,
            random_state=random_state
            # Note: subsample=0.6 is handled internally in the NGBoostClassifier implementation
        ),

        "NeuralNet": NeuralNetClassifier(
            hidden_layer_sizes=(5, 5, 5),  # 3 hidden layers with 5 neurons each
            learning_rate_init=0.03,       # Learning rate
            max_iter=200,
            random_state=random_state
        )
    }

    # Step 4: Run the pipeline
    results = []

    for imp_name, imputer in tqdm(imputation_methods.items(), desc="Imputation Methods"):
        # Measure imputation time
        start_time = time.time()

        # Fit imputer on training data
        print(f"\nApplying {imp_name} imputation...")
        imputer.fit(X_train)

        # Transform training and validation data
        X_train_imputed = imputer.transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)

        imputation_time = time.time() - start_time
        print(f"{imp_name} imputation completed in {imputation_time:.2f} seconds")

        # Train and evaluate classifiers
        for clf_name, classifier in tqdm(classifiers.items(), desc=f"Classifiers for {imp_name}"):
            print(f"\nTraining {clf_name} classifier with {imp_name} imputation...")

            # Train the classifier on the imputed training data
            classifier.fit(X_train_imputed, y_train)

            # Evaluate on validation set
            y_val_proba = classifier.predict_proba(X_val_imputed)
            val_metrics = evaluate_classifier(y_val, y_val_proba)

            # Evaluate on test set
            y_test_proba = classifier.predict_proba(X_test_imputed)
            test_metrics = evaluate_classifier(y_test, y_test_proba)

            # Extract classifier parameters
            if clf_name == "Logistic":
                parameters = {"max_iter": classifier.max_iter}
            elif clf_name == "RandomForest":
                parameters = {
                    "n_estimators": classifier.n_estimators,
                    "max_depth": classifier.max_depth,
                    "min_samples_split": classifier.min_samples_split,
                    "min_samples_leaf": classifier.min_samples_leaf
                }
            elif clf_name == "XGBoost":
                parameters = {
                    "n_estimators": classifier.n_estimators,
                    "max_depth": classifier.max_depth,
                    "learning_rate": classifier.learning_rate,
                    "subsample": classifier.subsample
                }
            elif clf_name == "NGBoost":
                parameters = {
                    "n_estimators": classifier.n_estimators,
                    "learning_rate": classifier.learning_rate,
                    "subsample": classifier.subsample
                }
            elif clf_name == "NeuralNet":
                parameters = {
                    "hidden_layers": len(classifier.hidden_layer_sizes),
                    "hidden_neurons": classifier.hidden_layer_sizes[0],
                    "learning_rate": classifier.learning_rate_init,
                    "max_iter": classifier.max_iter
                }

            # Create result dictionary
            result = {
                "imputation": imp_name,
                "classifier": clf_name,
                "missing_percentage": missing_percentage,
                "random_state": random_state,
                "imputation_time": imputation_time,
                "validation": val_metrics,
                "test": test_metrics,
                "parameters": parameters
            }

            # Save result
            results.append(result)

            # Save to JSON file
            result_filename = f"results/{imp_name}_{clf_name}_{missing_percentage}.json"
            with open(result_filename, 'w') as f:
                json.dump(result, f, indent=4)

            print(f"Results saved to {result_filename}")

    return results


def compare_imputation_methods(results, metric='auc'):
    """
    Compare different imputation methods based on a specific metric.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    metric : str, optional
        Metric to compare

    Returns:
    --------
    pandas DataFrame
        DataFrame with comparison results
    """
    # Extract relevant information
    comparison = []
    for result in results:
        comparison.append({
            'imputation': result['imputation'],
            'classifier': result['classifier'],
            'validation_' + metric: result['validation'][metric],
            'test_' + metric: result['test'][metric],
            'imputation_time': result['imputation_time']
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)

    # Compute average metrics across classifiers
    avg_metrics = comparison_df.groupby('imputation').agg({
        'validation_' + metric: 'mean',
        'test_' + metric: 'mean',
        'imputation_time': 'mean'
    }).reset_index()

    avg_metrics = avg_metrics.rename(columns={
        'validation_' + metric: 'avg_validation_' + metric,
        'test_' + metric: 'avg_test_' + metric,
        'imputation_time': 'avg_imputation_time'
    })

    return avg_metrics


def compare_classifiers(results, metric='auc'):
    """
    Compare different classifiers based on a specific metric.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    metric : str, optional
        Metric to compare

    Returns:
    --------
    pandas DataFrame
        DataFrame with comparison results
    """
    # Extract relevant information
    comparison = []
    for result in results:
        comparison.append({
            'imputation': result['imputation'],
            'classifier': result['classifier'],
            'validation_' + metric: result['validation'][metric],
            'test_' + metric: result['test'][metric]
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)

    # Compute average metrics across imputation methods
    avg_metrics = comparison_df.groupby('classifier').agg({
        'validation_' + metric: 'mean',
        'test_' + metric: 'mean'
    }).reset_index()

    avg_metrics = avg_metrics.rename(columns={
        'validation_' + metric: 'avg_validation_' + metric,
        'test_' + metric: 'avg_test_' + metric
    })

    return avg_metrics


def create_summary_plots(results, missing_percentage):
    """
    Create summary plots for visualization.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    missing_percentage : float
        Percentage of missing values

    Returns:
    --------
    None
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'imputation': r['imputation'],
            'classifier': r['classifier'],
            'validation_auc': r['validation']['auc'],
            'test_auc': r['test']['auc'],
            'validation_f1': r['validation']['F1'],
            'test_f1': r['test']['F1'],
            'imputation_time': r['imputation_time'],
            'missing_percentage': r['missing_percentage']
        }
        for r in results
    ])

    # Plot 1: Compare imputation methods (AUC)
    plt.figure(figsize=(14, 8))

    for clf in results_df['classifier'].unique():
        subset = results_df[results_df['classifier'] == clf]
        plt.plot(subset['imputation'], subset['test_auc'], 'o-', label=clf, linewidth=2, markersize=8)

    plt.title(f'Test AUC by Imputation Method (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/imputation_comparison_auc_{missing_percentage}.png", dpi=300)

    # Plot 2: Compare imputation methods (F1)
    plt.figure(figsize=(14, 8))

    for clf in results_df['classifier'].unique():
        subset = results_df[results_df['classifier'] == clf]
        plt.plot(subset['imputation'], subset['test_f1'], 'o-', label=clf, linewidth=2, markersize=8)

    plt.title(f'Test F1 Score by Imputation Method (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/imputation_comparison_f1_{missing_percentage}.png", dpi=300)

    # Plot 3: Compare classifiers (AUC)
    plt.figure(figsize=(14, 8))

    for imp in results_df['imputation'].unique():
        subset = results_df[results_df['imputation'] == imp]
        plt.plot(subset['classifier'], subset['test_auc'], 'o-', label=imp, linewidth=2, markersize=8)

    plt.title(f'Test AUC by Classifier (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Classifier', fontsize=14)
    plt.ylabel('Test AUC', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/classifier_comparison_auc_{missing_percentage}.png", dpi=300)

    # Plot 4: Compare classifiers (F1)
    plt.figure(figsize=(14, 8))

    for imp in results_df['imputation'].unique():
        subset = results_df[results_df['imputation'] == imp]
        plt.plot(subset['classifier'], subset['test_f1'], 'o-', label=imp, linewidth=2, markersize=8)

    plt.title(f'Test F1 Score by Classifier (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Classifier', fontsize=14)
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/classifier_comparison_f1_{missing_percentage}.png", dpi=300)

    # Plot 5: Imputation time comparison
    plt.figure(figsize=(12, 8))

    imputation_times = results_df.groupby('imputation')['imputation_time'].mean()
    ax = imputation_times.plot(kind='bar', color='skyblue', edgecolor='black')

    # Add time values on top of bars
    for i, v in enumerate(imputation_times):
        ax.text(i, v + 0.1, f"{v:.1f}s", ha='center', fontsize=10)

    plt.title(f'Imputation Time Comparison (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/imputation_time_{missing_percentage}.png", dpi=300)

    # Plot 6: Heatmap of AUC values
    plt.figure(figsize=(12, 10))

    # Create a pivot table
    heatmap_data = results_df.pivot_table(
        values='test_auc',
        index='classifier',
        columns='imputation'
    )

    # Plot heatmap
    import seaborn as sns
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap='YlGnBu',
        vmin=0.5,
        vmax=1.0,
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'AUC Score'}
    )

    plt.title(f'Test AUC Heatmap (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Classifier', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/auc_heatmap_{missing_percentage}.png", dpi=300)

    # Plot 7: Heatmap of F1 values
    plt.figure(figsize=(12, 10))

    # Create a pivot table
    heatmap_data_f1 = results_df.pivot_table(
        values='test_f1',
        index='classifier',
        columns='imputation'
    )

    # Plot heatmap
    ax = sns.heatmap(
        heatmap_data_f1,
        annot=True,
        cmap='RdYlGn',
        vmin=0,
        vmax=1.0,
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'F1 Score'}
    )

    plt.title(f'Test F1 Score Heatmap (Missing: {missing_percentage})', fontsize=16)
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Classifier', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/f1_heatmap_{missing_percentage}.png", dpi=300)

    print(f"Plots saved in the 'plots' directory")


def main():
    """
    Main function to run the pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the imputation and classification pipeline.')
    parser.add_argument('--missing', type=float, default=0.25,
                        help='Percentage of missing values (default: 0.25)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run pipeline with all missing percentages (0.25, 0.5)')
    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print(f"Starting Imputation and Classification Pipeline")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if args.run_all:
        # Run pipeline with all missing percentages
        missing_percentages = [0.25, 0.5]
        all_results = {}

        for missing_pct in missing_percentages:
            print(f"\n\n{'#' * 80}")
            print(f"# Running pipeline with missing percentage: {missing_pct}")
            print(f"{'#' * 80}")

            results = run_pipeline(missing_pct, args.random_state)
            all_results[missing_pct] = results

            # Create summary plots
            create_summary_plots(results, missing_pct)

            # Compare imputation methods
            imp_comparison = compare_imputation_methods(results)
            print("\nImputation Method Comparison (AUC):")
            print(imp_comparison)

            # Compare classifiers
            clf_comparison = compare_classifiers(results)
            print("\nClassifier Comparison (AUC):")
            print(clf_comparison)

        # Perform cross-missing-rate analysis
        print("\n\nCross-Missing-Rate Analysis:")
        for metric in ['auc', 'F1']:
            print(f"\n{metric.upper()} comparison across missing percentages:")

            for missing_pct, results in all_results.items():
                print(f"\nMissing percentage: {missing_pct}")

                # Imputation method comparison
                imp_comparison = compare_imputation_methods(results, metric)
                print(f"Top imputation methods by test {metric}:")
                print(imp_comparison.sort_values(f'avg_test_{metric}', ascending=False)[['imputation', f'avg_test_{metric}']])

                # Classifier comparison
                clf_comparison = compare_classifiers(results, metric)
                print(f"Top classifiers by test {metric}:")
                print(clf_comparison.sort_values(f'avg_test_{metric}', ascending=False)[['classifier', f'avg_test_{metric}']])

        # Create a combined analysis across all missing percentages
        print("\n\nCombined Analysis Across All Missing Percentages:")
        # Flatten results from all missing percentages
        all_flat_results = [r for results_list in all_results.values() for r in results_list]

        # Create a summary dataframe
        summary_df = pd.DataFrame([
            {
                'missing_percentage': r['missing_percentage'],
                'imputation': r['imputation'],
                'classifier': r['classifier'],
                'test_auc': r['test']['auc'],
                'test_f1': r['test']['F1'],
                'imputation_time': r['imputation_time']
            }
            for r in all_flat_results
        ])

        # Print overall best combinations
        print("\nTop 10 Imputation-Classifier Combinations (by AUC):")
        top_combos = summary_df.sort_values('test_auc', ascending=False).head(10)
        print(top_combos[['missing_percentage', 'imputation', 'classifier', 'test_auc', 'test_f1']])

        # Save the combined results to CSV
        summary_df.to_csv("results/combined_results_summary.csv", index=False)
        print("\nCombined results saved to 'results/combined_results_summary.csv'")

    else:
        # Run pipeline with specified missing percentage
        results = run_pipeline(args.missing, args.random_state)

        # Create summary plots
        create_summary_plots(results, args.missing)

        # Compare imputation methods
        imp_comparison = compare_imputation_methods(results)
        print("\nImputation Method Comparison (AUC):")
        print(imp_comparison)

        # Compare classifiers
        clf_comparison = compare_classifiers(results)
        print("\nClassifier Comparison (AUC):")
        print(clf_comparison)

        # Additional analysis - F1 score
        imp_comparison_f1 = compare_imputation_methods(results, 'F1')
        print("\nImputation Method Comparison (F1):")
        print(imp_comparison_f1)

        clf_comparison_f1 = compare_classifiers(results, 'F1')
        print("\nClassifier Comparison (F1):")
        print(clf_comparison_f1)

        # Save the result summary to CSV
        results_df = pd.DataFrame([
            {
                'imputation': r['imputation'],
                'classifier': r['classifier'],
                'test_auc': r['test']['auc'],
                'test_f1': r['test']['F1'],
                'imputation_time': r['imputation_time']
            }
            for r in results
        ])

        results_df.to_csv(f"results/summary_{args.missing}.csv", index=False)
        print(f"\nResults summary saved to 'results/summary_{args.missing}.csv'")

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()