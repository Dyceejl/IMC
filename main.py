import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import utilities
from utils.data_loader import load_mimic_dataset
from utils.evaluation import evaluate_classifier, plot_evaluation_results
from utils.visualization import plot_missing_values, compare_imputation_methods

# Import imputation methods
from imputation.mean import MeanImputer
from imputation.mice import MICEImputer
from imputation.forest import MissForestImputer
from imputation.miwae import MIWAE
from imputation.gain import GAIN

# Import classifiers
from classification.logistic import LogisticRegressionClassifier
from classification.random_forest import RandomForestClassifier
from classification.neural_net import NeuralNetClassifier


def main(args):
    """
    Main function to run the imputation and classification pipeline.

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Load data with missing values
    print(f"Loading MIMIC dataset with {args.missing_percentage * 100}% missing values...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_mimic_dataset(
            missing_percentage=args.missing_percentage,
            random_state=args.random_state
        )
        print(
            f"Data loaded successfully: {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check that the data directory structure is correct.")
        return

    # Select imputation method
    print(f"Using imputation method: {args.imputation}")
    if args.imputation == 'Mean':
        imputer = MeanImputer(random_state=args.random_state)
    elif args.imputation == 'MICE':
        imputer = MICEImputer(max_iter=10, random_state=args.random_state)
    elif args.imputation == 'MissForest':
        imputer = MissForestImputer(
            max_iter=10,
            n_estimators=100,
            random_state=args.random_state
        )
    elif args.imputation == 'MIWAE':
        imputer = MIWAE(
            n_epochs=100,
            batch_size=64,
            random_state=args.random_state
        )
    elif args.imputation == 'GAIN':
        imputer = GAIN(
            n_epochs=200,
            batch_size=128,
            random_state=args.random_state
        )
    else:
        raise ValueError(f"Imputation method '{args.imputation}' not recognized.")

    # Impute missing values
    print("Imputing missing values in training data...")
    start_time = time.time()
    X_train_imputed = imputer.fit_transform(X_train)
    train_time = time.time() - start_time
    print(f"Imputation completed in {train_time:.2f} seconds.")

    print("Imputing missing values in validation and test data...")
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Select classifier
    print(f"Using classifier: {args.classifier}")
    if args.classifier == 'LogisticRegression':
        classifier = LogisticRegressionClassifier(
            max_iter=args.max_iter,
            random_state=args.random_state
        )
    elif args.classifier == 'RandomForest':
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )

    elif args.classifier == 'NeuralNetwork':
        hidden_layer_sizes = tuple([args.hidden_neurons] * args.hidden_layers)
        classifier = NeuralNetClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=args.learning_rate,
            max_iter=args.max_iter,
            random_state=args.random_state
        )
    else:
        raise ValueError(f"Classifier '{args.classifier}' not recognized.")

    # Train classifier
    print("Training classifier...")
    start_time = time.time()
    classifier.fit(X_train_imputed, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    # Evaluate classifier
    print("Evaluating classifier on validation data...")
    val_probs = classifier.predict_proba(X_val_imputed)
    val_metrics = evaluate_classifier(y_val, val_probs)

    print("Evaluating classifier on test data...")
    test_probs = classifier.predict_proba(X_test_imputed)
    test_metrics = evaluate_classifier(y_test, test_probs)

    # Plot ROC curves and confusion matrices
    if args.plot:
        print("Plotting evaluation results...")
        fig_val = plot_evaluation_results(y_val, val_probs)
        fig_val.savefig(f'results/{args.imputation}_{args.classifier}_val.png')
        plt.close(fig_val)

        fig_test = plot_evaluation_results(y_test, test_probs)
        fig_test.savefig(f'results/{args.imputation}_{args.classifier}_test.png')
        plt.close(fig_test)

    # Save results
    results = {
        'imputation': args.imputation,
        'classifier': args.classifier,
        'missing_percentage': args.missing_percentage,
        'random_state': args.random_state,
        'imputation_time': train_time,
        'validation': val_metrics,
        'test': test_metrics,
        'parameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'hidden_layers': args.hidden_layers,
            'hidden_neurons': args.hidden_neurons,
            'max_iter': args.max_iter
        }
    }

    results_file = f'results/{args.imputation}_{args.classifier}_{args.missing_percentage}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

    # Print summary
    print("\nValidation Results:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run imputation and classification experiments on MIMIC data")

    # Imputation parameters
    parser.add_argument('--imputation', type=str, default='Mean',
                        choices=['Mean', 'MICE', 'MissForest', 'MIWAE', 'GAIN'],
                        help='Imputation method to use')
    parser.add_argument('--missing_percentage', type=float, default=0.25,
                        help='Percentage of missing values to introduce')

    # Classifier parameters
    parser.add_argument('--classifier', type=str, default='RandomForest',
                        choices=['LogisticRegression', 'RandomForest', 'XGBoost', 'NeuralNetwork'],
                        help='Classifier to use')

    # Model-specific parameters
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimators (for RandomForest and boosting methods)')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth of trees')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (for boosting and neural network)')
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='Number of hidden layers (for neural network)')
    parser.add_argument('--hidden_neurons', type=int, default=100,
                        help='Number of neurons per hidden layer (for neural network)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='Maximum number of iterations (for Logistic Regression and neural network)')

    # Other parameters
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots of results')

    args = parser.parse_args()
    main(args)