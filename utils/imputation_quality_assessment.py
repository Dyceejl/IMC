import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def compute_sample_wise_discrepancy(X_true, X_imputed, mask):
    """
    Compute Class A (sample-wise) discrepancy metrics.

    Parameters:
    -----------
    X_true : array-like
        Original complete data
    X_imputed : array-like
        Imputed data
    mask : array-like
        Boolean mask indicating missing values (True = missing)

    Returns:
    --------
    dict
        Dictionary with RMSE, MAE, and R² metrics
    """
    # Only compare the values that were actually imputed
    X_true_missing = X_true[mask]
    X_imputed_missing = X_imputed[mask]

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(X_true_missing, X_imputed_missing)),
        'MAE': mean_absolute_error(X_true_missing, X_imputed_missing),
        'R2': r2_score(X_true_missing, X_imputed_missing)
    }

    return metrics


def compute_feature_wise_discrepancy(X_true, X_imputed, mask=None):
    """
    Compute Class B (feature-wise) discrepancy metrics for each feature.

    Parameters:
    -----------
    X_true : array-like
        Original complete data
    X_imputed : array-like
        Imputed data
    mask : array-like, optional
        Boolean mask indicating missing values

    Returns:
    --------
    dict
        Dictionary with KL, KS, and Wasserstein metrics for min, median, max over features
    """
    n_features = X_true.shape[1]
    metrics = {
        'KL': np.zeros(n_features),
        'KS': np.zeros(n_features),
        'Wasserstein': np.zeros(n_features)
    }

    for j in range(n_features):
        # Get the true and imputed values for this feature
        feature_true = X_true[:, j]
        feature_imputed = X_imputed[:, j]

        # If mask is provided, only look at samples where the feature was missing
        if mask is not None:
            feature_mask = mask[:, j]
            if np.sum(feature_mask) > 0:  # Only if we have missing values for this feature
                feature_true = feature_true[feature_mask]
                feature_imputed = feature_imputed[feature_mask]

        # Compute histograms for KL divergence (need same bins)
        hist_range = (min(min(feature_true), min(feature_imputed)),
                      max(max(feature_true), max(feature_imputed)))
        bins = 20  # Number of bins for histogram

        # Add small epsilon to avoid division by zero in KL divergence
        hist_true, bin_edges = np.histogram(feature_true, bins=bins, range=hist_range, density=True)
        hist_imputed, _ = np.histogram(feature_imputed, bins=bins, range=hist_range, density=True)

        # Add small epsilon to avoid division by zero in KL divergence
        hist_true = hist_true + 1e-10
        hist_imputed = hist_imputed + 1e-10

        # Normalize to get proper probability distributions
        hist_true = hist_true / np.sum(hist_true)
        hist_imputed = hist_imputed / np.sum(hist_imputed)

        # Calculate KL divergence
        metrics['KL'][j] = entropy(hist_true, hist_imputed)

        # Calculate Kolmogorov-Smirnov statistic
        ks_stat, _ = ks_2samp(feature_true, feature_imputed)
        metrics['KS'][j] = ks_stat

        # Calculate Wasserstein distance (also called Earth Mover's Distance)
        # For 1D distributions, this is just the L1 distance between the CDFs
        from scipy.stats import wasserstein_distance
        metrics['Wasserstein'][j] = wasserstein_distance(feature_true, feature_imputed)

    # Compute summary statistics
    result = {}
    for metric_name, values in metrics.items():
        result[f'{metric_name}_min'] = np.min(values)
        result[f'{metric_name}_median'] = np.median(values)
        result[f'{metric_name}_max'] = np.max(values)

    return result


def compute_sliced_wasserstein(X_true, X_imputed, n_directions=50, n_partitions=10, random_state=42):
    """
    Compute Class C (sliced Wasserstein) discrepancy metrics.

    Parameters:
    -----------
    X_true : array-like
        Original complete data
    X_imputed : array-like
        Imputed data
    n_directions : int
        Number of random directions to project the data onto
    n_partitions : int
        Number of random partitions of the data
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with KL, KS, and Wasserstein metrics for the sliced distributions
    """
    np.random.seed(random_state)
    n_samples, n_features = X_true.shape

    # Normalize data (for consistent distance calculations)
    scaler = StandardScaler()
    X_true_norm = scaler.fit_transform(X_true)
    X_imputed_norm = scaler.transform(X_imputed)

    # Generate random unit vectors (directions)
    directions = []
    for _ in range(n_directions):
        direction = np.random.randn(n_features)
        direction = direction / np.linalg.norm(direction)
        directions.append(direction)

    # Generate random partitions of the data
    partitions = []
    for _ in range(n_partitions):
        # Randomly shuffle indices
        indices = np.random.permutation(n_samples)
        # Split into two approximately equal parts
        mid = n_samples // 2
        partition_A = indices[:mid]
        partition_B = indices[mid:]
        partitions.append((partition_A, partition_B))

    # Compute projected distances
    baseline_distances = []
    imputed_distances = []

    for direction in directions:
        for partition_A, partition_B in partitions:
            # Project original data
            projected_true_A = X_true_norm[partition_A].dot(direction)
            projected_true_B = X_true_norm[partition_B].dot(direction)

            # Project imputed data
            projected_imputed_B = X_imputed_norm[partition_B].dot(direction)

            # Normalize by standard deviation of partition A
            std_A = np.std(projected_true_A) + 1e-10  # Add small epsilon to avoid division by zero

            # Compute baseline Wasserstein distance (true A vs true B)
            from scipy.stats import wasserstein_distance
            baseline = wasserstein_distance(projected_true_A / std_A, projected_true_B / std_A)
            baseline_distances.append(baseline)

            # Compute imputed Wasserstein distance (true A vs imputed B)
            imputed = wasserstein_distance(projected_true_A / std_A, projected_imputed_B / std_A)
            imputed_distances.append(imputed)

    # Convert to numpy arrays
    baseline_distances = np.array(baseline_distances)
    imputed_distances = np.array(imputed_distances)

    # Compute distance ratios
    distance_ratios = imputed_distances / (baseline_distances + 1e-10)

    # Compute histograms for KL divergence
    hist_range = (min(min(baseline_distances), min(imputed_distances)),
                  max(max(baseline_distances), max(imputed_distances)))
    bins = 30

    hist_baseline, bin_edges = np.histogram(baseline_distances, bins=bins, range=hist_range, density=True)
    hist_imputed, _ = np.histogram(imputed_distances, bins=bins, range=hist_range, density=True)

    # Add small epsilon to avoid division by zero
    hist_baseline = hist_baseline + 1e-10
    hist_imputed = hist_imputed + 1e-10

    # Normalize
    hist_baseline = hist_baseline / np.sum(hist_baseline)
    hist_imputed = hist_imputed / np.sum(hist_imputed)

    # Calculate metrics for the sliced distributions
    kl_divergence = entropy(hist_baseline, hist_imputed)

    ks_stat, _ = ks_2samp(baseline_distances, imputed_distances)

    wass_dist = wasserstein_distance(baseline_distances, imputed_distances)

    result = {
        'Sliced_KL': kl_divergence,
        'Sliced_KS': ks_stat,
        'Sliced_Wasserstein': wass_dist,
        'Distance_ratio_mean': np.mean(distance_ratios),
        'Distance_ratio_median': np.median(distance_ratios),
        'Distance_ratio_std': np.std(distance_ratios)
    }

    return result, baseline_distances, imputed_distances, distance_ratios