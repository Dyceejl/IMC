from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_classifier_dependence(performance_df, metric='auc', save_path=None):
    """
    Plot Figure 3a: Dependence of downstream performance on classification method.

    Parameters:
    -----------
    performance_df : pandas DataFrame
        DataFrame with classifier performance metrics
    metric : str
        Metric to plot
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Group by relevant factors and compute mean and std
    grouped = performance_df.groupby(['dataset', 'train_miss_rate', 'test_miss_rate', 'classifier'])
    summary = grouped[metric].agg(['mean', 'std']).reset_index()

    # Format dataset names with missingness rates
    summary['dataset_label'] = summary.apply(
        lambda x: f"{x['dataset']} [{x['train_miss_rate']} / {x['test_miss_rate']}]", axis=1
    )

    # Create figure
    plt.figure(figsize=(15, 10))

    # Define color palette for classifiers
    palette = sns.color_palette("Set2", n_colors=len(summary['classifier'].unique()))

    # Create the plot
    ax = sns.scatterplot(
        data=summary,
        x='mean',
        y='dataset_label',
        hue='classifier',
        size='std',
        sizes=(50, 200),
        palette=palette,
        alpha=0.8
    )

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add titles and labels
    plt.title(f'Dependence of {metric.upper()} on Classification Method', fontsize=14)
    plt.xlabel(f'{metric.upper()} Value', fontsize=12)
    plt.ylabel('Dataset [train / test missingness]', fontsize=12)

    # Add a legend
    plt.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add dashed lines to separate datasets
    datasets = summary['dataset'].unique()
    for i in range(1, len(datasets)):
        idx = summary[summary['dataset'] == datasets[i]].index[0]
        plt.axhline(y=idx - 0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_imputation_dependence(performance_df, metric='auc', save_path=None):
    """
    Plot Figure 3b: Dependence of downstream performance on imputation method.

    Parameters:
    -----------
    performance_df : pandas DataFrame
        DataFrame with classifier performance metrics
    metric : str
        Metric to plot
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Group by relevant factors and compute mean and std
    grouped = performance_df.groupby(['dataset', 'train_miss_rate', 'test_miss_rate', 'imputation_method'])
    summary = grouped[metric].agg(['mean', 'std']).reset_index()

    # Format dataset names with missingness rates
    summary['dataset_label'] = summary.apply(
        lambda x: f"{x['dataset']} [{x['train_miss_rate']} / {x['test_miss_rate']}]", axis=1
    )

    # Create figure
    plt.figure(figsize=(15, 10))

    # Define color palette for imputation methods
    palette = sns.color_palette("Set1", n_colors=len(summary['imputation_method'].unique()))

    # Create the plot
    ax = sns.scatterplot(
        data=summary,
        x='mean',
        y='dataset_label',
        hue='imputation_method',
        size='std',
        sizes=(50, 200),
        palette=palette,
        alpha=0.8
    )

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add titles and labels
    plt.title(f'Dependence of {metric.upper()} on Imputation Method', fontsize=14)
    plt.xlabel(f'{metric.upper()} Value', fontsize=12)
    plt.ylabel('Dataset [train / test missingness]', fontsize=12)

    # Add a legend
    plt.legend(title='Imputation Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add dashed lines to separate datasets
    datasets = summary['dataset'].unique()
    for i in range(1, len(datasets)):
        idx = summary[summary['dataset'] == datasets[i]].index[0]
        plt.axhline(y=idx - 0.5, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_anova_results(anova_results, dataset_name=None, save_path=None):
    """
    Plot Figure 4: ANOVA analysis of factors affecting classifier performance.

    Parameters:
    -----------
    anova_results : dict
        Dictionary with ANOVA results
    dataset_name : str, optional
        Name of the dataset
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Extract ANOVA table
    anova_table = anova_results['anova_table'].copy()

    # Sort by deviance explained (descending)
    anova_table = anova_table.sort_values('deviance_explained', ascending=False)

    # Remove the Residual row if present
    if 'Residual' in anova_table.index:
        anova_table = anova_table.drop('Residual')

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create the bar plot
    bars = plt.bar(
        range(len(anova_table)),
        anova_table['deviance_explained'],
        color=sns.color_palette("tab10", n_colors=len(anova_table))
    )

    # Add text labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center', va='bottom',
            fontsize=10
        )

    # Set x-axis labels with factor names
    plt.xticks(range(len(anova_table)), anova_table.index, rotation=45, ha='right')

    # Add titles and labels
    title = f'Factors Influencing Classifier Performance - {dataset_name}' if dataset_name else 'Pooled ANOVA Analysis'
    plt.title(title, fontsize=14)
    plt.xlabel('Factors', fontsize=12)
    plt.ylabel('Deviance Explained', fontsize=12)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_sample_wise_metrics(imputation_quality_df, dataset_name, train_miss, test_miss, save_path=None):
    """
    Plot Figure 5 and Supplementary Figures 10-12: Sample-wise imputation quality metrics.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    dataset_name : str
        Name of the dataset
    train_miss : float
        Training set missingness rate
    test_miss : float
        Test set missingness rate
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data
    df = imputation_quality_df[
        (imputation_quality_df['dataset'] == dataset_name) &
        (imputation_quality_df['train_miss_rate'] == train_miss) &
        (imputation_quality_df['test_miss_rate'] == test_miss)
        ].copy()

    # Create figure with 3 subplots (RMSE, MAE, R²)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define metrics to plot
    metrics = ['RMSE', 'MAE', 'R2']

    for i, metric in enumerate(metrics):
        # Create boxplot
        ax = sns.boxplot(
            x='imputation_method',
            y=metric,
            hue='holdout_set',
            data=df,
            ax=axes[i],
            palette='Set3'
        )

        # Set title and labels
        axes[i].set_title(f'{metric} by Imputation Method', fontsize=12)
        axes[i].set_xlabel('Imputation Method', fontsize=10)
        axes[i].set_ylabel(metric, fontsize=10)

        # Rotate x-axis labels
        axes[i].set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add a grid
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a title for the entire figure
    plt.suptitle(
        f'Sample-wise Imputation Quality Metrics - {dataset_name} (Train: {train_miss}, Test: {test_miss})',
        fontsize=14
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_wise_metrics(imputation_quality_df, dataset_name, train_miss, test_miss, save_path=None):
    """
    Plot Figure 6 and Supplementary Figures 13-23: Feature-wise imputation quality metrics.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    dataset_name : str
        Name of the dataset
    train_miss : float
        Training set missingness rate
    test_miss : float
        Test set missingness rate
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data
    df = imputation_quality_df[
        (imputation_quality_df['dataset'] == dataset_name) &
        (imputation_quality_df['train_miss_rate'] == train_miss) &
        (imputation_quality_df['test_miss_rate'] == test_miss)
        ].copy()

    # Create figure with 3 rows (KL, KS, Wasserstein) and 3 columns (min, median, max)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Define metrics to plot
    metric_groups = ['KL', 'KS', 'Wasserstein']
    stat_types = ['min', 'median', 'max']

    for i, metric in enumerate(metric_groups):
        for j, stat in enumerate(stat_types):
            # Column name in the DataFrame
            col_name = f'{metric}_{stat}'

            # Create boxplot
            ax = sns.boxplot(
                x='imputation_method',
                y=col_name,
                hue='holdout_set',
                data=df,
                ax=axes[i, j],
                palette='Set3'
            )

            # Set title and labels
            axes[i, j].set_title(f'{metric} {stat.capitalize()}', fontsize=12)
            axes[i, j].set_xlabel('Imputation Method', fontsize=10)
            axes[i, j].set_ylabel(col_name, fontsize=10)

            # Rotate x-axis labels
            axes[i, j].set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add a grid
            axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a title for the entire figure
    plt.suptitle(
        f'Feature-wise Imputation Quality Metrics - {dataset_name} (Train: {train_miss}, Test: {test_miss})',
        fontsize=14
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sliced_wasserstein_metrics(imputation_quality_df, dataset_name, train_miss, test_miss, save_path=None):
    """
    Plot Figure 7 and Supplementary Figures 24-29: Sliced Wasserstein-derived imputation quality metrics.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    dataset_name : str
        Name of the dataset
    train_miss : float
        Training set missingness rate
    test_miss : float
        Test set missingness rate
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data
    df = imputation_quality_df[
        (imputation_quality_df['dataset'] == dataset_name) &
        (imputation_quality_df['train_miss_rate'] == train_miss) &
        (imputation_quality_df['test_miss_rate'] == test_miss)
        ].copy()

    # Create figure with 3 subplots (Sliced_KL, Sliced_KS, Sliced_Wasserstein)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define metrics to plot
    metrics = ['Sliced_KL', 'Sliced_KS', 'Sliced_Wasserstein']

    for i, metric in enumerate(metrics):
        # Create boxplot
        ax = sns.boxplot(
            x='imputation_method',
            y=metric,
            hue='holdout_set',
            data=df,
            ax=axes[i],
            palette='Set3'
        )

        # Set title and labels
        short_name = metric.split('_')[1]  # Extract KL, KS, Wasserstein
        axes[i].set_title(f'Sliced {short_name}', fontsize=12)
        axes[i].set_xlabel('Imputation Method', fontsize=10)
        axes[i].set_ylabel(metric, fontsize=10)

        # Rotate x-axis labels
        axes[i].set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add a grid
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a title for the entire figure
    plt.suptitle(
        f'Sliced Wasserstein Metrics - {dataset_name} (Train: {train_miss}, Test: {test_miss})',
        fontsize=14
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_distance_ratio(imputation_quality_df, dataset_name, save_path=None):
    """
    Plot Supplementary Figures 32-34: Ratio of Wasserstein distances for imputed vs original data.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data
    df = imputation_quality_df[imputation_quality_df['dataset'] == dataset_name].copy()

    # Create a grid of subplots for different missingness combinations
    train_miss_rates = sorted(df['train_miss_rate'].unique())
    test_miss_rates = sorted(df['test_miss_rate'].unique())

    fig, axes = plt.subplots(len(train_miss_rates), len(test_miss_rates),
                             figsize=(6 * len(test_miss_rates), 5 * len(train_miss_rates)))

    # Flatten axes if it's a 1D array
    if len(train_miss_rates) == 1 or len(test_miss_rates) == 1:
        axes = np.array([axes]).reshape(len(train_miss_rates), len(test_miss_rates))

    for i, train_miss in enumerate(train_miss_rates):
        for j, test_miss in enumerate(test_miss_rates):
            # Filter for this missingness combination
            subset = df[
                (df['train_miss_rate'] == train_miss) &
                (df['test_miss_rate'] == test_miss)
                ].copy()

            # Create boxplot
            ax = sns.boxplot(
                x='imputation_method',
                y='Distance_ratio_mean',
                hue='holdout_set',
                data=subset,
                ax=axes[i, j],
                palette='Set3'
            )

            # Set title and labels
            axes[i, j].set_title(f'Train {train_miss}, Test {test_miss}', fontsize=12)
            axes[i, j].set_xlabel('Imputation Method', fontsize=10)
            axes[i, j].set_ylabel('Distance Ratio', fontsize=10)

            # Rotate x-axis labels
            axes[i, j].set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add a grid
            axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a title for the entire figure
    plt.suptitle(
        f'Wasserstein Distance Ratio - {dataset_name}',
        fontsize=14
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_quality_vs_performance(imputation_quality_df, performance_df, metric, quality_metric,
                                dataset_name, save_path=None):
    """
    Plot Figure 9 and Supplementary Figures 30-31: Relationship between imputation quality and classifier performance.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    performance_df : pandas DataFrame
        DataFrame with classifier performance
    metric : str
        Performance metric to plot (e.g., 'auc')
    quality_metric : str
        Imputation quality metric to plot (e.g., 'RMSE')
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data
    qual_df = imputation_quality_df[imputation_quality_df['dataset'] == dataset_name].copy()
    perf_df = performance_df[performance_df['dataset'] == dataset_name].copy()

    # Group both datasets by common factors
    common_cols = ['imputation_method', 'train_miss_rate', 'test_miss_rate', 'holdout_set']
    qual_grouped = qual_df.groupby(common_cols)[quality_metric].mean().reset_index()
    perf_grouped = perf_df.groupby(common_cols)[metric].mean().reset_index()

    # Merge the two datasets
    merged = pd.merge(qual_grouped, perf_grouped, on=common_cols)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create scatter plot with regression line for each test missingness rate
    for test_miss in sorted(merged['test_miss_rate'].unique()):
        subset = merged[merged['test_miss_rate'] == test_miss]

        # Add scatter plot
        ax = sns.scatterplot(
            data=subset,
            x=quality_metric,
            y=metric,
            hue='imputation_method',
            style='test_miss_rate',
            s=100,
            alpha=0.8,
            label=f'Test Miss: {test_miss}'
        )

        # Add regression line
        if len(subset) > 2:  # Need at least 3 points for regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(subset[quality_metric], subset[metric])

            # Sort for line plotting
            x_sorted = sorted(subset[quality_metric])
            y_pred = [slope * x + intercept for x in x_sorted]

            plt.plot(x_sorted, y_pred, '--',
                     label=f'Test {test_miss}: R = {r_value:.2f}',
                     alpha=0.7)

    # Add grid
    plt.grid(linestyle='--', alpha=0.7)

    # Add titles and labels
    plt.title(f'Relationship between {quality_metric} and {metric} - {dataset_name}', fontsize=14)
    plt.xlabel(quality_metric, fontsize=12)
    plt.ylabel(metric, fontsize=12)

    # Add legend
    plt.legend(title='Imputation Method & Test Miss Rate', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_correlation_heatmap(imputation_quality_df, dataset_name, save_path=None):
    """
    Plot Figure 10 and Supplementary Figure 35: Correlation heatmap between different discrepancy metrics.

    Parameters:
    -----------
    imputation_quality_df : pandas DataFrame
        DataFrame with imputation quality metrics
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Filter data for the specified dataset
    df = imputation_quality_df[imputation_quality_df['dataset'] == dataset_name].copy()

    # Select metrics of interest
    metrics = [
        'RMSE', 'MAE', 'R2',  # Class A
        'KL_median', 'KS_median', 'Wasserstein_median',  # Class B (using median as representative)
        'Sliced_KL', 'Sliced_KS', 'Sliced_Wasserstein'  # Class C
    ]

    # Compute correlation matrix
    corr_matrix = df[metrics].corr()

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

    ax = sns.heatmap(
        corr_matrix,
        annot=False,  # Too many annotations may clutter the plot
        mask=None,  # Show full matrix
        cmap='RdYlGn',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )

    # Set title and axis labels
    plt.title(f'Correlation Between Discrepancy Metrics - {dataset_name}', fontsize=14)

    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_feature_importance_skew(feature_importance_df, save_path=None):
    """
    Plot Supplementary Figure 4: Absolute skew of Shapley values for feature importance.

    Parameters:
    -----------
    feature_importance_df : pandas DataFrame
        DataFrame with feature importance values
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    matplotlib Figure
        Figure object
    """
    # Create figure with one subplot per classifier
    classifiers = feature_importance_df['classifier'].unique()
    fig, axes = plt.subplots(1, len(classifiers), figsize=(6 * len(classifiers), 5))

    if len(classifiers) == 1:
        axes = [axes]

    for i, clf in enumerate(classifiers):
        # Filter data for this classifier
        clf_data = feature_importance_df[feature_importance_df['classifier'] == clf].copy()

        # Plot line graph of absolute skew values by feature
        for imp_method in clf_data['imputation_method'].unique():
            method_data = clf_data[clf_data['imputation_method'] == imp_method]

            axes[i].plot(
                method_data['feature_number'],
                method_data['abs_skew'],
                marker='o',
                label=imp_method,
                alpha=0.8
            )

        # Add titles and labels
        axes[i].set_title(clf, fontsize=12)
        axes[i].set_xlabel('Feature Number', fontsize=10)
        axes[i].set_ylabel('Absolute Skew of Shapley Values', fontsize=10)

        # Add grid and legend
        axes[i].grid(linestyle='--', alpha=0.7)
        axes[i].legend(title='Imputation Method')

    # Add a title for the entire figure
    plt.suptitle('Feature Importance Skew by Classifier and Imputation Method', fontsize=14)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig