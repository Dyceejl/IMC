def perform_anova_analysis(performance_df, dataset_name=None):
    """
    Perform multi-factor ANOVA analysis on classifier performance.

    Parameters:
    -----------
    performance_df : pandas DataFrame
        DataFrame with classifier performance metrics
    dataset_name : str, optional
        If provided, filter results to this dataset

    Returns:
    --------
    dict
        Dictionary with ANOVA results
    """
    # Filter by dataset if specified
    if dataset_name:
        df = performance_df[performance_df['dataset'] == dataset_name].copy()
    else:
        df = performance_df.copy()

    # Convert categorical variables to factors
    for col in ['classifier', 'imputation_method', 'train_miss_rate', 'test_miss_rate', 'holdout_set']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Build formula based on available columns
    formula_terms = []

    # Always include imputation_method and classifier
    formula_terms.append('imputation_method')
    formula_terms.append('classifier')

    # Add missingness rates if available
    if 'train_miss_rate' in df.columns:
        formula_terms.append('train_miss_rate')

    if 'test_miss_rate' in df.columns:
        formula_terms.append('test_miss_rate')

    # Add interactions if we have both missingness rates
    if 'train_miss_rate' in df.columns and 'test_miss_rate' in df.columns:
        formula_terms.append('train_miss_rate:test_miss_rate')
        formula_terms.append('imputation_method:train_miss_rate')
        formula_terms.append('imputation_method:test_miss_rate')
        formula_terms.append('classifier:train_miss_rate')
        formula_terms.append('classifier:test_miss_rate')

    # Build formula string
    formula = 'auc ~ ' + ' + '.join(formula_terms)

    # Fit the model
    try:
        # Try to fit using OLS for simplicity
        model = sm.formula.ols(formula, data=df).fit()
    except:
        # If that fails, try a more robust GLM with binomial family
        from statsmodels.genmod.families import Binomial
        model = sm.formula.glm(formula, data=df, family=Binomial()).fit()

    # Compute ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract deviance explained by each factor
    total_deviance = np.sum(anova_table['sum_sq'])
    anova_table['deviance_explained'] = anova_table['sum_sq'] / total_deviance

    # Create a dictionary with results
    anova_results = {
        'model': model,
        'anova_table': anova_table,
        'formula': formula
    }

    return anova_results