import numpy as np
import pandas as pd
from missingpy import MissForest
from .base import BaseImputer


class MissForestImputer(BaseImputer):
    """
    Impute missing values using the MissForest algorithm.

    MissForest is a non-parametric imputation method based on random forests.
    It can handle mixed-type data (both continuous and categorical).
    """

    def __init__(self, max_iter=10, n_estimators=100, random_state=None):
        """
        Initialize the MissForest imputer.

        Parameters:
        -----------
        max_iter : int, optional
            Maximum number of iterations
        n_estimators : int, optional
            Number of trees in the forest
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.imputer = MissForest(
            max_iter=max_iter,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_dataframe = False
        self.columns = None
        self.index = None

    def fit(self, X):
        """
        Fit the imputer on the data.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        self : returns self
        """
        # Check if input is a DataFrame
        self.is_dataframe = isinstance(X, pd.DataFrame)
        if self.is_dataframe:
            self.columns = X.columns
            self.index = X.index

        # MissForest doesn't have a separate fit method in missingpy
        # The fitting is done in transform
        return self

    def transform(self, X):
        """
        Impute missing values using MissForest.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        X_imputed : pandas DataFrame or numpy array
            Data with imputed values
        """
        # Check if input is a DataFrame
        is_df = isinstance(X, pd.DataFrame)

        # If input is a DataFrame, store column names and index
        if is_df:
            cols = X.columns
            idx = X.index

        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Convert back to DataFrame if input was a DataFrame
        if is_df:
            X_imputed = pd.DataFrame(X_imputed, columns=cols, index=idx)

        return X_imputed