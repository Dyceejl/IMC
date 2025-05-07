import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from .base import BaseImputer


class MICEImputer(BaseImputer):
    """
    Multiple Imputation by Chained Equations (MICE).

    MICE works by modeling each feature with missing values as a function of other features,
    and uses these models to impute the missing values in an iterative fashion.
    """

    def __init__(self, max_iter=10, random_state=None):
        """
        Initialize the MICE imputer.

        Parameters:
        -----------
        max_iter : int, optional
            Maximum number of iterations
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.max_iter = max_iter
        # We use BayesianRidge as the estimator for MICE
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
            verbose=0
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

        # Fit the imputer
        self.imputer.fit(X)
        return self

    def transform(self, X):
        """
        Impute missing values using MICE.

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
        X_imputed = self.imputer.transform(X)

        # Convert back to DataFrame if input was a DataFrame
        if is_df:
            X_imputed = pd.DataFrame(X_imputed, columns=cols, index=idx)

        return X_imputed