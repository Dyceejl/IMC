# Create our own MissForest-inspired imputer that doesn't rely on missingpy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


class CustomMissForestImputer:
    """A custom implementation of missing value imputation using Random Forests."""

    def __init__(self, max_iter=10, n_estimators=100, random_state=None):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.initial_imputer = SimpleImputer(strategy='mean')

    def fit(self, X):
        """Fit the imputer on the data."""
        if isinstance(X, pd.DataFrame):
            self.is_dataframe = True
            self.columns = X.columns
            self.index = X.index
            X_np = X.values
        else:
            self.is_dataframe = False
            X_np = X.copy()

        self.X_fitted = self._impute_missing(X_np)
        return self

    def transform(self, X):
        """Impute all missing values in X."""
        if not hasattr(self, 'X_fitted'):
            raise ValueError("You must call fit before transform.")

        if isinstance(X, pd.DataFrame):
            is_df = True
            cols = X.columns
            idx = X.index
            X_np = X.values.copy()
        else:
            is_df = False
            X_np = X.copy()

        # Use the same approach as in fit
        X_imputed = self._impute_missing(X_np)

        # Convert back to DataFrame if the input was a DataFrame
        if is_df:
            X_imputed = pd.DataFrame(X_imputed, columns=cols, index=idx)

        return X_imputed

    def fit_transform(self, X):
        """Fit the imputer and impute missing values."""
        self.fit(X)
        return self.transform(X)

    def _impute_missing(self, X):
        """The actual imputation algorithm using Random Forests."""
        # Initial imputation with mean
        X_filled = self.initial_imputer.fit_transform(X)

        # Create a mask of missing values
        mask = np.isnan(X)

        # If there are no missing values, return the original data
        if not np.any(mask):
            return X

        # For each column with missing values
        n_features = X.shape[1]

        for _ in range(self.max_iter):
            prev_X = X_filled.copy()

            # For each column
            for j in range(n_features):
                # Get the mask for this column
                mask_j = mask[:, j]

                # If no missing values in this column, skip
                if not np.any(mask_j):
                    continue

                # Get the observed values for this column
                obs_indices = np.where(~mask_j)[0]
                miss_indices = np.where(mask_j)[0]

                # If all values are missing, skip
                if len(obs_indices) == 0:
                    continue

                # Create the target vector and feature matrix for training
                y_obs = X[obs_indices, j]
                X_obs = np.delete(X_filled[obs_indices], j, axis=1)

                # Train a random forest regressor
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state
                )
                rf.fit(X_obs, y_obs)

                # Create the feature matrix for prediction
                X_miss = np.delete(X_filled[miss_indices], j, axis=1)

                # Predict the missing values
                y_pred = rf.predict(X_miss)

                # Fill in the predicted values
                X_filled[miss_indices, j] = y_pred

            # Check for convergence
            change = np.mean((X_filled - prev_X) ** 2)
            if change < 1e-6:
                break

        return X_filled