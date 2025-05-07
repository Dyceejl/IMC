import numpy as np
import pandas as pd
import xgboost as xgb
from .base import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    """
    XGBoost classifier.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 subsample=1.0, random_state=None):
        """
        Initialize the XGBoost classifier.

        Parameters:
        -----------
        n_estimators : int, optional
            Number of boosting rounds
        max_depth : int, optional
            Maximum depth of the trees
        learning_rate : float, optional
            Learning rate
        subsample : float, optional
            Subsample ratio of the training instances
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Fit the classifier on the data.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Training data
        y : pandas Series or numpy array
            Target values

        Returns:
        --------
        self : returns self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data to predict

        Returns:
        --------
        y_pred : numpy array
            Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data to predict

        Returns:
        --------
        y_proba : numpy array
            Predicted class probabilities
        """
        return self.model.predict_proba(X)[:, 1]