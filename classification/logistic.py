import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression classifier.
    """

    def __init__(self, max_iter=100, random_state=None):
        """
        Initialize the Logistic Regression classifier.

        Parameters:
        -----------
        max_iter : int, optional
            Maximum number of iterations
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.max_iter = max_iter
        self.model = LogisticRegression(
            penalty='none',
            max_iter=max_iter,
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