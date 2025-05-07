import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from .base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest classifier.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        """
        Initialize the Random Forest classifier.

        Parameters:
        -----------
        n_estimators : int, optional
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of the trees
        min_samples_split : int, optional
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.model = SklearnRFC(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            oob_score=True
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