import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from .base import BaseClassifier


class NeuralNetClassifier(BaseClassifier):
    """
    Neural Network classifier based on scikit-learn's MLPClassifier.
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001,
                 max_iter=200, random_state=None):
        """
        Initialize the Neural Network classifier.

        Parameters:
        -----------
        hidden_layer_sizes : tuple, optional
            Number of neurons in each hidden layer
        learning_rate_init : float, optional
            Initial learning rate
        max_iter : int, optional
            Maximum number of iterations
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
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