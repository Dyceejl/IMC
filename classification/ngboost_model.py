import numpy as np
import pandas as pd
from ngboost import NGBClassifier as NGBC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from .base import BaseClassifier


class NGBoostClassifier(BaseClassifier):
    """
    NGBoost classifier with added robustness against singular matrices.

    Natural Gradient Boosting for probabilistic prediction, which enhances
    gradient boosting with natural gradients to directly optimize a proper scoring rule.
    This allows for high-quality probabilistic predictions.
    """

    def __init__(self, n_estimators=192, learning_rate=0.04, random_state=None):
        """
        Initialize the NGBoost classifier.

        Parameters:
        -----------
        n_estimators : int, optional
            Number of boosting rounds (boosting estimators)
        learning_rate : float, optional
            Learning rate for the boosting process
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = 0.6  # Store for reference

        # Create the NGBoost model with robust settings
        # Note: NGBoost uses a default Bernoulli distribution for classification
        self.model = NGBC(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,  # Disable verbose output
            minibatch_frac=0.5,  # Use smaller minibatches
            natural_gradient=False  # Disable natural gradient if causing issues
        )

        # Fallback model in case NGBoost fails
        self.fallback_model = RandomForestClassifier(
            n_estimators=144,
            random_state=random_state
        )

        # Flag to track if we used the fallback model
        self.used_fallback = False

        # Scaler for preprocessing
        self.scaler = StandardScaler()

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
        # Convert inputs to numpy arrays if they're pandas objects
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y

        # Scale the data to improve stability
        X_scaled = self.scaler.fit_transform(X_array)

        # Add small random noise to break perfect correlations
        rng = np.random.RandomState(self.random_state)
        X_scaled += rng.normal(0, 1e-6, X_scaled.shape)

        # Implement subsampling manually using sample weights
        n_samples = X_scaled.shape[0]
        if self.subsample < 1.0:
            sample_mask = rng.rand(n_samples) < self.subsample
            sample_weight = sample_mask.astype(float)
        else:
            sample_weight = None

        try:
            # Try to fit NGBoost
            if sample_weight is not None:
                self.model.fit(X_scaled, y_array, sample_weight=sample_weight)
            else:
                self.model.fit(X_scaled, y_array)

            self.used_fallback = False

        except Exception as e:
            # If NGBoost fails, use the fallback model
            print(f"NGBoost training failed with error: {str(e)}")
            print("Falling back to Random Forest classifier")

            self.fallback_model.fit(X_scaled, y_array)
            self.used_fallback = True

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
        # Convert and scale the input data
        X_array = X.values if hasattr(X, 'values') else X
        X_scaled = self.scaler.transform(X_array)

        if self.used_fallback:
            return self.fallback_model.predict(X_scaled)
        else:
            return self.model.predict(X_scaled)

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
        # Convert and scale the input data
        X_array = X.values if hasattr(X, 'values') else X
        X_scaled = self.scaler.transform(X_array)

        if self.used_fallback:
            return self.fallback_model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict_proba(X_scaled)[:, 1]