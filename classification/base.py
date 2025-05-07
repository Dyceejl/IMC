class BaseClassifier:
    """
    Base class for all classifiers.
    All classifier methods should inherit from this class and implement
    fit, predict, and predict_proba methods.
    """

    def __init__(self, random_state=None):
        """
        Initialize the classifier.

        Parameters:
        -----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state

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
        raise NotImplementedError("Subclasses must implement this method.")

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
        raise NotImplementedError("Subclasses must implement this method.")

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
        raise NotImplementedError("Subclasses must implement this method.")