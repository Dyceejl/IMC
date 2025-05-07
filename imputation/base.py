class BaseImputer:
    """
    Base class for all imputation methods.
    All imputation methods should inherit from this class and implement
    fit and transform methods.
    """

    def __init__(self, random_state=None):
        """
        Initialize the imputer.

        Parameters:
        -----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state

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
        raise NotImplementedError("Subclasses must implement this method.")

    def transform(self, X):
        """
        Impute missing values in the data.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        X_imputed : pandas DataFrame or numpy array
            Data with imputed values
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def fit_transform(self, X):
        """
        Fit the imputer and impute missing values.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Data with missing values

        Returns:
        --------
        X_imputed : pandas DataFrame or numpy array
            Data with imputed values
        """
        self.fit(X)
        return self.transform(X)