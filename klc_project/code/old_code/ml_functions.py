# functions.py

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def to_pm1_labels(y: np.ndarray) -> np.ndarray:
    """
    Converts labels in {0, 1} to {-1, +1}. If the labels are already in {-1, +1},
    they remain unchanged.

    Args:
        y - A numpy array of shape (n_samples,) containing the input labels, which
            are expected to be in {0, 1} or {-1, +1}.

    Returns: A numpy array of shape (n_samples,) with labels in {-1, +1}.
    """
    unique_vals = np.unique(y)
    if set(unique_vals).issubset({0, 1}):
        return np.where(y == 0, -1, 1)
    elif set(unique_vals).issubset({-1, 1}):
        return y
    else:
        raise ValueError("Labels must be in {0,1} or {-1,+1}.")

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the 0-1 accuracy score.

    Args:
        y_true - A numpy array of shape (n_samples,) containing the true labels.
        y_pred - A numpy array of shape (n_samples,) containing the predicted labels.

    Returns: A float representing the fraction of samples where y_pred equals y_true.
    """
    return np.mean(y_true == y_pred)

def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the 0-1 loss, defined as the fraction of misclassified samples.

    Args:
        y_true - A numpy array of shape (n_samples,) with the true labels.
        y_pred - A numpy array of shape (n_samples,) with the predicted labels.

    Returns: A float representing the fraction of samples where y_pred does not equal y_true.
    """
    return np.mean(y_true != y_pred)

# ------------------------------------------------------------------------------
# Data Preprocessing Functions
# ------------------------------------------------------------------------------

def remove_outliers_from_file(file_path: str, threshold: float = 3.0, columns: list = None) -> None:
    """
    Removes outliers from a CSV dataset file by deleting rows with an absolute z-score
    greater than the specified threshold.

    Args:
        file_path - A string representing the path to the CSV file containing the dataset.
        threshold - A float representing the z-score threshold for identifying outliers (default is 3.0).
        columns - An optional list of column names to check for outliers. If None, all numeric columns are used.

    Returns: None. The function overwrites the original file with the cleaned dataset.
    """
    # Read the dataset from the CSV file.
    df = pd.read_csv(file_path)
    
    # If no specific columns are provided, use all numeric columns.
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute z-scores for the selected columns.
    z_scores = df[columns].apply(lambda x: (x - x.mean()) / x.std())
    
    # Keep only the rows where all selected columns have an absolute z-score below the threshold.
    df_clean = df[(z_scores.abs() < threshold).all(axis=1)]
    
    # Overwrite the original file with the cleaned dataset.
    df_clean.to_csv(file_path, index=False)
    
    print(f"Removed {len(df) - len(df_clean)} outliers from {file_path}")

def split_and_standardize_data(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = None
) -> tuple:
    """
    Splits the dataset into training and test sets and standardizes the features using
    the training set statistics. This prevents data leakage by ensuring that the test set
    is scaled based solely on information from the training set.

    Args:
        X - A numpy array of shape (n_samples, n_features) containing the input data.
        y - A numpy array of shape (n_samples,) containing the labels.
        test_ratio - A float representing the fraction of data to reserve as the test set (default is 0.2).
        random_state - An optional integer seed for reproducible splitting.

    Returns: A tuple (X_train, X_test, y_train, y_test) where:
        X_train - A numpy array of standardized training data features.
        X_test - A numpy array of standardized test data features.
        y_train - A numpy array of training labels.
        y_test - A numpy array of test labels.
    """
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Compute mean and standard deviation from the training set.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero.
    
    # Standardize using training set statistics.
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    
    return X_train_std, X_test_std, y_train, y_test

def impute_missing_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    strategy: str = 'median'
) -> tuple:
    """
    Imputes missing values in the training and test datasets based on the training data statistics.
    The chosen strategy is applied column-wise.

    Args:
        X_train - A numpy array of shape (n_samples_train, n_features) containing the training data.
        X_test - A numpy array of shape (n_samples_test, n_features) containing the test data.
        strategy - A string specifying the imputation strategy ('mean' or 'median', default is 'median').

    Returns: A tuple (X_train_imputed, X_test_imputed) with missing values filled.
    """
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    for i in range(X_train.shape[1]):
        if strategy == 'mean':
            fill_value = np.nanmean(X_train[:, i])
        elif strategy == 'median':
            fill_value = np.nanmedian(X_train[:, i])
        else:
            raise ValueError("Unsupported imputation strategy. Choose 'mean' or 'median'.")
        
        # Replace nan values in both training and test sets.
        X_train_imputed[np.isnan(X_train_imputed[:, i]), i] = fill_value
        X_test_imputed[np.isnan(X_test_imputed[:, i]), i] = fill_value
        
    return X_train_imputed, X_test_imputed

# ------------------------------------------------------------------------------
# Polynomial Feature Expansion
# ------------------------------------------------------------------------------

def polynomial_feature_expansion(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = True
) -> np.ndarray:
    """
    Produces polynomial feature expansions of the input data up to a specified degree.
    (Currently implemented only for degree=2.)

    Args:
        X - A numpy array of shape (n_samples, n_features) containing the input data.
        degree - An integer representing the maximum polynomial degree (default is 2).
        include_bias - A boolean indicating whether to include a bias (constant 1) term (default is True).

    Returns: A numpy array containing the transformed features. The shape is (n_samples, D),
    where D depends on the number of original features and the specified degree.
    """
    if degree != 2:
        raise NotImplementedError("Currently only degree=2 expansion is implemented.")

    n_samples, n_features = X.shape
    
    X_poly_list = []
    if include_bias:
        X_poly_list.append(np.ones(n_samples))
    
    # Linear terms.
    for f in range(n_features):
        X_poly_list.append(X[:, f])
    
    # Quadratic terms (pairwise products and squared terms).
    for i in range(n_features):
        for j in range(i, n_features):
            X_poly_list.append(X[:, i] * X[:, j])
    
    X_poly = np.vstack(X_poly_list).T
    return X_poly

# ------------------------------------------------------------------------------
# Linear Perceptron
# ------------------------------------------------------------------------------

class Perceptron:
    """
    A simple linear Perceptron classifier.
    """

    def __init__(self, max_iter: int = 10, shuffle: bool = True, random_state: int = None):
        """
        Initializes the Perceptron classifier.

        Args:
            max_iter - An integer representing the number of passes over the training dataset.
            shuffle - A boolean indicating whether to shuffle the data each epoch.
            random_state - An optional integer seed for reproducible shuffling.
        """
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Perceptron model on the training data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the training data.
            y - A numpy array of shape (n_samples,) with labels in {-1, +1} (convert using to_pm1_labels if needed).

        Returns: None
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for i in indices:
                if y[i] * np.dot(self.w, X[i]) <= 0:
                    self.w += y[i] * X[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input data using the Perceptron model.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the input data.

        Returns: A numpy array of shape (n_samples,) with predicted labels in {-1, +1}.
        """
        return np.sign(X @ self.w)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the 0-1 accuracy of the Perceptron model on the provided data.

        Args:
            X - A numpy array containing the test data.
            y - A numpy array containing the true labels in {-1, +1}.

        Returns: A float representing the accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# ------------------------------------------------------------------------------
# Pegasos SVM (Linear)
# ------------------------------------------------------------------------------

class PegasosSVM:
    """
    Implementation of the Pegasos algorithm for linear SVM with hinge loss.
    """

    def __init__(self, lambda_: float = 1e-4, max_iter: int = 1000, random_state: int = None):
        """
        Initializes the Pegasos SVM classifier.

        Args:
            lambda_ - A float representing the regularization parameter.
            max_iter - An integer representing the total number of stochastic updates.
            random_state - An optional integer seed for reproducible sampling.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Pegasos SVM on the given training data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the training data.
            y - A numpy array of shape (n_samples,) with labels in {-1, +1}.

        Returns: None
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)
        t = 0
        for iteration in range(1, self.max_iter + 1):
            i = rng.integers(n_samples)
            t += 1
            eta = 1.0 / (self.lambda_ * t)
            self.w = (1 - eta * self.lambda_) * self.w
            if y[i] * np.dot(self.w, X[i]) < 1:
                self.w += eta * y[i] * X[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input data using the trained Pegasos SVM.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the data.

        Returns: A numpy array of shape (n_samples,) with predicted labels in {-1, +1}.
        """
        return np.sign(X @ self.w)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the 0-1 accuracy of the Pegasos SVM on the provided data.

        Args:
            X - A numpy array containing the test data.
            y - A numpy array containing the true labels in {-1, +1}.

        Returns: A float representing the accuracy score.
        """
        return accuracy_score(y, self.predict(X))

# ------------------------------------------------------------------------------
# Pegasos Logistic (Linear)
# ------------------------------------------------------------------------------

class PegasosLogistic:
    """
    Implementation of a Pegasos-style algorithm for logistic loss.
    """

    def __init__(self, lambda_: float = 1e-4, max_iter: int = 1000, random_state: int = None):
        """
        Initializes the Pegasos logistic model.

        Args:
            lambda_ - A float representing the regularization parameter.
            max_iter - An integer representing the total number of stochastic updates.
            random_state - An optional integer seed for reproducible sampling.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Pegasos logistic model on the given data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the training data.
            y - A numpy array of shape (n_samples,) with labels in {-1, +1}.

        Returns: None
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)
        for t in range(1, self.max_iter + 1):
            i = rng.integers(n_samples)
            eta = 1.0 / (self.lambda_ * t)
            self.w = (1 - eta * self.lambda_) * self.w
            exponent = y[i] * np.dot(self.w, X[i])
            logistic_grad = - (y[i] * X[i]) / (1.0 + np.exp(exponent))
            self.w -= eta * logistic_grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input data based on the logistic model.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the input data.

        Returns: A numpy array of shape (n_samples,) with predicted labels in {-1, +1}.
        """
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of class +1 for the input data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the input data.

        Returns: A numpy array of shape (n_samples,) with the probability of class +1 for each sample.
        """
        scores = X @ self.w
        return 1.0 / (1.0 + np.exp(-scores))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the 0-1 accuracy of the logistic model on the given data.

        Args:
            X - A numpy array containing the test data.
            y - A numpy array containing the true labels in {-1, +1}.

        Returns: A float representing the accuracy score.
        """
        return accuracy_score(y, self.predict(X))

# ------------------------------------------------------------------------------
# Kernel Functions
# ------------------------------------------------------------------------------

def gaussian_kernel(X: np.ndarray, Z: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Computes the Gaussian (RBF) kernel matrix between two datasets.

    Args:
        X - A numpy array of shape (n_samples, n_features) for the first dataset.
        Z - A numpy array of shape (m_samples, n_features) for the second dataset.
        sigma - A float representing the bandwidth parameter for the Gaussian kernel.

    Returns: A numpy array of shape (n_samples, m_samples) representing the kernel matrix.
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
    dists = X_norm + Z_norm - 2 * X @ Z.T
    return np.exp(-dists / (2 * sigma**2))

def polynomial_kernel(X: np.ndarray, Z: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """
    Computes the polynomial kernel matrix between two datasets.

    Args:
        X - A numpy array of shape (n_samples, n_features) for the first dataset.
        Z - A numpy array of shape (m_samples, n_features) for the second dataset.
        degree - An integer representing the degree of the polynomial.
        c - A float representing the offset (constant term) in the polynomial kernel.

    Returns: A numpy array of shape (n_samples, m_samples) representing the kernel matrix.
    """
    return (X @ Z.T + c)**degree

# ------------------------------------------------------------------------------
# Kernel Perceptron
# ------------------------------------------------------------------------------

class KernelPerceptron:
    """
    Kernel Perceptron classifier using either a Gaussian or polynomial kernel.
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        sigma: float = 1.0,
        degree: int = 2,
        c: float = 1.0,
        max_iter: int = 10
    ):
        """
        Initializes the Kernel Perceptron classifier.

        Args:
            kernel - A string specifying the kernel type ("gaussian" or "polynomial").
            sigma - A float representing the parameter for the Gaussian kernel.
            degree - An integer representing the degree for the polynomial kernel.
            c - A float representing the offset for the polynomial kernel.
            max_iter - An integer representing the number of passes over the training set.
        """
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.c = c
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def _compute_kernel_matrix(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Computes the kernel matrix between two datasets using the specified kernel.

        Args:
            X - A numpy array of shape (n_samples, n_features) for the first dataset.
            Z - A numpy array of shape (m_samples, n_features) for the second dataset.

        Returns: A numpy array of shape (n_samples, m_samples) representing the kernel matrix.
        """
        if self.kernel == "gaussian":
            return gaussian_kernel(X, Z, sigma=self.sigma)
        elif self.kernel == "polynomial":
            return polynomial_kernel(X, Z, degree=self.degree, c=self.c)
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Kernel Perceptron on the provided training data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the training data.
            y - A numpy array of shape (n_samples,) with labels in {-1, +1}.

        Returns: None
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        K = self._compute_kernel_matrix(X, X)
        for _ in range(self.max_iter):
            for i in range(n_samples):
                pred = np.sign(np.sum(self.alpha * y * K[:, i]))
                if pred != y[i]:
                    self.alpha[i] += 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for new data using the Kernel Perceptron.

        Args:
            X - A numpy array of shape (m_samples, n_features) containing the input data.

        Returns: A numpy array of shape (m_samples,) with predicted labels in {-1, +1}.
        """
        K_test = self._compute_kernel_matrix(self.X_train, X)
        scores = (self.alpha * self.y_train)[:, None] * K_test
        scores = np.sum(scores, axis=0)
        return np.where(scores >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the 0-1 accuracy of the Kernel Perceptron on the provided data.

        Args:
            X - A numpy array containing the test data.
            y - A numpy array containing the true labels.

        Returns: A float representing the accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# ------------------------------------------------------------------------------
# Kernel Pegasos SVM
# ------------------------------------------------------------------------------

class KernelPegasosSVM:
    """
    Kernelized Pegasos SVM using either a Gaussian or polynomial kernel.
    """

    def __init__(
        self,
        lambda_: float = 1e-4,
        max_iter: int = 1000,
        kernel: str = "gaussian",
        sigma: float = 1.0,
        degree: int = 2,
        c: float = 1.0,
        random_state: int = None
    ):
        """
        Initializes the Kernel Pegasos SVM.

        Args:
            lambda_ - A float representing the regularization parameter.
            max_iter - An integer representing the total number of stochastic updates.
            kernel - A string specifying the kernel type ("gaussian" or "polynomial").
            sigma - A float representing the parameter for the Gaussian kernel.
            degree - An integer representing the degree for the polynomial kernel.
            c - A float representing the offset for the polynomial kernel.
            random_state - An optional integer seed for reproducible sampling.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.c = c
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.K = None

    def _compute_kernel_matrix(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Computes the kernel matrix between two datasets using the specified kernel.

        Args:
            X - A numpy array of shape (n_samples, n_features) for the first dataset.
            Z - A numpy array of shape (m_samples, n_features) for the second dataset.

        Returns: A numpy array of shape (n_samples, m_samples) representing the kernel matrix.
        """
        if self.kernel == "gaussian":
            return gaussian_kernel(X, Z, sigma=self.sigma)
        elif self.kernel == "polynomial":
            return polynomial_kernel(X, Z, degree=self.degree, c=self.c)
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Kernel Pegasos SVM on the given training data.

        Args:
            X - A numpy array of shape (n_samples, n_features) containing the training data.
            y - A numpy array of shape (n_samples,) with labels in {-1, +1}.

        Returns: None
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        self.K = self._compute_kernel_matrix(X, X)
        rng = np.random.default_rng(self.random_state)
        for t in range(1, self.max_iter + 1):
            i = rng.integers(n_samples)
            eta = 1.0 / (self.lambda_ * t)
            decision_value = np.sum(self.alpha * y * self.K[:, i])
            if y[i] * decision_value < 1:
                self.alpha[i] = (1.0 - eta * self.lambda_) * self.alpha[i] + eta
            else:
                self.alpha[i] = (1.0 - eta * self.lambda_) * self.alpha[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for new data using the trained Kernel Pegasos SVM.

        Args:
            X - A numpy array of shape (m_samples, n_features) containing the input data.

        Returns: A numpy array of shape (m_samples,) with predicted labels in {-1, +1}.
        """
        K_test = self._compute_kernel_matrix(self.X_train, X)
        scores = (self.alpha * self.y_train)[:, None] * K_test
        scores = np.sum(scores, axis=0)
        return np.where(scores >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the 0-1 accuracy of the Kernel Pegasos SVM on the provided data.

        Args:
            X - A numpy array containing the test data.
            y - A numpy array containing the true labels.

        Returns: A float representing the accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
