# functions.py

import numpy as np

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def to_pm1_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert {0,1} labels to {-1,+1} for use in models requiring +/-1.
    If the labels are already in {-1,+1}, they will be left unchanged.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        The input labels, assumed to be either in {0,1} or {-1,+1}.
    
    Returns
    -------
    y_converted : np.ndarray, shape (n_samples,)
        The output labels in {-1, +1}.
    """
    unique_vals = np.unique(y)
    if set(unique_vals).issubset({0, 1}):
        # Convert to +/-1
        y_converted = np.where(y == 0, -1, 1)
        return y_converted
    elif set(unique_vals).issubset({-1, 1}):
        # Already in +/-1
        return y
    else:
        raise ValueError("Labels must be in {0,1} or {-1,+1}.")

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute 0-1 accuracy score.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels in {-1, +1} or {0,1}.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels in the same domain.

    Returns
    -------
    accuracy : float
        The fraction of samples for which y_pred == y_true.
    """
    return np.mean(y_true == y_pred)

def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute 0-1 loss, i.e., the fraction of misclassifications.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
    y_pred : np.ndarray, shape (n_samples,)

    Returns
    -------
    loss : float
        The fraction of samples for which y_pred != y_true.
    """
    return np.mean(y_true != y_pred)


# ------------------------------------------------------------------------------
# Polynomial Feature Expansion
# ------------------------------------------------------------------------------

def polynomial_feature_expansion(
    X: np.ndarray,
    degree: int = 2,
    include_bias: bool = True
) -> np.ndarray:
    """
    Produce polynomial (up to 'degree') feature expansions of X.
    Currently implemented for degree=2 by default.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The input data.
    degree : int
        The maximum polynomial degree. For degree=2, expansions include
        squared terms and pairwise interactions.
    include_bias : bool
        If True, include a bias (constant 1) term at the beginning.
        
    Returns
    -------
    X_poly : np.ndarray
        The transformed input of shape (n_samples, D), where D depends
        on the number of original features and 'degree'.

    Example
    -------
    For a single sample with features [x1, x2], if degree=2 and include_bias=True,
    the new features will be: [1, x1, x2, x1^2, x1*x2, x2^2].
    """
    if degree != 2:
        raise NotImplementedError("Currently only degree=2 expansion is implemented.")

    n_samples, n_features = X.shape
    
    # Start building the feature list
    X_poly_list = []
    if include_bias:
        X_poly_list.append(np.ones(n_samples))

    # Linear terms
    for f in range(n_features):
        X_poly_list.append(X[:, f])

    # Quadratic terms (pairwise + squared)
    for i in range(n_features):
        for j in range(i, n_features):
            X_poly_list.append(X[:, i] * X[:, j])

    X_poly = np.vstack(X_poly_list).T  # shape (n_samples, new_dim)
    return X_poly


# ------------------------------------------------------------------------------
# Linear Perceptron
# ------------------------------------------------------------------------------

class Perceptron:
    """
    A simple linear Perceptron classifier implemented from scratch.
    """

    def __init__(self, max_iter: int = 10, shuffle: bool = True, random_state: int = None):
        """
        Parameters
        ----------
        max_iter : int
            Number of passes over the training dataset.
        shuffle : bool
            Whether to shuffle the data each epoch.
        random_state : int, optional
            Seed for reproducible shuffling.
        """
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Perceptron model on training data (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Labels in {+1, -1} (convert using to_pm1_labels if needed).
        """
        # Basic shape checks
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")

        # Initialize
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        rng = np.random.default_rng(self.random_state)

        # Perceptron training
        for _ in range(self.max_iter):
            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            
            for i in indices:
                # Perceptron update if misclassified
                if y[i] * np.dot(self.w, X[i]) <= 0:
                    self.w += y[i] * X[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels in {+1, -1}.
        """
        return np.sign(X @ self.w)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute 0-1 accuracy on the given data and labels.

        Parameters
        ----------
        X : np.ndarray
            Test data.
        y : np.ndarray
            True labels in {+1, -1}.

        Returns
        -------
        accuracy : float
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


# ------------------------------------------------------------------------------
# Pegasos SVM (Linear)
# ------------------------------------------------------------------------------

class PegasosSVM:
    """
    Pegasos algorithm for linear SVM with hinge loss from scratch.
    """

    def __init__(self, lambda_: float = 1e-4, max_iter: int = 1000, random_state: int = None):
        """
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        max_iter : int
            Total number of stochastic updates.
        random_state : int, optional
            Seed for reproducible sampling.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Pegasos SVM on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Labels in {+1, -1}.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        rng = np.random.default_rng(self.random_state)

        t = 0
        for iteration in range(1, self.max_iter + 1):
            # Sample one data point at random
            i = rng.integers(n_samples)
            t += 1
            eta = 1.0 / (self.lambda_ * t)

            # Scale w
            self.w = (1 - eta * self.lambda_) * self.w

            # Hinge check
            if y[i] * np.dot(self.w, X[i]) < 1:
                self.w += eta * y[i] * X[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X in {+1, -1}.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        return np.sign(X @ self.w)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        0-1 accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        accuracy : float
        """
        return accuracy_score(y, self.predict(X))


# ------------------------------------------------------------------------------
# Pegasos Logistic (Linear)
# ------------------------------------------------------------------------------

class PegasosLogistic:
    """
    Pegasos-style algorithm for logistic loss from scratch.
    """

    def __init__(self, lambda_: float = 1e-4, max_iter: int = 1000, random_state: int = None):
        """
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        max_iter : int
            Total number of stochastic updates.
        random_state : int, optional
            Seed for reproducible sampling.
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Pegasos logistic model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Labels in {+1, -1}.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        rng = np.random.default_rng(self.random_state)

        for t in range(1, self.max_iter + 1):
            i = rng.integers(n_samples)
            eta = 1.0 / (self.lambda_ * t)
            
            # Scale w
            self.w = (1 - eta * self.lambda_) * self.w
            
            exponent = y[i] * np.dot(self.w, X[i])
            logistic_grad = - (y[i] * X[i]) / (1.0 + np.exp(exponent))
            
            # Update
            self.w -= eta * logistic_grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels in {+1, -1} based on logistic model output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        scores = X @ self.w
        return np.where(scores >= 0, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities P(y=+1 | x).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples,)
            Probability of class = +1 for each sample.
        """
        scores = X @ self.w
        return 1.0 / (1.0 + np.exp(-scores))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        0-1 accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        accuracy : float
        """
        return accuracy_score(y, self.predict(X))


# ------------------------------------------------------------------------------
# Kernel Functions
# ------------------------------------------------------------------------------

def gaussian_kernel(X: np.ndarray, Z: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute the Gaussian (RBF) kernel matrix between X and Z.

    K(i, j) = exp( -||X[i] - Z[j]||^2 / (2 * sigma^2) ).
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    Z : np.ndarray, shape (m_samples, n_features)
    sigma : float
    
    Returns
    -------
    K : np.ndarray, shape (n_samples, m_samples)
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
    dists = X_norm + Z_norm - 2 * X @ Z.T
    return np.exp(-dists / (2 * sigma**2))

def polynomial_kernel(X: np.ndarray, Z: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """
    Compute the polynomial kernel matrix between X and Z.

    K(i, j) = (X[i] · Z[j] + c)^degree
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    Z : np.ndarray, shape (m_samples, n_features)
    degree : int
    c : float
    
    Returns
    -------
    K : np.ndarray, shape (n_samples, m_samples)
    """
    return (X @ Z.T + c)**degree


# ------------------------------------------------------------------------------
# Kernel Perceptron
# ------------------------------------------------------------------------------

class KernelPerceptron:
    """
    Kernel Perceptron classifier with either Gaussian or polynomial kernel.
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
        Parameters
        ----------
        kernel : {"gaussian", "polynomial"}
            Which kernel to use.
        sigma : float
            Parameter for Gaussian kernel.
        degree : int
            Degree for polynomial kernel.
        c : float
            Offset for polynomial kernel.
        max_iter : int
            Number of passes over the training set.
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
        if self.kernel == "gaussian":
            return gaussian_kernel(X, Z, sigma=self.sigma)
        elif self.kernel == "polynomial":
            return polynomial_kernel(X, Z, degree=self.degree, c=self.c)
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Kernel Perceptron using the specified kernel.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) in {+1, -1}
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")

        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        
        # Precompute kernel matrix among training points
        K = self._compute_kernel_matrix(X, X)  # shape (n_samples, n_samples)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Prediction is sign( sum_j (alpha_j * y_j * K(j, i)) )
                pred = np.sign(np.sum(self.alpha * y * K[:, i]))
                if pred != y[i]:
                    self.alpha[i] += 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels in {+1, -1} for new data.

        Parameters
        ----------
        X : np.ndarray, shape (m_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (m_samples,)
        """
        K_test = self._compute_kernel_matrix(self.X_train, X)  # shape (n_samples, m_samples)
        scores = (self.alpha * self.y_train)[:, None] * K_test
        scores = np.sum(scores, axis=0)
        return np.where(scores >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        0-1 accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        accuracy : float
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


# ------------------------------------------------------------------------------
# Kernel Pegasos SVM
# ------------------------------------------------------------------------------

class KernelPegasosSVM:
    """
    Kernelized Pegasos SVM with Gaussian or polynomial kernel.
    Reference: "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", Shalev-Shwartz et al.
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
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        max_iter : int
            Total number of stochastic updates.
        kernel : {"gaussian", "polynomial"}
            Which kernel to use.
        sigma : float
            Parameter for Gaussian kernel.
        degree : int
            Degree for polynomial kernel.
        c : float
            Offset for polynomial kernel.
        random_state : int, optional
            Seed for reproducible sampling in the stochastic loop.
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
        if self.kernel == "gaussian":
            return gaussian_kernel(X, Z, sigma=self.sigma)
        elif self.kernel == "polynomial":
            return polynomial_kernel(X, Z, degree=self.degree, c=self.c)
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Kernel Pegasos SVM on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) in {+1, -1}
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")

        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Precompute kernel matrix among training points
        self.K = self._compute_kernel_matrix(X, X)  # shape (n_samples, n_samples)
        rng = np.random.default_rng(self.random_state)

        for t in range(1, self.max_iter + 1):
            i = rng.integers(n_samples)  # pick random sample
            eta = 1.0 / (self.lambda_ * t)
            
            # Evaluate current decision function for sample i
            decision_value = np.sum(self.alpha * y * self.K[:, i])
            
            # Pegasos kernel update
            if y[i] * decision_value < 1:
                self.alpha[i] = (1.0 - eta * self.lambda_) * self.alpha[i] + eta
            else:
                self.alpha[i] = (1.0 - eta * self.lambda_) * self.alpha[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels in {+1, -1} for new data.

        Parameters
        ----------
        X : np.ndarray, shape (m_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (m_samples,)
        """
        # Kernel matrix between training points and new points
        K_test = self._compute_kernel_matrix(self.X_train, X)  # shape (n_samples, m_samples)
        # f(x) = sign( sum_i alpha_i * y_i * K(x_i, x) )
        scores = (self.alpha * self.y_train)[:, None] * K_test
        scores = np.sum(scores, axis=0)
        return np.where(scores >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        0-1 accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        accuracy : float
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
