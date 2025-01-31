# my_code/kernel_models.py

import numpy as np

class KernelFunctions:
    """
    A static container for kernel functions.
    """

    @staticmethod
    def rbf_kernel(X1, X2, gamma):
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * (X1 @ X2.T)
        return np.exp(-gamma * dist_sq)

    @staticmethod
    def polynomial_kernel(X1, X2, degree=2, c=1.0):
        return (X1 @ X2.T + c) ** degree


class KernelPerceptron:
    """
    Kernelized Perceptron classifier: alpha-based approach.
    """

    def __init__(self, kernel_func, epochs=10, **kernel_params):
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.epochs = epochs
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.K = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)

        # Precompute kernel matrix
        self.K = self.kernel_func(X, X, **self.kernel_params)

        for _ in range(self.epochs):
            for i in range(n_samples):
                decision = np.sum(self.alpha * self.y_train * self.K[:, i])
                pred = 1 if decision >= 0 else -1
                if pred != y[i]:
                    self.alpha[i] += 1

    def predict(self, X):
        K_test = self.kernel_func(self.X_train, X, **self.kernel_params)
        decision = np.dot(self.alpha * self.y_train, K_test)
        return np.where(decision >= 0, 1, -1)


class KernelPegasosSVM:
    """
    Kernelized Pegasos for SVM.
    """

    def __init__(self, kernel_func, lam=1e-4, max_iters=1000, **kernel_params):
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.lam = lam
        self.max_iters = max_iters

        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.K = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)

        self.K = self.kernel_func(X, X, **self.kernel_params)

        for t in range(1, self.max_iters + 1):
            i = np.random.randint(0, n_samples)
            eta_t = 1.0 / (self.lam * t)

            decision_i = np.sum(self.alpha * y * self.K[:, i])
            if y[i] * decision_i < 1:
                self.alpha[i] += eta_t

            alpha_y = self.alpha * self.y_train
            norm_factor = np.sqrt(self.lam) * np.sqrt(np.sum(alpha_y[:, None] * alpha_y[None, :] * self.K))
            if norm_factor > 1.0:
                self.alpha /= norm_factor

    def predict(self, X):
        K_test = self.kernel_func(self.X_train, X, **self.kernel_params)
        decision = np.dot(self.alpha * self.y_train, K_test)
        return np.where(decision >= 0, 1, -1)
