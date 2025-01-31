# my_code/models.py

import numpy as np

class BaseModel:
    """
    A simple base class for all linear models.
    """

    def __init__(self):
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @staticmethod
    def sign_function(scores):
        return np.where(scores >= 0, 1, -1)

    @staticmethod
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)


class Perceptron(BaseModel):
    """
    Simple linear Perceptron.
    """

    def __init__(self, epochs=10, eta=1.0):
        super().__init__()
        self.epochs = epochs
        self.eta = eta

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += self.eta * y[i] * X[i]
                    self.b += self.eta * y[i]

    def predict(self, X):
        scores = X @ self.w + self.b
        return self.sign_function(scores)


class PegasosSVM(BaseModel):
    """
    Pegasos algorithm for SVM (hinge loss).
    """

    def __init__(self, lam=1e-4, max_iters=1000, batch_size=1):
        super().__init__()
        self.lam = lam
        self.max_iters = max_iters
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for iteration in range(1, self.max_iters + 1):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            eta_t = 1.0 / (self.lam * iteration)
            self.w = (1 - eta_t * self.lam) * self.w

            for i in range(self.batch_size):
                if y_batch[i] * np.dot(self.w, X_batch[i]) < 1:
                    self.w += eta_t * y_batch[i] * X_batch[i]

        self.b = 0.0

    def predict(self, X):
        scores = X @ self.w
        return self.sign_function(scores)


class PegasosLogistic(BaseModel):
    """
    Pegasos-style logistic regression.
    """

    def __init__(self, lam=1e-4, max_iters=1000, batch_size=1):
        super().__init__()
        self.lam = lam
        self.max_iters = max_iters
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for t in range(1, self.max_iters + 1):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]

            eta_t = 1.0 / (self.lam * t)
            self.w = (1 - eta_t * self.lam) * self.w

            for i in range(self.batch_size):
                exponent = - y_batch[i] * np.dot(self.w, X_batch[i])
                grad_factor = -(y_batch[i] * X_batch[i]) * (np.exp(exponent) / (1.0 + np.exp(exponent)))
                self.w -= eta_t * grad_factor

    def predict(self, X):
        scores = X @ self.w
        return self.sign_function(scores)
