# my_code/data.py

import pandas as pd
import numpy as np

class DataManager:
    """
    This class will be responsible for loading and handling data.
    """

    def __init__(self, csv_path: str):
        """
        Initialize with a path to the CSV file. We do not immediately read or process.
        """
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None

        # Store mean/std for scaling (if needed)
        self.mean_ = None
        self.std_ = None

    def load_data(self):
        """
        Loads CSV into a pandas DataFrame and extracts features X and label y.
        Expects columns named x1 to x10 (for features) and y (for label).
        """
        self.df = pd.read_csv(self.csv_path)

        expected_features = [f"x{i}" for i in range(1, 11)]
        if not all(col in self.df.columns for col in expected_features + ["y"]):
            raise ValueError("CSV does not contain required columns x1..x10 and y.")

        self.X = self.df[expected_features].values
        self.y = self.df["y"].values

        # Convert y from {0,1} to {-1,+1} if needed
        unique_labels = set(self.y)
        if unique_labels == {0, 1}:
            self.y = np.where(self.y == 0, -1, 1)

        return self.df

    def train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                             shuffle=True, seed=42):
        """
        Splits (self.X, self.y) into train, validation, and test sets.
        Returns: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1."

        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_val, y_val = self.X[val_idx], self.y[val_idx]
        X_test, y_test = self.X[test_idx], self.y[test_idx]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def standard_scale_fit(self, X_train):
        """
        Computes mean and std of X_train for standard scaling.
        """
        self.mean_ = np.mean(X_train, axis=0)
        self.std_ = np.std(X_train, axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero

    def standard_scale_transform(self, X):
        """
        Applies previously computed mean_ and std_ for scaling X.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call standard_scale_fit before transform.")
        return (X - self.mean_) / self.std_

    def polynomial_feature_expansion_degree2(self, X):
        """
        Generates polynomial features up to degree 2:
        Original features + pairwise products (including squared terms).
        """
        n_samples, n_features = X.shape
        new_dim = n_features + (n_features * (n_features + 1)) // 2
        X_poly = np.zeros((n_samples, new_dim))

        # copy original
        X_poly[:, :n_features] = X

        idx = n_features
        for i in range(n_features):
            for j in range(i, n_features):
                X_poly[:, idx] = X[:, i] * X[:, j]
                idx += 1
        return X_poly
