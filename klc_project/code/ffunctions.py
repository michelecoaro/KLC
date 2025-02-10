import pandas as pd
import numpy as np

def load_data(csv_path):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    data = pd.read_csv(csv_path)
    return data

def train_test_split(df, test_size=0.2, random_state=42):
    """
    Splits a pandas DataFrame into train and test sets.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_cutoff = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_cutoff]
    train_indices = shuffled_indices[test_cutoff:]
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df

def standard_scaler_fit(train_df, features):
    """
    Learns mean and standard deviation for each feature in 'features' on train_df.
    """
    means = {}
    stds = {}
    for f in features:
        means[f] = train_df[f].mean()
        stds[f] = train_df[f].std()
    return means, stds

def standard_scaler_transform(df, features, means, stds):
    """
    Applies standard scaling (z-score) to the specified features using pre-computed means/stds.
    """
    df_copy = df.copy()
    for f in features:
        if stds[f] != 0:
            df_copy[f] = (df_copy[f] - means[f]) / stds[f]
        else:
            df_copy[f] = df_copy[f] - means[f]
    return df_copy

def detect_outliers_zscore(df, features, z_thresh=3.0):
    """
    Detects outliers in each of the specified features using a z-score threshold.
    Returns a set of row indices that contain outliers.
    """
    outlier_indices = set()
    for f in features:
        mean_f = df[f].mean()
        std_f = df[f].std()
        if std_f == 0:
            continue
        z_scores = ((df[f] - mean_f) / std_f).abs()
        f_outliers = z_scores[z_scores > z_thresh].index
        outlier_indices.update(f_outliers)
    return outlier_indices

def remove_outliers(df, outlier_indices):
    """
    Removes rows from df based on the set of outlier indices.
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)

def check_high_correlation(df, features, corr_threshold=0.95):
    """
    Checks for pairs of features that exceed a given correlation threshold.
    """
    corr_matrix = df[features].corr().abs()
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                f1 = features[i]
                f2 = features[j]
                high_corr_pairs.append((f1, f2, corr_matrix.iloc[i, j]))
    return high_corr_pairs

def k_fold_cross_validation(df, features, target, k=5, random_state=42, classifier_func=None):
    """
    Performs K-fold cross-validation on the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to be split into folds.
        features (list): List of feature column names.
        target (str): Name of the target column.
        k (int): Number of folds. Default = 5.
        random_state (int): Random seed for reproducibility.
        classifier_func (callable): A function that takes (train_df, test_df, features, target)
                                    and returns predicted labels for test_df.

    Returns:
        mean_accuracy (float): Average accuracy across K folds.
    """

    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // k
    accuracies = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size
        # Indices for validation fold
        val_indices = indices[start:end]
        # All other indices for training
        train_indices = np.concatenate([indices[:start], indices[end:]])

        train_fold = df.iloc[train_indices].reset_index(drop=True)
        val_fold = df.iloc[val_indices].reset_index(drop=True)

        # If no classifier_func provided, use a dummy classifier
        if classifier_func is None:
            # (Simple majority-class predictor)
            majority_class = train_fold[target].value_counts().idxmax()
            val_predictions = np.full(len(val_fold), majority_class)
        else:
            # Use the provided classifier function
            val_predictions = classifier_func(train_fold, val_fold, features, target)

        # Calculate accuracy for this fold
        correct = (val_predictions == val_fold[target]).sum()
        accuracy = correct / len(val_fold)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def pca_transform(df, features, n_components=2):
    """
    Performs PCA on the specified features of df using the covariance matrix.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be transformed.
        features (list): List of feature columns on which to run PCA.
        n_components (int): Number of principal components to retain.

    Returns:
        pca_data (pd.DataFrame): DataFrame with the principal component columns replacing the original features.
        pca_components (np.ndarray): The principal component vectors (eigenvectors).
        explained_variance (np.ndarray): The eigenvalues associated with each principal component.
    """

    # 1. Extract the feature matrix
    X = df[features].values

    # 2. Center the data (subtract mean)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    # 3. Compute covariance matrix
    cov_mat = np.cov(X_centered, rowvar=False)

    # 4. Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    # 5. Sort eigenvalues (and corresponding eigenvectors) in descending order
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]

    # 6. Select the top n_components
    selected_vecs = eigen_vecs[:, :n_components]
    selected_vals = eigen_vals[:n_components]

    # 7. Project data onto new components
    X_pca = np.dot(X_centered, selected_vecs)

    # Build a new DataFrame
    pca_cols = [f"PCA_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    # Combine with the rest of the DataFrame (drop original features)
    df_rest = df.drop(columns=features)
    pca_data = pd.concat([df_rest, pca_df], axis=1)

    return pca_data, selected_vecs, selected_vals

import numpy as np

def perceptron_train(train_df, features, target, epochs=5):
    """
    Trains a binary Perceptron classifier on the given training DataFrame.
    Labels must be -1 or +1.

    Args:
        train_df (pd.DataFrame): Training data containing features and target column.
        features (list): List of feature column names (strings).
        target (str): Name of the target column (must have values in {-1, +1}).
        epochs (int): Number of passes (epochs) over the training data.

    Returns:
        theta (np.ndarray): Learned weight vector (length = number of features).
        theta_0 (float): Learned bias term.
    """
    # Convert to numpy for faster operations
    X = train_df[features].values
    y = train_df[target].values

    # Number of features
    d = len(features)

    # Initialize parameters
    theta = np.zeros(d)
    theta_0 = 0.0

    for _ in range(epochs):
        # Shuffle the data each epoch to help convergence
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            # Perceptron update rule
            if y_i * (np.dot(theta, x_i) + theta_0) <= 0:
                theta = theta + y_i * x_i
                theta_0 = theta_0 + y_i

    return theta, theta_0

def perceptron_train_eta(train_df, features, target, epochs=5, eta=1.0):
    """
    Trains a binary Perceptron classifier on the given training DataFrame.
    Labels must be -1 or +1.

    Args:
        train_df (pd.DataFrame): Training data containing features and target column.
        features (list): List of feature column names (strings).
        target (str): Name of the target column (must have values in {-1, +1}).
        epochs (int): Number of passes (epochs) over the training data.
        eta (float): Learning rate for parameter updates.

    Returns:
        theta (np.ndarray): Learned weight vector (length = number of features).
        theta_0 (float): Learned bias term.
    """
    # Convert to numpy for faster operations
    X = train_df[features].values
    y = train_df[target].values

    # Number of features
    d = len(features)

    # Initialize parameters
    theta = np.zeros(d)
    theta_0 = 0.0

    for _ in range(epochs):
        # Shuffle the data each epoch to help convergence
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            # Perceptron update rule with learning rate
            if y_i * (np.dot(theta, x_i) + theta_0) <= 0:
                theta = theta + eta * y_i * x_i
                theta_0 = theta_0 + eta * y_i

    return theta, theta_0


def perceptron_classifier_func_eta(train_fold, val_fold, features, target, epochs=5, eta=1.0):
    """
    This function is designed to be passed as `classifier_func` to k_fold_cross_validation.
    It trains the Perceptron on train_fold and then returns predictions for val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names.
        target (str): Target column name.
        epochs (int): Number of epochs for the Perceptron.
        eta (float): Learning rate for parameter updates.

    Returns:
        val_predictions (np.ndarray): Predictions (-1 or +1) for the validation fold.
    """
    theta, theta_0 = perceptron_train_eta(train_fold, features, target, epochs=epochs, eta=eta)
    val_predictions = perceptron_predict(val_fold, features, theta, theta_0)
    return val_predictions



def perceptron_predict(df, features, theta, theta_0):
    """
    Uses the learned Perceptron parameters to predict labels (-1 or +1).

    Args:
        df (pd.DataFrame): Data containing the features to predict.
        features (list): List of feature names.
        theta (np.ndarray): Learned weight vector.
        theta_0 (float): Learned bias term.

    Returns:
        predictions (np.ndarray): 1D array of predicted labels (-1 or +1).
    """
    X = df[features].values
    # Linear scores
    scores = np.dot(X, theta) + theta_0
    # Convert scores to labels
    predictions = np.where(scores > 0, 1, -1)
    return predictions


def perceptron_classifier_func(train_fold, val_fold, features, target, epochs=5):
    """
    This function is designed to be passed as `classifier_func` to k_fold_cross_validation.
    It trains the Perceptron on train_fold and then returns predictions for val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names.
        target (str): Target column name.
        epochs (int): Number of epochs for the Perceptron.

    Returns:
        val_predictions (np.ndarray): Predictions (-1 or +1) for the validation fold.
    """
    theta, theta_0 = perceptron_train(train_fold, features, target, epochs=epochs)
    val_predictions = perceptron_predict(val_fold, features, theta, theta_0)
    return val_predictions

def pegasos_train(train_df, features, target, lambda_param=0.01, epochs=5):
    """
    Trains a binary Pegasos SVM on the given training DataFrame.
    Labels must be -1 or +1.

    Args:
        train_df (pd.DataFrame): Training data containing features and target column.
        features (list): List of feature column names (strings).
        target (str): Name of the target column (must have values in {-1, +1}).
        lambda_param (float): Regularization parameter (λ).
        epochs (int): Number of passes (epochs) over the training data.

    Returns:
        theta (np.ndarray): Learned weight vector (length = number of features).
        theta_0 (float): Learned bias term.
    """
    # Convert to numpy for faster operations
    X = train_df[features].values
    y = train_df[target].values

    # Number of features
    d = len(features)

    # Initialize parameters
    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1  # Iteration counter

    for _ in range(epochs):
        # Shuffle the data each epoch to help convergence
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            eta_t = 1 / (lambda_param * t)  # Step size
            t += 1

            # Check hinge-loss condition
            if y_i * (np.dot(theta, x_i) + theta_0) < 1:
                theta = (1 - eta_t * lambda_param) * theta + eta_t * y_i * x_i
                theta_0 = theta_0 + eta_t * y_i
            else:
                theta = (1 - eta_t * lambda_param) * theta

    return theta, theta_0


def pegasos_predict(df, features, theta, theta_0):
    """
    Uses the learned Pegasos SVM parameters to predict labels (-1 or +1).

    Args:
        df (pd.DataFrame): Data containing the features to predict.
        features (list): List of feature names.
        theta (np.ndarray): Learned weight vector.
        theta_0 (float): Learned bias term.

    Returns:
        predictions (np.ndarray): 1D array of predicted labels (-1 or +1).
    """
    X = df[features].values
    # Linear scores
    scores = np.dot(X, theta) + theta_0
    # Convert scores to labels
    predictions = np.where(scores > 0, 1, -1)
    return predictions


def pegasos_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5):
    """
    This function is designed to be passed as `classifier_func` to k_fold_cross_validation.
    It trains Pegasos SVM on train_fold and returns predictions for val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter (λ).
        epochs (int): Number of epochs for the Pegasos SVM.

    Returns:
        val_predictions (np.ndarray): Predictions (-1 or +1) for the validation fold.
    """
    theta, theta_0 = pegasos_train(train_fold, features, target, lambda_param=lambda_param, epochs=epochs)
    val_predictions = pegasos_predict(val_fold, features, theta, theta_0)
    return val_predictions
