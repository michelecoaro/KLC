'''import pandas as pd
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

def logistic_regression_train(train_df, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    Trains a regularized logistic regression model using SGD.
    Labels must be -1 or +1.

    Args:
        train_df (pd.DataFrame): Training data containing features and target column.
        features (list): List of feature column names (strings).
        target (str): Name of the target column (must have values in {-1, +1}).
        lambda_param (float): Regularization parameter (λ).
        epochs (int): Number of passes (epochs) over the training data.
        eta (float): Initial learning rate.

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
        # Shuffle the data each epoch
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]

            # Learning rate schedule
            eta_t = eta / np.sqrt(t)  # You can modify this as needed
            t += 1

            # Logistic loss gradient
            margin = y_i * (np.dot(theta, x_i) + theta_0)
            gradient_theta = lambda_param * theta - (y_i * x_i) / (1 + np.exp(margin))
            gradient_theta_0 = -y_i / (1 + np.exp(margin))

            # Update weights and bias
            theta -= eta_t * gradient_theta
            theta_0 -= eta_t * gradient_theta_0

    return theta, theta_0

def logistic_regression_predict(df, features, theta, theta_0):
    """
    Uses the learned logistic regression parameters to predict labels (-1 or +1).

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
    # Convert scores to probabilities and then to labels
    predictions = np.where(scores > 0, 1, -1)
    return predictions

def logistic_regression_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    This function is designed to be passed as `classifier_func` to k_fold_cross_validation.
    It trains logistic regression on train_fold and returns predictions for val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter (λ).
        epochs (int): Number of epochs.
        eta (float): Initial learning rate.

    Returns:
        val_predictions (np.ndarray): Predictions (-1 or +1) for the validation fold.
    """
    theta, theta_0 = logistic_regression_train(
        train_fold, features, target, lambda_param=lambda_param, epochs=epochs, eta=eta
    )
    val_predictions = logistic_regression_predict(val_fold, features, theta, theta_0)
    return val_predictions


'''

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
    Splits a pandas DataFrame into train and test sets, without data leakage.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_cutoff = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_cutoff]
    train_indices = shuffled_indices[test_cutoff:]
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df

def detect_outliers_zscore(df, features, z_thresh=3.0):
    """
    Detects outliers in each of the specified features using a z-score threshold.
    Returns a set of row indices (in df) that are outliers.
    """
    outlier_indices = set()
    for f in features:
        mean_f = df[f].mean()
        std_f = df[f].std()
        if std_f == 0:
            continue  # skip features with zero variance
        z_scores = ((df[f] - mean_f) / std_f).abs()
        f_outliers = z_scores[z_scores > z_thresh].index
        outlier_indices.update(f_outliers)
    return outlier_indices

def remove_outliers(df, outlier_indices):
    """
    Removes rows from df based on a set of outlier indices.
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)

def standard_scaler_fit(train_df, features):
    """
    Learns mean and std for each feature on the training data only.
    """
    means = {}
    stds = {}
    for f in features:
        means[f] = train_df[f].mean()
        stds[f] = train_df[f].std()
    return means, stds

def standard_scaler_transform(df, features, means, stds):
    """
    Applies the precomputed means/stds to scale the features in df.
    Avoid data leakage by using the means/stds from training data only.
    """
    df_copy = df.copy()
    for f in features:
        if stds[f] != 0:
            df_copy[f] = (df_copy[f] - means[f]) / stds[f]
        else:
            df_copy[f] = df_copy[f] - means[f]
    return df_copy

def check_high_correlation(df, features, corr_threshold=0.95):
    """
    Checks for pairs of features that exceed a given correlation threshold.
    Returns a list of (feature1, feature2, corr_value) for those above the threshold.
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
    Returns the mean accuracy across K folds.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    fold_size = len(df) // k
    accuracies = []

    for fold in range(k):
        start = fold * fold_size
        end = start + fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        train_fold = df.iloc[train_indices].reset_index(drop=True)
        val_fold = df.iloc[val_indices].reset_index(drop=True)

        if classifier_func is None:
            # Dummy classifier: predict the majority class in the training fold
            majority_class = train_fold[target].value_counts().idxmax()
            val_predictions = np.full(len(val_fold), majority_class)
        else:
            val_predictions = classifier_func(train_fold, val_fold, features, target)

        accuracy = (val_predictions == val_fold[target]).mean()
        accuracies.append(accuracy)

    return np.mean(accuracies)

##############################################################################
# Linear Models (Perceptron, Pegasos, Logistic)
##############################################################################

def perceptron_train(train_df, features, target, epochs=5):
    """
    Trains a binary Perceptron classifier. Labels must be -1 or +1.
    """
    X = train_df[features].values
    y = train_df[target].values

    d = len(features)
    theta = np.zeros(d)
    theta_0 = 0.0

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            # Perceptron update
            if y_i * (np.dot(theta, x_i) + theta_0) <= 0:
                theta += y_i * x_i
                theta_0 += y_i

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

def perceptron_predict(df, features, theta, theta_0):
    """
    Predicts -1 or +1 using the trained Perceptron parameters.
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    predictions = np.where(scores > 0, 1, -1)
    return predictions

def perceptron_classifier_func(train_fold, val_fold, features, target, epochs=5):
    """
    For k-fold CV usage: trains a Perceptron, returns predictions on val_fold.
    """
    theta, theta_0 = perceptron_train(train_fold, features, target, epochs=epochs)
    return perceptron_predict(val_fold, features, theta, theta_0)

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


def pegasos_train(train_df, features, target, lambda_param=0.01, epochs=5):
    """
    Trains a Pegasos SVM (linear) with hinge loss. y in {-1, +1}.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = len(features)

    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1  # global iteration counter

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            eta_t = 1 / (lambda_param * t)
            t += 1

            if y_i * (np.dot(theta, x_i) + theta_0) < 1:
                # update with hinge
                theta = (1 - eta_t * lambda_param) * theta + eta_t * y_i * x_i
                theta_0 += eta_t * y_i
            else:
                # no gradient from the hinge term
                theta = (1 - eta_t * lambda_param) * theta

    return theta, theta_0

def pegasos_predict(df, features, theta, theta_0):
    """
    Predicts -1 or +1 for Pegasos SVM (linear).
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores > 0, 1, -1)

def pegasos_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5):
    """
    For k-fold CV usage: trains Pegasos SVM, returns predictions on val_fold.
    """
    theta, theta_0 = pegasos_train(train_fold, features, target,
                                   lambda_param=lambda_param, epochs=epochs)
    return pegasos_predict(val_fold, features, theta, theta_0)

def logistic_regression_train(train_df, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    Trains a regularized logistic regression model with SGD.
    y in {-1, +1}.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = len(features)

    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]

            # learning rate schedule
            eta_t = eta / np.sqrt(t)
            t += 1

            margin = y_i * (np.dot(theta, x_i) + theta_0)
            gradient_theta = lambda_param * theta - (y_i * x_i) / (1 + np.exp(margin))
            gradient_theta_0 = -y_i / (1 + np.exp(margin))

            theta -= eta_t * gradient_theta
            theta_0 -= eta_t * gradient_theta_0

    return theta, theta_0

def logistic_regression_predict(df, features, theta, theta_0):
    """
    Predicts -1 or +1 for logistic regression using the sign of the linear score.
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores > 0, 1, -1)

def logistic_regression_classifier_func(train_fold, val_fold, features, target,
                                        lambda_param=0.01, epochs=5, eta=1.0):
    """
    For k-fold CV usage: trains logistic regression, returns predictions on val_fold.
    """
    theta, theta_0 = logistic_regression_train(
        train_fold, features, target, lambda_param=lambda_param, epochs=epochs, eta=eta
    )
    return logistic_regression_predict(val_fold, features, theta, theta_0)

def polynomial_feature_expansion(df, features, degree=2, include_bias=False):
    """
    Expands the given features in the DataFrame up to a specified polynomial degree (2 by default).
    Specifically for degree=2, this function creates:
      - All original features x_i
      - All squared terms x_i^2
      - All pairwise interaction terms x_i * x_j (for i < j)

    Args:
        df (pd.DataFrame): The DataFrame containing the features to expand.
        features (list): List of column names (strings) in df to be expanded.
        degree (int): Degree of polynomial expansion. Currently only supports degree=2 explicitly.
        include_bias (bool): If True, an additional 'bias' column (all ones) is added.

    Returns:
        pd.DataFrame: A new DataFrame that contains the expanded polynomial features.
                      The original non-expanded columns are dropped (to avoid duplication).
    """
    if degree != 2:
        raise ValueError("This function currently only supports degree=2 expansion.")

    # We'll work on a copy to avoid modifying the original DataFrame
    X_original = df[features].copy()

    # List to collect new feature columns
    poly_data = {}

    # 1. (Optional) add bias column
    if include_bias:
        poly_data['bias'] = np.ones(len(X_original))

    # 2. Add the original features
    for f in features:
        poly_data[f] = X_original[f]

    # 3. Add squared terms
    for f in features:
        new_col_name = f"{f}^2"
        poly_data[new_col_name] = X_original[f] ** 2

    # 4. Add pairwise interaction terms (for i < j to avoid duplicates)
    num_feats = len(features)
    for i in range(num_feats):
        for j in range(i+1, num_feats):
            f_i = features[i]
            f_j = features[j]
            new_col_name = f"{f_i}*{f_j}"
            poly_data[new_col_name] = X_original[f_i] * X_original[f_j]

    # Convert to a new DataFrame
    expanded_df = pd.DataFrame(poly_data, index=df.index)

    # If you want to keep the other (non-feature) columns (like target),
    # you can merge them back. For instance:
    other_cols = [col for col in df.columns if col not in features]
    return pd.concat([expanded_df, df[other_cols]], axis=1)

##############################################################################
# Kernel Functions
##############################################################################

def gaussian_kernel(x, y, sigma=1.0):
    """
    Gaussian (RBF) kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))
    """
    diff = x - y
    sq_dist = np.dot(diff, diff)
    return np.exp(-sq_dist / (2 * sigma**2))

def polynomial_kernel(x, y, degree=2, c=1.0):
    """
    Polynomial kernel: k(x, y) = (x^T y + c)^degree
    """
    return (np.dot(x, y) + c) ** degree

##############################################################################
# Kernelized Perceptron
##############################################################################

def kernelized_perceptron_train(X, y, kernel_func, kernel_params={}, epochs=5):
    """
    Trains a Kernelized Perceptron on data (X, y) using the given kernel_func.
    y should be in {-1, +1}.

    Args:
        X (np.ndarray): Training data of shape (n_samples, n_features).
        y (np.ndarray): Labels of shape (n_samples,).
        kernel_func (callable): A function k(x, x') returning a scalar kernel value.
        kernel_params (dict): Additional params for kernel_func (e.g., sigma, degree, c, etc.).
        epochs (int): Number of passes over the data.

    Returns:
        alpha (np.ndarray): Coefficients for each training example (length = n_samples).
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples)

    # Main loop
    for _ in range(epochs):
        for i in range(n_samples):
            # Compute the decision function f(x_i)
            # f(x_i) = sum_j [ alpha_j * y_j * k(x_j, x_i ) ]
            # If sign(f(x_i)) != y_i, increment alpha_i
            f_i = 0.0
            x_i = X[i]
            for j in range(n_samples):
                if alpha[j] != 0:  # small optimization
                    f_i += alpha[j] * y[j] * kernel_func(X[j], x_i, **kernel_params)

            # Check for misclassification
            if np.sign(f_i) != y[i]:
                alpha[i] += 1.0

    return alpha

def kernelized_perceptron_predict(X_train, y_train, alpha, X_test, kernel_func, kernel_params={}):
    """
    Predict labels for X_test using the trained Kernelized Perceptron.

    Args:
        X_train (np.ndarray): Training data (n_train, n_features).
        y_train (np.ndarray): Training labels (n_train,).
        alpha (np.ndarray): Coefficients from kernelized_perceptron_train (n_train,).
        X_test (np.ndarray): Test data (n_test, n_features).
        kernel_func (callable): Same kernel function used during training.
        kernel_params (dict): Params for kernel_func.

    Returns:
        predictions (np.ndarray): Predicted labels (-1 or +1) for the test set.
    """
    n_train = len(y_train)
    n_test = len(X_test)
    predictions = np.zeros(n_test)

    for i in range(n_test):
        f_i = 0.0
        x_i = X_test[i]
        for j in range(n_train):
            if alpha[j] != 0:
                f_i += alpha[j] * y_train[j] * kernel_func(X_train[j], x_i, **kernel_params)
        predictions[i] = np.sign(f_i)
        if predictions[i] == 0:
            # In case f_i = 0, decide +1 or -1 (choose +1 by convention)
            predictions[i] = 1.0

    return predictions.astype(int)

def kernelized_perceptron_classifier_func(
    train_fold, val_fold, features, target,
    kernel_func, kernel_params={}, epochs=5
):
    """
    A classifier_func for k_fold_cross_validation, specifically for the Kernelized Perceptron.

    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name.
        kernel_func (callable): Kernel function (gaussian_kernel or polynomial_kernel).
        kernel_params (dict): Additional kernel parameters.
        epochs (int): Number of epochs for training.

    Returns:
        val_preds (np.ndarray): Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values

    # Train kernelized perceptron
    alpha = kernelized_perceptron_train(
        X_train, y_train,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        epochs=epochs
    )

    # Predict on val_fold
    val_preds = kernelized_perceptron_predict(
        X_train, y_train,
        alpha,
        X_val,
        kernel_func=kernel_func,
        kernel_params=kernel_params
    )
    return val_preds

def kernelized_pegasos_train(
    X, y,
    kernel_func,
    kernel_params={},
    lambda_param=0.01,
    epochs=5
):
    """
    Trains a kernelized Pegasos SVM on (X, y) using the provided kernel_func (e.g. Gaussian, Polynomial).
    The pseudocode from the question is effectively:

    INPUT: S, λ, T
    INITIALIZE: alpha^1 = 0
    FOR t = 1 to T:
        pick i_t randomly
        margin = y[i_t] * (1/(lambda * t)) * sum_j [ alpha[j] * y[j] * K(x_j, x_{i_t}) ]
        if margin < 1:
            alpha[i_t] += 1
    OUTPUT: alpha^{T+1}

    We implement T = epochs * n, picking a random index each iteration.

    Args:
        X (np.ndarray): Training data, shape (n_samples, n_features).
        y (np.ndarray): Labels in {-1, +1}, shape (n_samples,).
        kernel_func (callable): Kernel function k(x1, x2, **kernel_params).
        kernel_params (dict): Additional parameters for the kernel function.
        lambda_param (float): Regularization parameter λ.
        epochs (int): Number of passes (epochs) over the data (multiplied by n to get T).

    Returns:
        alpha (np.ndarray): The learned alpha coefficients of shape (n_samples,).
        T (int): The total number of iterations used (useful for predicting).
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples, dtype=float)

    # We'll do T = epochs * n_samples total updates
    T = epochs * n_samples

    for t in range(1, T + 1):
        # Pick random sample i_t
        i_t = np.random.randint(0, n_samples)

        # Compute margin = y[i_t] * (1/(lambda * t)) * sum_j [ alpha[j]*y[j]*K(X_j, X_i_t) ]
        sum_k = 0.0
        x_i_t = X[i_t]
        for j in range(n_samples):
            if alpha[j] != 0:
                sum_k += alpha[j] * y[j] * kernel_func(X[j], x_i_t, **kernel_params)

        margin = y[i_t] * (1.0 / (lambda_param * t)) * sum_k

        # If margin < 1, then alpha[i_t] += 1
        if margin < 1:
            alpha[i_t] += 1.0

    return alpha, T


def kernelized_pegasos_predict(
    X_train, y_train,
    alpha, T,
    X_test,
    kernel_func,
    kernel_params={},
    lambda_param=0.01
):
    """
    Predicts labels for X_test using the trained kernelized Pegasos SVM.

    The decision function is:
        f(x) = (1 / (lambda * T)) * sum_j [ alpha[j] * y[j] * K(x_j, x) ]
    We take the sign of f(x).

    Args:
        X_train (np.ndarray): Training data (n_train, n_features).
        y_train (np.ndarray): Training labels (n_train,).
        alpha (np.ndarray): Learned alpha array from kernelized_pegasos_train (length = n_train).
        T (int): The total number of iterations used in training.
        X_test (np.ndarray): Test data (n_test, n_features).
        kernel_func (callable): Kernel function used in training.
        kernel_params (dict): Additional parameters for the kernel function.
        lambda_param (float): The same λ used in training.

    Returns:
        predictions (np.ndarray): Predicted labels (-1 or +1) for X_test.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    predictions = np.zeros(n_test)

    factor = 1.0 / (lambda_param * T)

    for i in range(n_test):
        x_i = X_test[i]
        f_x = 0.0
        for j in range(n_train):
            if alpha[j] != 0:
                f_x += alpha[j] * y_train[j] * kernel_func(X_train[j], x_i, **kernel_params)
        f_x *= factor
        # sign
        predictions[i] = np.sign(f_x)
        if predictions[i] == 0:
            predictions[i] = 1.0  # break ties with +1

    return predictions.astype(int)


def kernelized_pegasos_classifier_func(
    train_fold, val_fold,
    features, target,
    kernel_func,
    kernel_params={},
    lambda_param=0.01,
    epochs=5
):
    """
    A classifier_func for k-fold cross-validation. Trains kernelized Pegasos on train_fold,
    then predicts on val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names to use.
        target (str): Target column name (in {-1, +1}).
        kernel_func (callable): e.g. gaussian_kernel or polynomial_kernel.
        kernel_params (dict): Additional params for the kernel function (e.g. sigma, degree, c).
        lambda_param (float): Regularization parameter λ.
        epochs (int): Number of epochs.

    Returns:
        val_preds (np.ndarray): Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values

    # Train
    alpha, T = kernelized_pegasos_train(
        X_train, y_train,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        lambda_param=lambda_param,
        epochs=epochs
    )

    # Predict
    val_preds = kernelized_pegasos_predict(
        X_train, y_train, alpha, T,
        X_val,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        lambda_param=lambda_param
    )
    return val_preds
