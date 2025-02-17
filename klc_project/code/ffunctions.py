import numpy as np
import pandas as pd

# =============================================================================
# Data Loading & Preprocessing Functions
# =============================================================================
def load_data(csv_path, **kwargs):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file.
        **kwargs: Additional keyword arguments for pd.read_csv.
        
    Returns:
        pd.DataFrame: The loaded data.
        
    Raises:
        ValueError: If reading the CSV fails.
    """
    try:
        data = pd.read_csv(csv_path, **kwargs)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    return data

def train_test_split(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into training and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        test_size (float): Fraction of data for testing.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (train_df, test_df) as DataFrames.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df))
    test_count = int(len(df) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df

def detect_outliers_zscore(df, features, z_thresh=3.0):
    """
    Detect outlier rows in specified features using z-scores.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of feature names to check.
        z_thresh (float): Z-score threshold.
        
    Returns:
        set: Row indices where any feature's z-score exceeds z_thresh.
    """
    outlier_indices = set()
    for f in features:
        std_f = df[f].std()
        if std_f == 0:
            continue
        z_scores = ((df[f] - df[f].mean()) / std_f).abs()
        outlier_indices.update(z_scores[z_scores > z_thresh].index)
    return outlier_indices

def remove_outliers(df, outlier_indices):
    """
    Remove rows from a DataFrame based on provided indices.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        outlier_indices (set): Set of row indices to remove.
        
    Returns:
        pd.DataFrame: DataFrame with the specified rows removed.
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)

def standard_scaler_fit(train_df, features):
    """
    Compute means and standard deviations for specified features.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame.
        features (list): List of feature names.
        
    Returns:
        tuple: (means, stds) dictionaries mapping feature names to mean and std.
    """
    means = {f: train_df[f].mean() for f in features}
    stds = {f: train_df[f].std() for f in features}
    return means, stds

def standard_scaler_transform(df, features, means, stds):
    """
    Scale features in the DataFrame using provided means and stds.
    
    Args:
        df (pd.DataFrame): DataFrame to transform.
        features (list): List of feature names.
        means (dict): Dictionary of feature means.
        stds (dict): Dictionary of feature standard deviations.
        
    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_copy = df.copy()
    for f in features:
        if stds.get(f, 0) != 0:
            df_copy[f] = (df_copy[f] - means[f]) / stds[f]
        else:
            df_copy[f] = df_copy[f] - means[f]
    return df_copy

def check_high_correlation(df, features, corr_threshold=0.95):
    """
    Identify pairs of features with absolute correlation exceeding a threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of feature names.
        corr_threshold (float): Correlation threshold.
        
    Returns:
        list: Tuples of (feature1, feature2, correlation value).
    """
    corr_matrix = df[features].corr().abs()
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                high_corr_pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))
    return high_corr_pairs

def k_fold_cross_validation(df, features, target, k=5, random_state=42, classifier_func=None):
    """
    Perform k-fold cross validation on a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): Feature names.
        target (str): Target column name.
        k (int): Number of folds.
        random_state (int): RNG seed.
        classifier_func (callable): Function that takes (train_fold, val_fold, features, target)
                                    and returns predicted labels for val_fold.
                                    If None, a majority-class classifier is used.
        
    Returns:
        float: Mean accuracy across folds.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df))
    fold_sizes = np.full(k, len(df) // k, dtype=int)
    fold_sizes[:len(df) % k] += 1
    current = 0
    accuracies = []
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        train_fold = df.iloc[train_indices].reset_index(drop=True)
        val_fold = df.iloc[val_indices].reset_index(drop=True)
        if classifier_func is None:
            majority_class = train_fold[target].mode().iloc[0]
            val_predictions = np.full(len(val_fold), majority_class)
        else:
            val_predictions = classifier_func(train_fold, val_fold, features, target)
        accuracy = (val_predictions == val_fold[target]).mean()
        accuracies.append(accuracy)
        current = end
    return np.mean(accuracies)

# =============================================================================
# Linear Models: Perceptron, Pegasos, Logistic Regression
# =============================================================================
def perceptron_train(train_df, features, target, epochs=5):
    """
    Train a binary Perceptron classifier (labels in {-1, +1}).
    
    Args:
        train_df (pd.DataFrame): Training data with features and target.
        features (list): Feature column names.
        target (str): Target column name.
        epochs (int): Number of training epochs.
        
    Returns:
        tuple: (theta, theta_0) where theta is the weight vector and theta_0 is the bias.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = X.shape[1]
    theta = np.zeros(d)
    theta_0 = 0.0
    rng = np.random.default_rng()
    for _ in range(epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            if y[i] * (np.dot(theta, X[i]) + theta_0) <= 0:
                theta += y[i] * X[i]
                theta_0 += y[i]
    return theta, theta_0

def perceptron_train_eta(train_df, features, target, epochs=5, eta=1.0):
    """
    Train a binary Perceptron using a constant learning rate.
    
    Args:
        train_df (pd.DataFrame): Training data.
        features (list): Feature names.
        target (str): Target column name.
        epochs (int): Number of epochs.
        eta (float): Learning rate.
        
    Returns:
        tuple: (theta, theta_0) weight vector and bias.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = X.shape[1]
    theta = np.zeros(d)
    theta_0 = 0.0
    rng = np.random.default_rng()
    for _ in range(epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            if y[i] * (np.dot(theta, X[i]) + theta_0) <= 0:
                theta += eta * y[i] * X[i]
                theta_0 += eta * y[i]
    return theta, theta_0

def perceptron_predict(df, features, theta, theta_0):
    """
    Predict labels using a trained Perceptron model.
    
    Args:
        df (pd.DataFrame): Data for prediction.
        features (list): Feature names.
        theta (np.ndarray): Weight vector.
        theta_0 (float): Bias term.
        
    Returns:
        np.ndarray: Predicted labels (1 or -1).
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores >= 0, 1, -1)

def perceptron_classifier_func(train_fold, val_fold, features, target, epochs=5):
    """
    Classifier function using Perceptron for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name.
        epochs (int): Number of epochs.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    theta, theta_0 = perceptron_train(train_fold, features, target, epochs)
    return perceptron_predict(val_fold, features, theta, theta_0)

def perceptron_classifier_func_eta(train_fold, val_fold, features, target, epochs=5, eta=1.0):
    """
    Classifier function using Perceptron with learning rate eta for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name.
        epochs (int): Number of epochs.
        eta (float): Learning rate.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    theta, theta_0 = perceptron_train_eta(train_fold, features, target, epochs, eta)
    return perceptron_predict(val_fold, features, theta, theta_0)

def pegasos_train(train_df, features, target, lambda_param=0.01, epochs=5):
    """
    Train a linear Pegasos SVM using stochastic gradient descent.
    
    Args:
        train_df (pd.DataFrame): Training data.
        features (list): Feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        
    Returns:
        tuple: (theta, theta_0) weight vector and bias.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = X.shape[1]
    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1
    rng = np.random.default_rng()
    for _ in range(epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            eta_t = 1 / (lambda_param * t)
            t += 1
            if y[i] * (np.dot(theta, X[i]) + theta_0) < 1:
                theta = (1 - eta_t * lambda_param) * theta + eta_t * y[i] * X[i]
                theta_0 += eta_t * y[i]
            else:
                theta = (1 - eta_t * lambda_param) * theta
    return theta, theta_0

def pegasos_predict(df, features, theta, theta_0):
    """
    Predict labels using a trained Pegasos SVM model.
    
    Args:
        df (pd.DataFrame): Data for prediction.
        features (list): Feature names.
        theta (np.ndarray): Weight vector.
        theta_0 (float): Bias term.
        
    Returns:
        np.ndarray: Predicted labels (1 or -1).
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores >= 0, 1, -1)

def pegasos_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5):
    """
    Classifier function using Pegasos SVM for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    theta, theta_0 = pegasos_train(train_fold, features, target, lambda_param, epochs)
    return pegasos_predict(val_fold, features, theta, theta_0)

def logistic_regression_train(train_df, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    Train a logistic regression model using SGD with a decaying learning rate.
    
    Args:
        train_df (pd.DataFrame): Training data.
        features (list): Feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        eta (float): Initial learning rate.
        
    Returns:
        tuple: (theta, theta_0) weight vector and bias.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = X.shape[1]
    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1
    rng = np.random.default_rng()
    for _ in range(epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            eta_t = eta / np.sqrt(t)
            t += 1
            margin = y[i] * (np.dot(theta, X[i]) + theta_0)
            margin = np.clip(margin, -50, 50)  # ensure numerical stability
            exp_margin = np.exp(margin)
            grad_theta = lambda_param * theta - (y[i] * X[i]) / (1 + exp_margin)
            grad_theta_0 = -y[i] / (1 + exp_margin)
            theta -= eta_t * grad_theta
            theta_0 -= eta_t * grad_theta_0
    return theta, theta_0

def logistic_regression_predict(df, features, theta, theta_0):
    """
    Predict labels using a trained logistic regression model.
    
    Args:
        df (pd.DataFrame): Data for prediction.
        features (list): Feature names.
        theta (np.ndarray): Weight vector.
        theta_0 (float): Bias term.
        
    Returns:
        np.ndarray: Predicted labels (1 or -1).
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores >= 0, 1, -1)

def logistic_regression_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    Classifier function using logistic regression for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        eta (float): Initial learning rate.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    theta, theta_0 = logistic_regression_train(train_fold, features, target, lambda_param, epochs, eta)
    return logistic_regression_predict(val_fold, features, theta, theta_0)

# =============================================================================
# Polynomial Feature Expansion (Degree 2 Only)
# =============================================================================
def polynomial_feature_expansion(df, features, degree=2, include_bias=False):
    """
    Expand the specified features to include squared and interaction terms (degree=2 only).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): Feature column names.
        degree (int): Degree of expansion (must be 2).
        include_bias (bool): If True, include an additional bias column.
        
    Returns:
        pd.DataFrame: DataFrame with expanded polynomial features.
    """
    if degree != 2:
        raise ValueError("Only degree=2 expansion is supported.")
    X_orig = df[features].copy()
    poly_data = {}
    if include_bias:
        poly_data['bias'] = np.ones(len(X_orig))
    for f in features:
        poly_data[f] = X_orig[f]
        poly_data[f"{f}^2"] = X_orig[f] ** 2
    n_feats = len(features)
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            f_i, f_j = features[i], features[j]
            poly_data[f"{f_i}*{f_j}"] = X_orig[f_i] * X_orig[f_j]
    expanded_df = pd.DataFrame(poly_data, index=df.index)
    other_cols = [col for col in df.columns if col not in features]
    if other_cols:
        expanded_df = pd.concat([expanded_df, df[other_cols]], axis=1)
    return expanded_df

# =============================================================================
# Kernel Functions & Kernelized Models
# =============================================================================
def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute the Gaussian (RBF) kernel between two vectors.
    
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        sigma (float): Bandwidth parameter (must be > 0).
        
    Returns:
        float: Computed Gaussian kernel value.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    diff = x - y
    sq_dist = np.dot(diff, diff)
    return np.exp(-sq_dist / (2 * sigma**2))

def polynomial_kernel(x, y, degree=2, c=1.0):
    """
    Compute the polynomial kernel between two vectors.
    
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        degree (int): Degree of the polynomial.
        c (float): Coefficient term.
        
    Returns:
        float: Computed polynomial kernel value.
    """
    return (np.dot(x, y) + c) ** degree

def kernelized_perceptron_train(X, y, kernel_func, kernel_params={}, epochs=5):
    """
    Train a kernelized Perceptron using the provided kernel function.
    
    Args:
        X (np.ndarray): Training data of shape (n_samples, n_features).
        y (np.ndarray): Labels (n_samples,).
        kernel_func (callable): Kernel function (e.g., gaussian_kernel).
        kernel_params (dict): Additional parameters for kernel_func.
        epochs (int): Number of training epochs.
        
    Returns:
        np.ndarray: Alpha coefficients for each training sample.
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples)
    rng = np.random.default_rng()
    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        for i in indices:
            f_i = 0.0
            for j in range(n_samples):
                if alpha[j] != 0:
                    f_i += alpha[j] * y[j] * kernel_func(X[j], X[i], **kernel_params)
            if np.sign(f_i) != y[i]:
                alpha[i] += 1.0
    return alpha

def kernelized_perceptron_predict(X_train, y_train, alpha, X_test, kernel_func, kernel_params={}):
    """
    Predict labels using a kernelized Perceptron.
    
    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        alpha (np.ndarray): Learned alpha coefficients.
        X_test (np.ndarray): Test data.
        kernel_func (callable): Kernel function.
        kernel_params (dict): Additional parameters for kernel_func.
        
    Returns:
        np.ndarray: Predicted labels for X_test.
    """
    n_train = len(y_train)
    predictions = []
    for x in X_test:
        f_x = 0.0
        for j in range(n_train):
            if alpha[j] != 0:
                f_x += alpha[j] * y_train[j] * kernel_func(X_train[j], x, **kernel_params)
        pred = np.sign(f_x)
        if pred == 0:
            pred = 1.0
        predictions.append(int(pred))
    return np.array(predictions)

def kernelized_perceptron_classifier_func(train_fold, val_fold, features, target, kernel_func, kernel_params={}, epochs=5):
    """
    Classifier function using a kernelized Perceptron for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold.
        val_fold (pd.DataFrame): Validation fold.
        features (list): Feature names.
        target (str): Target column name (labels in {-1, +1}).
        kernel_func (callable): Kernel function (e.g., gaussian_kernel).
        kernel_params (dict): Additional parameters for kernel_func.
        epochs (int): Number of epochs.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values
    alpha = kernelized_perceptron_train(X_train, y_train, kernel_func, kernel_params, epochs)
    return kernelized_perceptron_predict(X_train, y_train, alpha, X_val, kernel_func, kernel_params)

def kernelized_pegasos_train(X, y, kernel_func, kernel_params={}, lambda_param=0.01, epochs=5):
    """
    Train a kernelized Pegasos SVM.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Training labels.
        kernel_func (callable): Kernel function.
        kernel_params (dict): Additional parameters for kernel_func.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        
    Returns:
        tuple: (alpha, T) where alpha are the coefficients and T is the total iteration count.
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples)
    T = epochs * n_samples
    rng = np.random.default_rng()
    for t in range(1, T + 1):
        i_t = rng.integers(0, n_samples)
        sum_k = 0.0
        for j in range(n_samples):
            if alpha[j] != 0:
                sum_k += alpha[j] * y[j] * kernel_func(X[j], X[i_t], **kernel_params)
        margin = y[i_t] * (1 / (lambda_param * t)) * sum_k
        if margin < 1:
            alpha[i_t] += 1.0
    return alpha, T

def kernelized_pegasos_predict(X_train, y_train, alpha, T, X_test, kernel_func, kernel_params={}, lambda_param=0.01):
    """
    Predict labels using a kernelized Pegasos SVM.
    
    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        alpha (np.ndarray): Learned alpha coefficients.
        T (int): Total iterations used in training.
        X_test (np.ndarray): Test data.
        kernel_func (callable): Kernel function.
        kernel_params (dict): Additional parameters for kernel_func.
        lambda_param (float): Regularization parameter.
        
    Returns:
        np.ndarray: Predicted labels for X_test.
    """
    factor = 1 / (lambda_param * T)
    n_train = len(y_train)
    predictions = []
    for x in X_test:
        f_x = 0.0
        for j in range(n_train):
            if alpha[j] != 0:
                f_x += alpha[j] * y_train[j] * kernel_func(X_train[j], x, **kernel_params)
        f_x *= factor
        pred = np.sign(f_x)
        if pred == 0:
            pred = 1.0
        predictions.append(int(pred))
    return np.array(predictions)

def kernelized_pegasos_classifier_func(train_fold, val_fold, features, target, kernel_func, kernel_params={}, lambda_param=0.01, epochs=5):
    """
    Classifier function using a kernelized Pegasos SVM for k-fold cross-validation.
    
    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): Feature names.
        target (str): Target column name (labels in {-1, +1}).
        kernel_func (callable): Kernel function (e.g., gaussian_kernel).
        kernel_params (dict): Additional parameters for kernel_func.
        lambda_param (float): Regularization parameter.
        epochs (int): Number of epochs.
        
    Returns:
        np.ndarray: Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values
    alpha, T = kernelized_pegasos_train(X_train, y_train, kernel_func, kernel_params, lambda_param, epochs)
    return kernelized_pegasos_predict(X_train, y_train, alpha, T, X_val, kernel_func, kernel_params, lambda_param)
