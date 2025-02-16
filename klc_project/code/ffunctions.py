import pandas as pd
import numpy as np

def load_data(csv_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(csv_path)
    return data

def train_test_split(df, test_size=0.2, random_state=42):
    """
    Split a pandas DataFrame into training and testing sets without data leakage.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test set (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): Training set.
            - test_df (pd.DataFrame): Testing set.
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
    Detect outliers in the specified features of a DataFrame using a z-score threshold.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to check for outliers.
        features (list): A list of column names in the DataFrame to check for outliers.
        z_thresh (float): The z-score threshold above which a data point is considered an outlier (default: 3.0).

    Returns:
        set: A set of row indices corresponding to detected outliers in the DataFrame.
    """
    outlier_indices = set()
    for f in features:
        mean_f = df[f].mean()
        std_f = df[f].std()
        if std_f == 0:
            continue  # Skip features with zero variance
        z_scores = ((df[f] - mean_f) / std_f).abs()
        f_outliers = z_scores[z_scores > z_thresh].index
        outlier_indices.update(f_outliers)
    return outlier_indices

def remove_outliers(df, outlier_indices):
    """
    Remove rows from a DataFrame based on a set of outlier indices.

    Args:
        df (pd.DataFrame): The input DataFrame from which outliers should be removed.
        outlier_indices (set): A set of row indices corresponding to the outliers to be removed.

    Returns:
        pd.DataFrame: A new DataFrame with the specified outliers removed, with the index reset.
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)

def standard_scaler_fit(train_df, features):
    """
    Compute the mean and standard deviation for each feature in the training data.

    Args:
        train_df (pd.DataFrame): The training DataFrame containing the features to scale.
        features (list): A list of column names (strings) representing the features to scale.

    Returns:
        means (dict): A dictionary where the keys are feature names and the values are their means.
        stds (dict): A dictionary where the keys are feature names and the values are their standard deviations.
    """
    means = {}
    stds = {}
    for f in features:
        means[f] = train_df[f].mean()
        stds[f] = train_df[f].std()
    return means, stds

def standard_scaler_transform(df, features, means, stds):
    """
    Scale the features in a DataFrame using precomputed means and standard deviations.

    This method ensures no data leakage by applying the statistics calculated from the training data only.

    Args:
        df (pd.DataFrame): The DataFrame containing the features to scale.
        features (list): A list of column names (strings) representing the features to scale.
        means (dict): A dictionary where the keys are feature names and the values are their precomputed means.
        stds (dict): A dictionary where the keys are feature names and the values are their precomputed standard deviations.

    Returns:
        pd.DataFrame: A new DataFrame with the specified features scaled. The original DataFrame remains unchanged.
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
    Identify pairs of highly correlated features in a DataFrame.

    This function calculates the correlation matrix for the specified features and
    identifies all pairs of features whose absolute correlation exceeds the given threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the features to check.
        features (list): A list of column names (strings) representing the features to evaluate.
        corr_threshold (float): The correlation threshold above which feature pairs are considered highly correlated.

    Returns:
        list of tuple: A list of tuples where each tuple contains:
                       - feature1 (str): The name of the first feature.
                       - feature2 (str): The name of the second feature.
                       - corr_value (float): The absolute correlation value between the two features.
    """
    corr_matrix = df[features].corr().abs()
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                f1 = features[i]
                f2 = features[j]
                high_corr_pairs.append((f1, f2, corr_matrix.iloc[i, j]))
    return high_corr_pairs

def k_fold_cross_validation(df, features, target, k=5, random_state=42, classifier_func=None):
    """
    Perform K-fold cross-validation on a dataset and compute the mean accuracy.

    This function splits the dataset into K folds, iteratively using one fold for validation
    and the remaining folds for training. It evaluates the accuracy of a classifier provided
    through `classifier_func`.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset to be evaluated.
        features (list): List of column names (strings) representing the features to use for training.
        target (str): The name of the target column in the DataFrame.
        k (int): The number of folds to use for cross-validation (default: 5).
        random_state (int): Random seed for reproducibility (default: 42).
        classifier_func (callable): A function that trains and predicts using the classifier.
                                    Should accept arguments (train_df, val_df, features, target) and return predictions.

    Returns:
        float: The mean accuracy across all K folds.
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

# ---------------------------------------------
# Linear Models (Perceptron, Pegasos, Logistic)
# ---------------------------------------------

def perceptron_train(train_df, features, target, epochs=5):
    """
    Train a binary Perceptron classifier using the training dataset.

    The Perceptron algorithm updates weights iteratively based on misclassified examples.
    Labels in the target column must be -1 or +1.

    Args:
        train_df (pd.DataFrame): The training dataset containing features and target labels.
        features (list): List of column names (strings) representing the features to use for training.
        target (str): The name of the target column in the DataFrame.
        epochs (int): The number of passes over the training data (default: 5).

    Returns:
        tuple:
            - theta (np.ndarray): The weight vector for the features (shape: [n_features]).
            - theta_0 (float): The bias term.
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
            if y_i * (np.dot(theta, x_i) + theta_0) <= 0:
                theta = theta + eta * y_i * x_i
                theta_0 = theta_0 + eta * y_i

    return theta, theta_0

def perceptron_predict(df, features, theta, theta_0):
    """
    Predict class labels (-1 or +1) using the trained Perceptron model.

    The prediction is based on the sign of the linear combination of features and the model's parameters.

    Args:
        df (pd.DataFrame): The dataset containing the features for prediction.
        features (list): List of column names (strings) representing the features used for prediction.
        theta (np.ndarray): The weight vector for the features (shape: [n_features]).
        theta_0 (float): The bias term of the trained model.

    Returns:
        np.ndarray: Predicted class labels (-1 or +1) for each instance in the dataset (shape: [n_samples]).
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    predictions = np.where(scores > 0, 1, -1)
    return predictions

def perceptron_classifier_func(train_fold, val_fold, features, target, epochs=5):
    """
    Trains a Perceptron model on the training fold and predicts labels for the validation fold.

    Specifically designed for use in K-fold cross-validation to assess Perceptron performance.

    Args:
        train_fold (pd.DataFrame): The training fold containing features and target labels.
        val_fold (pd.DataFrame): The validation fold for which predictions will be made.
        features (list): List of feature column names used for training and prediction.
        target (str): Column name representing the target variable.
        epochs (int): Number of training epochs for the Perceptron.

    Returns:
        np.ndarray: Predicted class labels (-1 or +1) for each instance in the validation fold.
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
    Trains a linear Pegasos SVM with hinge loss.

    Implements the Pegasos algorithm for training a linear support vector machine (SVM) with 
    regularization. The training labels must be in the range {-1, +1}.

    Args:
        train_df (pd.DataFrame): Training data containing features and target labels.
        features (list): List of feature column names used for training.
        target (str): Column name representing the target variable.
        lambda_param (float): Regularization parameter for controlling overfitting.
        epochs (int): Number of training epochs (passes through the dataset).

    Returns:
        tuple:
            - theta (np.ndarray): Learned weight vector for the features.
            - theta_0 (float): Learned bias term.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = len(features)

    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1  # Global iteration counter

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            eta_t = 1 / (lambda_param * t)
            t += 1

            if y_i * (np.dot(theta, x_i) + theta_0) < 1:
                theta = (1 - eta_t * lambda_param) * theta + eta_t * y_i * x_i
                theta_0 += eta_t * y_i
            else:
                theta = (1 - eta_t * lambda_param) * theta

    return theta, theta_0

def pegasos_predict(df, features, theta, theta_0):
    """
    Predicts labels using a trained linear Pegasos SVM model.

    Computes the decision scores for the input features using the learned weight vector (`theta`)
    and bias term (`theta_0`) and predicts the class labels as -1 or +1.

    Args:
        df (pd.DataFrame): Input data containing the features to predict labels for.
        features (list): List of feature column names used for prediction.
        theta (np.ndarray): Learned weight vector for the features.
        theta_0 (float): Learned bias term.

    Returns:
        np.ndarray: Predicted labels (-1 or +1) for the input data.
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores > 0, 1, -1)

def pegasos_classifier_func(train_fold, val_fold, features, target, lambda_param=0.01, epochs=5):
    """
    Trains a Pegasos SVM on the training fold and predicts labels on the validation fold.

    This function is designed for use in K-fold cross-validation. It trains the Pegasos SVM model
    on the `train_fold` using the specified features and target, and then predicts labels for
    the `val_fold` using the trained model.

    Args:
        train_fold (pd.DataFrame): Training data for the current fold.
        val_fold (pd.DataFrame): Validation data for the current fold.
        features (list): List of feature column names used for training and prediction.
        target (str): Name of the target column in the dataset.
        lambda_param (float, optional): Regularization parameter for Pegasos SVM. Defaults to 0.01.
        epochs (int, optional): Number of training epochs. Defaults to 5.

    Returns:
        np.ndarray: Predicted labels (-1 or +1) for the validation fold.
    """
    theta, theta_0 = pegasos_train(
        train_fold, features, target, lambda_param=lambda_param, epochs=epochs
    )
    return pegasos_predict(val_fold, features, theta, theta_0)

def logistic_regression_train(train_df, features, target, lambda_param=0.01, epochs=5, eta=1.0):
    """
    Trains a regularized logistic regression model using stochastic gradient descent (SGD).

    The model is trained to minimize the logistic loss with L2 regularization. The labels in 
    the target column must be encoded as -1 and +1. A learning rate schedule is applied during 
    the training process.

    Args:
        train_df (pd.DataFrame): Training data containing features and target.
        features (list): List of feature column names used for training.
        target (str): Name of the target column in the dataset.
        lambda_param (float, optional): Regularization parameter. Defaults to 0.01.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        eta (float, optional): Initial learning rate. Defaults to 1.0.

    Returns:
        tuple: 
            - theta (np.ndarray): Weight vector for the features.
            - theta_0 (float): Bias term for the model.
    """
    X = train_df[features].values
    y = train_df[target].values
    d = len(features)

    theta = np.zeros(d)
    theta_0 = 0.0
    t = 1  # Global iteration counter for learning rate schedule

    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]
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
    Predicts binary labels (-1 or +1) for a dataset using a trained logistic regression model.

    The prediction is based on the sign of the linear score computed as the dot product of 
    feature values and the weight vector, plus the bias term.

    Args:
        df (pd.DataFrame): DataFrame containing the features for prediction.
        features (list): List of feature column names used for prediction.
        theta (np.ndarray): Weight vector of the trained logistic regression model.
        theta_0 (float): Bias term of the trained logistic regression model.

    Returns:
        np.ndarray: Predicted labels (-1 or +1) for each row in the dataset.
    """
    X = df[features].values
    scores = np.dot(X, theta) + theta_0
    return np.where(scores > 0, 1, -1)

def logistic_regression_classifier_func(train_fold, val_fold, features, target,
                                        lambda_param=0.01, epochs=5, eta=1.0):
    """
    Trains a logistic regression model on the training fold and predicts labels for the validation fold.

    This function is designed to be used within k-fold cross-validation, where it trains a logistic
    regression model on the training fold using specified hyperparameters and evaluates it on the validation fold.

    Args:
        train_fold (pd.DataFrame): Training fold containing features and target column.
        val_fold (pd.DataFrame): Validation fold containing features and target column.
        features (list): List of feature column names.
        target (str): Name of the target column (labels in {-1, +1}).
        lambda_param (float): Regularization parameter (default: 0.01).
        epochs (int): Number of epochs for training (default: 5).
        eta (float): Initial learning rate for SGD (default: 1.0).

    Returns:
        np.ndarray: Predicted labels (-1 or +1) for the validation fold.
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

    X_original = df[features].copy()
    poly_data = {}

    if include_bias:
        poly_data['bias'] = np.ones(len(X_original))

    for f in features:
        poly_data[f] = X_original[f]
    for f in features:
        new_col_name = f"{f}^2"
        poly_data[new_col_name] = X_original[f] ** 2

    num_feats = len(features)
    for i in range(num_feats):
        for j in range(i+1, num_feats):
            f_i = features[i]
            f_j = features[j]
            new_col_name = f"{f_i}*{f_j}"
            poly_data[new_col_name] = X_original[f_i] * X_original[f_j]

    expanded_df = pd.DataFrame(poly_data, index=df.index)
    other_cols = [col for col in df.columns if col not in features]
    return pd.concat([expanded_df, df[other_cols]], axis=1)

#------------------
# Kernel Functions
#------------------

def gaussian_kernel(x, y, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel between two vectors.

    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        sigma (float): Kernel bandwidth parameter (default: 1.0).

    Returns:
        float: The computed Gaussian kernel value.
    """
    diff = x - y
    sq_dist = np.dot(diff, diff)
    return np.exp(-sq_dist / (2 * sigma**2))

def polynomial_kernel(x, y, degree=2, c=1.0):
    """
    Computes the Polynomial kernel between two vectors.

    k(x, y) = (x^T y + c)^degree

    Args:
        x (np.ndarray): First input vector.
        y (np.ndarray): Second input vector.
        degree (int): Degree of the polynomial (default: 2).
        c (float): Coefficient for the constant term (default: 1.0).

    Returns:
        float: The computed Polynomial kernel value.
    """
    return (np.dot(x, y) + c) ** degree

# Helper for kernel precomputation
def compute_kernel_matrix(X1, X2, kernel_func, kernel_params={}):
    """
    Computes the kernel matrix between two datasets.

    Args:
        X1 (np.ndarray): First dataset of shape (n_samples1, n_features).
        X2 (np.ndarray): Second dataset of shape (n_samples2, n_features).
        kernel_func (callable): Kernel function.
        kernel_params (dict): Additional parameters for the kernel function.

    Returns:
        np.ndarray: A matrix of shape (n_samples1, n_samples2) where entry (i, j) = kernel_func(X1[i], X2[j]).
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kernel_params)
    return K

#----------------------
# Kernelized Perceptron
#----------------------

def kernelized_perceptron_train(X, y, kernel_func, kernel_params={}, epochs=5, cache_kernel=False):
    """
    Trains a Kernelized Perceptron on data (X, y) using the given kernel_func.
    y should be in {-1, +1}.

    Args:
        X (np.ndarray): Training data of shape (n_samples, n_features).
        y (np.ndarray): Labels of shape (n_samples,).
        kernel_func (callable): A function k(x, x') returning a scalar kernel value.
        kernel_params (dict): Additional params for kernel_func (e.g., sigma, degree, c, etc.).
        epochs (int): Number of passes over the data.
        cache_kernel (bool): If True, precompute the kernel matrix to speed up computations.

    Returns:
        alpha (np.ndarray): Coefficients for each training example (length = n_samples).
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples)
    if cache_kernel:
        K = compute_kernel_matrix(X, X, kernel_func, kernel_params)
    for _ in range(epochs):
        for i in range(n_samples):
            f_i = 0.0
            if cache_kernel:
                for j in range(n_samples):
                    if alpha[j] != 0:
                        f_i += alpha[j] * y[j] * K[j, i]
            else:
                for j in range(n_samples):
                    if alpha[j] != 0:
                        f_i += alpha[j] * y[j] * kernel_func(X[j], X[i], **kernel_params)
            if np.sign(f_i) != y[i]:
                alpha[i] += 1.0
    return alpha

def kernelized_perceptron_predict(X_train, y_train, alpha, X_test, kernel_func, kernel_params={}, cache_kernel=False):
    """
    Predict labels for X_test using the trained Kernelized Perceptron.

    Args:
        X_train (np.ndarray): Training data (n_train, n_features).
        y_train (np.ndarray): Training labels (n_train,).
        alpha (np.ndarray): Coefficients from kernelized_perceptron_train (n_train,).
        X_test (np.ndarray): Test data (n_test, n_features).
        kernel_func (callable): Same kernel function used during training.
        kernel_params (dict): Params for kernel_func.
        cache_kernel (bool): If True, precompute the cross-kernel matrix between X_train and X_test.

    Returns:
        predictions (np.ndarray): Predicted labels (-1 or +1) for the test set.
    """
    n_train = len(y_train)
    n_test = len(X_test)
    predictions = np.zeros(n_test)
    
    if cache_kernel:
        K_test = compute_kernel_matrix(X_train, X_test, kernel_func, kernel_params)
    
    for i in range(n_test):
        f_i = 0.0
        if cache_kernel:
            for j in range(n_train):
                if alpha[j] != 0:
                    f_i += alpha[j] * y_train[j] * K_test[j, i]
        else:
            for j in range(n_train):
                if alpha[j] != 0:
                    f_i += alpha[j] * y_train[j] * kernel_func(X_train[j], X_test[i], **kernel_params)
        predictions[i] = np.sign(f_i)
        if predictions[i] == 0:
            predictions[i] = 1.0
    return predictions.astype(int)

def kernelized_perceptron_classifier_func(train_fold, val_fold, features, target,
                                            kernel_func, kernel_params={}, epochs=5, cache_kernel=False):
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
        cache_kernel (bool): If True, precompute the kernel matrix to speed up computations.

    Returns:
        val_preds (np.ndarray): Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values

    alpha = kernelized_perceptron_train(
        X_train, y_train,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        epochs=epochs,
        cache_kernel=cache_kernel
    )

    val_preds = kernelized_perceptron_predict(
        X_train, y_train,
        alpha,
        X_val,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        cache_kernel=cache_kernel
    )
    return val_preds

#----------------------
# Kernelized Pegasos SVM
#----------------------

def kernelized_pegasos_train(X, y, kernel_func, kernel_params={}, lambda_param=0.01, epochs=5, cache_kernel=False):
    """
    Trains a kernelized Pegasos SVM on (X, y) using the provided kernel_func (e.g. Gaussian, Polynomial).

    Args:
        X (np.ndarray): Training data, shape (n_samples, n_features).
        y (np.ndarray): Labels in {-1, +1}, shape (n_samples,).
        kernel_func (callable): Kernel function k(x1, x2, **kernel_params).
        kernel_params (dict): Additional parameters for the kernel function.
        lambda_param (float): Regularization parameter λ.
        epochs (int): Number of passes (epochs) over the data.
        cache_kernel (bool): If True, precompute the kernel matrix for training.

    Returns:
        alpha (np.ndarray): The learned alpha coefficients of shape (n_samples,).
        T (int): The total number of iterations used (useful for predicting).
    """
    n_samples = len(y)
    alpha = np.zeros(n_samples, dtype=float)
    T = epochs * n_samples
    if cache_kernel:
        K = compute_kernel_matrix(X, X, kernel_func, kernel_params)
    for t in range(1, T + 1):
        i_t = np.random.randint(0, n_samples)
        sum_k = 0.0
        if cache_kernel:
            for j in range(n_samples):
                if alpha[j] != 0:
                    sum_k += alpha[j] * y[j] * K[j, i_t]
        else:
            for j in range(n_samples):
                if alpha[j] != 0:
                    sum_k += alpha[j] * y[j] * kernel_func(X[j], X[i_t], **kernel_params)
        margin = y[i_t] * (1.0 / (lambda_param * t)) * sum_k
        if margin < 1:
            alpha[i_t] += 1.0
    return alpha, T

def kernelized_pegasos_predict(X_train, y_train, alpha, T, X_test, kernel_func, kernel_params={}, lambda_param=0.01, cache_kernel=False):
    """
    Predicts labels for X_test using the trained kernelized Pegasos SVM.

    Args:
        X_train (np.ndarray): Training data (n_train, n_features).
        y_train (np.ndarray): Training labels (n_train,).
        alpha (np.ndarray): Learned alpha array from kernelized_pegasos_train (length = n_train).
        T (int): The total number of iterations used in training.
        X_test (np.ndarray): Test data (n_test, n_features).
        kernel_func (callable): Kernel function used in training.
        kernel_params (dict): Additional parameters for the kernel function.
        lambda_param (float): The same λ used in training.
        cache_kernel (bool): If True, precompute the cross-kernel matrix between training and test data.

    Returns:
        predictions (np.ndarray): Predicted labels (-1 or +1) for X_test.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    predictions = np.zeros(n_test)
    factor = 1.0 / (lambda_param * T)
    
    if cache_kernel:
        K_test = compute_kernel_matrix(X_train, X_test, kernel_func, kernel_params)
    
    for i in range(n_test):
        f_x = 0.0
        if cache_kernel:
            for j in range(n_train):
                if alpha[j] != 0:
                    f_x += alpha[j] * y_train[j] * K_test[j, i]
        else:
            for j in range(n_train):
                if alpha[j] != 0:
                    f_x += alpha[j] * y_train[j] * kernel_func(X_train[j], X_test[i], **kernel_params)
        f_x *= factor
        predictions[i] = np.sign(f_x)
        if predictions[i] == 0:
            predictions[i] = 1.0
    return predictions.astype(int)

def kernelized_pegasos_classifier_func(train_fold, val_fold, features, target,
                                        kernel_func, kernel_params={}, lambda_param=0.01, epochs=5, cache_kernel=False):
    """
    A classifier_func for k_fold_cross_validation. Trains kernelized Pegasos on train_fold,
    then predicts on val_fold.

    Args:
        train_fold (pd.DataFrame): Training fold data.
        val_fold (pd.DataFrame): Validation fold data.
        features (list): List of feature names to use.
        target (str): Target column name (in {-1, +1}).
        kernel_func (callable): e.g. gaussian_kernel or polynomial_kernel.
        kernel_params (dict): Additional params for the kernel function.
        lambda_param (float): Regularization parameter λ.
        epochs (int): Number of epochs.
        cache_kernel (bool): If True, precompute kernel matrices to speed up computations.

    Returns:
        val_preds (np.ndarray): Predicted labels for val_fold.
    """
    X_train = train_fold[features].values
    y_train = train_fold[target].values
    X_val = val_fold[features].values

    alpha, T = kernelized_pegasos_train(
        X_train, y_train,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        lambda_param=lambda_param,
        epochs=epochs,
        cache_kernel=cache_kernel
    )

    val_preds = kernelized_pegasos_predict(
        X_train, y_train, alpha, T,
        X_val,
        kernel_func=kernel_func,
        kernel_params=kernel_params,
        lambda_param=lambda_param,
        cache_kernel=cache_kernel
    )
    return val_preds
