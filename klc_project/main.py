# main.py

import numpy as np
from my_code import (
    DataManager,
    Perceptron,
    PegasosSVM,
    PegasosLogistic,
    KernelPerceptron,
    KernelPegasosSVM,
    KernelFunctions,
    DataVisualizer
)

def confusion_matrix_binary(y_true, y_pred, pos_label=1, neg_label=-1):
    """
    Computes a binary confusion matrix [ [TN, FP],
                                        [FN, TP] ]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    tn = np.sum((y_true == neg_label) & (y_pred == neg_label))
    fp = np.sum((y_true == neg_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred == neg_label))

    return np.array([[tn, fp],
                     [fn, tp]])

def main():
    # 1. Load data
    csv_path = "dataset.csv"
    dm = DataManager(csv_path)
    df = dm.load_data()
    print("Data loaded. Shape:", df.shape)

    # 2. Visualize correlations
    DataVisualizer.correlation_heatmap(df, title="Dataset Correlation")

    # 3. Split data
    X_train, y_train, X_val, y_val, X_test, y_test = dm.train_val_test_split()
    print("Train size:", X_train.shape[0])
    print("Validation size:", X_val.shape[0])
    print("Test size:", X_test.shape[0])

    # 4. Scale data
    dm.standard_scale_fit(X_train)
    X_train_scaled = dm.standard_scale_transform(X_train)
    X_val_scaled   = dm.standard_scale_transform(X_val)
    X_test_scaled  = dm.standard_scale_transform(X_test)

    # 5. Train a Perceptron
    percep = Perceptron(epochs=5, eta=1.0)
    percep.fit(X_train_scaled, y_train)
    y_val_pred = percep.predict(X_val_scaled)
    acc_val = np.mean(y_val == y_val_pred)
    print(f"Perceptron validation accuracy: {acc_val:.4f}")

    # 6. Evaluate on Test
    y_test_pred = percep.predict(X_test_scaled)
    acc_test = np.mean(y_test == y_test_pred)
    print(f"Perceptron test accuracy: {acc_test:.4f}")

    # Confusion matrix
    cm = confusion_matrix_binary(y_test, y_test_pred, pos_label=1, neg_label=-1)
    DataVisualizer.confusion_matrix_plot(cm, labels=("Class -1","Class +1"), title="Perceptron Confusion Matrix")

    # 7. Example with Kernel Pegasos SVM (RBF kernel)
    kernel_svm = KernelPegasosSVM(
        kernel_func=KernelFunctions.rbf_kernel,
        lam=1e-4,
        max_iters=5000,
        gamma=0.05
    )
    kernel_svm.fit(X_train_scaled, y_train)
    y_test_pred_k = kernel_svm.predict(X_test_scaled)
    acc_test_k = np.mean(y_test == y_test_pred_k)
    print(f"Kernel Pegasos SVM (RBF) test accuracy: {acc_test_k:.4f}")

if __name__ == "__main__":
    main()
