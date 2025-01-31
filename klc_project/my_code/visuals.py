# my_code/visuals.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

class DataVisualizer:
    """
    A collection of static methods for data visualization.
    """

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, title="Correlation Heatmap"):
        plt.figure(figsize=(8, 6))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def scatter_matrix(df: pd.DataFrame, cols=None, color_col=None):
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        fig = px.scatter_matrix(df, dimensions=cols, color=color_col)
        fig.update_traces(
            diagonal_visible=True,
            showupperhalf=True,
            showlowerhalf=True
        )
        fig.show()

    @staticmethod
    def distribution_plots(df: pd.DataFrame, cols=None):
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        num_cols = len(cols)
        plt.figure(figsize=(10, 4 * num_cols))

        for i, col in enumerate(cols):
            plt.subplot(num_cols, 2, 2*i + 1)
            sns.histplot(data=df, x=col, kde=True, color='skyblue')
            plt.title(f"Distribution of {col}")

            plt.subplot(num_cols, 2, 2*i + 2)
            sns.boxplot(data=df, x=col, color='lightgreen')
            plt.title(f"Box Plot of {col}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def confusion_matrix_plot(cm, labels=("Negative", "Positive"), title="Confusion Matrix"):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()
