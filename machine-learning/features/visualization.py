import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_matrix(df, output_path):
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{output_path}/correlation_matrix.png')
    plt.close()

def plot_model_results(results_df, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='classificador', y='accuracy', data=results_df)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.savefig(f'{output_path}/model_accuracy_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='classificador', y='f1-score', data=results_df)
    plt.title('Model F1-Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('F1-Score')
    plt.savefig(f'{output_path}/model_f1_score_comparison.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, y_test, pred):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(pred))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/confusion_matrix.png')
    plt.close()

def main():
    df = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/features.xlsx")
    output_path = "C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results"

    # Plot distribution for each feature
    for feature in df.columns:
        if feature != 'label':
            plot_feature_distribution(df, feature, output_path)

    # Plot correlation matrix
    plot_correlation_matrix(df, output_path)

    # Load original and augmented data for comparison
    original_df = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/original_features.xlsx")
    augmented_df = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/augmented_features.xlsx")

    # Compare original and augmented data for each feature
    for feature in original_df.columns:
        if feature != 'label':
            compare_original_augmented(original_df, augmented_df, feature, output_path)

    # Plot model results
    results_df = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/metrics.xlsx")
    plot_model_results(results_df, output_path)