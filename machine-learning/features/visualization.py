import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_distribution(df, feature, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(f'{output_path}/{feature}_distribution.png')
    plt.close()

def plot_correlation_matrix(df, output_path):
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f'{output_path}/correlation_matrix.png')
    plt.close()

def plot_feature_importance(importances, feature_names, output_path):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.savefig(f'{output_path}/feature_importance.png')
    plt.close()

def compare_original_augmented(original_df, augmented_df, feature, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(original_df[feature], color='blue', label='Original', kde=True)
    sns.histplot(augmented_df[feature], color='red', label='Augmented', kde=True)
    plt.title(f'Comparison of Original and Augmented {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{output_path}/{feature}_comparison.png')
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

def plot_data_augmentation(original_counts, augmented_counts, output_path):
    labels = list(original_counts.keys())
    original_values = list(original_counts.values())
    augmented_values = list(augmented_counts.values())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, original_values, width, label='Original')
    ax.bar(x + width/2, augmented_values, width, label='Augmented')

    ax.set_xlabel('Features')
    ax.set_ylabel('Counts')
    ax.set_title('Data Augmentation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig(f'{output_path}/data_augmentation_comparison.png')
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

    # Plot data augmentation quantities
    original_counts = original_df.count().to_dict()
    augmented_counts = augmented_df.count().to_dict()
    plot_data_augmentation(original_counts, augmented_counts, output_path)

if __name__ == "__main__":
    main()
    print("Visualization graphs generated.")