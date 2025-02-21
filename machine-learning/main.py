import subprocess
from features.visualization import plot_model_results, plot_confusion_matrix
import pandas as pd


def main():
    # Executar o script features/main.py
    print("Iniciando a extração de features...")
    subprocess.run(["python", "C:/Users/walgn/OneDrive\/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/features/main.py"])
    
    # Executar o script model/main.py
    print("\nIniciando o treinamento e avaliação dos modelos...")
    subprocess.run(["python", "C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/model/main.py"])

    # Plotar resultados dos modelos
    print("\nPlotando resultados dos modelos...")
    results_df = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/metrics.xlsx")
    output_path = "C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results"
    plot_model_results(results_df, output_path)

    # Plotar matriz de confusão
    print("\nPlotando matriz de confusão...")
    conf_matrix = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/confusion_matrix.xlsx")
    y_test = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/y_test.xlsx")
    pred = pd.read_excel("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/pred.xlsx")
    plot_confusion_matrix(conf_matrix, y_test, pred)

if __name__ == "__main__":
    main()
