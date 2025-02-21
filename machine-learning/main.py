import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.visualization import plot_model_results
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

if __name__ == "__main__":
    main()

