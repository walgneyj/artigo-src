import subprocess

def main():
    # Executar o script features/main.py
    print("Iniciando a extração de features...")
    subprocess.run(["python", "features/main.py"])
    
    # Executar o script model/main.py
    print("\nIniciando o treinamento e avaliação dos modelos...")
    subprocess.run(["python", "model/main.py"])

if __name__ == "__main__":
    main()