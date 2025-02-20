from sklearn.preprocessing import MinMaxScaler as mms
import pandas as pd

def normalize_and_save_features(df_fea, output_path):
    """
    Normaliza as features de um DataFrame e as salva em um arquivo Excel.

    Parâmetros:
    df_fea (DataFrame): DataFrame contendo as features a serem normalizadas. 
                        A coluna 'label' é excluída da normalização.
    output_path (str): Caminho do arquivo onde os dados normalizados serão salvos.
    """
    scaler = mms()
    df_fea_normalized = df_fea.copy()
    df_fea_normalized[df_fea.columns.difference(['label'])] = scaler.fit_transform(df_fea[df_fea.columns.difference(['label'])])
    
    df_fea_normalized.to_excel(output_path, index=False)
    print(f"Features salvas em {output_path}")