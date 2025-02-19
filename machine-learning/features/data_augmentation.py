import pandas as pd
import numpy as np

def augment_keyboard_data(df_key, num_augmented_samples=5):
    augmented_data = []
    for _ in range(num_augmented_samples):
        augmented_sample = df_key.copy()
        # Convert 'epoch' column to numeric and handle non-numeric values
        augmented_sample['epoch'] = pd.to_numeric(
            augmented_sample['epoch'], errors='coerce'
        ).fillna(0).astype(np.int64)
        augmented_sample['epoch'] += np.random.randint(-50, 50, size=len(df_key))
        augmented_sample['timestamp'] = pd.to_datetime(
            augmented_sample['epoch'], unit='ms'
        )
        augmented_data.append(augmented_sample)
    return pd.concat(augmented_data, ignore_index=True)

def augment_mouse_data(df_mou, num_augmented_samples=5):
    augmented_data = []
    for _ in range(num_augmented_samples):
        augmented_sample = df_mou.copy()
        # Convert 'epoch' column to numeric
        augmented_sample['epoch'] = pd.to_numeric(
            augmented_sample['epoch'], errors='coerce'
        ).fillna(0).astype(np.int64)
        augmented_sample['epoch'] += np.random.randint(-50, 50, size=len(df_mou))
        augmented_sample['timestamp'] = pd.to_datetime(
            augmented_sample['epoch'], unit='ms'
        )
        augmented_sample['coordenadas'] = augmented_sample['coordenadas'].apply(
            lambda x: f"({int(x[1:-1].split(',')[0]) + np.random.randint(-5, 5)}, {int(x[1:-1].split(',')[1]) + np.random.randint(-5, 5)})"
        )
        augmented_data.append(augmented_sample)
    return pd.concat(augmented_data, ignore_index=True)

def augment_behavior_data(df_beh, num_augmented_samples=5):
    augmented_data = []
    for _ in range(num_augmented_samples):
        augmented_sample = df_beh.copy()
        # Convert 'formStartE' and 'formFinishE' to numeric
        augmented_sample['formStartE'] = pd.to_numeric(
            augmented_sample['formStartE'], errors='coerce'
        ).fillna(0).astype(np.int64)
        augmented_sample['formFinishE'] = pd.to_numeric(
            augmented_sample['formFinishE'], errors='coerce'
        ).fillna(0).astype(np.int64)
        augmented_sample['formStartE'] += np.random.randint(-50, 50, size=len(df_beh))
        augmented_sample['formFinishE'] += np.random.randint(-50, 50, size=len(df_beh))
        augmented_sample['formStartTs'] = pd.to_datetime(
            augmented_sample['formStartE'], unit='ms'
        )
        augmented_sample['formFinishTs'] = pd.to_datetime(
            augmented_sample['formFinishE'], unit='ms'
        )
        augmented_data.append(augmented_sample)
    return pd.concat(augmented_data, ignore_index=True)

from data_collection import collect_data
from feature_extraction import extract_features
from helper_functions import normalize_and_save_features

def main():
    df_key, df_mou, df_beh = collect_data("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/dataset/user*.csv")
    print("\nDataframes gerados!")

    # aplicação do data augmentation
    df_key = augment_keyboard_data(df_key)  # Direct call
    df_mou = augment_mouse_data(df_mou)     # Direct call
    df_beh = augment_behavior_data(df_beh)  # Direct call
    print("\nData augmentation applied!")

    df_fea = extract_features(df_key, df_mou, df_beh)
    print("\nFeatures calculadas!")

    normalize_and_save_features(df_fea, "C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/features.xlsx")

if __name__ == "__main__":
    main()