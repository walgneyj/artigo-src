from data_collection import collect_data
from feature_extraction import extract_features
from helper_functions import normalize_and_save_features
from data_augmentation import augment_keyboard_data, augment_mouse_data, augment_behavior_data

def main():
    df_key, df_mou, df_beh = collect_data("C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/dataset/user*.csv")
    print("\nDataframes gerados!")

    # Aplicação do data augmentation
    df_key = augment_keyboard_data(df_key)
    df_mou = augment_mouse_data(df_mou)
    df_beh = augment_behavior_data(df_beh)
    print("\nData augmentation applied!")

    df_fea = extract_features(df_key, df_mou, df_beh)
    print("\nFeatures calculadas!")

    normalize_and_save_features(df_fea, "C:/Users/walgn/OneDrive/Documentos/Trabalho artigo/autenticacao-de-sistemas-baseados-em-biometria-comportamental-main/machine-learning/results/features.xlsx")
    print("\nFeatures salvas em features.xlsx!")

if __name__ == "__main__":
    main()