"""
Script di Utilità per Generare e Salvare un Insieme Comune di Feature.

Questo script esegue la pipeline di pulizia e feature engineering su un client 
di riferimento (es. client 1) e salva i NOMI delle colonne risultanti.
Questo garantisce che tutti i partecipanti alla federazione utilizzino
lo stesso identico "vocabolario" di feature.
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

def create_and_save_common_features(client_id_ref=1):
    print(f"--- Creazione feature comuni basate sul client {client_id_ref} ---")

    # --- 1. Caricamento e Pulizia Iniziale ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assicurati che il percorso relativo al file dei dati sia corretto
    file_path = os.path.join(script_dir, "..", "..", "..", "data", "SmartGrid", f"data{client_id_ref}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato. Controlla il percorso.")

    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    
    # Rimuovi colonne con troppi NaN
    missing_threshold = 0.95
    missing_cols = X.columns[X.isnull().mean() > missing_threshold]
    if len(missing_cols) > 0:
        X = X.drop(columns=missing_cols)
        print(f"Rimosse {len(missing_cols)} colonne con >95% di valori mancanti.")

    # Rimuovi colonne con varianza zero
    zero_var_cols = X.columns[X.var() == 0]
    if len(zero_var_cols) > 0:
        X = X.drop(columns=zero_var_cols)
        print(f"Rimosse {len(zero_var_cols)} colonne con varianza zero.")

    # Gestione inf/nan e imputazione con mediana
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    if X.isnull().sum().sum() > 0: X = X.fillna(0)
    
    # --- 2. Feature Engineering ---
    X_fe = X.copy()
    X_fe['row_mean'] = X.mean(axis=1)
    X_fe['row_std'] = X.std(axis=1)
    X_fe['row_median'] = X.median(axis=1)
    X_fe['row_max'] = X.max(axis=1)
    X_fe['row_min'] = X.min(axis=1)
    X_fe['row_range'] = X_fe['row_max'] - X_fe['row_min']
    X_fe['row_q25'] = X.quantile(0.25, axis=1)
    X_fe['row_q75'] = X.quantile(0.75, axis=1)
    X_fe['row_iqr'] = X_fe['row_q75'] - X_fe['row_q25']
    X_fe['row_skew'] = X.skew(axis=1)
    X_fe['row_kurt'] = X.kurtosis(axis=1)
    X_fe['energy_sum'] = X.sum(axis=1)
    X_fe['energy_l2'] = np.sqrt((X**2).sum(axis=1))

    # Pulizia delle nuove feature create
    for col in X_fe.columns.difference(X.columns):
        X_fe[col] = X_fe[col].replace([np.inf, -np.inf], np.nan).fillna(X_fe[col].median())
    
    # --- 3. Salvataggio dei nomi delle colonne ---
    common_feature_names = X_fe.columns.tolist()
    
    print(f"Identificate {len(common_feature_names)} feature comuni.")

    # Salva la lista di nomi in un file pickle
    output_path = os.path.join(script_dir, "common_feature_names.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(common_feature_names, f)
        
    print(f"Nomi delle feature comuni salvati in '{output_path}'.")
    return output_path

if __name__ == "__main__":
    create_and_save_common_features()