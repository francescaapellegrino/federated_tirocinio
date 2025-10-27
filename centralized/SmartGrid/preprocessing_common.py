"""
Preprocessing SmartGrid - Versione FINALE senza PCA e con Feature Comuni
Francesca Pellegrino

Questa versione garantisce la coerenza totale tra i client:
1. Carica una lista predefinita di nomi di feature da 'common_feature_names.pkl'.
2. Elabora i dati locali di ogni client.
3. Allinea il DataFrame finale per avere esattamente le feature comuni,
   riempiendo eventuali valori mancanti.
Questo risolve gli errori di incoerenza del numero di feature.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def load_common_feature_names():
    """Carica i nomi delle feature comuni dal file .pkl."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(script_dir, "common_feature_names.pkl")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"File 'common_feature_names.pkl' non trovato. "
            f"Esegui prima lo script 'generate_common_features.py' per crearlo."
        )
    with open(features_path, "rb") as f:
        names = pickle.load(f)
    return names

def load_improved_client_data(client_id: int, config=None): # Manteniamo il nome per compatibilità
    """
    Carica e preprocessa i dati per un dato client, usando un set fisso di nomi di feature.
    """
    print(f"\nPREPROCESSING CLIENT {client_id} (NO-PCA, Feature Comuni)")

    # Carica la lista di nomi delle feature che deve essere usata da tutti
    common_feature_names = load_common_feature_names()

    # --- 1. Caricamento ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(np.float32)

    # --- 2. Pulizia e Feature Engineering ---
    # Semplifichiamo la pulizia: imputiamo prima e poi facciamo engineering.
    # L'allineamento finale gestirà le colonne mancanti.
    X = X.replace([np.inf, -np.inf], np.nan)
    X.fillna(X.median(numeric_only=True), inplace=True)
    X.fillna(0, inplace=True) 

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

    # --- 3. Allineamento alle Feature Comuni (MODIFICA CRUCIALE) ---
    # Il metodo .reindex() forza il DataFrame ad avere esattamente le colonne 
    # della lista `common_feature_names`, nell'ordine specificato.
    # - Le colonne che esistono in X_fe ma non nella lista comune vengono scartate.
    # - Le colonne che sono nella lista comune ma non in X_fe vengono aggiunte e riempite con 0.
    print(f"Allineamento a {len(common_feature_names)} feature comuni...")
    X_aligned = X_fe.reindex(columns=common_feature_names, fill_value=0)
    
    # --- 4. Outlier removal (eseguito sui dati allineati e coerenti) ---
    y.index = X_aligned.index # Assicura che y e X abbiano lo stesso indice
    normal_mask = (y == 0)
    if normal_mask.sum() > 50:
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        outlier_labels = iso_forest.fit_predict(X_aligned[normal_mask])
        keep_mask = np.ones(len(X_aligned), dtype=bool)
        normal_indices = np.where(normal_mask)[0]
        outlier_indices = normal_indices[outlier_labels == -1]
        keep_mask[outlier_indices] = False
        keep_mask[y == 1] = True
        X_aligned = X_aligned[keep_mask]
        y = y[keep_mask]

    # --- 5. Scaling, Split e Normalizzazione Finale ---
    # Da qui in poi, X_aligned ha un numero di colonne FISSO e UGUALE per tutti.
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_aligned)
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X_robust)
    
    stratify = y if len(np.unique(y)) > 1 else None
    X_temp_split, X_test, y_temp_split, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=stratify
    )
    stratify_temp = y_temp_split if len(np.unique(y_temp_split)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_split, y_temp_split, test_size=0.118, random_state=42, stratify=stratify_temp
    )
    
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train).astype(np.float32)
    X_val_final = final_scaler.transform(X_val).astype(np.float32)
    X_test_final = final_scaler.transform(X_test).astype(np.float32)

    print(f"DATI FINALI PRONTI (NO-PCA, {X_train_final.shape[1]} Feature Comuni):")
    print(f"- Train: {X_train_final.shape}")
    
    # Restituiamo 7 valori per mantenere la compatibilità con il codice del client
    return (X_train_final, y_train.to_numpy(dtype=np.float32), 
            X_val_final, y_val.to_numpy(dtype=np.float32), 
            X_test_final, y_test.to_numpy(dtype=np.float32),
            None)