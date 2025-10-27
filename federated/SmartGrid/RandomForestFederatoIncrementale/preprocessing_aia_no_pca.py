"""
Preprocessing SmartGrid per AIA - Versione senza PCA
Francesca Pellegrino

Questa versione si basa sulla logica di preprocessing per gli attacchi AIA,
ma omette il passo finale di riduzione dimensionale con PCA.
Mantiene la corretta separazione dei dati (split prima di scaling finale)
e la selezione forzata della feature target.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# PCA non è più necessaria in questo script
# from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
import os

warnings.filterwarnings('ignore')

TARGET_FEATURE_NAME_AIA = 'row_skew'

def load_data_for_aia(client_id: int):
    """
    Carica e preprocessa i dati per un dato client, omettendo la PCA.
    Mantiene la logica di split anticipato per prevenire data leakage.
    """
    print(f"\nPREPROCESSING PER AIA (Client {client_id}) - SENZA PCA")

    # 1. Caricamento e pulizia (identico)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "data", "SmartGrid", f"data{client_id}.csv")
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(np.float32)

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    if X.isnull().sum().sum() > 0: X = X.fillna(0)
    
    # 2. Feature Engineering (identico)
    X_temp = X.copy()
    X_temp['row_mean'] = X.mean(axis=1); X_temp['row_std'] = X.std(axis=1); X_temp['row_median'] = X.median(axis=1)
    X_temp['row_max'] = X.max(axis=1); X_temp['row_min'] = X.min(axis=1); X_temp['row_range'] = X_temp['row_max'] - X_temp['row_min']
    X_temp['row_q25'] = X.quantile(0.25, axis=1); X_temp['row_q75'] = X.quantile(0.75, axis=1); X_temp['row_iqr'] = X_temp['row_q75'] - X_temp['row_q25']
    X_temp['row_skew'] = X.skew(axis=1); X_temp['row_kurt'] = X.kurtosis(axis=1)
    for col in ['row_mean', 'row_std', 'row_median', 'row_max', 'row_min', 'row_range', 'row_q25', 'row_q75', 'row_iqr', 'row_skew', 'row_kurt']:
        if col in X_temp.columns:
            X_temp[col] = X_temp[col].replace([np.inf, -np.inf], np.nan).fillna(X_temp[col].median())

    # 3. Outlier removal (identico)
    normal_mask = (y == 0)
    if normal_mask.sum() > 50:
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        outlier_labels = iso_forest.fit_predict(X_temp[normal_mask])
        keep_mask = np.ones(len(X_temp), dtype=bool)
        normal_indices = np.where(normal_mask)[0]
        outlier_indices = normal_indices[outlier_labels == -1]
        keep_mask[outlier_indices] = False
        keep_mask[y == 1] = True
        X_temp = X_temp[keep_mask]
        y = y[keep_mask]

    # 4. Scaling e Selezione Feature (identici)
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_temp)
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X_robust)
    
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_temp.columns, index=X_temp.index)

    n_features_target = min(60, X_df_scaled.shape[1])
    if X_df_scaled.shape[1] > n_features_target:
        selector_f = SelectKBest(score_func=f_classif, k=n_features_target//2)
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features_target//2)
        selector_f.fit(X_df_scaled, y)
        selector_mi.fit(X_df_scaled, y)
        
        selected_cols_mask = selector_f.get_support() | selector_mi.get_support()
        
        if TARGET_FEATURE_NAME_AIA in X_df_scaled.columns:
            target_feature_idx = X_df_scaled.columns.get_loc(TARGET_FEATURE_NAME_AIA)
            selected_cols_mask[target_feature_idx] = True
            print(f"Selezione feature: forzata inclusione di '{TARGET_FEATURE_NAME_AIA}'.")
        
        X_selected_df = X_df_scaled.loc[:, selected_cols_mask]
    else:
        X_selected_df = X_df_scaled

    # --- MODIFICA CHIAVE: ESEGUIAMO LO SPLIT UNA SOLA VOLTA ---
    print("Esecuzione dello split unico dei dati...")
    
    stratify = y if len(np.unique(y)) > 1 else None
    X_temp_split, X_test, y_temp_split, y_test = train_test_split(
        X_selected_df, y, test_size=0.15, random_state=42, stratify=stratify
    )
    
    stratify_temp = y_temp_split if len(np.unique(y_temp_split)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp_split, y_temp_split, test_size=0.118, random_state=42, stratify=stratify_temp
    )
    
    # --- PCA RIMOSSA ---
    print("PCA: Step saltato come da configurazione.")

    # --- Normalizzazione finale post-split ---
    # Applichiamo lo scaler finale direttamente sui dati splittati (non trasformati da PCA)
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train).astype(np.float32)
    X_val_final = final_scaler.transform(X_val).astype(np.float32)
    X_test_final = final_scaler.transform(X_test).astype(np.float32)

    print(f"DATI FINALI PRONTI (SENZA PCA):")
    print(f"- Train: {X_train_final.shape}")
    print(f"- Val:   {X_val_final.shape}")
    print(f"- Test:  {X_test_final.shape}")
    
    # La funzione restituisce solo i set di dati necessari per l'addestramento standard,
    # omettendo i dati pre-PCA che servivano solo per l'AIA.
    return (X_train_final, y_train.to_numpy(dtype=np.float32), 
            X_val_final, y_val.to_numpy(dtype=np.float32), 
            X_test_final, y_test.to_numpy(dtype=np.float32),
            None) # Aggiungo None per mantenere la compatibilità di tupla con altre funzioni