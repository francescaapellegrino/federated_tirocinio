"""
Preprocessing SmartGrid modificato per supportare Attribute Inference Attacks.
Questa versione definitiva corregge il disallineamento dei dati causato 
dal doppio split, garantendo la coerenza tra i set di dati pre e post PCA.
Francesca Pellegrino
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

TARGET_FEATURE_NAME_AIA = 'row_skew'

def load_data_for_aia(client_id: int):
    print(f"\nPREPROCESSING PER AIA (Client {client_id})")

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(np.float32)

    # ... (Tutta la logica di pulizia e feature engineering è identica) ...
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    if X.isnull().sum().sum() > 0: X = X.fillna(0)
    
    X_temp = X.copy()
    # ... (feature engineering identico)
    X_temp['row_mean'] = X.mean(axis=1); X_temp['row_std'] = X.std(axis=1); X_temp['row_median'] = X.median(axis=1)
    X_temp['row_max'] = X.max(axis=1); X_temp['row_min'] = X.min(axis=1); X_temp['row_range'] = X_temp['row_max'] - X_temp['row_min']
    X_temp['row_q25'] = X.quantile(0.25, axis=1); X_temp['row_q75'] = X.quantile(0.75, axis=1); X_temp['row_iqr'] = X_temp['row_q75'] - X_temp['row_q25']
    X_temp['row_skew'] = X.skew(axis=1); X_temp['row_kurt'] = X.kurtosis(axis=1)
    for col in ['row_mean', 'row_std', 'row_median', 'row_max', 'row_min', 'row_range', 'row_q25', 'row_q75', 'row_iqr', 'row_skew', 'row_kurt']:
        if col in X_temp.columns:
            X_temp[col] = X_temp[col].replace([np.inf, -np.inf], np.nan).fillna(X_temp[col].median())

    # Outlier removal (identico)
    # ... (codice omesso per brevità)
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

    # Scaling e Selezione Feature (identici)
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_temp)
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X_robust)
    
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_temp.columns)

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
    
    # Dividiamo i dati *PRIMA* della PCA
    stratify = y if len(np.unique(y)) > 1 else None
    X_temp_pre_pca, X_test_pre_pca, y_temp, y_test = train_test_split(
        X_selected_df, y, test_size=0.15, random_state=42, stratify=stratify
    )
    
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
    X_train_pre_pca, X_val_pre_pca, y_train, y_val = train_test_split(
        X_temp_pre_pca, y_temp, test_size=0.118, random_state=42, stratify=stratify_temp
    )
    
    # --- Ora applichiamo la PCA sui set già splittati ---
    print("Applicazione della PCA sui set splittati...")
    
    n_components_target = 30
    # Adatta la PCA solo sui dati di TRAIN
    pca = PCA(n_components=min(n_components_target, X_train_pre_pca.shape[1]-1), random_state=42)
    X_train_pca = pca.fit_transform(X_train_pre_pca)
    
    # Trasforma i set di validazione e test con la PCA già addestrata
    X_val_pca = pca.transform(X_val_pre_pca)
    X_test_pca = pca.transform(X_test_pre_pca)

    # Padding per garantire 30 componenti
    def pad_to_30(arr):
        padding_needed = 30 - arr.shape[1]
        if padding_needed > 0:
            return np.hstack([arr, np.zeros((arr.shape[0], padding_needed))])
        return arr[:, :30]

    X_train_pca = pad_to_30(X_train_pca)
    X_val_pca = pad_to_30(X_val_pca)
    X_test_pca = pad_to_30(X_test_pca)

    # Normalizzazione finale post-split (come prima)
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train_pca).astype(np.float32)
    X_val_final = final_scaler.transform(X_val_pca).astype(np.float32)
    X_test_final = final_scaler.transform(X_test_pca).astype(np.float32)

    # Ora X_train_final e X_train_pre_pca sono perfettamente allineati
    assert len(X_train_final) == len(X_train_pre_pca), "Disallineamento TRAIN!"
    assert len(X_test_final) == len(X_test_pre_pca), "Disallineamento TEST!"

    return (X_train_final, y_train.astype(np.float32), X_val_final, y_val.astype(np.float32), 
            X_test_final, y_test.astype(np.float32),
            X_train_pre_pca, X_test_pre_pca)