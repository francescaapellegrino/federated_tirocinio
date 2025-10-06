"""
Preprocessing SmartGrid senza PCA
Francesca Pellegrino
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_improved_client_data(client_id: int, config):
    print(f"\nPREPROCESSING CLIENT {client_id} (no PCA, 132 features fissi)")

    # 1. Caricamento dati
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(np.float32)
    print(f"Dati raw: {X.shape[0]} campioni, {X.shape[1]} features")
    print(f"Attack ratio: {y.mean()*100:.1f}%")

    # 2. Pulizia avanzata
    print("Pulizia avanzata...")
    missing_threshold = 0.95
    missing_cols = X.columns[X.isnull().mean() > missing_threshold]
    if len(missing_cols) > 0:
        X = X.drop(columns=missing_cols)
        print(f"Rimosse {len(missing_cols)} colonne con >95% missing")
    zero_var_cols = X.columns[X.var() == 0]
    if len(zero_var_cols) > 0:
        X = X.drop(columns=zero_var_cols)
        print(f"Rimosse {len(zero_var_cols)} colonne varianza zero")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)
    print(f"Pulizia completata: {X.shape[1]} features pulite")

    # 3. Feature engineering
    print("Feature engineering...")
    X_temp = X.copy()
    # Statistiche per riga
    X_temp['row_mean'] = X.mean(axis=1)
    X_temp['row_std'] = X.std(axis=1)
    X_temp['row_median'] = X.median(axis=1)
    X_temp['row_max'] = X.max(axis=1)
    X_temp['row_min'] = X.min(axis=1)
    X_temp['row_range'] = X_temp['row_max'] - X_temp['row_min']
    X_temp['row_q25'] = X.quantile(0.25, axis=1)
    X_temp['row_q75'] = X.quantile(0.75, axis=1)
    X_temp['row_iqr'] = X_temp['row_q75'] - X_temp['row_q25']
    X_temp['row_skew'] = X.skew(axis=1)
    X_temp['row_kurt'] = X.kurtosis(axis=1)
    X_temp['energy_sum'] = X.sum(axis=1)
    X_temp['energy_l2'] = np.sqrt((X**2).sum(axis=1))
    for col in ['row_mean', 'row_std', 'row_median', 'row_max', 'row_min', 'row_range',
                'row_q25', 'row_q75', 'row_iqr', 'row_skew', 'row_kurt', 'energy_sum', 'energy_l2']:
        if col in X_temp.columns:
            X_temp[col] = X_temp[col].replace([np.inf, -np.inf], np.nan)
            X_temp[col] = X_temp[col].fillna(X_temp[col].median())
    print(f"Feature engineering: {X.shape[1]} → {X_temp.shape[1]} features temporanee")

    # Salva le colonne dopo feature engineering
    temp_columns = X_temp.columns.tolist()

    # 4. Outlier removal
    print("Outlier removal...")
    normal_mask = (y == 0)
    if normal_mask.sum() > 50:
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        normal_data = X_temp[normal_mask]
        outlier_labels = iso_forest.fit_predict(normal_data)
        normal_outliers = (outlier_labels == -1)
        keep_mask = np.ones(len(X_temp), dtype=bool)
        normal_indices = np.where(normal_mask)[0]
        outlier_indices = normal_indices[normal_outliers]
        keep_mask[outlier_indices] = False
        attack_mask = (y == 1)
        keep_mask[attack_mask] = True
        removed_count = (~keep_mask).sum()
        print(f"Rimossi {removed_count} outliers ({removed_count/len(X_temp)*100:.1f}%)")
        X_temp = X_temp[keep_mask]
        y = y[keep_mask]
    else:
        print("Troppi pochi dati normali per outlier detection!")

    # 5. Scaling
    print("Scaling...")
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_temp)
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X_robust)
    print("RobustScaler + MinMaxScaler applicati")

    # 6. Selezione feature: fissa a 132
    print("Selezione feature fissa (132)...")
    n_features_target = 132
    if X_scaled.shape[1] >= n_features_target:
        selector = SelectKBest(score_func=f_classif, k=n_features_target)
        X_selected = selector.fit_transform(X_scaled, y)
        selected_indices = selector.get_support(indices=True)
        selected_columns = [temp_columns[i] for i in selected_indices]
    else:
        print(f"ATTENZIONE: feature disponibili ({X_scaled.shape[1]}) < 132, verranno paddate con zeri.")
        # Pad con zeri se necessario
        X_selected = np.pad(X_scaled, ((0,0),(0, n_features_target - X_scaled.shape[1])), mode='constant')
        selected_columns = temp_columns + [f"pad_{i}" for i in range(n_features_target - X_scaled.shape[1])]
    print(f"Finale: {X_selected.shape[1]} features")

    # 7. Split stratificato
    print("Split stratificato...")
    attack_ratio = y.mean()
    print(f"Attack ratio finale: {attack_ratio*100:.1f}%")
    stratify = y if len(np.unique(y)) > 1 else None
    X_temp2, X_test, y_temp, y_test = train_test_split(
        X_selected, y, 
        test_size=0.15, 
        random_state=42,
        stratify=stratify
    )
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp2, y_temp,
        test_size=0.118,  # ~10% del totale
        random_state=42,
        stratify=stratify_temp
    )

    # 8. Normalizzazione finale post-split
    print("Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train).astype(np.float32)
    X_val_final = final_scaler.transform(X_val).astype(np.float32)
    X_test_final = final_scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    num_features = X_train_final.shape[1]
    print(f"COMPATIBILITÀ:")
    print(f"- Train: {len(X_train_final)} campioni, {num_features} features")
    print(f"- Val: {len(X_val_final)} campioni, {num_features} features")
    print(f"- Test: {len(X_test_final)} campioni, {num_features} features")
    print(f"- Tipi: X={X_train_final.dtype}, y={y_train.dtype}")

    # Verifica che il numero di feature sia costante
    assert X_train_final.shape[1] == n_features_target, f"ERRORE TRAIN: {X_train_final.shape[1]} features"
    assert X_val_final.shape[1] == n_features_target, f"ERRORE VAL: {X_val_final.shape[1]} features"
    assert X_test_final.shape[1] == n_features_target, f"ERRORE TEST: {X_test_final.shape[1]} features"

    dataset_info = {
        'client_id': client_id,
        'total_samples': len(X_train_final) + len(X_val_final) + len(X_test_final),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'test_samples': len(X_test_final),
        'features': num_features,
        'attack_ratio': y_train.mean(),
        'preprocessing': 'no_pca_132_features',
        'feature_engineering': True,
        'outlier_removal': True,
        'intelligent_selection': True,
        'advanced_scaling': True,
        'compatibility_guaranteed': True
    }

    print("PREPROCESSING COMPLETATO! (no PCA, 132 features fissi)")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, dataset_info