"""
Preprocessing SmartGrid
Francesca Pellegrino
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_improved_client_data(client_id: int, config):
    print(f"\nPREPROCESSING {client_id}")

    # 1. Caricamento
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

    # 2. PULIZIA AVANZATA
    print(f"Pulizia avanzata...")

    # Rimuovi colonne con troppi NaN
    missing_threshold = 0.95
    missing_cols = X.columns[X.isnull().mean() > missing_threshold]
    if len(missing_cols) > 0:
        X = X.drop(columns=missing_cols)
        print(f"Rimosse {len(missing_cols)} colonne con >95% missing")

    # Rimuovi colonne varianza zero
    zero_var_cols = X.columns[X.var() == 0]
    if len(zero_var_cols) > 0:
        X = X.drop(columns=zero_var_cols)
        print(f"Rimosse {len(zero_var_cols)} colonne varianza zero")

    # Gestione inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Imputazione intelligente con MEDIANA
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    # Verifica finale
    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)

    print(f"Pulizia completata: {X.shape[1]} features pulite")

    # 3. FEATURE ENGINEERING (come in versione base)
    X_temp = X.copy()
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

    # 4. OUTLIER REMOVAL
    print(f"Outlier removal...")

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
        print(f"Troppi pochi dati normali per outlier detection!")

    # 5. SCALING
    print(f"Scaling...")

    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_temp)
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X_robust)

    print(f"RobustScaler + MinMaxScaler applicati")

    # 6. SELEZIONE DELLE MIGLIORI FEATURE
    print(f"Selezione features...")

    n_features_target = min(60, X_scaled.shape[1])
    if X_scaled.shape[1] > n_features_target:
        selector_f = SelectKBest(score_func=f_classif, k=n_features_target//2)
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features_target//2)
        X_f = selector_f.fit_transform(X_scaled, y)
        X_mi = selector_mi.fit_transform(X_scaled, y)
        selected_features_f = selector_f.get_support()
        selected_features_mi = selector_mi.get_support()
        combined_features = selected_features_f | selected_features_mi
        X_selected = X_scaled[:, combined_features]
        print(f"Selezione: {X_scaled.shape[1]} → {X_selected.shape[1]} features")
    else:
        X_selected = X_scaled
        print(f"Features {X_scaled.shape[1]} <= target, skip selezione")

    # 7. PCA con 30 componenti
    print(f"PCA 30 componenti...")

    n_samples, n_features = X_selected.shape
    n_components_target = 30
    max_components = min(n_samples - 1, n_features, n_components_target)

    pca = PCA(n_components=max_components, random_state=42)
    X_pca = pca.fit_transform(X_selected)

    # 30 componenti
    if X_pca.shape[1] != 30:
        if X_pca.shape[1] > 30:
            X_pca = X_pca[:, :30]
            print(f"Tagliato a 30 componenti")
        else:
            padding = np.zeros((X_pca.shape[0], 30 - X_pca.shape[1]))
            X_pca = np.hstack([X_pca, padding])
            print(f"Paddato a 30 componenti")

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {X_selected.shape[1]} → {X_pca.shape[1]} componenti")
    print(f"Varianza spiegata: {variance_explained*100:.1f}%")

    assert X_pca.shape[1] == 30, f"ERRORE: Features finali {X_pca.shape[1]} != 30"

    # 8. Split stratificato
    print(f"Split stratificato...")

    attack_ratio = y.mean()
    print(f"Attack ratio finale: {attack_ratio*100:.1f}%")

    stratify = y if len(np.unique(y)) > 1 else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_pca, y, 
        test_size=0.15, 
        random_state=42,
        stratify=stratify
    )
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.118,
        random_state=42,
        stratify=stratify_temp
    )

    # 9. Normalizzazione finale post-split
    print(f"Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train).astype(np.float32)
    X_val_final = final_scaler.transform(X_val).astype(np.float32)
    X_test_final = final_scaler.transform(X_test).astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    assert X_train_final.shape[1] == 30, f"ERRORE TRAIN: {X_train_final.shape[1]} features"
    assert X_val_final.shape[1] == 30, f"ERRORE VAL: {X_val_final.shape[1]} features"
    assert X_test_final.shape[1] == 30, f"ERRORE TEST: {X_test_final.shape[1]} features"

    print(f"COMPATIBILITÀ:")
    print(f"- Train: {len(X_train_final)} campioni, {X_train_final.shape[1]} features")
    print(f"- Val: {len(X_val_final)} campioni, {X_val_final.shape[1]} features")
    print(f"- Test: {len(X_test_final)} campioni, {X_test_final.shape[1]} features")
    print(f"- Tipi: X={X_train_final.dtype}, y={y_train.dtype}")

    dataset_info = {
        'client_id': client_id,
        'total_samples': len(X_train_final) + len(X_val_final) + len(X_test_final),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'test_samples': len(X_test_final),
        'features': X_train_final.shape[1],
        'attack_ratio': y_train.mean(),
        'preprocessing': 'improved_kaggle_compatible',
        'feature_engineering': True,
        'outlier_removal': True,
        'intelligent_selection': True,
        'advanced_scaling': True,
        'compatibility_guaranteed': True,
        'smote_applied': False
    }

    print(f"PREPROCESSING COMPLETATO!")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, dataset_info