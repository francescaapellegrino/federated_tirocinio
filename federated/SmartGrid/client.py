"""
Client federato SmartGrid
Author: francescaapellegrino
"""

import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from scipy import stats
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🔧 CONFIGURAZIONE
# ============================================================================

class HybridConfig:
    """Configurazione."""
    
    # Training parameters
    EPOCHS_PER_ROUND = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Model architecture
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT_RATES = [0.2, 0.15, 0.1]

    # Data preprocessing
    PCA_COMPONENTS = 30
    STATISTICAL_FEATURES = 12  # Feature aggiuntive conservative
    TOTAL_FEATURES = 42        # 30 PCA + 12 statistical
    
    # Threshold optimization
    ENABLE_THRESHOLD_OPT = True
    THRESHOLD_SEARCH_POINTS = 100
    THRESHOLD_MIN = 0.1
    THRESHOLD_MAX = 0.9
    
    # Specificity optimization
    ENABLE_SPECIFICITY_OPT = True
    TARGET_SPECIFICITY = 0.15      # 15% obiettivo minimo
    MIN_SENSITIVITY = 0.85         # 85% sensitivity minima da mantenere
    
    # System info
    RANDOM_SEED = 42

# ============================================================================
# SPECIFICITY OPTIMIZER CLASS
# ============================================================================

class SpecificityOptimizer:
    """
    Ottimizzatore threshold per bilanciare Sensitivity e Specificity.
    """
    
    def __init__(self, target_specificity=0.15, min_sensitivity=0.85, verbose=True):
        self.target_specificity = target_specificity
        self.min_sensitivity = min_sensitivity
        self.verbose = verbose
        self.optimal_threshold = 0.5
        self.optimization_results = {}
        self.is_optimized = False
        
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Trova threshold ottimale per massimizzare specificity mantenendo sensitivity.
        
        Strategia:
        1. Testa range di threshold da 0.1 a 0.9
        2. Per ogni threshold, calcola sensitivity e specificity
        3. Mantieni solo threshold con sensitivity ≥ min_sensitivity
        4. Tra questi, scegli quello con specificity massima
        """
        if self.verbose:
            print(f"Ottimizzazione threshold per Specificity...")
            print(f"   - Target specificity: ≥{self.target_specificity*100:.1f}%")
            print(f"   - Sensitivity minima: ≥{self.min_sensitivity*100:.1f}%")
        
        thresholds = np.linspace(0.1, 0.95, 100)
        best_threshold = 0.5
        best_specificity = 0
        valid_results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                
                # Accetta threshold solo se mantiene sensitivity minima
                if sensitivity >= self.min_sensitivity:
                    valid_results.append({
                        'threshold': threshold,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'precision': precision,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                    })
                    
                    # Aggiorna best se specificity migliora
                    if specificity > best_specificity:
                        best_specificity = specificity
                        best_threshold = threshold
                        
            except ValueError:
                continue
        
        self.optimal_threshold = best_threshold
        
        # Trova risultati per threshold ottimale
        best_result = None
        for result in valid_results:
            if abs(result['threshold'] - best_threshold) < 1e-6:
                best_result = result
                break
        
        if best_result:
            self.optimization_results = best_result
            self.is_optimized = True
            
            if self.verbose:
                print(f"✅ Threshold ottimizzato: {best_threshold:.3f}")
                print(f"   - Specificity raggiunta: {best_specificity:.3f} ({best_specificity*100:.1f}%)")
                print(f"   - Sensitivity mantenuta: {best_result['sensitivity']:.3f} ({best_result['sensitivity']*100:.1f}%)")
                
                if best_specificity >= self.target_specificity:
                    print(f"   🎯 TARGET SPECIFICITY RAGGIUNTO!")
                else:
                    print(f"   ⚠️ Specificity sotto target ({best_specificity*100:.1f}% < {self.target_specificity*100:.1f}%)")
        else:
            self.is_optimized = False
            if self.verbose:
                print(f"⚠️ Impossibile trovare threshold che mantenga sensitivity ≥{self.min_sensitivity*100:.1f}%")
        
        return best_threshold, valid_results
    
    def predict_with_optimal_threshold(self, y_pred_proba):
        """Applica threshold ottimizzato per predizioni."""
        return (y_pred_proba >= self.optimal_threshold).astype(int)
    
    def evaluate_improvement(self, y_true, y_pred_proba):
        """
        Valuta miglioramento vs threshold standard 0.5.
        """
        if not self.is_optimized:
            return {"error": "Optimizer not trained"}
        
        # Predizioni con threshold standard
        y_pred_standard = (y_pred_proba >= 0.5).astype(int)
        tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_true, y_pred_standard).ravel()
        
        # Predizioni con threshold ottimizzato
        y_pred_opt = self.predict_with_optimal_threshold(y_pred_proba)
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()
        
        # Calcola metriche
        def calc_metrics(tp, tn, fp, fn):
            return {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_positives': fp,
                'true_negatives': tn
            }
        
        standard_metrics = calc_metrics(tp_std, tn_std, fp_std, fn_std)
        optimized_metrics = calc_metrics(tp_opt, tn_opt, fp_opt, fn_opt)
        
        # Calcola miglioramenti
        improvements = {
            'threshold_change': self.optimal_threshold - 0.5,
            'specificity_improvement': optimized_metrics['specificity'] - standard_metrics['specificity'],
            'sensitivity_change': optimized_metrics['sensitivity'] - standard_metrics['sensitivity'],
            'false_positive_reduction': standard_metrics['false_positives'] - optimized_metrics['false_positives'],
            'false_positive_reduction_pct': (standard_metrics['false_positives'] - optimized_metrics['false_positives']) / standard_metrics['false_positives'] * 100 if standard_metrics['false_positives'] > 0 else 0,
            'target_achieved': optimized_metrics['specificity'] >= self.target_specificity
        }
        
        if self.verbose:
            print(f"\n📊 RISULTATI OTTIMIZZAZIONE SPECIFICITY:")
            print(f"{'Metrica':<20} {'Standard':<10} {'Optimized':<10} {'Δ':<10}")
            print(f"{'-'*55}")
            print(f"{'Threshold':<20} {0.5:<10.3f} {self.optimal_threshold:<10.3f} {improvements['threshold_change']:+.3f}")
            print(f"{'Sensitivity':<20} {standard_metrics['sensitivity']:<10.3f} {optimized_metrics['sensitivity']:<10.3f} {improvements['sensitivity_change']:+.3f}")
            print(f"{'Specificity':<20} {standard_metrics['specificity']:<10.3f} {optimized_metrics['specificity']:<10.3f} {improvements['specificity_improvement']:+.3f}")
            print(f"{'Accuracy':<20} {standard_metrics['accuracy']:<10.3f} {optimized_metrics['accuracy']:<10.3f} {optimized_metrics['accuracy'] - standard_metrics['accuracy']:+.3f}")
            print(f"{'False Positives':<20} {standard_metrics['false_positives']:<10} {optimized_metrics['false_positives']:<10} {-improvements['false_positive_reduction']:+}")
            
            if improvements['target_achieved']:
                print(f"🎯 TARGET SPECIFICITY RAGGIUNTO! ({optimized_metrics['specificity']*100:.1f}% ≥ {self.target_specificity*100:.1f}%)")
            else:
                print(f"⚠️ Specificity migliorata ma sotto target ({optimized_metrics['specificity']*100:.1f}% < {self.target_specificity*100:.1f}%)")
        
        return {
            'standard': standard_metrics,
            'optimized': optimized_metrics,
            'improvements': improvements
        }

# ============================================================================
# 🔧 CONSERVATIVE FEATURE ENGINEERING
# ============================================================================

class ConservativeFeatureEngineer:
    """
    Feature engineering conservativa
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def add_statistical_features(self, X):
        """
        Aggiunge SOLO le 12 statistical features più discriminative.
        Base PCA (30) + Statistical (12) = 42 feature finali.
        """
        if self.verbose:
            print(f"   🔧 Statistical features: da {X.shape[1]} a {X.shape[1] + 12}")
        
        # Basic statistics per row
        mean_per_row = np.mean(X, axis=1).reshape(-1, 1)
        std_per_row = np.std(X, axis=1).reshape(-1, 1)
        var_per_row = np.var(X, axis=1).reshape(-1, 1)
        min_per_row = np.min(X, axis=1).reshape(-1, 1)
        max_per_row = np.max(X, axis=1).reshape(-1, 1)
        range_per_row = (max_per_row - min_per_row)
        
        # Distribution statistics per row
        skew_per_row = stats.skew(X, axis=1).reshape(-1, 1)
        kurtosis_per_row = stats.kurtosis(X, axis=1).reshape(-1, 1)
        
        # Percentiles per row
        p25_per_row = np.percentile(X, 25, axis=1).reshape(-1, 1)
        p75_per_row = np.percentile(X, 75, axis=1).reshape(-1, 1)
        p90_per_row = np.percentile(X, 90, axis=1).reshape(-1, 1)
        
        # L2 norm per row (energia del segnale)
        l2_norm_per_row = np.sqrt(np.sum(X**2, axis=1)).reshape(-1, 1)
        
        # Stack features (12 totali)
        X_enhanced = np.hstack([
            X,                  # 30 PCA originali
            mean_per_row,       # 1
            std_per_row,        # 2
            var_per_row,        # 3
            min_per_row,        # 4
            max_per_row,        # 5
            range_per_row,      # 6
            skew_per_row,       # 7
            kurtosis_per_row,   # 8
            p25_per_row,        # 9
            p75_per_row,        # 10
            p90_per_row,        # 11
            l2_norm_per_row     # 12
        ])
        
        # Pulizia NaN/Inf (conservative)
        X_enhanced = np.where(np.isnan(X_enhanced), 0, X_enhanced)
        X_enhanced = np.where(np.isinf(X_enhanced), 0, X_enhanced)
        
        if self.verbose:
            print(f"   ✅ Feature finali: {X_enhanced.shape[1]} (conservative approach)")
        
        return X_enhanced

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

class ThresholdOptimizer:
    """Ottimizzazione soglie."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        
    def optimize_threshold_youdens_j(self, y_true, y_pred_prob):
        """
        Ottimizzazione threshold.
        J = Sensitivity + Specificity - 1
        """
        thresholds = np.linspace(
            self.config.THRESHOLD_MIN, 
            self.config.THRESHOLD_MAX, 
            self.config.THRESHOLD_SEARCH_POINTS
        )
        
        best_threshold = 0.5
        best_j_score = 0
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j_score = sensitivity + specificity - 1
                
                if j_score > best_j_score:
                    best_j_score = j_score
                    best_threshold = threshold
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    
                    best_metrics = {
                        'threshold': threshold,
                        'j_score': j_score,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'precision': precision,
                        'accuracy': accuracy,
                        'f1_score': f1
                    }
                    
            except Exception:
                continue
        
        return best_threshold, best_metrics

# ============================================================================
# STABLE MODEL ARCHITECTURE
# ============================================================================

def create_hybrid_model_v25(input_shape: int, config: HybridConfig) -> keras.Model:
    """
    Modello: architettura con parametri ottimizzati.
    """
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Layer 1: 128 neuroni (da v1)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[0], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED),
            name="dense_1"
        ),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Dropout(config.DROPOUT_RATES[0], seed=config.RANDOM_SEED, name="dropout_1"),
        
        # Layer 2: 64 neuroni (da v1)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[1], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+1),
            name="dense_2"
        ),
        keras.layers.BatchNormalization(name="batch_norm_2"),
        keras.layers.Dropout(config.DROPOUT_RATES[1], seed=config.RANDOM_SEED+1, name="dropout_2"),
        
        # Layer 3: 32 neuroni (da v1)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[2], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+2),
            name="dense_3"
        ),
        keras.layers.BatchNormalization(name="batch_norm_3"),
        keras.layers.Dropout(config.DROPOUT_RATES[2], seed=config.RANDOM_SEED+2, name="dropout_3"),
        
        # Output layer
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+3),
            name="output"
        )
    ], name="SmartGrid_Model")
    
    # Compilation (da v1 - comprovata)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", curve='ROC')
        ]
    )

    print(f"🧠 Model creato:")
    print(f"   - Architettura: {config.HIDDEN_LAYERS[0]}→{config.HIDDEN_LAYERS[1]}→{config.HIDDEN_LAYERS[2]}→1")
    print(f"   - Input features: {input_shape}")
    print(f"   - Weight tensors: {len(model.get_weights())}")
    print(f"   - Parametri: {model.count_params():,}")
    
    return model

# ============================================================================
# 📊 DATA LOADING
# ============================================================================

def load_hybrid_client_data_v25(client_id: int, config: HybridConfig):
    """
    Carica dati per client.
    Pipeline: Raw → PCA (30) → Statistical Features (+12) → Total (42)
    """
    print(f"=== CARICAMENTO DATI CLIENT {client_id} ===")
    
    # Path file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    print(f"📂 Dataset raw: {len(X)} campioni, {X.shape[1]} feature")
    print(f"   - Attacchi: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   - Naturali: {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # STEP 1: Pulizia base
    print(f"🔧 Pulizia base...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.median(), inplace=True)
        print(f"   - NaN imputati con mediana")
    
    # STEP 2: PCA (da v1 - 30 componenti stabili)
    print(f"🎯 PCA {config.PCA_COMPONENTS} componenti...")
    
    # Normalizzazione pre-PCA
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"   - Riduzione: {X.shape[1]} → {X_pca.shape[1]} feature")
    print(f"   - Varianza spiegata: {variance_explained*100:.2f}%")
    
    # STEP 3: Conservative Feature Engineering
    print(f"🔧 Conservative feature engineering...")
    feature_engineer = ConservativeFeatureEngineer(verbose=True)
    X_enhanced = feature_engineer.add_statistical_features(X_pca)

    # STEP 4: Split train/val/test
    print(f"📊 Split train/val/test...")
    
    # Prima split: train+val / test (85% / 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_enhanced, y, test_size=0.15, random_state=config.RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Seconda split: train / val (75% / 10% del totale)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.118, random_state=config.RANDOM_SEED,  # 10/85 ≈ 0.118
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    
    # STEP 5: Normalizzazione finale
    print(f"⚡ Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train)
    X_val_final = final_scaler.transform(X_val)
    X_test_final = final_scaler.transform(X_test)
    
    # Statistiche finali
    print(f"✅ Dataset preparato:")
    print(f"   - Pipeline: {X.shape[1]} → {X_pca.shape[1]} → {X_enhanced.shape[1]} feature")
    print(f"   - Train: {len(X_train_final)} ({len(X_train_final)/len(X_enhanced)*100:.1f}%)")
    print(f"   - Val: {len(X_val_final)} ({len(X_val_final)/len(X_enhanced)*100:.1f}%)")
    print(f"   - Test: {len(X_test_final)} ({len(X_test_final)/len(X_enhanced)*100:.1f}%)")
    print("=" * 60)
    
    dataset_info = {
        'client_id': client_id,
        'total_samples': len(X_enhanced),
        'train_samples': len(X_train_final),
        'val_samples': len(X_val_final),
        'test_samples': len(X_test_final),
        'features': X_enhanced.shape[1],
        'pca_components': config.PCA_COMPONENTS,
        'statistical_features': config.STATISTICAL_FEATURES,
        'attack_ratio': y.mean(),
        'variance_explained': variance_explained,
        'version': config.VERSION
    }
    
    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, dataset_info

# ============================================================================
# CLIENT
# ============================================================================

class SmartGridClient(fl.client.NumPyClient):
    """
    Client Flower hybrid v2.5.1: base v1 + threshold optimization v3 + specificity optimization.
    """
    
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.config = HybridConfig()
        self.threshold_optimizer = ThresholdOptimizer(self.config)
        
        # 🎯 NUOVO: Specificity optimizer
        self.specificity_optimizer = SpecificityOptimizer(
            target_specificity=self.config.TARGET_SPECIFICITY,
            min_sensitivity=self.config.MIN_SENSITIVITY,
            verbose=True
        )
        
        # Carica dati
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.dataset_info = \
            load_hybrid_client_data_v25(client_id, self.config)
        
        # Crea modello
        self.model = create_hybrid_model_v25(self.X_train.shape[1], self.config)
        
        print(f"🔧 Client hybrid v2.5.1 {client_id} inizializzato")
        print(f"   - Features: {self.X_train.shape[1]} (30 PCA + 12 statistical)")
        print(f"   - Train samples: {len(self.X_train)}")
        print(f"   - Threshold optimization: {'✅' if self.config.ENABLE_THRESHOLD_OPT else '❌'}")
        print(f"   - Specificity optimization: {'✅' if self.config.ENABLE_SPECIFICITY_OPT else '❌'}")
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        print(f"\n[Client {self.client_id}] === TRAINING HYBRID v2.5.1 ===")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Training (parametri da v1)
        print(f"[Client {self.client_id}] Training: {len(self.X_train)} campioni, {self.config.EPOCHS_PER_ROUND} epochs")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.config.EPOCHS_PER_ROUND,
            batch_size=self.config.BATCH_SIZE,
            verbose=0
        )
        
        # Estrai metriche training
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_f1 = history.history['f1_score'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        val_f1 = history.history['val_f1_score'][-1]
        
        print(f"[Client {self.client_id}] Training completato:")
        print(f"   - Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}, F1={train_f1:.4f}")
        print(f"   - Val: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}, F1={val_f1:.4f}")
        
        # Metriche base
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_f1_score': float(train_f1),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_f1_score': float(val_f1),
            'client_id': int(self.client_id),
            'version': self.config.VERSION,
            'statistical_features_count': int(self.config.STATISTICAL_FEATURES),
            'total_features': int(self.dataset_info['features']),
            'specificity_optimization_enabled': self.config.ENABLE_SPECIFICITY_OPT
        }
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        print(f"\n[Client {self.client_id}] === EVALUATION HYBRID v2.5.1 WITH SPECIFICITY OPT ===")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Valutazione standard su test set
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss, accuracy, precision, recall, f1_keras, auc = results
        
        # Predizioni per threshold optimization
        y_pred_prob = self.model.predict(self.X_test, verbose=0).flatten()
        
        baseline_metrics = {
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1_keras),
            'test_auc': float(auc)
        }
        
        # 🎯 THRESHOLD OPTIMIZATION (Youden's J)
        threshold_metrics = {}
        if self.config.ENABLE_THRESHOLD_OPT:
            print(f"[Client {self.client_id}] 🎯 Threshold optimization (Youden's J)...")
            
            try:
                optimal_threshold, thresh_metrics = self.threshold_optimizer.optimize_threshold_youdens_j(
                    self.y_test, y_pred_prob
                )
                
                threshold_metrics = {
                    'optimal_threshold': float(optimal_threshold),
                    'threshold_j_score': float(thresh_metrics['j_score']),
                    'threshold_accuracy': float(thresh_metrics['accuracy']),
                    'threshold_f1_score': float(thresh_metrics['f1_score']),
                    'threshold_sensitivity': float(thresh_metrics['sensitivity']),
                    'threshold_specificity': float(thresh_metrics['specificity']),
                    'threshold_precision': float(thresh_metrics['precision'])
                }
                
                # Improvement vs baseline
                f1_improvement = thresh_metrics['f1_score'] - f1_keras
                acc_improvement = thresh_metrics['accuracy'] - accuracy
                
                threshold_metrics.update({
                    'threshold_f1_improvement': float(f1_improvement),
                    'threshold_acc_improvement': float(acc_improvement),
                    'threshold_optimization_success': True
                })
                
                print(f"[Client {self.client_id}] 🎯 Youden's J results:")
                print(f"   - Optimal threshold: {optimal_threshold:.3f}")
                print(f"   - J-score: {thresh_metrics['j_score']:.4f}")
                print(f"   - Sensitivity: {thresh_metrics['sensitivity']:.4f}")
                print(f"   - Specificity: {thresh_metrics['specificity']:.4f}")
                
            except Exception as e:
                print(f"[Client {self.client_id}] ⚠️ Threshold optimization fallita: {e}")
                threshold_metrics = {
                    'optimal_threshold': 0.5,
                    'threshold_optimization_success': False
                }
        
        # 🎯 NUOVO: SPECIFICITY OPTIMIZATION
        specificity_metrics = {}
        if self.config.ENABLE_SPECIFICITY_OPT:
            print(f"[Client {self.client_id}] 🎯 Specificity optimization (Target: {self.config.TARGET_SPECIFICITY*100:.1f}%)...")
            
            try:
                # Trova threshold ottimale per specificity
                spec_threshold, spec_results = self.specificity_optimizer.find_optimal_threshold(
                    self.y_test, y_pred_prob
                )
                
                # Valuta miglioramento
                improvement_results = self.specificity_optimizer.evaluate_improvement(
                    self.y_test, y_pred_prob
                )
                
                if improvement_results and 'error' not in improvement_results:
                    opt_metrics = improvement_results['optimized']
                    improvements = improvement_results['improvements']
                    
                    specificity_metrics = {
                        'specificity_optimized_threshold': float(spec_threshold),
                        'specificity_optimized_accuracy': float(opt_metrics['accuracy']),
                        'specificity_optimized_sensitivity': float(opt_metrics['sensitivity']),
                        'specificity_optimized_specificity': float(opt_metrics['specificity']),
                        'specificity_optimized_precision': float(opt_metrics['precision']),
                        'specificity_optimized_f1_score': float(2 * opt_metrics['precision'] * opt_metrics['sensitivity'] / (opt_metrics['precision'] + opt_metrics['sensitivity']) if (opt_metrics['precision'] + opt_metrics['sensitivity']) > 0 else 0),
                        'specificity_improvement': float(improvements['specificity_improvement']),
                        'sensitivity_change': float(improvements['sensitivity_change']),
                        'false_positive_reduction': int(improvements['false_positive_reduction']),
                        'false_positive_reduction_pct': float(improvements['false_positive_reduction_pct']),
                        'specificity_target_achieved': bool(improvements['target_achieved']),
                        'specificity_optimization_success': True
                    }
                    
                    print(f"[Client {self.client_id}] 🎯 Specificity optimization results:")
                    print(f"   - Optimized threshold: {spec_threshold:.3f}")
                    print(f"   - Specificity: {opt_metrics['specificity']:.4f} ({opt_metrics['specificity']*100:.1f}%)")
                    print(f"   - Sensitivity: {opt_metrics['sensitivity']:.4f} ({opt_metrics['sensitivity']*100:.1f}%)")
                    print(f"   - False positive reduction: {improvements['false_positive_reduction']} ({improvements['false_positive_reduction_pct']:.1f}%)")
                    
                    if improvements['target_achieved']:
                        print(f"   ✅ TARGET SPECIFICITY RAGGIUNTO!")
                    else:
                        print(f"   ⚠️ Specificity migliorata ma sotto target")
                
                else:
                    specificity_metrics = {
                        'specificity_optimization_success': False,
                        'specificity_optimization_error': 'evaluation_failed'
                    }
                    
            except Exception as e:
                print(f"[Client {self.client_id}] ⚠️ Specificity optimization fallita: {e}")
                specificity_metrics = {
                    'specificity_optimization_success': False,
                    'specificity_optimization_error': str(e)
                }
        
        # Combina tutte le metriche
        all_metrics = {**baseline_metrics, **threshold_metrics, **specificity_metrics}
        all_metrics.update({
            'client_id': int(self.client_id),
            'version': self.config.VERSION,
            'test_samples': int(len(self.X_test)),
            'statistical_features_count': int(self.config.STATISTICAL_FEATURES),
            'specificity_target': float(self.config.TARGET_SPECIFICITY),
            'sensitivity_minimum': float(self.config.MIN_SENSITIVITY)
        })
        
        print(f"[Client {self.client_id}] ✅ Evaluation completata:")
        print(f"   - Test Acc: {accuracy:.4f}, F1: {f1_keras:.4f}, AUC: {auc:.4f}")
        print(f"   - Threshold optimization: {'✅' if threshold_metrics.get('threshold_optimization_success') else '❌'}")
        print(f"   - Specificity optimization: {'✅' if specificity_metrics.get('specificity_optimization_success') else '❌'}")
        
        return loss, len(self.X_test), all_metrics

# ============================================================================
# 🚀 MAIN FUNCTION
# ============================================================================

def main():
    """Avvia client."""
    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        print("Esempio: python client.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)
    
    print(f"🚀 AVVIO CLIENT {client_id}")
    print("=" * 70)
    
    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridClient(client_id)
        )
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()