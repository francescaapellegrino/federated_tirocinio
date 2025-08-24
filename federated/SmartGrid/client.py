"""
Client federato SmartGrid
Francesca Pellegrino
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

from optimized_config_20250824_193626 import OptimizedConfig

# CONFIGURAZIONE CLIENT (in caso di errore modello ottimizzato)
class ClientConfig:

    # Architettura modello
    HIDDEN_LAYERS = [208, 48, 52, 22]   # numero neuroni per layer
    DROPOUT_RATES = [0.250, 0.500, 0.250, 0.450]
    LEARNING_RATE = 0.0032895272   # tasso di apprendimento
    L2_REG = 0.0000539478   # fattore di regolarizzazione L2 che penalizza i pesi grandi
    
    # Training parameters
    EPOCHS_PER_ROUND = 15
    BATCH_SIZE = 32

    # LEARNING RATE SCHEDULING OTTIMIZZATO
    ENABLE_LR_SCHEDULING = False

    ENABLE_EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MODE = 'min'
    EARLY_STOPPING_RESTORE_BEST = True

     # REDUCE LR ON PLATEAU
    ENABLE_REDUCE_LR = True                  
    REDUCE_LR_FACTOR = 0.7                   
    REDUCE_LR_PATIENCE = 2                   
    REDUCE_LR_MIN_LR = 1e-6

    # Data preprocessing
    PCA_COMPONENTS = 30
    STATISTICAL_FEATURES = 12    # numero feature statistiche aggiuntive
    TOTAL_FEATURES = 42

    # Threshold optimization
    ENABLE_THRESHOLD_OPT = True
    THRESHOLD_SEARCH_POINTS = 100
    THRESHOLD_MIN = 0.1
    THRESHOLD_MAX = 0.9
    
    # Specificity optimization
    ENABLE_SPECIFICITY_OPT = True
    TARGET_SPECIFICITY = 0.15      # 15% obiettivo minimo
    MIN_SENSITIVITY = 0.85         # 85% sensitivity minima da mantenere
    
    # Info sistema
    VERSION  = "1.0"
    RANDOM_SEED = 42

# CURRICULUM LEARNING PROGRESSIVO GESTIONE
class CurriculumManager:

    def __init__(self):
        self.current_stage = 0
        self.stage_targets = [0.60, 0.55, 0.50, 0.45, 0.40]    # Target val_loss per stage
        self.stage_lr_multipliers = [1.0, 1.2, 0.8, 1.5, 0.6]   # LR multiplier per stage
        self.stage_epochs = [15, 18, 20, 25, 30]  # Epoche crescenti
        self.rounds_in_stage = 0
        self.max_rounds_per_stage = 8
        
    def should_advance_stage(self, current_val_loss, rounds_in_current_stage):
        """Decide se avanzare al prossimo stage"""
        target = self.stage_targets[min(self.current_stage, len(self.stage_targets)-1)]
        # Avanza se il target è stato raggiunto o sono stati fatti troppi rounds nello stage
        if current_val_loss <= target or rounds_in_current_stage >= self.max_rounds_per_stage:
            return True
        return False
    
    def advance_stage(self):
        """Avanza al prossimo stage"""
        if self.current_stage < len(self.stage_targets) - 1:
            self.current_stage += 1
            self.rounds_in_stage = 0
            return True
        return False
    
    def get_current_config(self):
        """Restituisce configurazione per stage corrente"""
        stage = min(self.current_stage, len(self.stage_targets)-1)
        return {
            'target_loss': self.stage_targets[stage],
            'lr_multiplier': self.stage_lr_multipliers[stage],
            'epochs': self.stage_epochs[stage],
            'stage': stage
        }

# EARLY STOPPING CALLBACK
class FederatedEarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='val_loss', min_delta=0.001, patience=10, mode='min', restore_best_weights=True, client_id=None):
        super().__init__()
        self.monitor = monitor  # metrica da controllare
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.client_id = client_id or "Unknown"
        self.best_weights = None
        self.best_epoch = 0
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stopped = False
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - self.min_delta)
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > (best + self.min_delta)
            self.best_score = float('-inf')
    
    def on_train_begin(self, logs=None):
        """Inizializza variabili all'inizio del training"""
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stopped = False
        
    def on_epoch_end(self, epoch, logs=None):
        """Controlla la metrica alla fine di ogni epoca"""
        if logs is None:
            logs = {}
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.early_stopped = True
                self.model.stop_training = True
    
    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
    
    def get_summary(self):
        return {
            'early_stopped': self.early_stopped,    # se si è attivato l'early stopping
            'stopped_epoch': self.stopped_epoch if self.early_stopped else None,    # epoca in cui si è fermato
            'best_epoch': self.best_epoch + 1,  # epoca del miglior risultato
            'best_score': float(self.best_score) if self.best_score not in [float('inf'), float('-inf')] else None, # miglior valore della metrica
            'epochs_saved': (20 - self.stopped_epoch) if self.early_stopped else 0, # epoche risparmiate
            'monitor_metric': self.monitor  # metrica monitorata
        }

# OTTIMIZZAZIONE SPECIFICITY E SENSITIVITY
class SpecificityOptimizer:
    
    def __init__(self, target_specificity=0.15, min_sensitivity=0.85, verbose=True):
        self.target_specificity = target_specificity
        self.min_sensitivity = min_sensitivity
        self.verbose = verbose
        self.optimal_threshold = 0.5
        self.optimization_results = {}
        self.is_optimized = False
        
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Trova threshold ottimale per massimizzare specificity mantenendo sensitivity.
        Strategia:
        1. Testa range di threshold da 0.1 a 0.9
        2. Per ogni threshold, calcola sensitivity e specificity
        3. Mantieni solo threshold con sensitivity ≥ min_sensitivity
        4. Tra questi, scegli quello con specificity massima"""
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
        else:
            self.is_optimized = False
        
        return best_threshold, valid_results
    
    def predict_with_optimal_threshold(self, y_pred_proba):
        """Applica threshold ottimizzato per predizioni"""
        return (y_pred_proba >= self.optimal_threshold).astype(int)
    
    def evaluate_improvement(self, y_true, y_pred_proba):
        """Valuta miglioramento vs threshold standard 0.5"""
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
        
        return {
            'standard': standard_metrics,
            'optimized': optimized_metrics,
            'improvements': improvements
        }

# FEATURE ENGINEERING CONSERVATIVA
class ConservativeFeatureEngineer:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def add_statistical_features(self, X):
        """Aggiunge solo le * statistical features più discriminative.
        Base PCA (30) + Statistical (0) = 30 feature finali."""
        if self.verbose:
            print(f"Statistical features: da {X.shape[1]} a {X.shape[1] + 12}")
        
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
        
        # Stack features
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
            print(f"Feature finali: {X_enhanced.shape[1]} (conservative approach)")

        return X_enhanced

# OTTIMIZZAZIONE SOGLIE
class ThresholdOptimizer:
    
    def __init__(self, config: ClientConfig):
        self.config = config
        
    def optimize_threshold_youdens_j(self, y_true, y_pred_prob):
        """Ottimizzazione threshold. J = Sensitivity + Specificity - 1"""
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

# MODELLO OTTIMIZZATO SCIENTIFICAMENTE CON OPTUNA
def create_optimized_model(input_shape: int):

    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Configurazione ottimizzata
    optimized_config = OptimizedConfig()
    
    # Funzione attivazione ottimizzata
    if optimized_config.ACTIVATION_FUNCTION == 'leaky_relu':
        activation_layer = lambda: keras.layers.LeakyReLU(alpha=0.1)
        initializer = 'he_normal'
    elif optimized_config.ACTIVATION_FUNCTION == 'selu':
        activation_layer = lambda: keras.layers.Activation('selu')
        initializer = 'lecun_normal'
    elif optimized_config.ACTIVATION_FUNCTION == 'elu':
        activation_layer = lambda: keras.layers.ELU(alpha=1.0)
        initializer = 'he_normal'
    else:  # relu
        activation_layer = lambda: keras.layers.Activation('relu')
        initializer = 'he_normal'
    
    # Architettura ottimizzata
    model_layers = [
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Layer 1 ottimizzato
        keras.layers.Dense(
            optimized_config.HIDDEN_LAYERS[0], 
            kernel_regularizer=keras.regularizers.L2(optimized_config.L2_REG),
            kernel_initializer=initializer,
            name="dense_1"
        ),
        activation_layer(),
    ]
    
    if optimized_config.USE_BATCH_NORM:
        model_layers.append(keras.layers.BatchNormalization(name="batch_norm_1"))
    
    model_layers.extend([
        keras.layers.Dropout(optimized_config.DROPOUT_RATES[0], name="dropout_1"),
        
        # Layer 2 ottimizzato
        keras.layers.Dense(
            optimized_config.HIDDEN_LAYERS[1], 
            kernel_regularizer=keras.regularizers.L2(optimized_config.L2_REG),
            kernel_initializer=initializer,
            name="dense_2"
        ),
        activation_layer(),
    ])
    
    if optimized_config.USE_BATCH_NORM:
        model_layers.append(keras.layers.BatchNormalization(name="batch_norm_2"))
    
    model_layers.extend([
        keras.layers.Dropout(optimized_config.DROPOUT_RATES[1], name="dropout_2"),
        
        # Layer 3 ottimizzato
        keras.layers.Dense(
            optimized_config.HIDDEN_LAYERS[2], 
            kernel_regularizer=keras.regularizers.L2(optimized_config.L2_REG),
            kernel_initializer=initializer,
            name="dense_3"
        ),
        activation_layer(),
    ])
    
    if optimized_config.USE_BATCH_NORM:
        model_layers.append(keras.layers.BatchNormalization(name="batch_norm_3"))
    
    model_layers.extend([
        keras.layers.Dropout(optimized_config.DROPOUT_RATES[2], name="dropout_3"),
        
        # Layer 4 ottimizzato
        keras.layers.Dense(
            optimized_config.HIDDEN_LAYERS[3], 
            kernel_regularizer=keras.regularizers.L2(optimized_config.L2_REG),
            kernel_initializer=initializer,
            name="dense_4"
        ),
        activation_layer(),
    ])
    
    if optimized_config.USE_BATCH_NORM:
        model_layers.append(keras.layers.BatchNormalization(name="batch_norm_4"))
    
    model_layers.extend([
        keras.layers.Dropout(optimized_config.DROPOUT_RATES[3], name="dropout_4"),
        
        # Output layer
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name="output"
        )
    ])
    
    model = keras.Sequential(model_layers, name="SmartGrid_Optimized_v26")
    
    # Ottimizzatore
    if optimized_config.OPTIMIZER_TYPE == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=optimized_config.LEARNING_RATE,
            weight_decay=optimized_config.L2_REG * 0.1,
            beta_1=optimized_config.BETA_1,
            beta_2=optimized_config.BETA_2,
            clipnorm=optimized_config.CLIPNORM
        )
    elif optimized_config.OPTIMIZER_TYPE == 'nadam':
        optimizer = keras.optimizers.Nadam(
            learning_rate=optimized_config.LEARNING_RATE,
            beta_1=optimized_config.BETA_1,
            beta_2=optimized_config.BETA_2,
            clipnorm=optimized_config.CLIPNORM
        )
    else:  # adam
        optimizer = keras.optimizers.Adam(
            learning_rate=optimized_config.LEARNING_RATE,
            beta_1=optimized_config.BETA_1,
            beta_2=optimized_config.BETA_2,
            clipnorm=optimized_config.CLIPNORM
        )
    
    # Compilazione ottimizzata
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", curve='ROC')
        ]
    )

    print(f"Modello ottimizzato scientificamente creato:")
    print(f"   - Architettura: {optimized_config.ARCHITECTURE_SUMMARY}")
    print(f"   - LR ottimizzato: {optimized_config.LEARNING_RATE:.6f}")
    print(f"   - L2 ottimizzato: {optimized_config.L2_REG:.6f}")
    print(f"   - Optimizer: {optimized_config.OPTIMIZER_TYPE}")
    print(f"   - Activation: {optimized_config.ACTIVATION_FUNCTION}")
    print(f"   - BatchNorm: {optimized_config.USE_BATCH_NORM}")
    print(f"   - Score ottimizzazione: {optimized_config.OPTIMIZATION_SCORE:.6f}")
    print(f"   - Parametri totali: {model.count_params():,}")
    
    return model

# STABLE MODEL ARCHITECTURE OTTIMIZZATO
def create_model(input_shape: int, config: ClientConfig) -> keras.Model:

    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # L2 regularization (Optuna optimized)
    l2_reg = keras.regularizers.L2(config.L2_REG)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Architettura
        # Layer 1
        keras.layers.Dense(
            config.HIDDEN_LAYERS[0],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED),
            name="dense_1"
        ),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Dropout(config.DROPOUT_RATES[0], seed=config.RANDOM_SEED, name="dropout_1"),
        
        # Layer 2
        keras.layers.Dense(
            config.HIDDEN_LAYERS[1],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+1),
            name="dense_2"
        ),
        keras.layers.BatchNormalization(name="batch_norm_2"),
        keras.layers.Dropout(config.DROPOUT_RATES[1], seed=config.RANDOM_SEED+1, name="dropout_2"),
        
        # Layer 3
        keras.layers.Dense(
            config.HIDDEN_LAYERS[2],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+2),
            name="dense_3"
        ),
        keras.layers.BatchNormalization(name="batch_norm_3"),
        keras.layers.Dropout(config.DROPOUT_RATES[2], seed=config.RANDOM_SEED+2, name="dropout_3"),
        
        # Layer 4
        keras.layers.Dense(
            config.HIDDEN_LAYERS[3],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+3),
            name="dense_4"
        ),
        keras.layers.BatchNormalization(name="batch_norm_4"),
        keras.layers.Dropout(config.DROPOUT_RATES[3], seed=config.RANDOM_SEED+3, name="dropout_4"),
        
        # Output layer
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+4),
            name="output"
        )
    ], name="SmartGrid_Model")
    
    # OPTUNA OPTIMIZED OPTIMIZER (learning rate fisso)
    optimizer = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", curve='ROC')
        ]
    )

    print(f"Optimized Model creato:")
    print(f"   - Architettura: {config.HIDDEN_LAYERS[0]}→{config.HIDDEN_LAYERS[1]}→{config.HIDDEN_LAYERS[2]}→{config.HIDDEN_LAYERS[3]}→1")
    print(f"   - Input features: {input_shape}")
    print(f"   - Learning Rate: {config.LEARNING_RATE:.6f} (OPTUNA)")
    print(f"   - L2 Reg: {config.L2_REG:.6f}")
    print(f"   - Parametri: {model.count_params():,}")
    
    return model

# CALLBACK OTTIMIZZATI
def create_callbacks(config: ClientConfig):

    callbacks = []
    
    # Early Stopping
    if config.ENABLE_EARLY_STOPPING:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config.EARLY_STOPPING_MONITOR,  # 'val_loss'
                patience=config.EARLY_STOPPING_PATIENCE,  # 10
                restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
                verbose=0,
                mode=config.EARLY_STOPPING_MODE,  # 'min'
                min_delta=config.EARLY_STOPPING_MIN_DELTA  # 0.001
            )
        )

    # Reduce LR on Plateau
    if config.ENABLE_REDUCE_LR:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.REDUCE_LR_MIN_LR,
                verbose=0,
                mode='min'
            )
        )

    # Reduce LR on Plateau
    if config.ENABLE_REDUCE_LR:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.REDUCE_LR_MIN_LR,
                verbose=0,
                mode='min'
            )
        )

    print(f"Callbacks configurati:")
    print(f"   - EarlyStopping: monitor={config.EARLY_STOPPING_MONITOR}, patience={config.EARLY_STOPPING_PATIENCE}")
    print(f"   - ReduceLROnPlateau: factor={config.REDUCE_LR_FACTOR}, patience={config.REDUCE_LR_PATIENCE}")
    
    return callbacks

# CARICAMENTO DATI CLIENT
def load_client_data(client_id: int, config: ClientConfig):
  
    print(f"CARICAMENTO DATI CLIENT... {client_id}")
    
    # Path file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    print(f"Dataset raw: {len(X)} campioni, {X.shape[1]} feature")
    print(f"-> Attacchi: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"-> Naturali: {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
    
    # STEP 1: Pulizia base
    print(f"🔧 Pulizia base...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.median(), inplace=True)
        print(f"NaN imputati con mediana")
    
    # STEP 2: PCA
    print(f"PCA {config.PCA_COMPONENTS} componenti...")
    
    # Normalizzazione pre-PCA
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"Riduzione: {X.shape[1]} → {X_pca.shape[1]} feature")
    print(f"Varianza spiegata: {variance_explained*100:.2f}%")
    
    # STEP 3: no statistical features
    print(f"Usando solo PCA features...")
    X_enhanced = X_pca  # no statistical features
    print(f"-> Feature finali: {X_enhanced.shape[1]} (solo PCA)")

    # STEP 4: Split train/val/test
    # Primo split: train+val / test (85% / 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_enhanced, y, test_size=0.15, random_state=config.RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    # Secondo split: train / val (75% / 10% del totale)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.118, random_state=config.RANDOM_SEED,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    
    # STEP 5: Normalizzazione finale
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train)
    X_val_final = final_scaler.transform(X_val)
    X_test_final = final_scaler.transform(X_test)
    
    # Statistiche finali
    print(f"Dataset preparato:")
    print(f"Pipeline: {X.shape[1]} → {X_pca.shape[1]} → {X_enhanced.shape[1]} feature")
    print(f"Train: {len(X_train_final)} ({len(X_train_final)/len(X_enhanced)*100:.1f}%)")
    print(f"Val: {len(X_val_final)} ({len(X_val_final)/len(X_enhanced)*100:.1f}%)")
    print(f"Test: {len(X_test_final)} ({len(X_test_final)/len(X_enhanced)*100:.1f}%)")
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
    }
    
    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, dataset_info

# CLIENT
class SmartGridClient(fl.client.NumPyClient):

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.config = ClientConfig()
        self.curriculum_manager = CurriculumManager()
        self.threshold_optimizer = ThresholdOptimizer(self.config)

        # Early Stopping Callback
        self.early_stopping_callback = None
        if self.config.ENABLE_EARLY_STOPPING:
            self.early_stopping_callback = FederatedEarlyStopping(
                monitor=self.config.EARLY_STOPPING_MONITOR,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                patience=self.config.EARLY_STOPPING_PATIENCE,
                mode=self.config.EARLY_STOPPING_MODE,
                restore_best_weights=self.config.EARLY_STOPPING_RESTORE_BEST,
                client_id=self.client_id
            )

        # Specificity optimizer
        self.specificity_optimizer = SpecificityOptimizer(
            target_specificity=self.config.TARGET_SPECIFICITY,
            min_sensitivity=self.config.MIN_SENSITIVITY,
            verbose=True
        )
        
        # Carica dati
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.dataset_info = \
            load_client_data(client_id, self.config)

        # Usa modello ottimizzato invece di quello manuale
        try:
            self.model = create_optimized_model(self.X_train.shape[1])
            print(f"Usando modello ottimizzato scientificamente con Optuna!")
        except Exception as e:
            print(f"!!! Errore caricamento modello ottimizzato: {e}")
            print(f"Fallback al modello manuale...")
            self.model = create_model(self.X_train.shape[1], self.config)
        
        print(f"Client {client_id} inizializzato")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Train samples: {len(self.X_train)}")
        print(f"Threshold optimization: {'SI' if self.config.ENABLE_THRESHOLD_OPT else 'NO'}")
        print(f"Specificity optimization: {'SI' if self.config.ENABLE_SPECIFICITY_OPT else 'NO'}")
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        print(f"\n[Client {self.client_id}]")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Estrai server round
        server_round = config.get("server_round", 1)
        
        # CURRICULUM LEARNING
        curriculum_config = self.curriculum_manager.get_current_config()
        dynamic_epochs = curriculum_config['epochs']
        lr_multiplier = curriculum_config['lr_multiplier']
        current_stage = curriculum_config['stage']
        target_loss = curriculum_config['target_loss']

        # ADAPTIVE LEARNING RATE BASATO SU CURRICULUM
        try:
            optimized_config = OptimizedConfig()
            base_lr = optimized_config.LEARNING_RATE
        except:
            base_lr = self.config.LEARNING_RATE
            
        adapted_lr = base_lr * lr_multiplier
        
        print(f"[Client {self.client_id}] Training Round {server_round}:")
        print(f"Curriculum Stage: {current_stage}/4")
        print(f"Target Loss: {target_loss:.3f}")
        print(f"Dynamic Epochs: {dynamic_epochs}")
        print(f"Adapted LR: {adapted_lr:.6f}")
        print(f"Campioni: {len(self.X_train)}")

        # Callback con parametri dinamici
        callbacks = []
        
        # Early Stopping più aggressivo per stage avanzati
        early_patience = 3 if current_stage <= 1 else 5
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_patience,
                restore_best_weights=True,
                verbose=0,
                mode='min',
                min_delta=0.001
            )
        )
        
        # ReduceLR più aggressivo per stage avanzati
        reduce_factor = 0.7 if current_stage <= 1 else 0.5
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_factor,
                patience=2,
                min_lr=1e-7,
                verbose=0,
                mode='min'
            )
        )
        
        # Training con curriculum
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=dynamic_epochs,  # epoche dinamiche
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        
        # Estrai metriche
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_f1 = history.history['f1_score'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        val_f1 = history.history['val_f1_score'][-1]

        # CURRICULUM ADVANCEMENT CHECK
        self.curriculum_manager.rounds_in_stage += 1

        if self.curriculum_manager.should_advance_stage(val_loss, self.curriculum_manager.rounds_in_stage):
            if self.curriculum_manager.advance_stage():
                print(f"[Client {self.client_id}] CURRICULUM ADVANCED: Stage {self.curriculum_manager.current_stage}")
                print(f"New target: {self.curriculum_manager.stage_targets[self.curriculum_manager.current_stage]:.3f}")
            else:
                print(f"[Client {self.client_id}] CURRICULUM COMPLETED: Final stage reached")
        
        print(f"[Client {self.client_id}] Training completato (CURRICULUM STAGE {current_stage}):")
        print(f"Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}, F1={train_f1:.4f}")
        print(f"Val: Loss={val_loss:.4f}, Acc={val_accuracy:.4f}, F1={val_f1:.4f}")
        print(f"Target: {target_loss:.3f} ({'REACHED' if val_loss <= target_loss else 'IN PROGRESS'})")
        
        # Metriche estese con curriculum
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_f1_score': float(train_f1),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_f1_score': float(val_f1),
            'client_id': int(self.client_id),
            'total_features': int(self.dataset_info['features']),
            'server_round': int(server_round),
            'architecture_type': 'optuna_optimized_curriculum',
            'learning_rate': float(adapted_lr),  # LR ADAPTED
            'pca_components': int(self.config.PCA_COMPONENTS),

            # CURRICULUM METRICS
            'curriculum_stage': int(current_stage),
            'curriculum_target': float(target_loss),
            'curriculum_epochs': int(dynamic_epochs),
            'curriculum_lr_multiplier': float(lr_multiplier),
            'curriculum_target_reached': bool(val_loss <= target_loss)
        }
        
        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"\n[Client {self.client_id}] === EVALUATION OTTIMIZZATO ===")
        
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
        
        # THRESHOLD OPTIMIZATION
        threshold_metrics = {}
        if self.config.ENABLE_THRESHOLD_OPT:
            print(f"[Client {self.client_id}] Threshold optimization...")

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

            except Exception as e:
                print(f"[Client {self.client_id}] Threshold optimization fallita: {e}")
                threshold_metrics = {
                    'optimal_threshold': 0.5,
                    'threshold_optimization_success': False
                }
        
        # SPECIFICITY OPTIMIZATION
        specificity_metrics = {}
        if self.config.ENABLE_SPECIFICITY_OPT:
            print(f"[Client {self.client_id}] Specificity optimization (Target: {self.config.TARGET_SPECIFICITY*100:.1f}%)...")
            
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
                        'specificity_improvement': float(improvements['specificity_improvement']),
                        'sensitivity_change': float(improvements['sensitivity_change']),
                        'false_positive_reduction': int(improvements['false_positive_reduction']),
                        'false_positive_reduction_pct': float(improvements['false_positive_reduction_pct']),
                        'specificity_target_achieved': bool(improvements['target_achieved']),
                        'specificity_optimization_success': True
                    }
                    
                    print(f"[Client {self.client_id}] Specificity optimization results:")
                    print(f"   - Optimized threshold: {spec_threshold:.3f}")
                    print(f"   - Specificity: {opt_metrics['specificity']:.4f} ({opt_metrics['specificity']*100:.1f}%)")
                    print(f"   - Sensitivity: {opt_metrics['sensitivity']:.4f} ({opt_metrics['sensitivity']*100:.1f}%)")
                    
                    if improvements['target_achieved']:
                        print(f"TARGET SPECIFICITY RAGGIUNTO!")
                    else:
                        print(f"Specificity migliorata ma sotto target")
                
                else:
                    specificity_metrics = {
                        'specificity_optimization_success': False,
                        'specificity_optimization_error': 'evaluation_failed'
                    }
                    
            except Exception as e:
                print(f"[Client {self.client_id}] Specificity optimization fallita: {e}")
                specificity_metrics = {
                    'specificity_optimization_success': False,
                    'specificity_optimization_error': str(e)
                }
        
        # Combina tutte le metriche
        all_metrics = {**baseline_metrics, **threshold_metrics, **specificity_metrics}
        all_metrics.update({
            'client_id': int(self.client_id),
            'version': 'optimized_v26',
            'test_samples': int(len(self.X_test)),
            'model_type': 'optuna_optimized'
        })

        print(f"[Client {self.client_id}] Evaluation completata:")
        print(f"Test Acc: {accuracy:.4f}, F1: {f1_keras:.4f}, AUC: {auc:.4f}")
        print(f"Threshold optimization: {'SI' if threshold_metrics.get('threshold_optimization_success', True) else 'NO'}")
        print(f"Specificity optimization: {'SI' if specificity_metrics.get('specificity_optimization_success') else 'NO'}")

        return loss, len(self.X_test), all_metrics


# MAIN FUNCTION
def main():

    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)

    print(f"AVVIO CLIENT {client_id}")
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