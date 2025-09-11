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
from improved_preprocessing import load_improved_client_data
from improved_model import create_improved_model, create_advanced_callbacks
from sklearn.utils.class_weight import compute_class_weight

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
            mean_per_row,       
            std_per_row,        
            var_per_row,        
            min_per_row,        
            max_per_row,        
            range_per_row,      
            skew_per_row,       
            kurtosis_per_row,   
            p25_per_row,        
            p75_per_row,        
            p90_per_row,        
            l2_norm_per_row     
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
    print(f"- Architettura: {optimized_config.ARCHITECTURE_SUMMARY}")
    print(f"- LR ottimizzato: {optimized_config.LEARNING_RATE:.6f}")
    print(f"- L2 ottimizzato: {optimized_config.L2_REG:.6f}")
    print(f"- Optimizer: {optimized_config.OPTIMIZER_TYPE}")
    print(f"- Activation: {optimized_config.ACTIVATION_FUNCTION}")
    print(f"- BatchNorm: {optimized_config.USE_BATCH_NORM}")
    print(f"- Score ottimizzazione: {optimized_config.OPTIMIZATION_SCORE:.6f}")
    print(f"- Parametri totali: {model.count_params():,}")

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
    print(f"- Architettura: {config.HIDDEN_LAYERS[0]}→{config.HIDDEN_LAYERS[1]}→{config.HIDDEN_LAYERS[2]}→{config.HIDDEN_LAYERS[3]}→1")
    print(f"- Input features: {input_shape}")
    print(f"- Learning Rate: {config.LEARNING_RATE:.6f} (OPTUNA)")
    print(f"- L2 Reg: {config.L2_REG:.6f}")
    print(f"- Parametri: {model.count_params():,}")

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
    print(f"- EarlyStopping: monitor={config.EARLY_STOPPING_MONITOR}, patience={config.EARLY_STOPPING_PATIENCE}")
    print(f"- ReduceLROnPlateau: factor={config.REDUCE_LR_FACTOR}, patience={config.REDUCE_LR_PATIENCE}")
    
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
    y = (df["marker"] != "Natural").astype(np.float32)
    
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
class ImprovedSmartGridClient(fl.client.NumPyClient):
    """
    Client migliorato che sostituisce SmartGridClient
    """
    
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.config = ClientConfig()  # Usa la tua config esistente
        
        print(f"🎯 IMPROVED CLIENT {client_id} - TARGET: >90% METRICHE")
        
        # Carica dati con preprocessing migliorato
        from improved_preprocessing import load_improved_client_data
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.dataset_info = load_improved_client_data(client_id, self.config)
        
        # Calcola class weights per dataset sbilanciato
        self.class_weights = self._compute_class_weights()
        
        # Crea modello migliorato
        from improved_model import create_improved_model
        self.model = create_improved_model(self.X_train.shape[1], self.config)
        
        print(f"✅ Client {client_id} migliorato inizializzato")
        print(f"   📊 Features: {self.X_train.shape[1]}")
        print(f"   📊 Train: {len(self.X_train)} samples")
        print(f"   📊 Attack ratio: {self.y_train.mean()*100:.1f}%")
    
    def _compute_class_weights(self):
        """Calcola pesi per bilanciare le classi"""
        if len(np.unique(self.y_train)) == 2:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            print(f"   ⚖️ Class weights: Normal={class_weights[0]:.2f}, Attack={class_weights[1]:.2f}")
            return weight_dict
        else:
            return None
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        print(f"\n[IMPROVED Client {self.client_id}] Training...")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Callbacks avanzati
        callbacks = create_advanced_callbacks(self.config)

        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        
        # Training con class weights e callback
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=25,  # Aumentato per convergenza migliore
            batch_size=64,  # Batch size più grande
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1  # Mostra progresso
        )
        
        # CORREZIONE: Estrai metriche finali con conversione sicura
        final_epoch = len(history.history['loss']) - 1
        
        # Conversione sicura per tutte le metriche
        def safe_extract(metric_history, epoch_idx):
            """Estrae valore sicuro da history"""
            try:
                value = metric_history[epoch_idx]
                # Se è un array, prendi il primo elemento; altrimenti usa direttamente
                if hasattr(value, '__len__') and len(value) > 0:
                    return float(value[0])
                else:
                    return float(value)
            except (IndexError, TypeError, ValueError):
                return 0.0
        
        # Estrazione sicura di tutte le metriche di training
        train_loss = safe_extract(history.history['loss'], final_epoch)
        train_acc = safe_extract(history.history['accuracy'], final_epoch)
        train_precision = safe_extract(history.history['precision'], final_epoch)
        train_recall = safe_extract(history.history['recall'], final_epoch)
        train_f1 = safe_extract(history.history['f1_score'], final_epoch)
        
        # Estrazione sicura di tutte le metriche di validation
        val_loss = safe_extract(history.history['val_loss'], final_epoch)
        val_acc = safe_extract(history.history['val_accuracy'], final_epoch)
        val_precision = safe_extract(history.history['val_precision'], final_epoch)
        val_recall = safe_extract(history.history['val_recall'], final_epoch)
        val_f1 = safe_extract(history.history['val_f1_score'], final_epoch)
        
        print(f"[IMPROVED Client {self.client_id}] Training completato:")
        print(f"   🎯 Train - Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"   🎯 Val - Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Metriche enhanced con valori garantiti scalari
        metrics = {
            # Training metrics
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            
            # Validation metrics
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1),
            
            # Client info
            'client_id': int(self.client_id),
            'total_features': int(self.dataset_info['features']),
            'epochs_trained': final_epoch + 1,
            'architecture_type': 'improved_kaggle_style',
            'preprocessing': 'advanced_feature_engineering',
            'class_weights_used': self.class_weights is not None,
            
            # Target tracking
            'target_accuracy_90': float(val_acc >= 0.90),
            'target_precision_90': float(val_precision >= 0.90),
            'target_recall_90': float(val_recall >= 0.90),
            'target_f1_90': float(val_f1 >= 0.90),
            'all_targets_met': float(all([
                val_acc >= 0.90,
                val_precision >= 0.90,
                val_recall >= 0.90,
                val_f1 >= 0.90
            ]))
        }
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        print(f"\n[IMPROVED Client {self.client_id}] Evaluation...")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Valutazione standard
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # CORREZIONE: Estrazione sicura dei valori
        def safe_extract_result(results_list, index, default=0.0):
            """Estrae valore sicuro da results"""
            try:
                if len(results_list) > index:
                    value = results_list[index]
                    # Se è un array, prendi il primo elemento; altrimenti usa direttamente
                    if hasattr(value, '__len__') and len(value) > 0:
                        return float(value[0])
                    else:
                        return float(value)
                else:
                    return default
            except (IndexError, TypeError, ValueError):
                return default
        
        # Estrazione sicura delle metriche
        loss = safe_extract_result(results, 0, 1.0)
        accuracy = safe_extract_result(results, 1, 0.0)
        precision = safe_extract_result(results, 2, 0.0)
        recall = safe_extract_result(results, 3, 0.0)
        f1_score = safe_extract_result(results, 4, 0.0)
        auc_roc = safe_extract_result(results, 5, 0.5)
        auc_pr = safe_extract_result(results, 6, 0.5)
        
        # Resto del codice rimane identico...
        y_pred_prob = self.model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Matrice di confusione
        from sklearn.metrics import confusion_matrix, classification_report
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Metriche aggiuntive
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        print(f"[IMPROVED Client {self.client_id}] Results:")
        print(f"   🎯 Accuracy: {accuracy:.4f} ({'✅' if accuracy >= 0.90 else '❌'} target: >90%)")
        print(f"   🎯 Precision: {precision:.4f} ({'✅' if precision >= 0.90 else '❌'} target: >90%)")
        print(f"   🎯 Recall: {recall:.4f} ({'✅' if recall >= 0.90 else '❌'} target: >90%)")
        print(f"   🎯 F1-Score: {f1_score:.4f} ({'✅' if f1_score >= 0.90 else '❌'} target: >90%)")
        print(f"   📊 AUC-ROC: {auc_roc:.4f}")
        print(f"   📊 Specificity: {specificity:.4f}")
        print(f"   📊 Balanced Acc: {balanced_accuracy:.4f}")
        print(f"   🔢 Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Verifica target raggiunti
        targets_met = {
            'accuracy_90': accuracy >= 0.90,
            'precision_90': precision >= 0.90,
            'recall_90': recall >= 0.90,
            'f1_90': f1_score >= 0.90
        }
        
        all_targets = all(targets_met.values())
        
        if all_targets:
            print(f"   🎉 TUTTI I TARGET >90% RAGGIUNTI! 🎉")
        else:
            missed = [k for k, v in targets_met.items() if not v]
            print(f"   ⚠️ Target mancati: {missed}")
        
        metrics = {
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1_score),
            'test_auc_roc': float(auc_roc),
            'test_auc_pr': float(auc_pr),
            'test_specificity': float(specificity),
            'test_sensitivity': float(sensitivity),
            'test_balanced_accuracy': float(balanced_accuracy),
            
            # Confusion matrix
            'test_tn': int(tn),
            'test_fp': int(fp),
            'test_fn': int(fn),
            'test_tp': int(tp),
            
            # Target tracking
            'target_accuracy_90': float(targets_met['accuracy_90']),
            'target_precision_90': float(targets_met['precision_90']),
            'target_recall_90': float(targets_met['recall_90']),
            'target_f1_90': float(targets_met['f1_90']),
            'all_targets_met': float(all_targets),
            
            # Client info
            'client_id': int(self.client_id),
            'test_samples': int(len(self.X_test)),
            'model_type': 'improved_kaggle_style',
            'version': 'v2_improved'
        }
        
        return loss, len(self.X_test), metrics


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
            client=ImprovedSmartGridClient(client_id)
        )
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()