"""
Server federato SmartGrid
Francesca Pellegrino
"""

from datetime import datetime
import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from optimized_config_20250824_193626 import OptimizedConfig
GLOBAL_METRICS_TRACKER = None

# TRACKER PER LE METRICHE
class CompleteMetricsTracker:
    
    def __init__(self):
        self.round_metrics = {}  # Dizionario: {round_num: {metriche}}
        self.target_metrics = [
            'val_loss',
            'global_accuracy',
            'global_precision', 
            'global_recall',
            'global_f1_score',
            'global_auc_roc',
            'global_specificity',
            'global_sensitivity'
        ]
        
        # Path del file di output
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(script_dir, f"metrics_complete_report_{timestamp}.txt")
        
        print(f"CompleteMetricsTracker inizializzato: {self.output_file}")
    
    def add_round_metrics(self, round_num: int, fit_metrics: Dict = None, evaluate_metrics: Dict = None):
        """Aggiunge metriche per un round specifico"""
        try:
            # Inizializza round
            if round_num not in self.round_metrics:
                self.round_metrics[round_num] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'val_loss': None,
                    'global_accuracy': None,
                    'global_precision': None,
                    'global_recall': None,
                    'global_f1_score': None,
                    'global_auc_roc': None,
                    'global_specificity': None,
                    'global_sensitivity': None
                }
            
            # Aggiungi val_loss dai client (fit_metrics)
            if fit_metrics and 'val_loss' in fit_metrics:
                self.round_metrics[round_num]['val_loss'] = fit_metrics['val_loss']
            
            # Aggiungi metriche globali dal server (evaluate_metrics)
            if evaluate_metrics:
                for metric in ['global_accuracy', 'global_precision', 'global_recall', 
                              'global_f1_score', 'global_auc_roc', 'global_specificity', 'global_sensitivity']:
                    if metric in evaluate_metrics:
                        self.round_metrics[round_num][metric] = evaluate_metrics[metric]
            
            # Debug
            available_metrics = [k for k, v in self.round_metrics[round_num].items() 
                               if v is not None and k != 'timestamp']
            print(f"Round {round_num}: {len(available_metrics)} metriche salvate")
            
        except Exception as e:
            print(f"Errore salvataggio round {round_num}: {e}")
    
    def generate_final_report(self):
        """Genera il resoconto finale completo"""
        try:
            if not self.round_metrics:
                print("Nessuna metrica da salvare")
                return
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("RESOCONTO ADDESTRAMENTO FEDERATO SMARTGRID\n")
                f.write(f"Francesca Pellegrino\n")
                f.write(f"Rounds totali: {len(self.round_metrics)}\n")
                
                # Tabella riassuntiva
                self._write_summary_table(f)
                
                # Statistiche finali
                self._write_final_statistics(f)
            
            print(f"Resoconto completo generato: {self.output_file}")
            print(f"Rounds tracciati: {len(self.round_metrics)}")
            
        except Exception as e:
            print(f"Errore generazione resoconto: {e}")
    
    def _write_summary_table(self, f):
        """Scrive tabella riassuntiva"""
        f.write("\n\nTABELLA RIASSUNTIVA METRICHE:\n")
        f.write("=" * 120 + "\n")
        
        # Header tabella
        header = f"{'Round':<6} {'Loss':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1_Score':<10} {'AUC_ROC':<10} {'Specificity':<10} {'Sensitivity':<10}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        # Righe dati
        for round_num in sorted(self.round_metrics.keys()):
            metrics = self.round_metrics[round_num]
            
            val_loss = f"{metrics['val_loss']:.6f}" if metrics['val_loss'] is not None else "N/A"
            accuracy = f"{metrics['global_accuracy']:.6f}" if metrics['global_accuracy'] is not None else "N/A"
            precision = f"{metrics['global_precision']:.6f}" if metrics['global_precision'] is not None else "N/A"
            recall = f"{metrics['global_recall']:.6f}" if metrics['global_recall'] is not None else "N/A"
            f1_score = f"{metrics['global_f1_score']:.6f}" if metrics['global_f1_score'] is not None else "N/A"
            auc_roc = f"{metrics['global_auc_roc']:.6f}" if metrics['global_auc_roc'] is not None else "N/A"
            specificity = f"{metrics['global_specificity']:.6f}" if metrics['global_specificity'] is not None else "N/A"
            sensitivity = f"{metrics['global_sensitivity']:.6f}" if metrics['global_sensitivity'] is not None else "N/A"
            
            row = f"{round_num:<6} {val_loss:<10} {accuracy:<10} {precision:<11} {recall:<10} {f1_score:<10} {auc_roc:<10} {specificity:<10} {sensitivity:<10}"
            f.write(row + "\n")
        
        f.write("=" * 120 + "\n\n")
    
    def _write_final_statistics(self, f):
        """Scrive statistiche finali"""
        f.write("STATISTICHE FINALI:\n")
        f.write("=" * 60 + "\n")
        
        # Calcola statistiche per ogni metrica
        metrics_stats = {}
        
        for metric in ['val_loss', 'global_accuracy', 'global_precision', 'global_recall', 
                      'global_f1_score', 'global_auc_roc', 'global_specificity', 'global_sensitivity']:
            values = [self.round_metrics[r][metric] for r in self.round_metrics 
                     if self.round_metrics[r][metric] is not None]
            
            if values:
                metrics_stats[metric] = {
                    'count': len(values),
                    'first': values[0],
                    'last': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'improvement': values[-1] - values[0] if len(values) > 1 else 0
                }
        
        # Scrivi statistiche
        for metric, stats in metrics_stats.items():
            f.write(f"\n🔹 {metric.upper()}:\n")
            f.write(f"   Rounds disponibili  : {stats['count']}\n")
            f.write(f"   Valore iniziale     : {stats['first']:.6f}\n")
            f.write(f"   Valore finale       : {stats['last']:.6f}\n")
            f.write(f"   Valore minimo       : {stats['min']:.6f}\n")
            f.write(f"   Valore massimo      : {stats['max']:.6f}\n")
            f.write(f"   Valore medio        : {stats['avg']:.6f}\n")
            if stats['improvement'] != 0:
                direction = "📈" if stats['improvement'] > 0 else "📉"
                f.write(f"   Miglioramento       : {stats['improvement']:+.6f} {direction}\n")
        
        f.write(f"\nADDESTRAMENTO COMPLETATO!\n")

# CONFIGURAZIONE SERVER (in caso di errore modello ottimizzato)
class ServerConfig:
    
    # Architettura modello
    HIDDEN_LAYERS = [208, 48, 52, 22]  # numero neuroni per layer
    DROPOUT_RATES = [0.250, 0.500, 0.250, 0.450]
    LEARNING_RATE = 0.0032895272    # tasso di apprendimento
    L2_REG = 0.0000539478   # fattore di regolarizzazione L2 che penalizza i pesi grandi
    
    # Data preprocessing
    PCA_COMPONENTS = 30  
    STATISTICAL_FEATURES = 12    # numero feature statistiche aggiuntive
    TOTAL_FEATURES = 30

    # Server specific
    NUM_ROUNDS = 15    # invio pesi, aggiornamento, aggregazione
    MIN_CLIENTS = 2

    ENABLE_FEDERATED_EARLY_STOPPING = False    # False: il training prosegue per tutti i round previsti
    FEDERATED_PATIENCE = 10    # numero di round senza miglioramento prima di fermare il training
    FEDERATED_MIN_DELTA = 0.001    # minimo miglioramento per considerare un progresso
    FEDERATED_MONITOR = 'val_loss'  # metrica da monitorare per early stopping
    FEDERATED_MODE = 'min'
    FEDERATED_MIN_ROUNDS = 25

    # Info sistema
    VERSION = "1.0"
    RANDOM_SEED = 42

# EARLY STOPPING
class FederatedEarlyStopping:
    
    def __init__(self, monitor='val_loss', min_delta=0.003, patience=8, mode='min', min_rounds=12):
        self.monitor = monitor  # metrica da controllare
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.mode = mode
        self.min_rounds = min_rounds
        self.best_score = None
        self.best_round = 0
        self.wait = 0
        self.should_stop = False
        self.rounds_completed = 0
        
        # DEBUG PRINT
        print(f"FederatedEarlyStopping inizializzato:")
        print(f"monitor='{self.monitor}' (validation loss from clients)")
        print(f"min_delta={self.min_delta}")
        print(f"patience={self.patience}")
        print(f"mode={self.mode}")
        print(f"min_rounds={self.min_rounds}")

        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - self.min_delta)
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > (best + self.min_delta)
            self.best_score = float('-inf')
    
    def update(self, round_num: int, metrics: Dict[str, float]) -> bool:
        """Aggiorna early stopping. Returns True se deve fermare."""
        print(f"\nEARLY STOPPING UPDATE ROUND {round_num}")
        
        self.rounds_completed = round_num
        
        print(f"Available metrics: {list(metrics.keys())}")
        print(f"Looking for monitor: '{self.monitor}'")
        
        if self.monitor not in metrics:
            print(f"Monitor '{self.monitor}' NOT FOUND!")
            return False
        
        current_score = metrics[self.monitor]
        print(f"Monitor found: {self.monitor} = {current_score:.6f}")
        print(f"Best score so far: {self.best_score:.6f}")
        print(f"Current wait: {self.wait}/{self.patience}")
        
        # Non attivare early stopping nei primi rounds
        if round_num < self.min_rounds:
            print(f"Warm-up phase: round {round_num} < min_rounds {self.min_rounds}")
            if self.monitor_op(current_score, self.best_score):
                old_best = self.best_score
                self.best_score = current_score
                self.best_round = round_num
                print(f"NEW BEST in warm-up: {old_best:.6f} → {current_score:.6f}")
            return False
        
        # Verifica miglioramento
        improvement_threshold = self.best_score + self.min_delta
        print(f"Improvement needed: > {improvement_threshold:.6f}")
        
        if self.monitor_op(current_score, self.best_score):
            old_best = self.best_score
            self.best_score = current_score
            self.best_round = round_num
            self.wait = 0
            print(f"IMPROVEMENT! {old_best:.6f} → {current_score:.6f}")
            print(f"Wait reset to 0")
        else:
            self.wait += 1
            print(f"NO IMPROVEMENT: {current_score:.6f} ≤ {improvement_threshold:.6f}")
            print(f"Wait increased to {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                print(f"STOPPING TRIGGERED! Wait {self.wait} >= patience {self.patience}")
                print(f"Best was: {self.best_score:.6f} at round {self.best_round}")
                self.should_stop = True
                return True
        
        print(f"Continue training")
        return False
    
    def get_summary(self):
        return {
            'federated_early_stopped': self.should_stop,
            'federated_stopped_round': self.rounds_completed if self.should_stop else None,
            'federated_best_round': self.best_round,
            'federated_best_score': float(self.best_score) if self.best_score not in [float('inf'), float('-inf')] else None,
            'federated_rounds_saved': (50 - self.rounds_completed) if self.should_stop else 0
        }

# VARIABILE GLOBALE PER EARLY STOPPING
GLOBAL_EARLY_STOPPING = None

# SERVER FEATURE ENGINEERING
class ServerFeatureEngineer:

    def add_statistical_features(self, X):
        """12 statistical features"""
        mean_per_row = np.mean(X, axis=1).reshape(-1, 1)
        std_per_row = np.std(X, axis=1).reshape(-1, 1)
        var_per_row = np.var(X, axis=1).reshape(-1, 1)
        min_per_row = np.min(X, axis=1).reshape(-1, 1)
        max_per_row = np.max(X, axis=1).reshape(-1, 1)
        range_per_row = (max_per_row - min_per_row)
        skew_per_row = stats.skew(X, axis=1).reshape(-1, 1)
        kurtosis_per_row = stats.kurtosis(X, axis=1).reshape(-1, 1)
        p25_per_row = np.percentile(X, 25, axis=1).reshape(-1, 1)
        p75_per_row = np.percentile(X, 75, axis=1).reshape(-1, 1)
        p90_per_row = np.percentile(X, 90, axis=1).reshape(-1, 1)
        l2_norm_per_row = np.sqrt(np.sum(X**2, axis=1)).reshape(-1, 1)
        
        X_enhanced = np.hstack([
            X, mean_per_row, std_per_row, var_per_row, min_per_row, max_per_row,
            range_per_row, skew_per_row, kurtosis_per_row, p25_per_row, 
            p75_per_row, p90_per_row, l2_norm_per_row
        ])
        
        # Pulizia
        X_enhanced = np.where(np.isnan(X_enhanced), 0, X_enhanced)
        X_enhanced = np.where(np.isinf(X_enhanced), 0, X_enhanced)
        
        return X_enhanced

# CARICAMENTO DATASET PER SERVER CON PREPROCESSING
def load_server_data():
    """Carica dati server"""
    print("CARICAMENTO DATASET GLOBALE SERVER")
    config = ServerConfig()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    validation_clients = [14, 15]
    df_list = []

    for client_id in validation_clients:
        file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"Caricato data{client_id}.csv: {len(df)} campioni")
        except FileNotFoundError:
            print(f"File data{client_id}.csv non trovato, saltato")
            continue

    if not df_list:
        print("ATTENZIONE: Usando fallback data1.csv per server")
        fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
        try:
            df_fallback = pd.read_csv(fallback_path)
            df_list = [df_fallback.sample(n=min(1000, len(df_fallback)), random_state=42)]
        except FileNotFoundError:
            raise FileNotFoundError("Impossibile caricare dati per server")
    
    # Combina dataset
    df_global = pd.concat(df_list, ignore_index=True)
    X = df_global.drop(columns=["marker"])
    y = (df_global["marker"] != "Natural").astype(np.float32)
    
    print(f"Dataset grezzo: {len(X)} campioni, {X.shape[1]} feature")
    print(f"Distribuzione: {y.sum():.0f} attacchi ({y.mean()*100:.1f}%)")
    
    # STEP 1: Pulizia
    print(f"Pulizia base...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.median(), inplace=True)
        print(f"NaN imputati con mediana")

    # STEP 2: PCA
    print(f"PCA {config.PCA_COMPONENTS} componenti...")
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X)
    
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {X.shape[1]} → {X_pca.shape[1]} feature")
    print(f"Varianza spiegata: {variance_explained*100:.2f}%")

    # STEP 3: no statistical features
    print(f"Approccio: usando solo PCA features...")
    X_enhanced = X_pca
    print(f"Features: {X_pca.shape[1]} (solo PCA)")

    # STEP 4: Normalizzazione finale
    print(f"Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X_enhanced).astype(np.float32)
    y = y.astype(np.float32)
    
    print(f"Dataset server preparato:")
    print(f"Pipeline: {X.shape[1]} → {X_pca.shape[1]} → {X_final.shape[1]} feature")
    print(f"Campioni finali: {len(X_final)}")
    print(f"Tipi corretti: X={X_final.dtype}, y={y.dtype}")
    print("=" * 70)
    
    return X_final, y

# MODELLO SERVER OTTIMIZZATO SCIENTIFICAMENTE CON OPTUNA
def create_server_model(input_shape: int):
    """Crea modello server ottimizzato"""

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
    
    model = keras.Sequential(model_layers, name="SmartGrid_Server_Optimized_v26")
    
    # Ottimizzatore ottimizzato
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
    
    # Compilazione con metriche 
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    print(f"Server Model ottimizzato scientificamente creato:")
    print(f"Architettura: {optimized_config.ARCHITECTURE_SUMMARY}")
    print(f"Input shape: {input_shape}")
    print(f"LR ottimizzato: {optimized_config.LEARNING_RATE:.6f}")
    print(f"L2 ottimizzato: {optimized_config.L2_REG:.6f}")
    print(f"Optimizer: {optimized_config.OPTIMIZER_TYPE}")
    print(f"Activation: {optimized_config.ACTIVATION_FUNCTION}")
    print(f"BatchNorm: {optimized_config.USE_BATCH_NORM}")
    print(f"Parametri: {model.count_params():,}")

    return model

def create_server_model_fallback(input_shape: int) -> keras.Model:
    """Crea modello server fallback"""

    config = ServerConfig()
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # L2 regularization
    l2_reg = keras.regularizers.L2(config.L2_REG)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Architettura
        keras.layers.Dense(
            config.HIDDEN_LAYERS[0],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED),
            name="dense_1"
        ),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Dropout(config.DROPOUT_RATES[0], seed=config.RANDOM_SEED, name="dropout_1"),
        
        keras.layers.Dense(
            config.HIDDEN_LAYERS[1],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+1),
            name="dense_2"
        ),
        keras.layers.BatchNormalization(name="batch_norm_2"),
        keras.layers.Dropout(config.DROPOUT_RATES[1], seed=config.RANDOM_SEED+1, name="dropout_2"),
        
        keras.layers.Dense(
            config.HIDDEN_LAYERS[2],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+2),
            name="dense_3"
        ),
        keras.layers.BatchNormalization(name="batch_norm_3"),
        keras.layers.Dropout(config.DROPOUT_RATES[2], seed=config.RANDOM_SEED+2, name="dropout_3"),
        
        keras.layers.Dense(
            config.HIDDEN_LAYERS[3],
            activation="relu",
            kernel_regularizer=l2_reg,
            kernel_initializer=keras.initializers.HeNormal(seed=config.RANDOM_SEED+3),
            name="dense_4"
        ),
        keras.layers.BatchNormalization(name="batch_norm_4"),
        keras.layers.Dropout(config.DROPOUT_RATES[3], seed=config.RANDOM_SEED+3, name="dropout_4"),
        
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+4),
            name="output"
        )
    ], name="SmartGrid_Server_Model")
    
    # Otimizzatore
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
            keras.metrics.Recall(name="recall")
        ]
    )
    
    print(f"Server Model fallback creato:")
    print(f"Architettura: {config.HIDDEN_LAYERS[0]}→{config.HIDDEN_LAYERS[1]}→{config.HIDDEN_LAYERS[2]}→{config.HIDDEN_LAYERS[3]}→1")
    print(f"Input shape: {input_shape}")
    print(f"Learning Rate: {config.LEARNING_RATE:.6f}")
    print(f"Parametri: {model.count_params():,}")
    print(f"🔧 Metriche: accuracy, precision, recall (versione stabile)")
    print(f"Compatibilità: 100%")

    return model

# ADVANCED METRICS AGGREGATION
def weighted_average(metrics):

    if not metrics:
        return {}
    
    print(f"Aggregating evaluate metrics from {len(metrics)} clients...")
    
    metrics_sum = {}
    total_examples = 0
    
    for i, (num_examples, metrics_dict) in enumerate(metrics):
        total_examples += num_examples
        print(f"Client {i+1}: {num_examples} samples, EVALUATE metrics: {list(metrics_dict.keys())}")
        
        for key, value in metrics_dict.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metrics_sum[key] += num_examples * value
    
    # Calcola medie pesate
    aggregated = {}
    for key, value in metrics_sum.items():
        if total_examples > 0:
            aggregated[key] = value / total_examples
    
    # Meta-statistiche
    aggregated['total_clients'] = len(metrics)
    aggregated['total_samples'] = total_examples

    print(f"evaluate metrics aggregated: {list(aggregated.keys())}")
    return aggregated

def print_client_metrics(fit_results):
    """Stampa metriche client"""
    if not fit_results:
        return

    print(f"\nMETRICHE CLIENT")
    
    total_samples = 0
    total_train_acc = 0
    total_val_acc = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1} :")
        print(f"Campioni: {client_samples}")
        
        if 'train_accuracy' in client_metrics:
            train_acc = client_metrics['train_accuracy']
            total_train_acc += train_acc * client_samples
            print(f"Train Acc: {train_acc:.4f}")
        
        if 'val_accuracy' in client_metrics:
            val_acc = client_metrics['val_accuracy']
            total_val_acc += val_acc * client_samples
            print(f"Val Acc: {val_acc:.4f}")
        
        if 'total_features' in client_metrics:
            total_features = client_metrics['total_features']
            print(f"Total features: {total_features}")

        if 'architecture_type' in client_metrics:
            arch_type = client_metrics['architecture_type']
            print(f"Architecture: {arch_type}")
    
    # Statistiche aggregate
    if total_samples > 0:
        avg_train_acc = total_train_acc / total_samples
        avg_val_acc = total_val_acc / total_samples
        
        print(f"\nSTATISTICHE AGGREGATE OTTIMIZZATE")
        print(f"Media Train Accuracy: {avg_train_acc:.4f}")
        print(f"Media Val Accuracy: {avg_val_acc:.4f}")
        print(f"Gap Train-Val: {avg_train_acc - avg_val_acc:.4f}")
    
    print("=" * 60)

# FEDERATED STRATEGY (FedAvg)
class Strategy(FedAvg):
    
    def __init__(self, **kwargs):

        # Inizializza MetricsTracker
        global GLOBAL_METRICS_TRACKER
        GLOBAL_METRICS_TRACKER = CompleteMetricsTracker()

        # Genera parametri iniziali per evitare GrpcBridgeClosed
        self.initial_parameters = self.generate_initial_parameters()
        super().__init__(**kwargs)
    
    def generate_initial_parameters(self):
        """Genera parametri iniziali per il modello."""
        print("Generazione parametri iniziali server ottimizzati...")
        
        config = ServerConfig()
        
        # PROVA PRIMA MODELLO OTTIMIZZATO
        try:
            temp_model = create_server_model(input_shape=config.TOTAL_FEATURES)
            print(f"Usando modello ottimizzato per parametri iniziali")
        except Exception as e:
            print(f"Fallback al modello manuale: {e}")
            temp_model = create_server_model_fallback(input_shape=config.TOTAL_FEATURES)
            
        initial_weights = temp_model.get_weights()
        
        print(f"Parametri generati: {len(initial_weights)} tensori")
        print(f"Compatibilità: Client ottimizzati")
        
        return fl.common.ndarrays_to_parameters(initial_weights)
    
    def initialize_parameters(self, client_manager):
        """Restituisce parametri iniziali"""
        print("Inizializzazione parametri server ottimizzati")
        return self.initial_parameters
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregazione con metriche e early stopping su val_loss"""
        print(f"\n=== AGGREGAZIONE FIT OTTIMIZZATO ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")

        print_client_metrics(results)

        # AGGREGAZIONE MANUALE DELLE METRICHE FIT
        fit_metrics = []
        for client_proxy, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                fit_metrics.append((fit_res.num_examples, fit_res.metrics))
                # DEBUG LEARNING RATE
                if 'learning_rate' in fit_res.metrics:
                    print(f"Client {fit_res.metrics.get('client_id', '?')} LR: {fit_res.metrics['learning_rate']:.6f}")
                print(f"Client {fit_res.metrics.get('client_id', '?')} FIT metrics: {list(fit_res.metrics.keys())}")
        
        # Aggrega manualmente le metriche FIT
        aggregated_fit_metrics = {}
        if fit_metrics:
            aggregated_fit_metrics = self.aggregate_fit_metrics_manual(fit_metrics)
            print(f"Aggregated FIT metrics: {list(aggregated_fit_metrics.keys())}")
        
        # Chiama l'aggregazione standard dei parametri
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            parameters, _ = aggregated_result  # Ignora le metriche standard
            
            # SALVA METRICHE FIT NEL TRACKER
            global GLOBAL_METRICS_TRACKER
            if GLOBAL_METRICS_TRACKER and aggregated_fit_metrics:
                GLOBAL_METRICS_TRACKER.add_round_metrics(
                    round_num=server_round,
                    fit_metrics=aggregated_fit_metrics,
                    evaluate_metrics=None
                )
            
            # EARLY STOPPING QUI SU VAL_LOSS DAI CLIENT
            global GLOBAL_EARLY_STOPPING
            if GLOBAL_EARLY_STOPPING is not None and aggregated_fit_metrics:
                should_stop = GLOBAL_EARLY_STOPPING.update(server_round, aggregated_fit_metrics)
                
                if should_stop:
                    print(f"!!! FEDERATED EARLY STOPPING ACTIVATED")
                    print(f"Validation Loss stopped improving at round {server_round}")
                    print(f"Best validation loss: {GLOBAL_EARLY_STOPPING.best_score:.6f} at round {GLOBAL_EARLY_STOPPING.best_round}")
                    aggregated_fit_metrics.update(GLOBAL_EARLY_STOPPING.get_summary())
                    
                    # Forza stop con eccezione
                    raise KeyboardInterrupt(f"Federated Early Stopping triggered at round {server_round}")
            
            print(f"Aggregazione fit ottimizzata completata per round {server_round}")
            return parameters, aggregated_fit_metrics
    
    def aggregate_fit_metrics_manual(self, metrics):
        """Aggregazione manuale delle metriche FIT"""
        if not metrics:
            print("No FIT metrics to aggregate")
            return {}
        
        print(f"Aggregating FIT metrics from {len(metrics)} clients...")
        
        metrics_sum = {}
        total_examples = 0
        
        for i, (num_examples, metrics_dict) in enumerate(metrics):
            total_examples += num_examples
            print(f"Client {i+1}: {num_examples} samples, FIT metrics: {list(metrics_dict.keys())}")
            
            for key, value in metrics_dict.items():
                if key not in metrics_sum:
                    metrics_sum[key] = 0
                
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    metrics_sum[key] += num_examples * value
                    if key == 'val_loss':
                        print(f" -> val_loss: {value:.6f}")
        
        # Calcola medie pesate
        aggregated = {}
        for key, value in metrics_sum.items():
            if total_examples > 0:
                aggregated[key] = value / total_examples
        
        # DEBUG PER VALIDATION LOSS
        if 'val_loss' in aggregated:
            print(f"Aggregated FIT Validation Loss: {aggregated['val_loss']:.6f}")
        else:
            print(f"val_loss not found in FIT aggregated metrics")
            print(f"Available FIT metrics: {list(aggregated.keys())}")

        aggregated['total_clients'] = len(metrics)
        aggregated['total_samples'] = total_examples

        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregazione evaluate con controllo None"""
        print(f"\n=== AGGREGATE_EVALUATE OTTIMIZZATO ROUND {server_round}===")
        
        # Debug input
        print(f"DEBUG AGGREGATE_EVALUATE Round {server_round}:")
        print(f"results ricevuti: {len(results) if results else 0}")
        print(f"failures: {len(failures) if failures else 0}")

        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        # Debug aggregated_result con controllo None
        print(f"DEBUG aggregated_result:")
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            
            # Controllo None per loss
            if loss is not None:
                print(f" Loss: {loss:.6f}")
            else:
                print(f" Loss: None (nessun client ha restituito risultati validi)")
                
            print(f"Metrics ricevute: {metrics is not None}")
            if metrics:
                print(f"Metrics keys: {list(metrics.keys())}")
            else:
                print(f"Metrics è None!")
        else:
            print(f"aggregated_result è None!")

        # SALVA METRICHE EVALUATE NEL TRACKER
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            
            global GLOBAL_METRICS_TRACKER
            if GLOBAL_METRICS_TRACKER and metrics and loss is not None:
                # Aggiunge loss alle metriche
                eval_metrics = metrics.copy()
                eval_metrics['global_loss'] = loss

                print(f"DEBUG: Chiamata TRACKER con evaluate_metrics")
                print(f"eval_metrics keys: {list(eval_metrics.keys())}")
                
                GLOBAL_METRICS_TRACKER.add_round_metrics(
                    round_num=server_round,
                    fit_metrics=None,
                    evaluate_metrics=eval_metrics
                )
            else:
                print(f"DEBUG: NON chiamo tracker perché:")
                print(f"GLOBAL_METRICS_TRACKER: {GLOBAL_METRICS_TRACKER is not None}")
                print(f"metrics: {metrics is not None}")
                print(f"loss: {loss is not None}")

        return aggregated_result

# VALUTAZIONE GLOBALE OTTIMIZZATA
def get_evaluate():
    global GLOBAL_EARLY_STOPPING

    # Carica dati globali
    try:
        X_global, y_global = load_server_data()
        input_shape = X_global.shape[1]
    except Exception as e:
        print(f"Errore caricamento dati server: {e}")
        X_global = np.random.random((100, 30)).astype(np.float32)
        y_global = np.random.randint(0, 2, 100).astype(np.float32)
        input_shape = 30
        print("Usando dati fittizi per server")
    
    def evaluate(server_round, parameters, config):
        """Valutazione globale ottimizzata con early stopping"""
        print(f"\n🔥 DEBUG: EVALUATE CHIAMATA PER ROUND {server_round}")
        print(f"\n=== VALUTAZIONE GLOBALE OTTIMIZZATA ROUND {server_round} ===")
        
        try:
            # PROVA PRIMA MODELLO OTTIMIZZATO
            try:
                model = create_server_model(input_shape)
                print(f"Usando modello ottimizzato per valutazione")
            except Exception as e:
                print(f"Fallback al modello manuale per valutazione: {e}")
                model = create_server_model_fallback(input_shape)
            
            # Verifica compatibilità pesi
            if len(parameters) != len(model.get_weights()):
                print(f"Incompatibilità pesi: ricevuti {len(parameters)}, attesi {len(model.get_weights())}")
                return 1.0, {"error": "weight_mismatch", "global_samples": 0}
            
            model.set_weights(parameters)

            # Valutazione con metriche corrette
            results = model.evaluate(X_global, y_global, verbose=0)
            loss = results[0]
            accuracy = results[1] if len(results) > 1 else 0.0
            precision = results[2] if len(results) > 2 else 0.0
            recall = results[3] if len(results) > 3 else 0.0
            
            # Predizioni per analisi dettagliata
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            
            # Matrice confusione
            tn, fp, fn, tp = confusion_matrix(y_global, y_pred_binary).ravel()
            
            # Metriche aggiuntive
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 calcolato manualmente
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            try:
                auc_roc_manual = roc_auc_score(y_global, y_pred_prob)
            except Exception:
                auc_roc_manual = 0.5

            print(f"RISULTATI GLOBALI OTTIMIZZATI:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.1f}%)")
            print(f"  - F1-Score: {f1:.4f} ({f1*100:.1f}%)")
            print(f"  - AUC-ROC: {auc_roc_manual:.4f} ({auc_roc_manual*100:.1f}%)")
            print(f"  - Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
            print(f"  - Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
            print(f" Confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            eval_metrics = {
                "global_accuracy": float(accuracy),
                "global_precision": float(precision),
                "global_recall": float(recall),
                "global_f1_score": float(f1),
                "global_auc_roc": float(auc_roc_manual),
                "global_specificity": float(specificity),
                "global_sensitivity": float(sensitivity),
                "global_samples": int(len(X_global)),
            }
            
            print(f"DEBUG: EVALUATE RITORNA METRICHE ROUND {server_round}")
            print(f"Loss: {float(loss):.6f}")
            print(f"Metriche keys: {list(eval_metrics.keys())}")
            
            # SALVA METRICHE DIRETTAMENTE NEL TRACKER
            global GLOBAL_METRICS_TRACKER
            if GLOBAL_METRICS_TRACKER:
                # Aggiunge loss alle metriche
                eval_metrics_with_loss = eval_metrics.copy()
                eval_metrics_with_loss['global_loss'] = float(loss)
                
                print(f"DEBUG: Salvataggio diretto nel TRACKER Round {server_round}")
                print(f"eval_metrics keys: {list(eval_metrics_with_loss.keys())}")
                
                GLOBAL_METRICS_TRACKER.add_round_metrics(
                    round_num=server_round,
                    fit_metrics=None,
                    evaluate_metrics=eval_metrics_with_loss
                )
            else:
                print(f"DEBUG: GLOBAL_METRICS_TRACKER è None!")

            return float(loss), eval_metrics
            
        except Exception as e:
            print(f"Errore valutazione globale ottimizzata: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {"error": str(e), "global_samples": 0}
    
    return evaluate

# MAIN FUNCTION
def main():

    global GLOBAL_EARLY_STOPPING

    print(f"\nAVVIO SERVER FEDERATO OTTIMIZZATO")
    print("=" * 80)
    print("- Architettura: Ottimizzata per SmartGrid")
    print("=" * 80)
    
    config = ServerConfig()

    # Inizializza early stopping globale
    if config.ENABLE_FEDERATED_EARLY_STOPPING:
        GLOBAL_EARLY_STOPPING = FederatedEarlyStopping(
            monitor=config.FEDERATED_MONITOR,
            min_delta=config.FEDERATED_MIN_DELTA,
            patience=config.FEDERATED_PATIENCE,
            mode=config.FEDERATED_MODE,
            min_rounds=config.FEDERATED_MIN_ROUNDS
        )
        print(f"Global Early Stopping ENABLED!")
    else:
        print(f"Global Early Stopping DISABLED!")

    # Strategia con parametri iniziali ottimizzati
    strategy = Strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.MIN_CLIENTS,
        min_evaluate_clients=config.MIN_CLIENTS,
        min_available_clients=config.MIN_CLIENTS,
        evaluate_fn=get_evaluate(),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    server_config = fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS)
    
    print("Server ottimizzato pronto!")
    print("Connettere client ottimizzati")
    print("\nIl training inizierà quando almeno 2 client saranno connessi.")
    print("=" * 80)
    
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=server_config,
            strategy=strategy,
        )
        
    except KeyboardInterrupt:
        print(f"\nServer ottimizzato fermato dall'utente")
    except Exception as e:
        print(f"Errore durante l'avvio del server ottimizzato: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # GENERA RESOCONTO FINALE
        global GLOBAL_METRICS_TRACKER
        if GLOBAL_METRICS_TRACKER:
            print(f"\nGenerazione resoconto finale...")
            GLOBAL_METRICS_TRACKER.generate_final_report()

if __name__ == "__main__":
    main()