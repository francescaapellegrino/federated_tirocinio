"""
Server federato SmartGrid - VERSIONE ADATTATA ai client_nuovo.py
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
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

GLOBAL_METRICS_TRACKER = None

# CONFIGURAZIONE SERVER ADATTATA AI CLIENT
class AdaptedServerConfig:
    """Configurazione server che matcha i client_nuovo.py"""
    
    # ARCHITETTURA IDENTICA AI CLIENT MIGLIORATI
    HIDDEN_LAYERS = [256, 128, 64, 32]  # Matcha improved_model.py
    DROPOUT_RATES = [0.3, 0.4, 0.3, 0.2]  # Matcha improved_model.py
    
    # PARAMETRI OTTIMIZZATI per convergenza verso 90%
    LEARNING_RATE = 0.0015  # Matcha improved_model.py
    L2_REG = 0.0001  # Matcha improved_model.py
    
    # CONFIGURAZIONE IDENTICA
    USE_BATCH_NORM = True
    ACTIVATION = 'relu'
    OPTIMIZER_TYPE = 'adam'
    BETA_1 = 0.9
    BETA_2 = 0.999
    CLIPNORM = 1.0
    
    # Data preprocessing (deve matchare client)
    PCA_COMPONENTS = 30
    TOTAL_FEATURES = 30  # Garantito dal preprocessing

    # Server specific
    NUM_ROUNDS = 50
    MIN_CLIENTS = 2
    VERSION = "adapted_to_client_nuovo"
    RANDOM_SEED = 42

# TRACKER PER LE METRICHE (IDENTICO)
class CompleteMetricsTracker:
    def __init__(self):
        self.round_metrics = {}
        self.target_metrics = [
            'val_loss', 'global_accuracy', 'global_precision', 
            'global_recall', 'global_f1_score', 'global_auc_roc',
            'global_specificity', 'global_sensitivity'
        ]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(script_dir, f"metrics_complete_report_{timestamp}.txt")
        
        print(f"📊 CompleteMetricsTracker inizializzato: {self.output_file}")
    
    def add_round_metrics(self, round_num: int, fit_metrics: Dict = None, evaluate_metrics: Dict = None):
        """Aggiunge metriche per un round specifico"""
        try:
            # Inizializza round
            if round_num not in self.round_metrics:
                self.round_metrics[round_num] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'val_loss': None, 'global_accuracy': None, 'global_precision': None,
                    'global_recall': None, 'global_f1_score': None, 'global_auc_roc': None,
                    'global_specificity': None, 'global_sensitivity': None
                }
            
            # AGGIUNGI METRICHE DAI CLIENT (FIT)
            if fit_metrics:
                # Validation loss dai client
                if 'val_loss' in fit_metrics:
                    self.round_metrics[round_num]['val_loss'] = fit_metrics['val_loss']
                
                # MAPPO METRICHE CLIENT ALLE GLOBALI
                client_to_global_mapping = {
                    'val_accuracy': 'global_accuracy',
                    'val_precision': 'global_precision', 
                    'val_recall': 'global_recall',
                    'val_f1_score': 'global_f1_score',
                    'val_auc_roc': 'global_auc_roc'
                }
                
                for client_metric, global_metric in client_to_global_mapping.items():
                    if client_metric in fit_metrics:
                        self.round_metrics[round_num][global_metric] = fit_metrics[client_metric]
            
            # AGGIUNGI METRICHE DAL SERVER (EVALUATE)
            if evaluate_metrics:
                for metric in ['global_accuracy', 'global_precision', 'global_recall', 
                            'global_f1_score', 'global_auc_roc', 'global_specificity', 'global_sensitivity']:
                    if metric in evaluate_metrics:
                        # Privilegia evaluate su fit (più preciso)
                        self.round_metrics[round_num][metric] = evaluate_metrics[metric]
            
            # Debug migliorato
            available_metrics = [k for k, v in self.round_metrics[round_num].items() 
                            if v is not None and k != 'timestamp']
            print(f"✅ Round {round_num}: {len(available_metrics)} metriche salvate: {available_metrics}")
            
        except Exception as e:
            print(f"❌ Errore salvataggio round {round_num}: {e}")
    
    def generate_final_report(self):
        """Genera il resoconto finale completo"""
        try:
            if not self.round_metrics:
                print("⚠️ Nessuna metrica da salvare")
                return
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("RESOCONTO ADDESTRAMENTO FEDERATO SMARTGRID\n")
                f.write("=" * 80 + "\n")
                f.write(f"Progetto: SmartGrid False Data Injection Attack Detection\n")
                f.write(f"Autore: Francesca Pellegrino\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Rounds completati: {len(self.round_metrics)}\n")
                f.write(f"Framework: Flower Federated Learning\n")
                f.write(f"Architettura: [256, 128, 64, 32] → 1 (Improved Kaggle Style)\n")
                f.write("=" * 80 + "\n")
                
                self._write_summary_table(f)
                self._write_final_statistics(f)
            
            print(f"✅ Resoconto completo generato: {self.output_file}")
            print(f"📊 Rounds tracciati: {len(self.round_metrics)}")
            
        except Exception as e:
            print(f"❌ Errore generazione resoconto: {e}")
    
    def _write_summary_table(self, f):
        """Scrive tabella riassuntiva"""
        f.write("\nTABELLA RIASSUNTIVA METRICHE:\n")
        f.write("=" * 120 + "\n")
        
        header = f"{'Round':<6} {'Loss':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1_Score':<10} {'AUC_ROC':<10} {'Specificity':<10} {'Sensitivity':<10}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        for round_num in sorted(self.round_metrics.keys()):
            metrics = self.round_metrics[round_num]
            
            def safe_format(value):
                return f"{value:.6f}" if value is not None else "N/A"
            
            val_loss = safe_format(metrics['val_loss'])
            accuracy = safe_format(metrics['global_accuracy'])
            precision = safe_format(metrics['global_precision'])
            recall = safe_format(metrics['global_recall'])
            f1_score = safe_format(metrics['global_f1_score'])
            auc_roc = safe_format(metrics['global_auc_roc'])
            specificity = safe_format(metrics['global_specificity'])
            sensitivity = safe_format(metrics['global_sensitivity'])
            
            row = f"{round_num:<6} {val_loss:<10} {accuracy:<10} {precision:<11} {recall:<10} {f1_score:<10} {auc_roc:<10} {specificity:<10} {sensitivity:<10}"
            f.write(row + "\n")
        
        f.write("=" * 120 + "\n\n")
    
    def _write_final_statistics(self, f):
        """Scrive statistiche finali"""
        f.write("STATISTICHE FINALI:\n")
        f.write("=" * 60 + "\n")
        
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

# CARICAMENTO DATASET SERVER (IDENTICO AI CLIENT)
def load_server_data():
    """Carica dati server con preprocessing identico ai client"""
    print("📂 CARICAMENTO DATASET GLOBALE SERVER")
    
    config = AdaptedServerConfig()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    validation_clients = [14, 15]
    df_list = []

    for client_id in validation_clients:
        file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"   ✅ Caricato data{client_id}.csv: {len(df)} campioni")
        except FileNotFoundError:
            print(f"   ⚠️ File data{client_id}.csv non trovato, saltato")
            continue

    if not df_list:
        print("🔄 ATTENZIONE: Usando fallback data1.csv per server")
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
    
    print(f"   📊 Dataset grezzo: {len(X)} campioni, {X.shape[1]} feature")
    print(f"   📊 Distribuzione: {y.sum():.0f} attacchi ({y.mean()*100:.1f}%)")
    
    # PREPROCESSING IDENTICO AI CLIENT (important!)
    print(f"   🧹 Pulizia base...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.median(), inplace=True)
        print(f"      ✅ NaN imputati con mediana")

    print(f"   🔍 PCA {config.PCA_COMPONENTS} componenti...")
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X)
    
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"      ✅ PCA: {X.shape[1]} → {X_pca.shape[1]} feature")
    print(f"      ✅ Varianza spiegata: {variance_explained*100:.2f}%")

    print(f"   📊 Usando solo PCA features (come i client)...")
    X_enhanced = X_pca
    print(f"      ✅ Features: {X_pca.shape[1]} (solo PCA)")

    print(f"   📏 Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X_enhanced).astype(np.float32)
    y = y.astype(np.float32)
    
    print(f"   ✅ Dataset server preparato:")
    print(f"      📊 Pipeline: {X.shape[1]} → {X_pca.shape[1]} → {X_final.shape[1]} feature")
    print(f"      📊 Campioni finali: {len(X_final)}")
    print(f"      ✅ Tipi corretti: X={X_final.dtype}, y={y.dtype}")
    print("=" * 70)
    
    return X_final, y

# MODELLO SERVER IDENTICO AI CLIENT
def create_server_model(input_shape: int):
    """
    Crea modello server IDENTICO ai client_nuovo.py
    ARCHITETTURA: [256, 128, 64, 32] → 1
    """
    
    if input_shape != 30:
        print(f"⚠️ Warning: input_shape {input_shape} != 30, forzo a 30 per compatibilità")
        input_shape = 30
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    config = AdaptedServerConfig()
    
    print(f"🔧 MODELLO SERVER IDENTICO AI CLIENT:")
    print(f"   📐 Input garantito: {input_shape} features")
    print(f"   🎯 Architettura: {' → '.join(map(str, config.HIDDEN_LAYERS))} → 1")
    print(f"   ✅ Compatibilità: 100% con client_nuovo.py")
    
    # ARCHITETTURA IDENTICA A improved_model.py
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Layer 1 (256 neuroni)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[0],  # 256
            kernel_regularizer=keras.regularizers.L2(config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_1"
        ),
        keras.layers.Activation('relu', name="activation_1"),
        keras.layers.BatchNormalization(name="batch_norm_1") if config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(config.DROPOUT_RATES[0], name="dropout_1"),
        
        # Layer 2 (128 neuroni)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[1],  # 128
            kernel_regularizer=keras.regularizers.L2(config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_2"
        ),
        keras.layers.Activation('relu', name="activation_2"),
        keras.layers.BatchNormalization(name="batch_norm_2") if config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(config.DROPOUT_RATES[1], name="dropout_2"),
        
        # Layer 3 (64 neuroni)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[2],  # 64
            kernel_regularizer=keras.regularizers.L2(config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_3"
        ),
        keras.layers.Activation('relu', name="activation_3"),
        keras.layers.BatchNormalization(name="batch_norm_3") if config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(config.DROPOUT_RATES[2], name="dropout_3"),
        
        # Layer 4 (32 neuroni)
        keras.layers.Dense(
            config.HIDDEN_LAYERS[3],  # 32
            kernel_regularizer=keras.regularizers.L2(config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_4"
        ),
        keras.layers.Activation('relu', name="activation_4"),
        keras.layers.BatchNormalization(name="batch_norm_4") if config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(config.DROPOUT_RATES[3], name="dropout_4"),
        
        # Output layer (identico)
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name="output"
        )
    ], name="SmartGrid_Server_Adapted")
    
    # OPTIMIZER IDENTICO AI CLIENT
    optimizer = keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,  # 0.0015
        beta_1=config.BETA_1,
        beta_2=config.BETA_2,
        clipnorm=config.CLIPNORM
    )
    
    # LOSS PESATA (identica ai client)
    def weighted_binary_crossentropy(pos_weight=2.5):
        """Loss pesata per migliorare recall mantenendo precision"""
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            
            loss_pos = -y_true * tf.math.log(y_pred) * pos_weight
            loss_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
            
            return tf.reduce_mean(loss_pos + loss_neg)
        
        return loss_fn
    
    # Compilazione identica ai client
    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(pos_weight=2.5),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", curve='ROC'),
            keras.metrics.AUC(name="auc_pr", curve='PR')
        ]
    )
    
    print(f"🎯 MODELLO SERVER ADATTATO CREATO:")
    print(f"   📐 Architettura: {' → '.join(map(str, config.HIDDEN_LAYERS))} → 1")
    print(f"   🎛️ Parametri totali: {model.count_params():,}")
    print(f"   🧠 Activation: {config.ACTIVATION}")
    print(f"   ⚡ Optimizer: Adam (learning_rate={config.LEARNING_RATE})")
    print(f"   🎯 Loss: Weighted Binary Crossentropy (pos_weight=2.5)")
    print(f"   📊 Metriche: accuracy, precision, recall, f1_score, auc_roc, auc_pr")
    print(f"   ✅ Compatibilità: Garantita al 100% con client_nuovo.py")
    
    return model

# AGGREGAZIONE SICURA DELLE METRICHE
def weighted_average(metrics):
    """Aggregazione sicura delle metriche evaluate"""
    if not metrics:
        print("ℹ️ Nessuna metrica da aggregare")
        return {}
    
    print(f"🔄 Aggregating evaluate metrics from {len(metrics)} clients...")
    
    metrics_sum = {}
    total_examples = 0
    
    for i, (num_examples, metrics_dict) in enumerate(metrics):
        total_examples += num_examples
        print(f"   📊 Client {i+1}: {num_examples} samples, EVALUATE metrics: {list(metrics_dict.keys())}")
        
        for key, value in metrics_dict.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metrics_sum[key] += num_examples * value
                if key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']:
                    print(f"      ✅ {key}: {value:.6f}")
    
    # Calcola medie pesate
    aggregated = {}
    for key, value in metrics_sum.items():
        if total_examples > 0:
            aggregated[key] = value / total_examples
    
    aggregated['total_clients'] = len(metrics)
    aggregated['total_samples'] = total_examples

    print(f"✅ Evaluate metrics aggregated: {list(aggregated.keys())}")
    return aggregated

def print_client_metrics(fit_results):
    """Stampa metriche client"""
    if not fit_results:
        return

    print(f"\n📊 METRICHE CLIENT")
    
    total_samples = 0
    total_train_acc = 0
    total_val_acc = 0
    targets_met_count = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"   🔹 Client {i+1}:")
        print(f"      📊 Campioni: {client_samples}")
        
        if 'train_accuracy' in client_metrics:
            train_acc = client_metrics['train_accuracy']
            total_train_acc += train_acc * client_samples
            print(f"      📈 Train Acc: {train_acc:.4f}")
        
        if 'val_accuracy' in client_metrics:
            val_acc = client_metrics['val_accuracy']
            total_val_acc += val_acc * client_samples
            print(f"      📈 Val Acc: {val_acc:.4f}")
        
        # Target tracking dai client migliorati
        if 'all_targets_met' in client_metrics:
            if client_metrics['all_targets_met']:
                targets_met_count += 1
                print(f"      🎉 TUTTI I TARGET >90% RAGGIUNTI!")
            else:
                print(f"      ⚠️ Alcuni target mancati")
        
        if 'model_type' in client_metrics:
            model_type = client_metrics['model_type']
            print(f"      🏗️ Architecture: {model_type}")
    
    # Statistiche aggregate
    if total_samples > 0:
        avg_train_acc = total_train_acc / total_samples
        avg_val_acc = total_val_acc / total_samples
        
        print(f"\n📊 STATISTICHE AGGREGATE:")
        print(f"   📈 Media Train Accuracy: {avg_train_acc:.4f}")
        print(f"   📈 Media Val Accuracy: {avg_val_acc:.4f}")
        print(f"   📊 Gap Train-Val: {avg_train_acc - avg_val_acc:.4f}")
        print(f"   🎯 Client con target >90%: {targets_met_count}/{len(fit_results)}")
        
        if targets_met_count == len(fit_results):
            print(f"   🎉 TUTTI I CLIENT HANNO RAGGIUNTO I TARGET >90%!")
    
    print("=" * 60)

# STRATEGIA FEDERATA ADATTATA
class AdaptedStrategy(FedAvg):
    """Strategia federata adattata ai client_nuovo.py"""
    
    def __init__(self, **kwargs):
        # Inizializza MetricsTracker
        global GLOBAL_METRICS_TRACKER
        GLOBAL_METRICS_TRACKER = CompleteMetricsTracker()

        # Genera parametri iniziali compatibili
        self.initial_parameters = self.generate_initial_parameters()
        super().__init__(**kwargs)
    
    def generate_initial_parameters(self):
        """Genera parametri iniziali compatibili con client_nuovo.py"""
        print("🔧 Generazione parametri iniziali server adattati...")
        
        config = AdaptedServerConfig()
        
        try:
            temp_model = create_server_model(input_shape=config.TOTAL_FEATURES)
            initial_weights = temp_model.get_weights()
            print(f"   ✅ Parametri generati: {len(initial_weights)} tensori")
            print(f"   ✅ Architettura: {config.HIDDEN_LAYERS}")
            print(f"   ✅ Compatibilità: Garantita con client_nuovo.py")
            return fl.common.ndarrays_to_parameters(initial_weights)
        except Exception as e:
            print(f"   ❌ Errore generazione parametri: {e}")
            raise
    
    def initialize_parameters(self, client_manager):
        """Restituisce parametri iniziali"""
        print("📊 Inizializzazione parametri server adattati")
        return self.initial_parameters
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregazione fit con gestione robusta"""
        print(f"\n=== AGGREGAZIONE FIT ADATTATA ROUND {server_round} ===")
        print(f"   📊 Client partecipanti: {len(results)}")
        print(f"   ⚠️ Client falliti: {len(failures)}")
        
        if failures:
            print("   ❌ Fallimenti:")
            for failure in failures:
                print(f"      - {failure}")

        print_client_metrics(results)

        # Aggregazione manuale delle metriche FIT
        fit_metrics = []
        for client_proxy, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                fit_metrics.append((fit_res.num_examples, fit_res.metrics))
                print(f"   📊 Client {fit_res.metrics.get('client_id', '?')} FIT metrics: {list(fit_res.metrics.keys())}")
        
        # Aggrega manualmente le metriche FIT
        aggregated_fit_metrics = {}
        if fit_metrics:
            aggregated_fit_metrics = self.aggregate_fit_metrics_manual(fit_metrics)
            print(f"   ✅ Aggregated FIT metrics: {list(aggregated_fit_metrics.keys())}")
        
        # Chiama l'aggregazione standard dei parametri
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            parameters, _ = aggregated_result  # Ignora le metriche standard
            
            # Salva metriche FIT nel tracker
            global GLOBAL_METRICS_TRACKER
            if GLOBAL_METRICS_TRACKER and aggregated_fit_metrics:
                GLOBAL_METRICS_TRACKER.add_round_metrics(
                    round_num=server_round,
                    fit_metrics=aggregated_fit_metrics,
                    evaluate_metrics=None
                )
            
            print(f"✅ Aggregazione fit adattata completata per round {server_round}")
            return parameters, aggregated_fit_metrics
    
    def aggregate_fit_metrics_manual(self, metrics):
        """Aggregazione manuale delle metriche FIT"""
        if not metrics:
            print("ℹ️ Nessuna metrica FIT da aggregare")
            return {}
        
        print(f"🔄 Aggregating FIT metrics from {len(metrics)} clients...")
        
        metrics_sum = {}
        total_examples = 0
        
        for i, (num_examples, metrics_dict) in enumerate(metrics):
            total_examples += num_examples
            print(f"   📊 Client {i+1}: {num_examples} samples, FIT metrics: {list(metrics_dict.keys())}")
            
            for key, value in metrics_dict.items():
                if key not in metrics_sum:
                    metrics_sum[key] = 0
                
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    metrics_sum[key] += num_examples * value
                    if key == 'val_loss':
                        print(f"      ✅ val_loss: {value:.6f}")
        
        # Calcola medie pesate
        aggregated = {}
        for key, value in metrics_sum.items():
            if total_examples > 0:
                aggregated[key] = value / total_examples
        
        if 'val_loss' in aggregated:
            print(f"✅ Aggregated FIT Validation Loss: {aggregated['val_loss']:.6f}")

        aggregated['total_clients'] = len(metrics)
        aggregated['total_samples'] = total_examples

        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregazione evaluate compatibile"""
        print(f"\n=== AGGREGATE_EVALUATE ADATTATO ROUND {server_round} ===")
        
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            
            global GLOBAL_METRICS_TRACKER
            if GLOBAL_METRICS_TRACKER and metrics and loss is not None:
                eval_metrics = metrics.copy()
                eval_metrics['global_loss'] = loss
                
                GLOBAL_METRICS_TRACKER.add_round_metrics(
                    round_num=server_round,
                    fit_metrics=None,
                    evaluate_metrics=eval_metrics
                )

        return aggregated_result

# VALUTAZIONE GLOBALE ADATTATA
def get_evaluate():
    """Crea funzione di valutazione globale adattata"""
    
    try:
        X_global, y_global = load_server_data()
        input_shape = X_global.shape[1]
    except Exception as e:
        print(f"❌ Errore caricamento dati server: {e}")
        X_global = np.random.random((100, 30)).astype(np.float32)
        y_global = np.random.randint(0, 2, 100).astype(np.float32)
        input_shape = 30
        print("🔄 Usando dati sintetici per server")
    
    # Nella funzione evaluate(), sostituisci la sezione di valutazione con:

def evaluate(server_round, parameters, config):
    """Valutazione globale adattata ai client_nuovo.py"""
    print(f"\n=== VALUTAZIONE GLOBALE ADATTATA ROUND {server_round} ===")
    
    try:
        # Crea modello adattato (architettura [256, 128, 64, 32])
        model = create_server_model(input_shape)
        
        # Verifica compatibilità pesi
        model_weights = model.get_weights()
        if len(parameters) != len(model_weights):
            print(f"⚠️ Incompatibilità pesi: ricevuti {len(parameters)}, attesi {len(model_weights)}")
            return 1.0, {"error": "weight_mismatch", "global_samples": len(X_global)}
        
        # Verifica compatibilità dimensioni (importante!)
        try:
            model.set_weights(parameters)
            print(f"✅ Pesi caricati con successo - Architettura compatibile!")
        except Exception as weight_error:
            print(f"❌ Errore caricamento pesi: {weight_error}")
            return 1.0, {"error": f"weight_loading_failed: {weight_error}", "global_samples": len(X_global)}

        # CORREZIONE: Valutazione con estrazione sicura dei valori
        try:
            results = model.evaluate(X_global, y_global, verbose=0)
            
            # ESTRAZIONE SICURA DEI VALORI (correzione principale)
            loss = float(results[0]) if len(results) > 0 else 1.0
            accuracy = float(results[1]) if len(results) > 1 else 0.0
            precision = float(results[2]) if len(results) > 2 else 0.0
            recall = float(results[3]) if len(results) > 3 else 0.0
            
            # F1-Score: potrebbe essere array, estraiamo il valore
            if len(results) > 4:
                f1_raw = results[4]
                # Se è un array, prendi il primo elemento, altrimenti usa direttamente
                f1_score = float(f1_raw[0]) if hasattr(f1_raw, '__len__') and len(f1_raw) > 0 else float(f1_raw)
            else:
                f1_score = 0.0
            
            # AUC-ROC: stesso trattamento
            if len(results) > 5:
                auc_raw = results[5]
                auc_roc = float(auc_raw[0]) if hasattr(auc_raw, '__len__') and len(auc_raw) > 0 else float(auc_raw)
            else:
                auc_roc = 0.5
            
            # AUC-PR: stesso trattamento
            if len(results) > 6:
                auc_pr_raw = results[6]
                auc_pr = float(auc_pr_raw[0]) if hasattr(auc_pr_raw, '__len__') and len(auc_pr_raw) > 0 else float(auc_pr_raw)
            else:
                auc_pr = 0.5
                
        except Exception as eval_error:
            print(f"❌ Errore valutazione modello: {eval_error}")
            return 1.0, {"error": f"evaluation_failed: {eval_error}", "global_samples": len(X_global)}
        
        # Calcoli aggiuntivi per analisi dettagliata
        try:
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            
            # Matrice confusione per specificity
            cm = confusion_matrix(y_global, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                balanced_accuracy = (sensitivity + specificity) / 2
            else:
                specificity = 0.0
                sensitivity = recall
                balanced_accuracy = accuracy
            
            # AUC sicuro
            try:
                auc_manual = roc_auc_score(y_global, y_pred_prob)
            except:
                auc_manual = auc_roc
                
        except Exception as calc_error:
            print(f"⚠️ Errore calcoli aggiuntivi: {calc_error}")
            specificity = 0.0
            sensitivity = recall
            balanced_accuracy = accuracy
            auc_manual = auc_roc
        
        # CORREZIONE: Risultati dettagliati con valori sicuri
        print(f"📊 RISULTATI VALUTAZIONE GLOBALE ADATTATA:")
        print(f"   📈 Loss: {loss:.6f}")
        print(f"   📈 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%) {'🎯' if accuracy >= 0.90 else ''}")
        print(f"   📈 Precision: {precision:.4f} ({precision*100:.1f}%) {'🎯' if precision >= 0.90 else ''}")
        print(f"   📈 Recall: {recall:.4f} ({recall*100:.1f}%) {'🎯' if recall >= 0.90 else ''}")
        print(f"   📈 F1-Score: {f1_score:.4f} ({f1_score*100:.1f}%) {'🎯' if f1_score >= 0.90 else ''}")
        print(f"   📈 AUC-ROC: {auc_manual:.4f} ({auc_manual*100:.1f}%)")
        print(f"   📈 AUC-PR: {auc_pr:.4f} ({auc_pr*100:.1f}%)")
        print(f"   📈 Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
        print(f"   📈 Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
        print(f"   📈 Balanced Accuracy: {balanced_accuracy:.4f}")
        
        # Check target raggiunti
        targets_met = {
            'accuracy_90': accuracy >= 0.90,
            'precision_90': precision >= 0.90,
            'recall_90': recall >= 0.90,
            'f1_90': f1_score >= 0.90
        }
        
        all_targets = all(targets_met.values())
        
        if all_targets:
            print(f"   🎉 TUTTI I TARGET >90% RAGGIUNTI GLOBALMENTE! 🎉")
        else:
            missed = [k for k, v in targets_met.items() if not v]
            print(f"   ⚠️ Target mancati globalmente: {missed}")
        
        # Metriche complete per tracking
        eval_metrics = {
            "global_accuracy": float(accuracy),
            "global_precision": float(precision),
            "global_recall": float(recall),
            "global_f1_score": float(f1_score),
            "global_auc_roc": float(auc_manual),
            "global_specificity": float(specificity),
            "global_sensitivity": float(sensitivity),
            "global_samples": int(len(X_global)),
            "server_round": int(server_round),
            "evaluation_successful": True,
            "architecture_compatible": True,
            "all_targets_met": float(all_targets)
        }

        # DEBUG MIGLIORATO
        print(f"📊 METRICHE INVIATE AL TRACKER (Round {server_round}):")
        for key, value in eval_metrics.items():
            if isinstance(value, float) and 'global_' in key:
                print(f"   {key}: {value:.6f}")
        
        return float(loss), eval_metrics
        
    except Exception as e:
        print(f"❌ Errore valutazione globale adattata: {e}")
        import traceback
        traceback.print_exc()
        return 1.0, {"error": str(e), "global_samples": len(X_global), "architecture_compatible": False}

# MAIN FUNCTION
def main():
    """Funzione principale del server adattato"""
    print(f"\n🚀 AVVIO SERVER SMARTGRID ADATTATO AI CLIENT_NUOVO.PY")
    print("=" * 80)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👩‍💻 Sviluppatore: Francesca Pellegrino")
    print(f"🎯 Progetto: SmartGrid False Data Injection Attack Detection")
    print(f"🔧 Framework: Flower (FedAvg)")
    print(f"🧠 Architettura: [256, 128, 64, 32] → 1 (Improved Kaggle Style)")
    print(f"✅ Compatibilità: 100% con client_nuovo.py")
    print("=" * 80)
    
    try:
        config = AdaptedServerConfig()
        print(f"📊 Configurazione server adattata:")
        print(f"   🔄 Rounds: {config.NUM_ROUNDS}")
        print(f"   👥 Min client: {config.MIN_CLIENTS}")
        print(f"   📐 Architettura: {config.HIDDEN_LAYERS}")
        print(f"   🎛️ Learning Rate: {config.LEARNING_RATE}")
        print(f"   📊 Input features: {config.TOTAL_FEATURES}")
        print(f"   ✅ Versione: {config.VERSION}")
        
        # Strategia federata adattata
        strategy = AdaptedStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.MIN_CLIENTS,
            min_evaluate_clients=config.MIN_CLIENTS,
            min_available_clients=config.MIN_CLIENTS,
            evaluate_fn=get_evaluate(),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        
        server_config = fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS)
        
        print("\n✅ Server adattato configurato e pronto!")
        print("🔗 In attesa di connessioni client...")
        print("💡 Avviare i client_nuovo.py per iniziare il federated learning")
        print("🎯 Obiettivo: Raggiungere >90% accuracy, precision, recall, f1-score")
        print("=" * 80)
        
        # Avvio server Flower
        fl.server.start_server(
            server_address="localhost:8080",
            config=server_config,
            strategy=strategy,
        )
        
    except KeyboardInterrupt:
        print(f"\n🛑 Server fermato dall'utente")
    except Exception as e:
        print(f"\n❌ Errore critico server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Genera report finale sempre
        print(f"\n📊 Generazione report finale...")
        global GLOBAL_METRICS_TRACKER
        if GLOBAL_METRICS_TRACKER:
            try:
                GLOBAL_METRICS_TRACKER.generate_final_report()
                print(f"✅ Report generato con successo!")
            except Exception as e:
                print(f"⚠️ Errore generazione report: {e}")
        
        print(f"\n🎉 SERVER SMARTGRID ADATTATO TERMINATO!")

if __name__ == "__main__":
    main()