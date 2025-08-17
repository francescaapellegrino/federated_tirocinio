"""
Server federato SmartGrid 
Author: francescaapellegrino
"""

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

# ============================================================================
# 🔧 SERVER CONFIGURATION 
# ============================================================================

class HybridServerConfig:
    """Configurazione server."""
    
    # Model architecture
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT_RATES = [0.2, 0.15, 0.1]
    LEARNING_RATE = 0.001
    
    # Data preprocessing
    PCA_COMPONENTS = 30
    STATISTICAL_FEATURES = 12
    TOTAL_FEATURES = 42
    
    # Server specific
    NUM_ROUNDS = 100
    MIN_CLIENTS = 2
    
    # System info
    VERSION = "2.5"
    RANDOM_SEED = 42

# ============================================================================
# 🔧 SERVER FEATURE ENGINEERING
# ============================================================================

class ServerFeatureEngineer:
    """Feature engineering server."""
    
    def add_statistical_features(self, X):
        """12 statistical features."""
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

# ============================================================================
# 📊 GLOBAL DATA LOADING
# ============================================================================

def load_hybrid_server_data_v25():
    """
    Carica dataset globale per server con preprocessing.
    """
    print("=== CARICAMENTO DATASET GLOBALE SERVER ===")
    
    config = HybridServerConfig()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    validation_clients = [14, 15]
    df_list = []

    for client_id in validation_clients:
        file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"  - Caricato data{client_id}.csv: {len(df)} campioni")
        except FileNotFoundError:
            print(f"  - File data{client_id}.csv non trovato, saltato")
            continue

    if not df_list:
        print("  - ATTENZIONE: Usando fallback data1.csv per server")
        fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
        try:
            df_fallback = pd.read_csv(fallback_path)
            df_list = [df_fallback.sample(n=min(1000, len(df_fallback)), random_state=42)]
        except FileNotFoundError:
            raise FileNotFoundError("Impossibile caricare dati per server")
    
    # Combina dataset
    df_global = pd.concat(df_list, ignore_index=True)
    X = df_global.drop(columns=["marker"])
    y = (df_global["marker"] != "Natural").astype(int)
    
    print(f"  - Dataset grezzo: {len(X)} campioni, {X.shape[1]} feature")
    print(f"  - Distribuzione: {y.sum()} attacchi ({y.mean()*100:.1f}%)")
    
    # STEP 1: Pulizia (IDENTICA al client)
    print(f"  🔧 Pulizia base...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.median(), inplace=True)
        print(f"     - NaN imputati con mediana")
    
    # STEP 2: PCA (IDENTICO al client)
    print(f"  🎯 PCA {config.PCA_COMPONENTS} componenti...")
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(X)
    
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"     - PCA: {X.shape[1]} → {X_pca.shape[1]} feature")
    print(f"     - Varianza spiegata: {variance_explained*100:.2f}%")
    
    # STEP 3: Statistical Features (IDENTICHE al client)
    print(f"  🔧 Statistical features...")
    feature_engineer = ServerFeatureEngineer()
    X_enhanced = feature_engineer.add_statistical_features(X_pca)
    print(f"     - Features: {X_pca.shape[1]} → {X_enhanced.shape[1]} (+{config.STATISTICAL_FEATURES})")
    
    # STEP 4: Normalizzazione finale (IDENTICA al client)
    print(f"  ⚡ Normalizzazione finale...")
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X_enhanced)
    
    print(f"✅ Dataset server preparato:")
    print(f"   - Pipeline: {X.shape[1]} → {X_pca.shape[1]} → {X_final.shape[1]} feature")
    print(f"   - Campioni finali: {len(X_final)}")
    print(f"   - Preprocessing: IDENTICO ai client per consistenza")
    print("=" * 70)
    
    return X_final, y

# ============================================================================
# 🧠 SERVER MODEL (IDENTICO AL CLIENT)
# ============================================================================

def create_hybrid_server_model_v25(input_shape: int) -> keras.Model:
    """
    Crea modello server IDENTICO all'architettura client.
    """
    config = HybridServerConfig()
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # IDENTICA architettura al client
        keras.layers.Dense(
            config.HIDDEN_LAYERS[0], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED),
            name="dense_1"
        ),
        keras.layers.BatchNormalization(name="batch_norm_1"),
        keras.layers.Dropout(config.DROPOUT_RATES[0], seed=config.RANDOM_SEED, name="dropout_1"),
        
        keras.layers.Dense(
            config.HIDDEN_LAYERS[1], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+1),
            name="dense_2"
        ),
        keras.layers.BatchNormalization(name="batch_norm_2"),
        keras.layers.Dropout(config.DROPOUT_RATES[1], seed=config.RANDOM_SEED+1, name="dropout_2"),
        
        keras.layers.Dense(
            config.HIDDEN_LAYERS[2], 
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+2),
            name="dense_3"
        ),
        keras.layers.BatchNormalization(name="batch_norm_3"),
        keras.layers.Dropout(config.DROPOUT_RATES[2], seed=config.RANDOM_SEED+2, name="dropout_3"),
        
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer=keras.initializers.GlorotUniform(seed=config.RANDOM_SEED+3),
            name="output"
        )
    ], name="SmartGrid_Server_Model")
    
    # IDENTICA compilation al client
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
    
    print(f"🧠 Server Model creato:")
    print(f"   - Architettura: {config.HIDDEN_LAYERS[0]}→{config.HIDDEN_LAYERS[1]}→{config.HIDDEN_LAYERS[2]}→1")
    print(f"   - Input shape: {input_shape}")
    print(f"   - Weight tensors: {len(model.get_weights())}")
    print(f"   - Compatibilità: 100% con client")
    
    return model

# ============================================================================
# 📊 ADVANCED METRICS AGGREGATION
# ============================================================================

def weighted_average_hybrid_v25(metrics):
    """Aggregazione metriche."""
    if not metrics:
        return {}
    
    metrics_sum = {}
    total_examples = 0
    threshold_opt_count = 0
    
    for num_examples, metrics_dict in metrics:
        total_examples += num_examples
        
        # Conta client con threshold optimization
        if metrics_dict.get('threshold_optimization_success', False):
            threshold_opt_count += 1
        
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
    
    # Aggiungi meta-statistiche
    aggregated['total_clients'] = len(metrics)
    aggregated['total_samples'] = total_examples
    aggregated['threshold_optimization_clients'] = threshold_opt_count
    
    return aggregated

def print_client_metrics_hybrid_v25(fit_results):
    """Stampa metriche client."""
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ===")
    
    total_samples = 0
    total_train_acc = 0
    total_val_acc = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1} (v2.5):")
        print(f"  - Campioni: {client_samples}")
        
        if 'train_accuracy' in client_metrics:
            train_acc = client_metrics['train_accuracy']
            total_train_acc += train_acc * client_samples
            print(f"  - Train Acc: {train_acc:.4f}")
        
        if 'val_accuracy' in client_metrics:
            val_acc = client_metrics['val_accuracy']
            total_val_acc += val_acc * client_samples
            print(f"  - Val Acc: {val_acc:.4f}")
        
        if 'statistical_features_count' in client_metrics:
            stat_features = client_metrics['statistical_features_count']
            print(f"  - 🔧 Statistical features: {stat_features}")
        
        if 'total_features' in client_metrics:
            total_features = client_metrics['total_features']
            print(f"  - 📊 Total features: {total_features}")
    
    # Statistiche aggregate
    if total_samples > 0:
        avg_train_acc = total_train_acc / total_samples
        avg_val_acc = total_val_acc / total_samples
        
        print(f"\n=== STATISTICHE AGGREGATE ===")
        print(f"Media Train Accuracy: {avg_train_acc:.4f}")
        print(f"Media Val Accuracy: {avg_val_acc:.4f}")
        print(f"Gap Train-Val: {avg_train_acc - avg_val_acc:.4f}")
    
    print("=" * 60)

# ============================================================================
# 🚀 FEDERATED STRATEGY
# ============================================================================

class HybridFedAvgV25(FedAvg):
    """Strategia FedAvg con initial parameters."""
    
    def __init__(self, **kwargs):
        # Genera parametri iniziali per evitare GrpcBridgeClosed
        self.initial_parameters = self._generate_initial_parameters()
        super().__init__(**kwargs)
    
    def _generate_initial_parameters(self):
        """Genera parametri iniziali per il modello."""
        print("🔧 Generazione parametri iniziali server...")
        
        config = HybridServerConfig()
        temp_model = create_hybrid_server_model_v25(input_shape=config.TOTAL_FEATURES)
        initial_weights = temp_model.get_weights()
        
        print(f"   - Parametri generati: {len(initial_weights)} tensori")
        print(f"   - Compatibilità: Client")
        
        return fl.common.ndarrays_to_parameters(initial_weights)
    
    def initialize_parameters(self, client_manager):
        """Restituisce parametri iniziali."""
        print("🔧 Inizializzazione parametri server")
        return self.initial_parameters
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregazione con metriche."""
        print(f"\n=== AGGREGAZIONE ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("❌ Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        print_client_metrics_hybrid_v25(results)
        
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            print(f"✅ Aggregazione completata per round {server_round}")
        else:
            print(f"❌ Aggregazione fallita per round {server_round}")
        
        return aggregated_result

# ============================================================================
# 🔧 GLOBAL EVALUATION FUNCTION
# ============================================================================

def get_hybrid_evaluate_fn_v25():
    """Funzione di valutazione globale."""
    
    # Carica dati globali
    try:
        X_global, y_global = load_hybrid_server_data_v25()
        input_shape = X_global.shape[1]
    except Exception as e:
        print(f"❌ Errore caricamento dati server: {e}")
        X_global = np.random.random((100, 42))
        y_global = np.random.randint(0, 2, 100)
        input_shape = 42
        print("⚠️ Usando dati fittizi per server")
    
    def evaluate(server_round, parameters, config):
        """Valutazione globale."""
        print(f"\n=== VALUTAZIONE GLOBALE ROUND {server_round} ===")
        
        try:
            # Crea modello identico ai client
            model = create_hybrid_server_model_v25(input_shape)
            
            # Verifica compatibilità pesi
            if len(parameters) != len(model.get_weights()):
                print(f"⚠️ Incompatibilità pesi: ricevuti {len(parameters)}, attesi {len(model.get_weights())}")
                return 1.0, {"error": "weight_mismatch", "global_samples": 0}
            
            model.set_weights(parameters)
            
            # Valutazione
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, f1, auc = results
            
            # Predizioni per analisi dettagliata
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.72).astype(int)
            
            # Matrice confusione
            tn, fp, fn, tp = confusion_matrix(y_global, y_pred_binary).ravel()
            
            # Metriche aggiuntive
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            try:
                auc_roc_manual = roc_auc_score(y_global, y_pred_prob)
            except Exception:
                auc_roc_manual = 0.5

            print(f"🔧 RISULTATI GLOBALI:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.1f}%)")
            print(f"  - F1-Score: {f1:.4f} ({f1*100:.1f}%)")
            print(f"  - AUC-ROC: {auc_roc_manual:.4f} ({auc_roc_manual*100:.1f}%)")
            print(f"  - Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
            print(f"  - Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
            print(f"📊 Confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # Valutazione vs obiettivi v2.5
            auc_vs_v1 = auc_roc_manual - 0.52  # v1 baseline
            acc_vs_v1 = accuracy - 0.687       # v1 baseline
            f1_vs_v1 = f1 - 0.817              # v1 baseline
            
            print(f"📈 PROGRESSO vs v1 baseline:")
            print(f"  - AUC: {auc_roc_manual:.4f} (Δ{auc_vs_v1:+.3f}, target +0.03)")
            print(f"  - Acc: {accuracy:.4f} (Δ{acc_vs_v1:+.3f}, target +0.013)")
            print(f"  - F1:  {f1:.4f} (Δ{f1_vs_v1:+.3f}, target +0.013)")
            
            # Valutazione successo
            success_score = 0
            if auc_roc_manual >= 0.55: success_score += 1
            if accuracy >= 0.70: success_score += 1
            if f1 >= 0.83: success_score += 1
            if specificity >= 0.15: success_score += 1
            
            if success_score >= 3:
                quality = "🟢 OBIETTIVI RAGGIUNTI"
            elif success_score >= 2:
                quality = "🔵 BUONI PROGRESSI"
            elif success_score >= 1:
                quality = "🟡 MIGLIORAMENTI PARZIALI"
            else:
                quality = "🔴 OBIETTIVI MANCATI"
            
            print(f"🎯 Valutazione: {quality} ({success_score}/4 obiettivi)")
            print("=" * 70)
            sys.stdout.flush()
            
            return float(loss), {
                "global_accuracy": float(accuracy),
                "global_precision": float(precision),
                "global_recall": float(recall),
                "global_f1_score": float(f1),
                "global_auc_roc": float(auc_roc_manual),
                "global_specificity": float(specificity),
                "global_sensitivity": float(sensitivity),
                "global_samples": int(len(X_global)),
                "auc_improvement_vs_v1": float(auc_vs_v1),
                "accuracy_improvement_vs_v1": float(acc_vs_v1),
                "f1_improvement_vs_v1": float(f1_vs_v1),
                "success_score": int(success_score),
            }
            
        except Exception as e:
            print(f"❌ Errore valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {"error": str(e), "global_samples": 0}
    
    return evaluate

# ============================================================================
# 🚀 MAIN FUNCTION
# ============================================================================

def main():
    """Avvia server federato."""
    print(f"\n🚀 AVVIO SERVER FEDERATO")
    print("=" * 80)
    print("📋 CONFIGURAZIONE:")
    print("   - Base: v1 (architettura stabile 128→64→32→1)")
    print("   - Enhancement: Threshold optimization v3")
    print("   - Features: 30 PCA + 12 statistical = 42 totali")
    print("   - Rounds: 10")
    print("   - Preprocessing: StandardScaler + PCA (testato)")
    print("   - Compatibilità: Client")
    print("   - Dataset globale: Client 14-15")
    print("   - Inizializzazione: Parametri autonomi (no GrpcBridgeClosed)")
    print("=" * 80)
    print("🎯 OBIETTIVI v2.5:")
    print("   - AUC-ROC: 52% → 55%+ (miglioramento graduale)")
    print("   - Accuracy: 68.7% → 70%+")
    print("   - F1-Score: 81.7% → 83%+")
    print("   - Specificity: 0% → 15%+ (fix problema v4)")
    print("=" * 80)
    
    config = HybridServerConfig()

    # Strategia con parametri iniziali
    strategy = HybridFedAvgV25(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.MIN_CLIENTS,
        min_evaluate_clients=config.MIN_CLIENTS,
        min_available_clients=config.MIN_CLIENTS,
        evaluate_fn=get_hybrid_evaluate_fn_v25(),
        evaluate_metrics_aggregation_fn=weighted_average_hybrid_v25,
    )
    
    server_config = fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS)
    
    print("🔗 Server pronto!")
    print("Connettere client")
    print("\nIl training inizierà quando almeno 2 client saranno connessi.")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=server_config,
            strategy=strategy,
        )
        
    except KeyboardInterrupt:
        print(f"\n🛑 Server fermato dall'utente")
    except Exception as e:
        print(f"❌ Errore durante l'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()