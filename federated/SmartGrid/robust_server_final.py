#!/usr/bin/env python3
"""
Robust Federated Server Final v4 - SmartGrid
============================================

Server federato avanzato compatibile con client v4 che utilizza:
- Feature engineering avanzata (40 features finali)
- Architettura con Attention Mechanism (256→128→64→1)
- QuantileTransformer preprocessing
- Valutazione globale robusta su client 14-15
- Logging dettagliato e gestione errori

Autore: Sistema AI per francescaapellegrino
Data: 2025-01-27
Versione: v4.0 - Final
"""

import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
import logging
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_server_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Sopprimi warnings non critici
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def create_v4_feature_engineering(X_pca: np.ndarray) -> np.ndarray:
    """
    Applica feature engineering avanzata per portare da 30 componenti PCA a 40 features finali.
    
    Args:
        X_pca: Array con 30 componenti PCA
        
    Returns:
        Array con 40 features finali (30 originali + 10 engineered)
    """
    logger.info(f"Applicando feature engineering: {X_pca.shape[1]} → 40 features")
    
    # Mantieni le 30 features PCA originali
    features = [X_pca]
    
    # FEATURE ENGINEERING AVANZATA (10 nuove features)
    
    # 1. Statistiche aggregate
    feature_means = np.mean(X_pca, axis=1, keepdims=True)  # 1 feature
    feature_stds = np.std(X_pca, axis=1, keepdims=True)   # 1 feature
    feature_mins = np.min(X_pca, axis=1, keepdims=True)   # 1 feature
    feature_maxs = np.max(X_pca, axis=1, keepdims=True)   # 1 feature
    
    # 2. Ratio e interazioni
    feature_range = feature_maxs - feature_mins  # 1 feature
    feature_cv = feature_stds / (feature_means + 1e-8)  # coefficient of variation - 1 feature
    
    # 3. Features polinomiali (quadratiche) - selezioniamo le prime 2
    poly_features = X_pca[:, :2] ** 2  # 2 features
    
    # 4. Interazioni tra le prime 2 componenti principali
    interaction_feature = (X_pca[:, 0] * X_pca[:, 1]).reshape(-1, 1)  # 1 feature
    
    # 5. Feature di energia/potenza
    energy_feature = np.sum(X_pca ** 2, axis=1, keepdims=True)  # 1 feature
    
    # Combina tutte le features
    features.extend([
        feature_means, feature_stds, feature_mins, feature_maxs,
        feature_range, feature_cv, poly_features, interaction_feature, energy_feature
    ])
    
    X_engineered = np.concatenate(features, axis=1)
    
    logger.info(f"Feature engineering completata: {X_engineered.shape[1]} features finali")
    return X_engineered


def create_v4_model_with_attention(input_shape: int = 40) -> keras.Model:
    """
    Crea il modello v4 con Attention Mechanism: 40 → 256+Attention → 128 → 64 → 1
    
    Args:
        input_shape: Numero di features in input (40)
        
    Returns:
        Modello Keras compilato con architettura v4
    """
    logger.info(f"Creando modello v4 con input shape: {input_shape}")
    
    # Input layer
    inputs = keras.layers.Input(shape=(input_shape,), name="input_layer")
    
    # Dense layer 256 con BatchNormalization
    x = keras.layers.Dense(256, activation="relu", name="dense_256")(inputs)
    x = keras.layers.BatchNormalization(name="batch_norm_256")(x)
    x = keras.layers.Dropout(0.3, name="dropout_256")(x)
    
    # ATTENTION MECHANISM
    # Attention weights: Dense(256) → sigmoid
    attention_weights = keras.layers.Dense(256, activation="sigmoid", name="attention_dense")(x)
    # Apply attention: Multiply layer
    x_attended = keras.layers.Multiply(name="attention_multiply")([x, attention_weights])
    
    # Dense layer 128
    x = keras.layers.Dense(128, activation="relu", name="dense_128")(x_attended)
    x = keras.layers.BatchNormalization(name="batch_norm_128")(x)
    x = keras.layers.Dropout(0.2, name="dropout_128")(x)
    
    # Dense layer 64
    x = keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = keras.layers.BatchNormalization(name="batch_norm_64")(x)
    x = keras.layers.Dropout(0.1, name="dropout_64")(x)
    
    # Output layer
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output_layer")(x)
    
    # Crea il modello
    model = keras.Model(inputs=inputs, outputs=outputs, name="SmartGrid_v4_AttentionModel")
    
    # Compila il modello
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve='ROC')
        ]
    )
    
    # Log delle informazioni del modello
    logger.info(f"Modello v4 creato con {model.count_params()} parametri")
    logger.info(f"Numero di tensori di peso: {len(model.get_weights())}")
    
    return model


def load_and_preprocess_global_validation_data() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Carica e preprocessa i dati globali di validazione (client 14-15) con pipeline v4.
    
    Returns:
        Tuple con (X_processed, y, preprocessing_info)
    """
    logger.info("=== CARICAMENTO DATASET GLOBALE DI VALIDAZIONE v4 ===")
    
    # Directory del dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Usa client 14-15 per validazione globale (mai usati nel training federato)
    validation_clients = [14, 15]
    df_list = []
    
    for client_id in validation_clients:
        file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            logger.info(f"Caricato data{client_id}.csv: {len(df)} campioni")
        except FileNotFoundError:
            logger.error(f"File {file_path} non trovato!")
            continue
    
    if not df_list:
        raise FileNotFoundError("Nessun file di validazione trovato!")
    
    # Combina i dataset
    df_global = pd.concat(df_list, ignore_index=True)
    logger.info(f"Dataset globale combinato: {len(df_global)} campioni, {df_global.shape[1]} colonne")
    
    # PREPROCESSING PIPELINE V4
    
    # 1. Separazione features e target
    X = df_global.drop(columns=["marker"])
    y = (df_global["marker"] != "Natural").astype(int)
    
    logger.info(f"Distribuzione classi: {y.sum()} attacchi ({y.mean()*100:.2f}%), {(~y.astype(bool)).sum()} naturali")
    
    # 2. Pulizia dati
    logger.info("Pulizia dati...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Trovati {nan_count} valori NaN, applicando imputazione mediana")
        X.fillna(X.median(), inplace=True)
    
    # 3. QuantileTransformer (invece di StandardScaler)
    logger.info("Applicando QuantileTransformer...")
    quantile_transformer = QuantileTransformer(n_quantiles=1000, random_state=42)
    X_quantile = quantile_transformer.fit_transform(X)
    
    # 4. PCA a 30 componenti
    logger.info("Applicando PCA a 30 componenti...")
    pca = PCA(n_components=30, random_state=42)
    X_pca = pca.fit_transform(X_quantile)
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA completata - Varianza spiegata: {variance_explained*100:.2f}%")
    
    # 5. Feature Engineering v4 (30 → 40 features)
    X_final = create_v4_feature_engineering(X_pca)
    
    logger.info(f"Preprocessing completato: {X.shape[1]} → {X_final.shape[1]} features finali")
    logger.info(f"Shape finale: {X_final.shape}")
    
    # Informazioni preprocessing
    preprocessing_info = {
        "original_features": X.shape[1],
        "pca_components": 30,
        "final_features": X_final.shape[1],
        "variance_explained": variance_explained,
        "samples": len(X_final),
        "attack_ratio": y.mean()
    }
    
    return X_final, y, preprocessing_info


def get_v4_global_validation_fn():
    """
    Crea la funzione di valutazione globale per il server v4.
    
    Returns:
        Funzione di valutazione che sarà chiamata ad ogni round
    """
    # Carica i dati globali una volta sola
    try:
        X_global, y_global, preprocessing_info = load_and_preprocess_global_validation_data()
        logger.info(f"Dati globali caricati: {preprocessing_info}")
    except Exception as e:
        logger.error(f"Errore nel caricamento dati globali: {e}")
        return None
    
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Funzione di valutazione globale chiamata ad ogni round.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
            
        Returns:
            Tuple con (loss, metriche)
        """
        logger.info(f"\n=== VALUTAZIONE GLOBALE v4 - ROUND {server_round} ===")
        
        try:
            # Crea il modello v4 per la valutazione
            model = create_v4_model_with_attention(input_shape=40)
            
            # Imposta i pesi aggregati
            if len(parameters) != len(model.get_weights()):
                logger.error(f"Mismatch parametri: ricevuti {len(parameters)}, attesi {len(model.get_weights())}")
                return 1.0, {"error": "parameter_mismatch", "global_accuracy": 0.0}
            
            model.set_weights(parameters)
            
            # Valutazione sul dataset globale
            logger.info(f"Valutando su {len(X_global)} campioni...")
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Predizioni per metriche aggiuntive
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            
            # Calcola F1-score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # AUC-ROC manuale per maggiore precisione
            try:
                auc_roc = roc_auc_score(y_global, y_pred_prob)
            except Exception:
                auc_roc = 0.0
                logger.warning("Impossibile calcolare AUC-ROC")
            
            # Matrice di confusione
            tn, fp, fn, tp = confusion_matrix(y_global, y_pred_binary).ravel()
            
            # LOGGING DETTAGLIATO
            logger.info("RISULTATI VALIDAZIONE GLOBALE v4:")
            logger.info(f"  💰 Loss: {loss:.4f}")
            logger.info(f"  🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  🔍 Precision: {precision:.4f} ({precision*100:.2f}%)")
            logger.info(f"  📡 Recall: {recall:.4f} ({recall*100:.2f}%)")
            logger.info(f"  ⚖️  F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            logger.info(f"  📊 AUC-ROC: {auc_roc:.4f} ({auc_roc*100:.2f}%)")
            logger.info(f"  📈 Campioni: {len(X_global)}")
            
            logger.info("MATRICE DI CONFUSIONE:")
            logger.info(f"  ✅ True Negative: {tn}")
            logger.info(f"  ❌ False Positive: {fp}")
            logger.info(f"  ❌ False Negative: {fn}")
            logger.info(f"  ✅ True Positive: {tp}")
            
            # Distribuzione delle predizioni
            logger.info("DISTRIBUZIONE PREDIZIONI:")
            logger.info(f"  🔍 Prob min: {y_pred_prob.min():.4f}")
            logger.info(f"  📊 Prob media: {y_pred_prob.mean():.4f}")
            logger.info(f"  🔍 Prob max: {y_pred_prob.max():.4f}")
            logger.info(f"  🎯 Predizioni positive: {y_pred_binary.sum()}/{len(y_pred_binary)} ({y_pred_binary.mean()*100:.2f}%)")
            
            # Metriche per Flower
            metrics = {
                "global_accuracy": float(accuracy),
                "global_precision": float(precision),
                "global_recall": float(recall),
                "global_f1_score": float(f1_score),
                "global_auc_roc": float(auc_roc),
                "global_samples": len(X_global),
                "round": server_round,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
            
            logger.info(f"✅ Valutazione globale round {server_round} completata con successo")
            return float(loss), metrics
            
        except Exception as e:
            logger.error(f"❌ Errore durante valutazione globale round {server_round}: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {
                "error": str(e),
                "global_accuracy": 0.0,
                "round": server_round,
                "global_samples": 0
            }
    
    return evaluate


def print_client_metrics_v4(fit_results: List[Tuple[Any, int, Dict[str, Any]]]) -> None:
    """
    Stampa le metriche dei client con formattazione migliorata per v4.
    
    Args:
        fit_results: Lista dei risultati di fit dai client
    """
    logger.info("\n=== METRICHE CLIENT v4 ===")
    
    total_samples = 0
    total_weighted_acc = 0.0
    
    for i, (_, client_samples, client_metrics) in enumerate(fit_results):
        client_id = i + 1  # Assumi ID client sequenziale
        logger.info(f"\n📱 CLIENT {client_id}:")
        logger.info(f"  📊 Campioni: {client_samples}")
        
        total_samples += client_samples
        
        # Metriche di training
        if 'accuracy' in client_metrics:
            acc = client_metrics['accuracy']
            total_weighted_acc += acc * client_samples
            logger.info(f"  🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        if 'loss' in client_metrics:
            logger.info(f"  💰 Loss: {client_metrics['loss']:.4f}")
            
        if 'precision' in client_metrics:
            logger.info(f"  🔍 Precision: {client_metrics['precision']:.4f}")
            
        if 'recall' in client_metrics:
            logger.info(f"  📡 Recall: {client_metrics['recall']:.4f}")
    
    # Statistiche aggregate
    if total_samples > 0:
        avg_weighted_acc = total_weighted_acc / total_samples
        logger.info(f"\n📈 STATISTICHE AGGREGATE:")
        logger.info(f"  🌐 Media pesata Accuracy: {avg_weighted_acc:.4f} ({avg_weighted_acc*100:.2f}%)")
        logger.info(f"  👥 Totale campioni: {total_samples}")
        logger.info(f"  🏢 Client partecipanti: {len(fit_results)}")


class RobustFedAvgV4(FedAvg):
    """
    Strategia FedAvg robusta personalizzata per SmartGrid v4 con logging avanzato.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Inizializzata strategia RobustFedAvgV4")
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Aggrega i risultati dell'addestramento con logging dettagliato.
        
        Args:
            server_round: Numero del round
            results: Risultati dei client
            failures: Fallimenti dei client
            
        Returns:
            Risultato aggregato e metriche
        """
        logger.info(f"\n🔄 AGGREGAZIONE ROUND {server_round}")
        logger.info(f"  ✅ Client riusciti: {len(results)}")
        logger.info(f"  ❌ Client falliti: {len(failures)}")
        
        if failures:
            logger.warning("FALLIMENTI RILEVATI:")
            for i, failure in enumerate(failures):
                logger.warning(f"  ❌ Fallimento {i+1}: {failure}")
        
        # Stampa metriche dettagliate dei client
        if results:
            print_client_metrics_v4(results)
        
        # Chiama l'aggregazione standard
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            logger.info(f"✅ Aggregazione round {server_round} completata con successo")
        else:
            logger.error(f"❌ ATTENZIONE: Aggregazione round {server_round} fallita!")
        
        return aggregated_result
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[Any, Any]], failures: List[Any]) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggrega i risultati della valutazione.
        """
        logger.info(f"\n📊 AGGREGAZIONE VALUTAZIONE ROUND {server_round}")
        return super().aggregate_evaluate(server_round, results, failures)


def main():
    """
    Funzione principale per avviare il server SmartGrid federato v4.
    """
    print("=" * 80)
    print("🚀 ROBUST FEDERATED SERVER v4 - SMARTGRID")
    print("=" * 80)
    print("🔧 Configurazione v4:")
    print("  📊 Architettura: 40 → 256+Attention → 128 → 64 → 1")
    print("  🧠 Preprocessing: QuantileTransformer + PCA(30) + Feature Engineering")
    print("  🔄 Rounds: 10 (ottimizzato)")
    print("  🎯 Features finali: 40 (30 PCA + 10 engineered)")
    print("  🔍 Attention Mechanism: Dense(256, sigmoid) + Multiply")
    print("  📈 Validazione globale: Client 14-15")
    print("  📊 Metriche: accuracy, precision, recall, f1-score, AUC-ROC")
    print("  🛡️  Gestione errori robusta")
    print("=" * 80)
    
    logger.info("Avvio server federato SmartGrid v4")
    
    try:
        # Test di creazione modello per verificare architettura
        logger.info("🧪 Test architettura modello v4...")
        test_model = create_v4_model_with_attention(40)
        weight_tensors = len(test_model.get_weights())
        logger.info(f"✅ Modello v4 creato - Tensori di peso: {weight_tensors}")
        
        if weight_tensors != 22:
            logger.warning(f"⚠️  ATTENZIONE: Attesi 22 tensori, trovati {weight_tensors}")
        else:
            logger.info("✅ Architettura v4 verificata: 22 tensori di peso")
        
        # Test preprocessing
        logger.info("🧪 Test pipeline preprocessing...")
        eval_fn = get_v4_global_validation_fn()
        if eval_fn is None:
            raise Exception("Impossibile creare funzione di valutazione globale")
        logger.info("✅ Pipeline preprocessing v4 inizializzata")
        
        # Configurazione server per 10 rounds ottimizzati
        config = fl.server.ServerConfig(num_rounds=10)
        logger.info("📊 Server configurato per 10 rounds")
        
        # Strategia federata v4 robusta
        strategy = RobustFedAvgV4(
            fraction_fit=1.0,                    # Usa tutti i client disponibili
            fraction_evaluate=1.0,               # Usa tutti i client per valutazione
            min_fit_clients=2,                   # Minimo 2 client per training
            min_evaluate_clients=2,              # Minimo 2 client per valutazione
            min_available_clients=2,             # Minimo 2 client totali
            evaluate_fn=eval_fn,                 # Valutazione globale v4
            initial_parameters=None,             # Lascia che Flower inizializzi
        )
        
        logger.info("🔄 Strategia RobustFedAvgV4 configurata")
        
        # Avvio server
        logger.info("🌐 Avvio server su localhost:8080...")
        print("\n🎯 SERVER PRONTO - In attesa di client v4...")
        print("💡 Per testare, avvia i client v4 con:")
        print("   python robust_client_final_v4.py <client_id>")
        print("=" * 80)
        
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server fermato dall'utente")
        print("\n🛑 Server fermato dall'utente")
        
    except Exception as e:
        logger.error(f"❌ Errore critico durante avvio server: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ ERRORE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()