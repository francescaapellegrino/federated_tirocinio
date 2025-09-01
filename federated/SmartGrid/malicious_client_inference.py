"""
Client Malevolo - Inference Attack
Francesca Pellegrino
"""

import warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, List
import time
import json
from datetime import datetime
import traceback
from optimized_config_20250824_193626 import OptimizedConfig

# Import ART per attacchi
ART_AVAILABLE = False
try:
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    from art.estimators.classification import TensorFlowV2Classifier
    ART_AVAILABLE = True
    print("✅ ART disponibile per attacchi avanzati")
except ImportError as e:
    print(f"⚠️ ART non disponibile: {e}")
    print("🔄 Usando solo attacchi fallback/statistici")

def sanitize_json(obj):
    """
    Ricorsivamente converte tutti i valori in tipi serializzabili e sostituisce NaN/inf con None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif obj is None:
        return None
    else:
        return obj

class EnhancedMaliciousClient(fl.client.NumPyClient):
    """
    Client malevolo con attacchi anti-privacy migliorati.
    Include modelli shadow/statistici, analisi dettagliata e export risultati.
    """
    def __init__(self, client_id: int, is_malicious: bool = True):
        self.client_id = client_id
        self.is_malicious = is_malicious
        
        print(f"🚀 Client Malevolo {client_id} - {'MALEVOLO' if is_malicious else 'NORMALE'}")
        self.load_and_preprocess_data()
        self.create_model()
        
    def load_and_preprocess_data(self):
        """Carica e preprocessa i dati come il client normale."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{self.client_id}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} non trovato")
        df = pd.read_csv(file_path)
        X = df.drop(columns=["marker"])
        y = (df["marker"] != "Natural").astype(np.float32)
        # Preprocessing
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isnull().sum().sum() > 0:
            X.fillna(X.median(), inplace=True)
        scaler_pca = StandardScaler()
        X_scaled = scaler_pca.fit_transform(X)
        pca = PCA(n_components=30, random_state=42)
        X_pca = pca.fit_transform(X_scaled).astype(np.float32)
        # Split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_pca, y, test_size=0.15, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.118, random_state=42, stratify=y_temp if len(np.unique(y_temp)) > 1 else None)
        # Normalizzazione finale
        final_scaler = StandardScaler()
        self.X_train = final_scaler.fit_transform(self.X_train).astype(np.float32)
        self.X_val = final_scaler.transform(self.X_val).astype(np.float32)
        self.X_test = final_scaler.transform(self.X_test).astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_val = self.y_val.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        print(f"Dati pronti: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")

    def create_model(self):
        """Crea il modello con architettura identica al client normale."""
        config = OptimizedConfig()
        self.model = keras.Sequential([
            keras.layers.Input(shape=(30,)),
            keras.layers.Dense(config.HIDDEN_LAYERS[0], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(config.DROPOUT_RATES[0]),
            keras.layers.Dense(config.HIDDEN_LAYERS[1], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(config.DROPOUT_RATES[1]),
            keras.layers.Dense(config.HIDDEN_LAYERS[2], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(config.DROPOUT_RATES[2]),
            keras.layers.Dense(config.HIDDEN_LAYERS[3], activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(config.DROPOUT_RATES[3]),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Modello creato: {self.model.count_params():,} parametri")

    def shadow_mia_attack(self):
        """
        Esegue un attacco Membership Inference avanzato usando ART.
        Si allena il classificatore shadow su dati membri (train) e non membri (test).
        """
        try:
            from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
            from art.estimators.classification import TensorFlowV2Classifier
        except ImportError as e:
            print(f"⚠️ ART non disponibile: {e}")
            return {
                "error": "ART library not available",
                "attack_success": False
            }

        print("⚔️ Esecuzione MIA avanzato con ART...")

        # Crea il wrapper ART per il modello Keras
        art_classifier = TensorFlowV2Classifier(
            model=self.model,
            nb_classes=2,
            input_shape=self.X_train.shape[1:],
            loss_object=tf.keras.losses.BinaryCrossentropy()
        )

        # Crea l'attacco MIA
        mia = MembershipInferenceBlackBox(art_classifier)

        # Concatenazione dati
        X_mia = np.concatenate([self.X_train, self.X_test])
        y_mia = np.concatenate([np.ones(len(self.X_train)), np.zeros(len(self.X_test))])

        print(f"Forma X_mia: {X_mia.shape}")    # Debug: verifica che non sia vuoto
        print(f"Forma y_mia: {y_mia.shape}")    # Debug: verifica che non sia vuoto

        # Fit dell'attacco (parametri x, y)
        mia.fit(X_mia, y_mia)

        # Inferenza MIA
        shadow_preds = mia.infer(X_mia)

        # Metriche
        from sklearn.metrics import accuracy_score, f1_score
        attack_accuracy = accuracy_score(y_mia, shadow_preds)
        attack_f1 = f1_score(y_mia, shadow_preds)
        attack_success = attack_accuracy > 0.6

        print(f"Accuratezza MIA (ART): {attack_accuracy:.3f}, F1: {attack_f1:.3f}, Successo: {attack_success}")

        return {
            "method": "ART_blackbox_shadow",
            "accuracy": float(attack_accuracy),
            "f1_score": float(attack_f1),
            "attack_success": attack_success,
            "samples_tested": int(len(y_mia))
        }

    def property_inference_attack(self):
        """
        Property Inference Attack migliorato.
        """
        print("🔍 Property Inference Attack avanzato...")
        preds = self.model.predict(self.X_test, verbose=0).flatten()
        est = RandomForestClassifier(n_estimators=50, random_state=42)
        preds_reshape = preds.reshape(-1, 1)
        est.fit(preds_reshape, self.y_test)
        y_pred = est.predict(preds_reshape)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        predicted_attack_ratio = np.mean(y_pred)
        actual_attack_ratio = np.mean(self.y_test)
        ratio_error = abs(predicted_attack_ratio - actual_attack_ratio)
        attack_success = ratio_error < 0.15
        return {
            "predicted_attack_ratio": float(predicted_attack_ratio),
            "actual_attack_ratio": float(actual_attack_ratio),
            "f1_score": float(f1),
            "acc": float(acc),
            "ratio_error": float(ratio_error),
            "attack_success": bool(attack_success)
        }

    def model_behavior_analysis(self):
        """
        Analizza la stabilità del modello su porzioni diverse e ripetute del test set.
        """
        print("📊 Model Behavior Analysis...")
        test_portions = [self.X_test[i*30:(i+1)*30] for i in range(5)]
        scores = []
        for portion in test_portions:
            if len(portion) > 0:
                preds = self.model.predict(portion, verbose=0).flatten()
                avg_conf = np.mean(np.maximum(preds, 1 - preds))
                scores.append(avg_conf)
        variance = np.var(scores) if len(scores) > 1 else 0
        return {
            "behavior_consistency_scores": [float(x) for x in scores],
            "behavior_variance": float(variance),
            "model_stability": float(1.0 - variance),
            "analysis_success": True
        }

    def execute_attacks(self):
        """
        Esegue tutti gli attacchi e produce un report dettagliato.
        La funzione è robusta a errori: in caso di eccezione, salva un report con l'errore.
        """
        results = {}
        try:
            if not self.is_malicious:
                return results
            # Membership Inference Attack
            results['membership_inference'] = self.shadow_mia_attack()
            # Property Inference Attack
            results['property_inference'] = self.property_inference_attack()
            # Model Behavior Analysis
            results['model_behavior'] = self.model_behavior_analysis()
            # Riassunto
            succ = sum([
                bool(results['membership_inference'].get('attack_success', False)),
                bool(results['property_inference'].get('attack_success', False)),
                1  # model_behavior è sempre eseguito
            ])
            results['attack_summary'] = {
                'total_attacks': 3,
                'successful_attacks': int(succ),
                'success_rate': float(succ/3),
                'client_id': int(self.client_id),
                'timestamp': datetime.now().isoformat(),
                'privacy_score': float(1.0 - (succ/3))
            }
            print(f"📊 Summary: {succ}/3 attacchi riusciti ({100*succ/3:.1f}%)")
        except Exception as e:
            # Gestione robusta degli errori: salva sempre un file con l'errore
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        return sanitize_json(results)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"\n[Client {self.client_id}] Training...")
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=15, batch_size=32, verbose=0
        )
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'client_id': int(self.client_id),
            'client_type': 'malicious_enhanced'
        }
        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"\n[Client {self.client_id}] Evaluate...")
        self.model.set_weights(parameters)
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss = results[0]
        accuracy = results[1] if len(results) > 1 else 0.0
        # Esegui attacchi
        attack_results = {}
        if self.is_malicious:
            try:
                attack_results = self.execute_attacks()
            except Exception as e:
                attack_results = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            # Salva SEMPRE il file, anche in caso di errore
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"attack_results_client_{self.client_id}_{timestamp}.json"
            try:
                with open(results_file, 'w') as f:
                    json.dump(attack_results, f, indent=2)
                print(f"🔥 Risultati attacchi salvati: {results_file}")
            except Exception as e:
                print(f"⚠️ Errore salvataggio JSON: {e}")
                print(f"Contenuto non serializzabile: {attack_results}")
        metrics = {
            'client_id': int(self.client_id),
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'test_samples': int(len(self.X_test))
        }
        return loss, len(self.X_test), metrics

def main():
    if len(sys.argv) != 3:
        print("Uso: python3 malicious_client_inference.py <client_id> <is_malicious>")
        sys.exit(1)
    try:
        client_id = int(sys.argv[1])
        is_malicious = sys.argv[2].lower() == 'true'
    except (ValueError, IndexError) as e:
        print(f"Errore: {e}")
        sys.exit(1)
    print(f"\n🚀 CLIENT MALEVOLO POTENZIATO {client_id}")
    print(f"Modalità: {'MALEVOLO' if is_malicious else 'NORMALE'}")
    print(f"✅ Attacchi: MIA avanzato, Property Inference, Model Behavior")
    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=EnhancedMaliciousClient(client_id, is_malicious)
        )
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()