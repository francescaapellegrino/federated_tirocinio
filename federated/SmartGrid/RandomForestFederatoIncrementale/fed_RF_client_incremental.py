"""
Client federato SmartGrid con Random Forest incrementale
Francesca Pellegrino
"""

import flwr as fl
import numpy as np
import pickle
import zlib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
)
import sys
import warnings
warnings.filterwarnings("ignore")
from preprocessing_common import load_improved_client_data

# CONFIGURAZIONE
N_NEW_TREES = 8
CRITERION = "gini"
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class IncrementalFederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id, n_new_trees=N_NEW_TREES, criterion=CRITERION, random_state=RANDOM_STATE):
        self.client_id = client_id
        self.n_new_trees = n_new_trees
        self.criterion = criterion
        self.random_state = random_state
        #self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, _ = load_improved_client_data(
            #client_id, None
        #)
        #(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test,
         #_, _) = load_data_for_aia(client_id)
        #self.model = None

        (
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            self.X_test, self.y_test, 
            _ # Il settimo valore viene ignorato
        ) = load_improved_client_data(client_id, None)

    def get_parameters(self, config):
        # Invia una struttura vuota per compatibilità con il primo round del server
        empty_model_data = {
            "new_estimators": [],
            "n_features_in_": self.X_train.shape[1],
            "classes_": np.unique(self.y_train)
        }
        pickled_model = pickle.dumps(empty_model_data)
        compressed_model = zlib.compress(pickled_model)
        return [np.frombuffer(compressed_model, dtype=np.uint8)]

    def fit(self, parameters, config):
        print(f"\n[CLIENT {self.client_id}] FIT (Addestramento Incrementale)...")
        
        # 1. Ricostruisci il modello globale ricevuto
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        existing_estimators_pickled = agg_model_data.get("estimators", [])
        existing_estimators = [pickle.loads(est) for est in existing_estimators_pickled]
        n_existing_trees = len(existing_estimators)
        print(f"[CLIENT {self.client_id}] Ricevuti {n_existing_trees} alberi dal server.")

        # Inizializza il modello con gli alberi esistenti
        self.model = RandomForestClassifier(
            n_estimators=n_existing_trees,
            warm_start=True,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=-1
        )
        if n_existing_trees > 0:
            self.model.estimators_ = existing_estimators
            self.model.n_features_in_ = agg_model_data["n_features_in_"]
            self.model.classes_ = agg_model_data["classes_"]
        
        # 2. Aggiungi nuovi alberi
        self.model.n_estimators += self.n_new_trees
        print(f"[CLIENT {self.client_id}] Addestro e aggiungo {self.n_new_trees} nuovi alberi...")
        self.model.fit(self.X_train, self.y_train)
        
        # 3. Isola e invia solo i nuovi alberi
        newly_trained_estimators = self.model.estimators_[n_existing_trees:]
        print(f"[CLIENT {self.client_id}] Invio {len(newly_trained_estimators)} nuovi alberi al server.")
        
        model_to_send = {
            "new_estimators": [pickle.dumps(est) for est in newly_trained_estimators],
            "n_features_in_": self.model.n_features_in_,
            "classes_": self.model.classes_
        }
        pickled_model = pickle.dumps(model_to_send)
        compressed_model = zlib.compress(pickled_model)

        # Calcola metriche di validazione sul modello locale completo (vecchi + nuovi alberi)
        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)
        
        acc = accuracy_score(self.y_val, y_pred)
        prec = precision_score(self.y_val, y_pred, zero_division=0)
        rec = recall_score(self.y_val, y_pred, zero_division=0)
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        loss = log_loss(self.y_val, y_pred_proba)

        # Invia un set completo di metriche al server per il logging
        metrics = {
            "client_id": self.client_id,
            "val_accuracy": float(acc),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1_score": float(f1),
            "val_loss": float(loss),
        }
        return [np.frombuffer(compressed_model, dtype=np.uint8)], len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"\n[CLIENT {self.client_id}] EVALUATE (Test su Modello Globale)...")
        
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        all_estimators_pickled = agg_model_data.get("estimators", [])
        all_estimators = [pickle.loads(est) for est in all_estimators_pickled]

        if not all_estimators:
            print(f"[CLIENT {self.client_id}] Nessun albero da valutare.")
            return 1.0, len(self.X_test), {"test_accuracy": 0.0}

        agg_model = RandomForestClassifier(n_estimators=len(all_estimators), n_jobs=-1)
        agg_model.estimators_ = all_estimators
        agg_model.n_features_in_ = agg_model_data["n_features_in_"]
        agg_model.classes_ = agg_model_data["classes_"]
        agg_model.n_classes_ = len(agg_model.classes_)
        agg_model.n_outputs_ = 1

        y_pred = agg_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        print(f"[CLIENT {self.client_id}] TEST su modello globale con {len(all_estimators)} alberi:")
        print(f" - Accuracy : {acc:.4f}")
        print(f" - Precision: {prec:.4f}")
        print(f" - Recall   : {rec:.4f}")
        print(f" - F1-Score : {f1:.4f}")
        
        metrics = {
            "client_id": self.client_id,
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1_score": float(f1)
        }
        loss = 1.0 - acc
        return loss, len(self.X_test), metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fed_RF_client_incremental.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    
    print(f"\n=== CLIENT RF INCREMENTALE {client_id} ===")
    
    max_message_length = 1024 * 1024 * 1024

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=IncrementalFederatedClient(client_id=client_id),
        grpc_max_message_length=max_message_length,
    )