"""
Client federato MALEVOLO per Membership Inference Attack (MIA)
Implementazione manuale dell'attacco.
FUNZIONANTE -> mia_accuracy = 0.83
Francesca Pellegrino
"""

import flwr as fl
import numpy as np
import pickle
import zlib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss
)
# --- RIMOSSE LE IMPORTAZIONI DI ART ---
# from art.estimators.classification import SklearnClassifier
# from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
import sys
import warnings
warnings.filterwarnings("ignore")
from preprocessing import load_improved_client_data

# CONFIGURAZIONE
N_NEW_TREES = 8
CRITERION = "gini"
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class MaliciousMIAClient(fl.client.NumPyClient):
    def __init__(self, client_id, n_new_trees=N_NEW_TREES, criterion=CRITERION, random_state=RANDOM_STATE):
        self.client_id = client_id
        self.n_new_trees = n_new_trees
        self.criterion = criterion
        self.random_state = random_state
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, _ = load_improved_client_data(
            15, None
        )
        self.model = None

    def get_parameters(self, config):
        empty_model_data = {
            "new_estimators": [],
            "n_features_in_": self.X_train.shape[1],
            "classes_": np.unique(self.y_train)
        }
        pickled_model = pickle.dumps(empty_model_data)
        compressed_model = zlib.compress(pickled_model)
        return [np.frombuffer(compressed_model, dtype=np.uint8)]

    def fit(self, parameters, config):
        print(f"\n[CLIENT MALEVOLO {self.client_id}] FIT (Addestramento come client onesto)...")
        
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        existing_estimators_pickled = agg_model_data.get("estimators", [])
        existing_estimators = [pickle.loads(est) for est in existing_estimators_pickled]
        n_existing_trees = len(existing_estimators)
        print(f"[CLIENT MALEVOLO {self.client_id}] Ricevuti {n_existing_trees} alberi dal server.")

        self.model = RandomForestClassifier(
            n_estimators=n_existing_trees,
            warm_start=True,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=1 
        )
        if n_existing_trees > 0:
            self.model.estimators_ = existing_estimators
            self.model.n_features_in_ = agg_model_data["n_features_in_"]
            self.model.classes_ = agg_model_data["classes_"]
        
        self.model.n_estimators += self.n_new_trees
        print(f"[CLIENT MALEVOLO {self.client_id}] Addestro e aggiungo {self.n_new_trees} nuovi alberi...")
        self.model.fit(self.X_train, self.y_train)
        
        newly_trained_estimators = self.model.estimators_[n_existing_trees:]
        print(f"[CLIENT MALEVOLO {self.client_id}] Invio {len(newly_trained_estimators)} nuovi alberi al server.")
        
        model_to_send = {
            "new_estimators": [pickle.dumps(est) for est in newly_trained_estimators],
            "n_features_in_": self.model.n_features_in_,
            "classes_": self.model.classes_
        }
        pickled_model = pickle.dumps(model_to_send)
        compressed_model = zlib.compress(pickled_model)

        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)
        
        acc = accuracy_score(self.y_val, y_pred)
        prec = precision_score(self.y_val, y_pred, zero_division=0)
        rec = recall_score(self.y_val, y_pred, zero_division=0)
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        loss = log_loss(self.y_val, y_pred_proba)

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
        print(f"\n[CLIENT MALEVOLO {self.client_id}] EVALUATE (Test E ATTACCO)...")
        
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        global_estimators = [pickle.loads(est) for est in agg_model_data.get("estimators", [])]

        if not global_estimators:
            return 1.0, len(self.X_test), {"test_accuracy": 0.0, "mia_accuracy": 0.0}

        # --- RICOSTRUZIONE DEL MODELLO GLOBALE (FUNZIONANTE) ---
        # Questo blocco è corretto e permette di usare il modello per le previsioni.
        agg_model = RandomForestClassifier(n_estimators=len(global_estimators), n_jobs=1, random_state=self.random_state)
        agg_model.estimators_ = global_estimators
        agg_model.n_features_in_ = agg_model_data["n_features_in_"]
        agg_model.classes_ = agg_model_data["classes_"]
        agg_model.n_classes_ = len(agg_model.classes_)
        agg_model.n_outputs_ = 1

        y_pred_test = agg_model.predict(self.X_test)
        acc_test = accuracy_score(self.y_test, y_pred_test)
        print(f"[CLIENT MALEVOLO {self.client_id}] TEST su modello globale con {len(global_estimators)} alberi: Accuracy={acc_test:.4f}")
        
        print(f"[CLIENT MALEVOLO {self.client_id}] Avvio Membership Inference Attack (implementazione manuale)...")
        
        attack_accuracy = 0.0
        try:
            # --- IMPLEMENTAZIONE MANUALE DELL'ATTACCO ---
            
            # 1. Ottenere le previsioni (probabilità) del modello globale sia sui dati
            #    usati per il training (membri) sia su dati nuovi (non-membri).
            # I dati "membri" per l'attaccante sono il suo set di training.
            # I dati "non-membri" sono il suo set di test.
            member_predictions = agg_model.predict_proba(self.X_train)
            non_member_predictions = agg_model.predict_proba(self.X_test)

            # 2. Creare un dataset per addestrare il "meta-modello" dell'attacco.
            # Le feature sono le probabilità restituite dal modello.
            # Le etichette sono: 1 per i membri, 0 per i non-membri.
            attack_X = np.vstack((member_predictions, non_member_predictions))
            attack_y = np.concatenate((np.ones(len(member_predictions)), np.zeros(len(non_member_predictions))))

            # 3. Addestrare un classificatore (il meta-modello) per distinguere
            #    le previsioni dei membri da quelle dei non-membri.
            #    Un RandomForest è una buona scelta anche qui.
            attack_model = RandomForestClassifier(random_state=self.random_state, n_jobs=1)
            attack_model.fit(attack_X, attack_y)

            # 4. Valutare l'accuracy dell'attacco.
            attack_accuracy = attack_model.score(attack_X, attack_y)

            print("\n--- RISULTATI MEMBERSHIP INFERENCE ATTACK (MANUALE) ---")
            print(f"  - ACCURACY TOTALE DELL'ATTACCO: {attack_accuracy:.4f}")
            print("-------------------------------------------------------")

        except Exception as e:
            print(f"[CLIENT MALEVOLO {self.client_id}] ERRORE durante l'attacco manuale: {e}")
            attack_accuracy = 0.0

        metrics = {
            "client_id": self.client_id,
            "test_accuracy": float(acc_test),
            "mia_accuracy": float(attack_accuracy) # Invia il risultato dell'attacco
        }
        
        loss = 1.0 - acc_test
        return loss, len(self.X_test), metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fed_RF_client_malicious_mia.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    
    print(f"\n=== CLIENT MALEVOLO (MIA) RF INCREMENTALE {client_id} ===")
    
    max_message_length = 1024 * 1024 * 1024

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MaliciousMIAClient(client_id=client_id),
        grpc_max_message_length=max_message_length,
    )
