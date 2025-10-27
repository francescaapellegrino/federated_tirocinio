"""
Client federato MALEVOLO per Attribute Inference Attack (AIA)
Sfrutta la libreria ART per sferrare l'attacco.
Questa versione implementa una pipeline di attacco manuale per garantire
robustezza e superare le incoerenze dell'API di ART.
Francesca Pellegrino
"""

import flwr as fl
import numpy as np
import pandas as pd
import pickle
import zlib
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, mean_squared_error
)
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
import sys
import warnings
warnings.filterwarnings("ignore")

# Usiamo la funzione di preprocessing unificata che garantisce dati coerenti
from preprocessing_common import load_improved_client_data

# CONFIGURAZIONE
N_NEW_TREES = 8
CRITERION = "gini"
RANDOM_STATE = 42
# Scegliamo di attaccare la prima componente principale (indice 0)
ATTACK_FEATURE_INDEX = 0

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class MaliciousAIAClient(fl.client.NumPyClient):
    """
    Client malevolo per Attribute Inference. Carica tutti i dati necessari 
    nel costruttore per garantire coerenza e usa una pipeline manuale per l'attacco.
    """
    def __init__(self, client_id):
        self.client_id = client_id
        self.n_new_trees = N_NEW_TREES
        self.criterion = CRITERION
        self.random_state = RANDOM_STATE
        
        (
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            self.X_test, self.y_test, 
            _
        ) = load_improved_client_data(15, None)

        # Carichiamo tutti i dati UNA SOLA VOLTA qui per garantire coerenza
        # (self.X_train, self.y_train, self.X_val, self.y_val, 
        # self.X_test, self.y_test, _, _) = load_data_for_aia(15)
        
        #self.model = None

    def get_parameters(self, config):
        # Comportamento standard
        empty_model_data = {
            "new_estimators": [],
            "n_features_in_": self.X_train.shape[1],
            "classes_": np.unique(self.y_train)
        }
        pickled_model = pickle.dumps(empty_model_data)
        compressed_model = zlib.compress(pickled_model)
        return [np.frombuffer(compressed_model, dtype=np.uint8)]

    def fit(self, parameters, config):
        # Il metodo fit rimane invariato
        print(f"\n[CLIENT MALEVOLO-AIA {self.client_id}] FIT (Addestramento come client onesto)...")
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        existing_estimators_pickled = agg_model_data.get("estimators", [])
        existing_estimators = [pickle.loads(est) for est in existing_estimators_pickled]
        n_existing_trees = len(existing_estimators)

        self.model = RandomForestClassifier(
            n_estimators=n_existing_trees, warm_start=True, criterion=self.criterion,
            random_state=self.random_state, n_jobs=1 
        )
        if n_existing_trees > 0:
            self.model.estimators_ = existing_estimators
            self.model.n_features_in_ = agg_model_data["n_features_in_"]
            self.model.classes_ = agg_model_data["classes_"]
        
        self.model.n_estimators += self.n_new_trees
        self.model.fit(self.X_train, self.y_train)
        
        newly_trained_estimators = self.model.estimators_[n_existing_trees:]
        model_to_send = {
            "new_estimators": [pickle.dumps(est) for est in newly_trained_estimators],
            "n_features_in_": self.model.n_features_in_,
            "classes_": self.model.classes_
        }
        pickled_model = pickle.dumps(model_to_send)
        compressed_model = zlib.compress(pickled_model)

        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)
        
        metrics = {
            "client_id": self.client_id,
            "val_accuracy": float(accuracy_score(self.y_val, y_pred)),
            "val_f1_score": float(f1_score(self.y_val, y_pred, zero_division=0)),
            "val_loss": float(log_loss(self.y_val, y_pred_proba)),
        }
        return [np.frombuffer(compressed_model, dtype=np.uint8)], len(self.X_train), metrics

    def evaluate(self, parameters, config):
        print(f"\n[CLIENT MALEVOLO-AIA {self.client_id}] EVALUATE (Test E ATTACCO AIA)...")
        
        # --- Parte 1: Ricostruzione del modello globale (con la tecnica del "Battesimo") ---
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        
        global_estimators_unfitted = [pickle.loads(est) for est in agg_model_data.get("estimators", [])]

        if not global_estimators_unfitted:
            return 1.0, len(self.X_test), {"test_accuracy": 0.0, "aia_mse": -1.0}

        n_features = agg_model_data["n_features_in_"]
        classes = agg_model_data["classes_"]
        fake_X_fit = np.zeros((len(classes), n_features))
        fake_y_fit = np.array(classes)

        baptized_estimators = []
        for unfitted_tree in global_estimators_unfitted:
            baptized_tree = DecisionTreeClassifier(random_state=self.random_state)
            baptized_tree.fit(fake_X_fit, fake_y_fit)
            baptized_tree.tree_ = unfitted_tree.tree_
            baptized_estimators.append(baptized_tree)
        
        agg_model = RandomForestClassifier(n_estimators=len(baptized_estimators), n_jobs=1, random_state=self.random_state)
        agg_model.estimators_ = baptized_estimators
        agg_model.n_features_in_ = n_features
        agg_model.classes_ = classes
        agg_model.n_classes_ = len(classes)
        agg_model.n_outputs_ = 1

        y_pred_test = agg_model.predict(self.X_test)
        acc_test = accuracy_score(self.y_test, y_pred_test)
        print(f"[CLIENT MALEVOLO-AIA {self.client_id}] TEST su modello globale: Accuracy={acc_test:.4f}")
        
        print(f"[CLIENT MALEVOLO-AIA {self.client_id}] Avvio Attribute Inference Attack...")
        
        attack_mse = -1.0
        try:
            # --- Parte 2: Sferrare l'Attribute Inference Attack (Pipeline Manuale Definitiva) ---

            art_classifier = SklearnClassifier(model=agg_model)
            
            # --- MODIFICA CHIAVE: Pipeline di attacco manuale per assecondare l'API di ART ---

            # 2.1 Generiamo le previsioni del modello-vittima. Queste saranno le "feature" per l'attacco.
            print("  - Step 1: Generazione previsioni del modello vittima...")
            predictions_train = art_classifier.predict(self.X_train)
            predictions_test = art_classifier.predict(self.X_test)

            # 2.2 Creiamo e addestriamo manualmente il nostro modello di attacco.
            # Questo modello impara a mappare le previsioni (es. [0.1, 0.9]) al valore della feature segreta.
            print("  - Step 2: Addestramento del modello di attacco (RandomForestRegressor)...")
            attack_model = RandomForestRegressor(random_state=self.random_state)
            
            # Le feature per l'attacco sono le previsioni, le etichette sono i veri valori della feature segreta.
            attack_model.fit(X=predictions_train, y=self.X_train[:, ATTACK_FEATURE_INDEX])

            # 2.3 Eseguiamo l'inferenza usando il nostro modello di attacco.
            print("  - Step 3: Inferenza dei valori della feature segreta...")
            # Usiamo il modello addestrato per predire la feature, usando le previsioni sui dati di test.
            inferred_values = attack_model.predict(predictions_test)
            
            # 2.4 Valutazione dell'attacco
            true_values = self.X_test[:, ATTACK_FEATURE_INDEX]
            attack_mse = mean_squared_error(true_values, inferred_values)

            print("\n--- RISULTATI ATTRIBUTE INFERENCE ATTACK ---")
            print(f"  - Feature Target: Componente PCA #{ATTACK_FEATURE_INDEX}")
            print(f"  - Mean Squared Error (MSE) dell'attacco: {attack_mse:.6f}")
            print("---------------------------------------------")

        except Exception as e:
            print(f"[CLIENT MALEVOLO-AIA {self.client_id}] ERRORE durante l'attacco con ART: {e}")
            attack_mse = -1.0

        metrics = {
            "client_id": self.client_id,
            "test_accuracy": float(acc_test),
            "aia_mse": float(attack_mse)
        }
        
        loss = 1.0 - acc_test
        return loss, len(self.X_test), metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fed_RF_client_malicious_aia.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    
    print(f"\n=== CLIENT MALEVOLO (AIA) RF INCREMENTALE {client_id} ===")
    
    max_message_length = 1024 * 1024 * 1024

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MaliciousAIAClient(client_id=client_id),
        grpc_max_message_length=max_message_length,
    )