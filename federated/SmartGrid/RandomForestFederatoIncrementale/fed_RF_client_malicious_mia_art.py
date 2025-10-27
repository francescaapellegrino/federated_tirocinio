"""
Client federato MALEVOLO per Attribute Inference Attack (AIA)
Sfrutta la libreria ART per sferrare l'attacco.
L'obiettivo è inferire una feature statistica usata nel preprocessing.
Francesca Pellegrino
"""

import flwr as fl
import numpy as np
import pandas as pd
import pickle
import zlib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss, mean_squared_error
)
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
import sys
import warnings
warnings.filterwarnings("ignore")

from preprocessing_aia import load_data_for_aia

# CONFIGURAZIONE
N_NEW_TREES = 8
CRITERION = "gini"
RANDOM_STATE = 42
TARGET_FEATURE_NAME = 'row_skew'

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class MaliciousAIAClient(fl.client.NumPyClient):
    """
    Questo client si comporta come un client onesto durante il fit,
    ma usa il modello globale ricevuto in evaluate per sferrare un 
    Attribute Inference Attack.
    """
    def __init__(self, client_id):
        self.client_id = client_id
        self.n_new_trees = N_NEW_TREES
        self.criterion = CRITERION
        self.random_state = RANDOM_STATE
        
        # Carichiamo tutti i dati UNA SOLA VOLTA qui per garantire coerenza
        (self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test,
         X_train_pre_pca, X_test_pre_pca) = load_data_for_aia(15)
        
        try:
            # Salviamo i valori della feature target come attributi dell'oggetto
            self.attack_feature_values_train = X_train_pre_pca[TARGET_FEATURE_NAME].to_numpy()
            self.attack_feature_values_test = X_test_pre_pca[TARGET_FEATURE_NAME].to_numpy()
        except KeyError:
            raise ValueError(f"La feature target '{TARGET_FEATURE_NAME}' non è stata trovata nelle colonne dei dati pre-PCA.")

        self.model = None

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
            # --- Parte 2: Sferrare l'Attribute Inference Attack con ART (Logica Corretta) ---

            art_classifier = SklearnClassifier(model=agg_model)
            
            # 2.1 Inizializzazione dell'attacco
            attack = AttributeInferenceBlackBox(
                estimator=art_classifier, 
                attack_model_type='rf'
            )

            # 2.2 Addestramento del modello di attacco
            # Generiamo le previsioni del modello vittima che serviranno come input per l'attacco
            predictions_train = art_classifier.predict(self.X_train)
            
            # Le "etichette" per l'addestramento dell'attacco sono i valori reali della feature segreta
            y_train_attack = self.attack_feature_values_train

            # Chiamata al metodo di addestramento corretto, passandogli le previsioni
            attack.fit_attack_model(pred=predictions_train, y=y_train_attack)
            
            # 2.3 Inferenza della feature target sui dati di test
            inferred_values = attack.infer(self.X_test, y=self.y_test)
            
            # 2.4 Valutazione dell'attacco
            true_values = self.attack_feature_values_test
            attack_mse = mean_squared_error(true_values, inferred_values)

            print("\n--- RISULTATI ATTRIBUTE INFERENCE ATTACK ---")
            print(f"  - Feature Target: '{TARGET_FEATURE_NAME}'")
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