"""
Client federato SmartGrid con Random Forest
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
from federated.SmartGrid.RandomForestFederatoIncrementale.preprocessing import load_improved_client_data

# CONFIGURAZIONE
N_NEW_TREES = 8            # Numero di nuovi alberi da addestrare per round
CRITERION = "gini"         # Splitting rule: "gini" oppure "entropy"
RANDOM_STATE = 42          # Per riproducibilità

# Imposta seed globale per numpy e random per ulteriore riproducibilità
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class FederatedRandomForestClient(fl.client.NumPyClient):
    """
    Client federato Random Forest.
    Ogni round:
    - Addestra solo N nuovi alberi sui dati locali.
    - Invia solo i nuovi alberi al server.
    - Riceve modello globale aggregato per valutazione.
    - Stampa e invia metriche di validazione e test.
    """
    def __init__(self, client_id, n_new_trees=N_NEW_TREES, criterion=CRITERION, random_state=RANDOM_STATE):
        self.client_id = client_id
        self.n_new_trees = n_new_trees
        self.criterion = criterion
        self.random_state = random_state
        config = None
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, _ = load_improved_client_data(
            client_id, config
        )
        self.model = None

    def get_parameters(self, config):
        # Inizializza struttura modello vuota da inviare al server
        empty_model_data = {
            "new_estimators": [],
            "n_features_in_": self.X_train.shape[1],
            "classes_": np.unique(self.y_train)
        }
        pickled_model = pickle.dumps(empty_model_data)
        compressed_model = zlib.compress(pickled_model)
        return [np.frombuffer(compressed_model, dtype=np.uint8)]

    def fit(self, parameters, config):
        print(f"\n[CLIENT {self.client_id}] Federated RandomForest...")
        print(f"Configurazione: n_new_trees={self.n_new_trees}, criterion={self.criterion}")

        # Decodifica modello globale
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)

        # round_num = config.get("server_round", 0) if config else 0

        # Addestra solo nuovi alberi (random_state fisso)
        self.model = RandomForestClassifier(
            n_estimators=self.n_new_trees,
            criterion=self.criterion,
            warm_start=False,
            bootstrap=True,
            random_state=self.random_state,  # <-- RANDOM STATE FISSO
            n_jobs=-1
        )
        print(f"[CLIENT {self.client_id}] Addestro {self.n_new_trees} nuovi alberi con splitting rule '{self.criterion}' e random_state={self.random_state}.")
        self.model.fit(self.X_train, self.y_train)

        model_data = {
            "new_estimators": [pickle.dumps(est) for est in self.model.estimators_],
            "n_features_in_": self.model.n_features_in_,
            "classes_": self.model.classes_
        }
        pickled_model = pickle.dumps(model_data)
        compressed_model = zlib.compress(pickled_model)

        # METRICHE VALIDAZIONE SUI DATI LOCALI
        y_pred = self.model.predict(self.X_val)
        acc = accuracy_score(self.y_val, y_pred)
        prec = precision_score(self.y_val, y_pred)
        rec = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        cm = confusion_matrix(self.y_val, y_pred)
        specificity = 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
            val_loss = log_loss(self.y_val, y_pred_proba)
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Errore calcolo val_loss: {e}")
            val_loss = None

        print(f"[CLIENT {self.client_id}] METRICHE VALIDAZIONE:")
        print(f" - Accuracy   : {acc:.4f}")
        print(f" - Precision  : {prec:.4f}")
        print(f" - Recall     : {rec:.4f}")
        print(f" - F1-Score   : {f1:.4f}")
        print(f" - Specificity: {specificity:.4f}")
        if val_loss is not None:
            print(f" - Val Loss   : {val_loss:.4f}")
        else:
            print(f" - Val Loss   : N/A")

        # Metriche inviate al server
        metrics = {
            "val_accuracy": float(acc),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1_score": float(f1),
            "val_specificity": float(specificity),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "client_id": int(self.client_id),
            "n_new_trees": self.n_new_trees,
            "criterion": self.criterion,
        }
        return [np.frombuffer(compressed_model, dtype=np.uint8)], len(self.X_train), metrics

    def evaluate(self, parameters, config):
        # Riceve il modello globale aggregato dal server
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        all_estimators = [pickle.loads(est) for est in agg_model_data["estimators"]]

        if len(all_estimators) == 0:
            print(f"[CLIENT {self.client_id}] Nessun albero aggregato dal server! Test impossibile.")
            metrics = {
                "test_accuracy": 0.0,
                "test_precision": 0.0,
                "test_recall": 0.0,
                "test_f1_score": 0.0,
                "test_specificity": 0.0,
                "client_id": int(self.client_id),
                "empty_global_model": True
            }
            return 1.0, len(self.X_test), metrics

        # Ricostruisci il modello globale Random Forest
        agg_model = RandomForestClassifier(n_estimators=len(all_estimators))
        agg_model.estimators_ = all_estimators
        agg_model.n_features_in_ = agg_model_data["n_features_in_"]
        agg_model.classes_ = agg_model_data["classes_"]
        agg_model.n_classes_ = len(agg_model.classes_)
        agg_model.n_outputs_ = 1

        # Valutazione su test locale
        y_pred = agg_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        specificity = 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"[CLIENT {self.client_id}] METRICHE TEST SUL MODELLO GLOBALE:")
        print(f" - Accuracy   : {acc:.4f}")
        print(f" - Precision  : {prec:.4f}")
        print(f" - Recall     : {rec:.4f}")
        print(f" - F1-Score   : {f1:.4f}")
        print(f" - Specificity: {specificity:.4f}")
        print(f" - Confusion  : {cm.ravel() if cm.size == 4 else cm}")

        metrics = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1_score": float(f1),
            "test_specificity": float(specificity),
            "client_id": int(self.client_id),
            "n_global_trees": len(all_estimators),
            "criterion": self.criterion
        }
        return 1.0 - acc, len(self.X_test), metrics

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Uso: python federated_random_forest_client.py <client_id> [n_new_trees] [criterion]")
        print("Esempio: python federated_random_forest_client.py 1 10 entropy")
        sys.exit(1)
    try:
        client_id = int(sys.argv[1])
        n_new_trees = int(sys.argv[2]) if len(sys.argv) > 2 else N_NEW_TREES
        criterion = sys.argv[3] if len(sys.argv) > 3 else CRITERION
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
        if criterion not in ["gini", "entropy"]:
            raise ValueError("criterion deve essere 'gini' o 'entropy'")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)

    print(f"\n=== CLIENT RANDOM FOREST INCREMENTALE MIGLIORATO {client_id} ===")
    print("=" * 60)
    print(f"Modello: RandomForestClassifier (sklearn, n_jobs=-1)")
    print(f"Preprocessing: improved_preprocessing.py")
    print(f"Numero nuovi alberi per round: {n_new_trees}")
    print(f"Splitting rule: {criterion}")
    print(f"Connessione al server in corso...")
    print("=" * 60)

    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=FederatedRandomForestClient(client_id, n_new_trees, criterion)
        )
    except Exception as e:
        print(f"Errore client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"\nCLIENT RANDOM FOREST INCREMENTALE MIGLIORATO {client_id} TERMINATO!")

if __name__ == "__main__":
    main()