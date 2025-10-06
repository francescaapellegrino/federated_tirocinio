import flwr as fl
import numpy as np
import pickle
import zlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_auc_score
import sys
import warnings
warnings.filterwarnings("ignore")

from preprocessing import load_improved_client_data

class FederatedRandomForestMaliciousClient(fl.client.NumPyClient):
    def __init__(self, client_id, n_new_trees=5, random_state=42):
        self.client_id = client_id
        self.n_new_trees = n_new_trees
        self.random_state = random_state
        config = None
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, _ = load_improved_client_data(
            client_id, config
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
        print(f"\n[MALEVOLENT CLIENT {self.client_id}] Federated RandomForest...")
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        self.model = RandomForestClassifier(
            n_estimators=self.n_new_trees,
            warm_start=False,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        model_data = {
            "new_estimators": [pickle.dumps(est) for est in self.model.estimators_],
            "n_features_in_": self.model.n_features_in_,
            "classes_": self.model.classes_
        }
        pickled_model = pickle.dumps(model_data)
        compressed_model = zlib.compress(pickled_model)

        # METRICHE VALIDAZIONE
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

        metrics = {
            "val_accuracy": float(acc),
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1_score": float(f1),
            "val_specificity": float(specificity),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "client_id": int(self.client_id),
            "n_new_trees": self.n_new_trees
        }
        return [np.frombuffer(compressed_model, dtype=np.uint8)], len(self.X_train), metrics

    def evaluate(self, parameters, config):
        # Ricostruisci il modello aggregato dal server
        compressed_bytes = parameters[0].tobytes()
        param_bytes = zlib.decompress(compressed_bytes)
        agg_model_data = pickle.loads(param_bytes)
        all_estimators = [pickle.loads(est) for est in agg_model_data["estimators"]]

        if len(all_estimators) == 0:
            print(f"[MALEVOLENT CLIENT {self.client_id}] Nessun albero aggregato dal server! Test impossibile.")
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

        agg_model = RandomForestClassifier(n_estimators=len(all_estimators))
        agg_model.estimators_ = all_estimators
        agg_model.n_features_in_ = agg_model_data["n_features_in_"]
        agg_model.classes_ = agg_model_data["classes_"]
        agg_model.n_classes_ = len(agg_model.classes_)
        agg_model.n_outputs_ = 1

        # METRICHE STANDARD
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

        # --- ATTACCHI DI INFERENZA ---
        print(f"\n[MALEVOLENT CLIENT {self.client_id}] Attacchi di inferenza in corso...")

        # 1. Membership Inference Attack (didattico)
        train_probs = agg_model.predict_proba(self.X_train)[:, 1]
        test_probs = agg_model.predict_proba(self.X_test)[:, 1]
        train_conf = np.abs(train_probs - 0.5)
        test_conf = np.abs(test_probs - 0.5)
        threshold = np.percentile(np.concatenate([train_conf, test_conf]), 90)
        train_membership_pred = (train_conf > threshold).astype(int)
        test_membership_pred = (test_conf > threshold).astype(int)
        y_membership_true = np.concatenate([np.ones_like(train_membership_pred), np.zeros_like(test_membership_pred)])
        y_membership_pred = np.concatenate([train_membership_pred, test_membership_pred])
        mia_acc = accuracy_score(y_membership_true, y_membership_pred)
        print(f"- Membership Inference Attack accuracy: {mia_acc:.4f}")

        # 2. Property Inference Attack (attack ratio)
        pred_mean = np.mean(test_probs)
        actual_attack_ratio = np.mean(self.y_test)
        property_error = abs(pred_mean - actual_attack_ratio)
        print(f"- Property Inference: pred_mean={pred_mean:.4f}, actual_attack_ratio={actual_attack_ratio:.4f}, error={property_error:.4f}")

        # 3. Model Inversion Attack (semplificato): "prototipo" che massimizza la probabilità di attacco
        # Per Random Forest, puoi generare un campione fittizio e iterare modificando le feature per massimizzare la predizione
        # Qui facciamo una versione base: testiamo i massimi/minimi delle feature
        feature_max = np.max(self.X_train, axis=0)
        feature_min = np.min(self.X_train, axis=0)
        proto_attack = feature_max
        proto_normal = feature_min
        proto_attack_pred = agg_model.predict_proba(proto_attack.reshape(1, -1))[0, 1]
        proto_normal_pred = agg_model.predict_proba(proto_normal.reshape(1, -1))[0, 1]
        print(f"- Model Inversion: attack prototype prob={proto_attack_pred:.4f}, normal prototype prob={proto_normal_pred:.4f}")

        # Salva risultati su file (didattico)
        with open(f"malicious_client_inference_results_{self.client_id}.txt", "w", encoding="utf-8") as f:
            f.write(f"Membership Inference Accuracy: {mia_acc:.4f}\n")
            f.write(f"Property Inference Error: {property_error:.4f}\n")
            f.write(f"Model Inversion - Attack Prototype Prob: {proto_attack_pred:.4f}\n")
            f.write(f"Model Inversion - Normal Prototype Prob: {proto_normal_pred:.4f}\n")

        metrics = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1_score": float(f1),
            "test_specificity": float(specificity),
            "client_id": int(self.client_id),
            "mia_accuracy": float(mia_acc),
            "property_inference_error": float(property_error),
            "model_inversion_attack_prob": float(proto_attack_pred),
            "model_inversion_normal_prob": float(proto_normal_pred)
        }
        return 1.0 - acc, len(self.X_test), metrics

def main():
    if len(sys.argv) != 2:
        print("Uso: python federated_random_forest_malicious_client.py <client_id>")
        sys.exit(1)
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)

    print(f"\n=== MALEVOLENT CLIENT RANDOM FOREST {client_id} ===")
    print("=" * 60)
    print("Questo client esegue attacchi di inferenza sul modello federato!")
    print("=" * 60)

    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=FederatedRandomForestMaliciousClient(client_id)
        )
    except Exception as e:
        print(f"❌ Errore client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"\nMALEVOLENT CLIENT RANDOM FOREST {client_id} TERMINATO!")

if __name__ == "__main__":
    main()