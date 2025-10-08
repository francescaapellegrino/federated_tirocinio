"""
Server federato SmartGrid con Random Forest
Francesca Pellegrino
"""

import flwr as fl
import pickle
import zlib
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import warnings
warnings.filterwarnings("ignore")

# Imposta seed globale per numpy e random per riproducibilità
np.random.seed(42)
random.seed(42)

ALL_CLIENT_METRICS = []
GLOBAL_METRICS = []

def load_validation_data():
    # Usa il preprocessing del progetto per caricare dati di validazione centrale.
    from federated.SmartGrid.RandomForestFederatoIncrementale.preprocessing import load_improved_client_data
    # Prendi il validation set dal client 1 (puoi modificarlo)
    _, _, X_val, y_val, _, _, _ = load_improved_client_data(1, None)
    return X_val, y_val

def aggregate_random_forest(client_parameters_list, max_global_trees=100, X_val=None, y_val=None):
    all_estimators = []
    n_features_in_ = None
    classes_ = None

    # Estrai solo i nuovi alberi dei client
    for i, params in enumerate(client_parameters_list):
        try:
            arr_list = fl.common.parameters_to_ndarrays(params)
            compressed_bytes = arr_list[0].tobytes()
            param_bytes = zlib.decompress(compressed_bytes)
            model_data = pickle.loads(param_bytes)
            new_estimators_pickled = model_data["new_estimators"]

            if not new_estimators_pickled:
                print(f"[SERVER] Client {i+1}: nessun nuovo albero, saltato.")
                continue

            for est in new_estimators_pickled:
                tree = pickle.loads(est)
                all_estimators.append(tree)

            if n_features_in_ is None:
                n_features_in_ = model_data.get("n_features_in_", None)
            if classes_ is None:
                classes_ = model_data.get("classes_", None)

        except Exception as e:
            print(f"[SERVER] ERRORE UNPICKLING client {i+1}: {e}. Parametri saltati.")

    print(f"[SERVER] Totale alberi aggregati (prima della selezione): {len(all_estimators)}")

    # Seleziona i migliori alberi su validation centrale
    if X_val is not None and y_val is not None and all_estimators:
        print("[SERVER] Selezione dei migliori alberi su validation centrale...")
        tree_scores = []
        for tree in all_estimators:
            y_pred = tree.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            tree_scores.append((tree, acc))
        tree_scores.sort(key=lambda x: x[1], reverse=True)
        all_estimators = [t[0] for t in tree_scores[:max_global_trees]]
        print(f"[SERVER] Selezionati i migliori {len(all_estimators)} alberi.")
    else:
        # Fallback: random selection con random seed fisso per riproducibilità
        if len(all_estimators) > max_global_trees:
            random.seed(42)  # <-- FISSO
            print(f"[SERVER] Limito a {max_global_trees} alberi (selezione casuale riproducibile)")
            all_estimators = random.sample(all_estimators, max_global_trees)

    agg_model_data = {
        "estimators": [pickle.dumps(est) for est in all_estimators],
        "n_features_in_": n_features_in_,
        "classes_": classes_
    }
    pickled_agg = pickle.dumps(agg_model_data)
    compressed_agg = zlib.compress(pickled_agg)
    return fl.common.ndarrays_to_parameters([np.frombuffer(compressed_agg, dtype=np.uint8)]), agg_model_data

def evaluate_global_model(agg_model_data, X_val, y_val, round_num, filename="federated_rf_global_metrics_summary.txt"):
    """Valuta il modello globale aggregato su validation centrale e salva le metriche su file"""
    all_estimators = [pickle.loads(est) for est in agg_model_data["estimators"]]
    if len(all_estimators) == 0:
        print("[SERVER] Nessun albero aggregato!")
        return

    # Ricostruisci il modello globale federato
    model = RandomForestClassifier(n_estimators=len(all_estimators), random_state=42)
    model.estimators_ = all_estimators
    model.n_features_in_ = agg_model_data["n_features_in_"]
    model.classes_ = agg_model_data["classes_"]
    model.n_classes_ = len(model.classes_)
    model.n_outputs_ = 1

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    specificity = 0.0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        val_loss = log_loss(y_val, y_pred_proba)
    except Exception:
        val_loss = None

    print(f"[SERVER][GLOBAL MODEL][Round {round_num}]")
    print(f" - Accuracy   : {acc:.4f}")
    print(f" - Precision  : {prec:.4f}")
    print(f" - Recall     : {rec:.4f}")
    print(f" - F1-Score   : {f1:.4f}")
    print(f" - Specificity: {specificity:.4f}")
    print(f" - Val Loss   : {val_loss:.4f}" if val_loss is not None else " - Val Loss   : N/A")

    # Salva su file (append)
    GLOBAL_METRICS.append({
        "round": round_num,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "loss": val_loss if val_loss is not None else "N/A"
    })
    with open(filename, "a", encoding="utf-8") as f:
        if round_num == 1:
            f.write("round\taccuracy\tprecision\trecall\tf1\tspecificity\tloss\n")
        f.write(f"{round_num}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{specificity:.4f}\t{val_loss if val_loss is not None else 'N/A'}\n")

class FederatedRandomForestStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        self.X_val_central, self.y_val_central = load_validation_data()
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n=== AGGREGAZIONE FIT ROUND {server_round} ===")
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics is not None:
                fit_res.metrics['round'] = server_round
                ALL_CLIENT_METRICS.append(fit_res.metrics.copy())
        client_params = [fit_res.parameters for _, fit_res in results]
        agg_params, agg_model_data = aggregate_random_forest(
            client_params,
            max_global_trees=100,
            X_val=self.X_val_central,
            y_val=self.y_val_central
        )
        # Valuta e salva metriche del modello globale federato
        evaluate_global_model(agg_model_data, self.X_val_central, self.y_val_central, server_round)
        return agg_params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        for _, eval_res in results:
            if hasattr(eval_res, 'metrics') and eval_res.metrics is not None:
                print(f"[SERVER] METRICHE CLIENT TEST (round {server_round}): {eval_res.metrics}")
        return super().aggregate_evaluate(server_round, results, failures)

def save_metrics_to_file(filename="federated_random_forest_metrics_summary_improved.txt"):
    if not ALL_CLIENT_METRICS:
        print("Nessuna metrica da salvare.")
        return
    all_keys = set()
    for m in ALL_CLIENT_METRICS:
        all_keys.update(m.keys())
    all_keys = sorted(list(all_keys))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\t".join(all_keys) + "\n")
        for m in ALL_CLIENT_METRICS:
            row = [str(m.get(k, "")) for k in all_keys]
            f.write("\t".join(row) + "\n")
    print(f"Metriche client salvate in {filename}")

def main():
    print("\nAVVIO SERVER FEDERATED RANDOM FOREST (MIGLIORATO)")
    print("=" * 60)
    print("Aspetto connessione dei client...")

    strategy = FederatedRandomForestStrategy(
        fraction_fit=5/15,
        min_fit_clients=5,
        min_available_clients=15,
    )

    server_config = fl.server.ServerConfig(num_rounds=200)

    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=server_config,
            strategy=strategy,
        )
    finally:
        save_metrics_to_file()

    print("\nSERVER FEDERATED RANDOM FOREST TERMINATO!")

if __name__ == "__main__":
    main()