"""
Server federato SmartGrid con Random Forest incrementale
Francesca Pellegrino
"""

import flwr as fl
import pickle
import zlib
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import warnings
warnings.filterwarnings("ignore")

# Imposta seed globale per riproducibilità
np.random.seed(42)
random.seed(42)

# --- Liste globali per il logging e per mantenere lo stato del modello ---
ALL_CLIENT_METRICS = []
GLOBAL_METRICS = []
GLOBAL_ESTIMATORS = [] 

def load_validation_data():
    """Carica un set di validazione centrale per la selezione degli alberi."""
    # Il server usa il suo preprocessing stabile e originale.
    from preprocessing_common import load_improved_client_data
    print("[SERVER] Caricamento validation set centrale (dal client 1)...")
    
    # --- MODIFICA CHIAVE: Corretto lo spacchettamento a 7 valori ---
    # La funzione `load_improved_client_data` restituisce 7 valori.
    # A noi interessano solo X_val e y_val, quindi ignoriamo gli altri.
    #_, _, X_val, y_val, _, _, _ = load_improved_client_data(1, None)

    #_, _, X_val, y_val, _, _, _ = load_data_for_aia(client_id=1)
    #return X_val, y_val

    # ... all'interno di load_validation_data()
    _, _, X_val, y_val, _, _, _ = load_improved_client_data(1, None)
    return X_val, y_val

def evolve_random_forest(current_estimators, client_parameters_list, max_global_trees, X_val, y_val):
    """
    Funzione chiave dell'approccio evolutivo.
    1. Raccoglie i nuovi alberi dai client.
    2. Crea una "piscina" con i vecchi alberi (current_estimators) e i nuovi.
    3. Valuta ogni singolo albero della piscina sul set di validazione.
    4. Seleziona i migliori `max_global_trees` per formare la nuova generazione del modello.
    """
    newly_received_estimators = []
    n_features_in_ = None
    classes_ = None

    for i, params in enumerate(client_parameters_list):
        try:
            arr_list = fl.common.parameters_to_ndarrays(params)
            compressed_bytes = arr_list[0].tobytes()
            param_bytes = zlib.decompress(compressed_bytes)
            model_data = pickle.loads(param_bytes)
            
            for est_pickled in model_data["new_estimators"]:
                tree = pickle.loads(est_pickled)
                newly_received_estimators.append(tree)

            if n_features_in_ is None: n_features_in_ = model_data.get("n_features_in_")
            if classes_ is None: classes_ = model_data.get("classes_")
        except Exception as e:
            print(f"[SERVER] ERRORE UNPICKLING client {i+1}: {e}.")

    candidate_pool = current_estimators + newly_received_estimators
    print(f"[SERVER] Piscina di candidati: {len(current_estimators)} vecchi + {len(newly_received_estimators)} nuovi = {len(candidate_pool)} totali.")

    if not candidate_pool:
        return fl.common.ndarrays_to_parameters([]), {}, []

    print("[SERVER] Valutazione e selezione dei migliori alberi (evoluzione)...")
    tree_scores = []
    for tree in candidate_pool:
        try:
            y_pred = tree.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            tree_scores.append((tree, acc))
        except Exception as e:
            print(f"  - Errore valutazione albero singolo: {e}")

    tree_scores.sort(key=lambda x: x[1], reverse=True)
    best_estimators = [t[0] for t in tree_scores[:max_global_trees]]
    
    print(f"[SERVER] Selezionati i migliori {len(best_estimators)} alberi per la nuova foresta globale.")

    agg_model_data = {
        "estimators": [pickle.dumps(est) for est in best_estimators],
        "n_features_in_": n_features_in_,
        "classes_": classes_
    }
    pickled_agg = pickle.dumps(agg_model_data)
    compressed_agg = zlib.compress(pickled_agg)
    
    return fl.common.ndarrays_to_parameters([np.frombuffer(compressed_agg, dtype=np.uint8)]), agg_model_data, best_estimators

def evaluate_global_model(agg_model_data, X_val, y_val, round_num, filename="fed_RF_incremental_global_metrics_final.txt"):
    """Valuta il modello globale aggregato e salva le metriche, inclusa la Log Loss."""
    all_estimators = [pickle.loads(est) for est in agg_model_data.get("estimators", [])]
    if not all_estimators:
        print("[SERVER] Nessun albero aggregato da valutare!")
        return

    model = RandomForestClassifier(n_estimators=len(all_estimators), random_state=42, n_jobs=-1)
    model.estimators_ = all_estimators
    model.n_features_in_ = agg_model_data["n_features_in_"]
    model.classes_ = agg_model_data["classes_"]
    model.n_classes_ = len(model.classes_)
    model.n_outputs_ = 1

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    loss = log_loss(y_val, y_pred_proba) 
    
    print(f"[SERVER][GLOBAL MODEL][Round {round_num}]")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall   : {rec:.4f}")
    print(f" - F1-Score : {f1:.4f}")
    print(f" - Log Loss : {loss:.4f}")

    GLOBAL_METRICS.append({"round": round_num, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "loss": loss})
    
    # Aggiorna il file di log per includere la loss
    with open(filename, "a", encoding="utf-8") as f:
        if round_num == 1:
            f.write("round\taccuracy\tprecision\trecall\tf1_score\tloss\n")
        f.write(f"{round_num}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{loss:.4f}\n")


class FederatedRandomForestStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        self.X_val_central, self.y_val_central = load_validation_data()
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        """Sovrascrive l'aggregazione per implementare la logica evolutiva."""
        global GLOBAL_ESTIMATORS
        print(f"\n=== AGGREGAZIONE FIT ROUND {server_round} ===")
        
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                fit_res.metrics['round'] = server_round
                ALL_CLIENT_METRICS.append(fit_res.metrics.copy())

        client_params = [fit_res.parameters for _, fit_res in results]
        
        agg_params, agg_model_data, new_global_estimators = evolve_random_forest(
            current_estimators=GLOBAL_ESTIMATORS,
            client_parameters_list=client_params,
            max_global_trees=100,
            X_val=self.X_val_central,
            y_val=self.y_val_central
        )
        
        GLOBAL_ESTIMATORS = new_global_estimators
        
        if agg_model_data:
            evaluate_global_model(agg_model_data, self.X_val_central, self.y_val_central, server_round)
            
        return agg_params, {}
    
        # --- NUOVO METODO ---
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Sovrascrive l'aggregazione di evaluate per raccogliere le metriche di test
        e, soprattutto, le metriche dell'attacco (es. mia_accuracy).
        """
        print(f"--- AGGREGAZIONE EVALUATE ROUND {server_round} ---")
        
        # Raccogli le metriche di test (test_...) e dell'attacco dai client
        for _, eval_res in results:
            if hasattr(eval_res, 'metrics') and eval_res.metrics:
                metrics = eval_res.metrics.copy()
                metrics['round'] = server_round
                # Cerca una metrica di validazione esistente per questo client e round
                # per unire tutto in un'unica riga nel file di log.
                found = False
                for existing_metric in ALL_CLIENT_METRICS:
                    if existing_metric.get('client_id') == metrics.get('client_id') and \
                       existing_metric.get('round') == server_round:
                        existing_metric.update(metrics)
                        found = True
                        break
                if not found:
                    ALL_CLIENT_METRICS.append(metrics)
        
        # Chiama il metodo originale della superclasse per ottenere la loss aggregata
        return super().aggregate_evaluate(server_round, results, failures)

def save_client_metrics_to_file(filename="fed_RF_incremental_client_metrics_final.txt"):
    if not ALL_CLIENT_METRICS: return
    all_keys = sorted(list(set(k for m in ALL_CLIENT_METRICS for k in m.keys())))
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\t".join(all_keys) + "\n")
        for m in ALL_CLIENT_METRICS:
            f.write("\t".join([str(m.get(k, "")) for k in all_keys]) + "\n")
    print(f"Metriche client salvate in {filename}")

def main():
    print("\nAVVIO SERVER FEDERATED RANDOM FOREST INCREMENTALE")
    
    strategy = FederatedRandomForestStrategy(
        fraction_fit=5/14,
        min_fit_clients=5,
        min_available_clients=14,
        fraction_evaluate=1.0,
        min_evaluate_clients=14,
    )
    
    server_config = fl.server.ServerConfig(num_rounds=200)
    max_message_length = 1024 * 1024 * 1024

    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=server_config,
            strategy=strategy,
            grpc_max_message_length=max_message_length,
        )
    finally:
        print("\n[SERVER] Salvataggio finale delle metriche client...")
        save_client_metrics_to_file()
        print("[SERVER] Salvataggio completato.")

    print("\nSERVER TERMINATO!")

if __name__ == "__main__":
    main()