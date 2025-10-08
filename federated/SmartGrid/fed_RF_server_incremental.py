"""
Server federato SmartGrid con Random Forest (Approccio Evolutivo)
Francesca Pellegrino

VERSIONE 4: Implementa un approccio ibrido. Il server mantiene una popolazione
fissa di alberi e ad ogni round seleziona i migliori tra i vecchi e i nuovi
arrivati, permettendo al modello di "evolvere".
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

# Imposta seed globale per riproducibilità
np.random.seed(42)
random.seed(42)

# --- Liste globali per il logging e per mantenere lo stato del modello ---
ALL_CLIENT_METRICS = []
GLOBAL_METRICS = []
# Questa lista ora conterrà la popolazione di alberi della foresta globale,
# che si evolve ad ogni round.
GLOBAL_ESTIMATORS = [] 

def load_validation_data():
    """Carica un set di validazione centrale per la selezione degli alberi."""
    from preprocessing import load_improved_client_data
    print("[SERVER] Caricamento validation set centrale (dal client 1)...")
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

    # 1. Estrai i nuovi alberi inviati dai client
    for i, params in enumerate(client_parameters_list):
        try:
            arr_list = fl.common.parameters_to_ndarrays(params)
            compressed_bytes = arr_list[0].tobytes()
            param_bytes = zlib.decompress(compressed_bytes)
            model_data = pickle.loads(param_bytes)
            
            for est_pickled in model_data["new_estimators"]:
                tree = pickle.loads(est_pickled)
                newly_received_estimators.append(tree)

            if n_features_in_ is None:
                n_features_in_ = model_data.get("n_features_in_")
            if classes_ is None:
                classes_ = model_data.get("classes_")
        except Exception as e:
            print(f"[SERVER] ERRORE UNPICKLING client {i+1}: {e}.")

    # 2. Crea la "piscina" di candidati: i vecchi alberi + i nuovi
    candidate_pool = current_estimators + newly_received_estimators
    print(f"[SERVER] Piscina di candidati: {len(current_estimators)} vecchi + {len(newly_received_estimators)} nuovi = {len(candidate_pool)} totali.")

    # 3. Valuta ogni albero e assegna un punteggio
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

    # 4. Ordina e seleziona i migliori per la "prossima generazione"
    tree_scores.sort(key=lambda x: x[1], reverse=True)
    best_estimators = [t[0] for t in tree_scores[:max_global_trees]]
    
    print(f"[SERVER] Selezionati i migliori {len(best_estimators)} alberi per la nuova foresta globale.")

    # Prepara il modello aggiornato per l'invio
    agg_model_data = {
        "estimators": [pickle.dumps(est) for est in best_estimators],
        "n_features_in_": n_features_in_,
        "classes_": classes_
    }
    pickled_agg = pickle.dumps(agg_model_data)
    compressed_agg = zlib.compress(pickled_agg)
    
    # Restituisce i parametri per Flower, i dati del modello per la valutazione e la nuova popolazione di alberi
    return fl.common.ndarrays_to_parameters([np.frombuffer(compressed_agg, dtype=np.uint8)]), agg_model_data, best_estimators

# La funzione evaluate_global_model non cambia, è già perfetta per il logging
def evaluate_global_model(agg_model_data, X_val, y_val, round_num, filename="federated_rf_evolutionary_global_metrics.txt"):
    """Valuta il modello globale aggregato su validation centrale e salva le metriche su file."""
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
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    print(f"[SERVER][GLOBAL MODEL][Round {round_num}]")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall   : {rec:.4f}")
    print(f" - F1-Score : {f1:.4f}")

    GLOBAL_METRICS.append({"round": round_num, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    with open(filename, "a", encoding="utf-8") as f:
        if round_num == 1:
            f.write("round\taccuracy\tprecision\trecall\tf1_score\n")
        f.write(f"{round_num}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\n")


class FederatedRandomForestStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        self.X_val_central, self.y_val_central = load_validation_data()
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        """Sovrascrive l'aggregazione per implementare la logica evolutiva."""
        global GLOBAL_ESTIMATORS
        print(f"\n=== AGGREGAZIONE FIT ROUND {server_round} (Approccio Evolutivo) ===")
        
        # Raccogli le metriche (non cambia)
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                fit_res.metrics['round'] = server_round
                ALL_CLIENT_METRICS.append(fit_res.metrics.copy())

        client_params = [fit_res.parameters for _, fit_res in results]
        
        # --- MODIFICA CHIAVE ---
        # Chiama la nuova funzione `evolve_random_forest` passando gli alberi correnti
        agg_params, agg_model_data, new_global_estimators = evolve_random_forest(
            current_estimators=GLOBAL_ESTIMATORS,
            client_parameters_list=client_params,
            max_global_trees=100,  # Manteniamo la foresta a 100 alberi
            X_val=self.X_val_central,
            y_val=self.y_val_central
        )
        
        # Aggiorna la popolazione globale di alberi con la nuova generazione
        GLOBAL_ESTIMATORS = new_global_estimators
        
        # Valuta e logga il modello globale aggiornato
        if agg_model_data:
            evaluate_global_model(agg_model_data, self.X_val_central, self.y_val_central, server_round)
            
        return agg_params, {}

# Il resto del codice (salvataggio metriche client, main, etc.) non necessita di modifiche.
# Assicurati solo di usare nomi file diversi per non sovrascrivere i risultati precedenti.

def save_client_metrics_to_file(filename="federated_rf_evolutionary_client_metrics.txt"):
    if not ALL_CLIENT_METRICS: return
    all_keys = sorted(list(set(k for m in ALL_CLIENT_METRICS for k in m.keys())))
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\t".join(all_keys) + "\n")
        for m in ALL_CLIENT_METRICS:
            f.write("\t".join([str(m.get(k, "")) for k in all_keys]) + "\n")
    print(f"Metriche client salvate in {filename}")

def main():
    print("\nAVVIO SERVER FEDERATED RANDOM FOREST (EVOLUTIVO - v4)")
    
    strategy = FederatedRandomForestStrategy(
        fraction_fit=5/15,
        min_fit_clients=5,
        min_available_clients=15,
    )
    
    server_config = fl.server.ServerConfig(num_rounds=200) # Possiamo tornare a 200 round
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