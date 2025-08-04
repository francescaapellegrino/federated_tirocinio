import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def weighted_average(metrics):
    """Calcola la media pesata delle metriche dei client."""
    if not metrics:
        return {}
    
    metrics_sum = {}
    total_examples = 0
    
    for num_examples, metrics_dict in metrics:
        total_examples += num_examples
        for key, value in metrics_dict.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += num_examples * value
    
    return {
        key: value / total_examples 
        for key, value in metrics_sum.items()
    }

def get_smartgrid_evaluate_fn():
    """Crea la funzione di valutazione globale per il server."""
    
    def load_global_test_data():
        """Carica dataset globale per valutazione server."""
        print("Caricamento dataset globale per valutazione server...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_clients = [14, 15]
        df_list = []

        for client_id in test_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"  - Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"  - File data{client_id}.csv non trovato, saltato")
                continue

        if not df_list:
            print("  - ATTENZIONE: Nessun file di test globale trovato!")
            return None, None

        df_global = pd.concat(df_list, ignore_index=True)
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        X_global.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_global.dropna(inplace=True)
        y_global = y_global.loc[X_global.index]
        
        scaler_global = StandardScaler()
        X_global_scaled = scaler_global.fit_transform(X_global)
        
        print(f"Dataset globale preparato: {len(X_global_scaled)} campioni")
        print(f"Distribuzione classi: Attacchi={y_global.sum()}, Naturali={(y_global==0).sum()}")
        
        return X_global_scaled, y_global
    
    # Carica i dati una volta sola
    X_global, y_global = load_global_test_data()
    
    if X_global is None:
        print("Usando dati fittizi per valutazione globale")
        X_global = np.random.random((100, 128))
        y_global = np.random.randint(0, 2, 100)
    
    input_shape = X_global.shape[1]
    
    def evaluate(server_round, parameters, config):
        """Valuta il modello globale."""
        print(f"\n=== VALUTAZIONE GLOBALE ROUND {server_round} ===")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", "precision", "recall"]
        )
        
        model.set_weights(parameters)
        loss, accuracy, precision, recall = model.evaluate(X_global, y_global, verbose=0)
        
        predictions = model.predict(X_global, verbose=0)
        predictions_binary = (predictions > 0.5).astype(int).flatten()
        
        tn = np.sum((y_global == 0) & (predictions_binary == 0))
        fp = np.sum((y_global == 0) & (predictions_binary == 1))
        fn = np.sum((y_global == 1) & (predictions_binary == 0))
        tp = np.sum((y_global == 1) & (predictions_binary == 1))
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Metriche globali:")
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
        print(f"Matrice confusione:")
        print(f"  - True Negative: {tn}")
        print(f"  - False Positive: {fp}")
        print(f"  - False Negative: {fn}")
        print(f"  - True Positive: {tp}")
        
        return loss, {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)
        }
    
    return evaluate

def main():
    """Avvia il server federato."""
    print("\n=== AVVIO SERVER FEDERATO SMARTGRID ===")
    print("Configurazione:")
    print("  - Rounds: 50")
    print("  - Client minimi: 2")
    print("  - Valutazione: Dataset globale (client 14-15)")
    print("=" * 50)
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_smartgrid_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )

if __name__ == "__main__":
    main()