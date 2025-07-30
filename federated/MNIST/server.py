import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Funzione per valutare le metriche aggregate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Estrae le accuratezze e i pesi (numero di esempi)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calcola la media pesata
    return {"accuracy": sum(accuracies) / sum(examples)}

# Funzione di valutazione centralizzata
def get_evaluate_fn():
    # Carica il dataset di test completo
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  # Normalizzazione
    
    # Crea il modello di valutazione
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    
    # Funzione di valutazione
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Server-side evaluation - Round: {server_round}")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, {"accuracy": accuracy}
    
    return evaluate

# Configura la strategia di federazione
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Usa tutti i client disponibili per il training
    fraction_evaluate=1.0,  # Usa tutti i client disponibili per la valutazione
    min_fit_clients=2,  # Numero minimo di client per il training
    min_evaluate_clients=2,  # Numero minimo di client per la valutazione
    min_available_clients=2,  # Numero minimo di client che devono essere disponibili
    evaluate_fn=get_evaluate_fn(),  # Funzione di valutazione centralizzata
    evaluate_metrics_aggregation_fn=weighted_average,  # Funzione di aggregazione delle metriche
)

# Avvia il server
def main():
    print("Server Federated Learning in avvio...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # 5 round di training federato
        strategy=strategy
    )

if __name__ == "__main__":
    main()