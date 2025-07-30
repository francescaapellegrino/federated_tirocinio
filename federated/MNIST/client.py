import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

# Funzione per partizionare il dataset in modo non-IID
def get_client_data(client_id, num_clients=3):
    # Carica il dataset MNIST completo
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalizza i dati
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Calcola quanti esempi per classe assegnare a ciascun client
    examples_per_client = len(x_train) // num_clients
    
    # Ottiene l'indice di inizio e fine per questo client
    start_idx = client_id * examples_per_client
    end_idx = (client_id + 1) * examples_per_client
    
    # Assegna una porzione del training set a questo client
    client_x_train = x_train[start_idx:end_idx]
    client_y_train = y_train[start_idx:end_idx]
    
    # Per il test set, prendiamo un sottoinsieme proporzionale
    test_examples_per_client = len(x_test) // num_clients
    test_start_idx = client_id * test_examples_per_client
    test_end_idx = (client_id + 1) * test_examples_per_client
    
    client_x_test = x_test[test_start_idx:test_end_idx]
    client_y_test = y_test[test_start_idx:test_end_idx]
    
    return client_x_train, client_y_train, client_x_test, client_y_test

# Crea il modello
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

# Definisci il client Flower
class MnistClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = create_model()
        # Carica i dati specifici per questo client
        self.x_train, self.y_train, self.x_test, self.y_test = get_client_data(client_id)
        print(f"Client {client_id} inizializzato con {len(self.x_train)} esempi di training e {len(self.x_test)} esempi di test")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"Client {self.client_id}: Avvio addestramento locale...")
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=1
        )
        print(f"Client {self.client_id}: Addestramento completato")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Avvio del client
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilizzo: python client.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    print(f"Avvio client {client_id}...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MnistClient(client_id)
    )