'''
Federated Learning con Flower e Tensorflow, usando il dataset MNIST
'''

import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf

# Funzione di valutazione lato server
def evaluate_model(server_round, parameters, fit_results):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test, y_test = x_test / 255.0, y_test

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    model.set_weights(parameters)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Round {server_round}: Loss: {loss}, Accuracy: {accuracy}")
    return loss, {"accuracy": accuracy}

# Configura e avvia il server
strategy = FedAvg(
    fraction_fit=1.0, # Tutti i clien disponibili parteciapano ad ogni round
    min_fit_clients=2, # Almeno 2 client contribuiscono al training
    min_available_clients=2,
    evaluate_fn=evaluate_model # Usa la funzione evaluate_mode1 sopra definita
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3), # Fa partire il training per 3 round federati
    strategy=strategy, # Usa la strategy FedAvg
)