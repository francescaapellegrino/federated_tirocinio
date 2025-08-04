import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_client_data(client_id):
    """Carica i dati per il client."""
    print(f"=== CARICAMENTO DATI CLIENT {client_id} ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
        
    df = pd.read_csv(file_path)
    
    # Prepara feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    # Pulizia dati
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    # Normalizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"Dataset preparato:")
    print(f"  - Training: {len(X_train)} campioni")
    print(f"  - Test: {len(X_test)} campioni")
    
    return X_train, y_train, X_test, y_test

def create_model(input_shape):
    """Crea il modello."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model

class SmartGridClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)
        self.model = create_model(self.X_train.shape[1])
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        print(f"\n[Client {self.client_id}] Avvio training locale...")
        self.model.set_weights(parameters)
    
        history = self.model.fit(
        self.X_train, self.y_train,
        epochs=5,
        batch_size=32,
        verbose=0
        )
    
        # Prendi solo l'ultimo valore di ogni metrica
        results = {
        'loss': float(history.history['loss'][-1]),
        'accuracy': float(history.history['accuracy'][-1]),
        'precision': float(history.history['precision'][-1]),
        'recall': float(history.history['recall'][-1])
        }
    
        print(f"[Client {self.client_id}] Training completato:")
        print(f"  - Loss: {results['loss']:.4f}")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
    
        return self.model.get_weights(), len(self.X_train), results
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        print(f"\n[Client {self.client_id}] Valutazione locale:")
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def main():
    if len(sys.argv) != 2:
        print("Uso: python client_sg.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=SmartGridClient(client_id)
    )

if __name__ == "__main__":
    main()