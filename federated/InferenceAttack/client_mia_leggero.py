import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import KerasClassifier

def load_client_data(client_id):
    """Carica i dati per il client malevolo."""
    print(f"=== CARICAMENTO DATI CLIENT MALEVOLO {client_id} ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
        
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"Dataset preparato per attacco MIA:")
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

class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)
        self.model = create_model(self.X_train.shape[1])
        print("[MALICIOUS] Client inizializzato per attacco MIA")
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def perform_mia_attack(self):
        """Esegue l'attacco Membership Inference."""
        print("\n[MALICIOUS] Esecuzione attacco MIA...")
        
        try:
            # Wrapper del modello per ART
            classifier = KerasClassifier(
                model=self.model,
                clip_values=(0, 1),
                use_logits=False
            )
            
            # Preparazione dati per l'attacco
            n_samples = 100
            X_train_attack = self.X_train[:n_samples]
            y_train_attack = self.y_train[:n_samples].to_numpy()  # Convertiamo in numpy array
            X_test_attack = self.X_test[:n_samples]
            y_test_attack = self.y_test[:n_samples].to_numpy()    # Convertiamo in numpy array
            
            print("[MALICIOUS] Preparazione dati di attacco...")
            
            # Predizioni usando il modello direttamente
            train_pred = self.model.predict(X_train_attack, verbose=0)
            test_pred = self.model.predict(X_test_attack, verbose=0)
            
            # Assicuriamoci che le predizioni abbiano la forma corretta
            if len(train_pred.shape) == 1:
                train_pred = train_pred.reshape(-1, 1)
            if len(test_pred.shape) == 1:
                test_pred = test_pred.reshape(-1, 1)
                
            # Convertiamo e rimodelliamo le label
            y_train_attack = y_train_attack.reshape(-1, 1)
            y_test_attack = y_test_attack.reshape(-1, 1)
            
            print("[MALICIOUS] Creazione modello di attacco...")
            
            # Creazione dell'attacco
            mia = MembershipInferenceBlackBox(
                classifier,
                attack_model_type='rf',
                input_type='prediction'
            )
            
            print("[MALICIOUS] Addestramento modello di attacco...")
            
            # Addestra il modello di attacco
            mia.fit(
                x=X_train_attack,
                y=y_train_attack,
                test_x=X_test_attack,
                test_y=y_test_attack,
                pred=train_pred,
                test_pred=test_pred
            )
            
            print("[MALICIOUS] Esecuzione inferenza...")
            
            # Prepara i dati per l'inferenza
            X_all = np.concatenate([X_train_attack, X_test_attack])
            y_all = np.concatenate([y_train_attack, y_test_attack])
            pred_all = np.concatenate([train_pred, test_pred])
            membership = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
            
            # Esegui l'attacco
            inferred = mia.infer(x=X_all, y=y_all, pred=pred_all)
            attack_acc = np.mean(inferred == membership)
            
            # Analisi dettagliata dei risultati
            true_positives = np.sum((inferred == 1) & (membership == 1))
            false_positives = np.sum((inferred == 1) & (membership == 0))
            true_negatives = np.sum((inferred == 0) & (membership == 0))
            false_negatives = np.sum((inferred == 0) & (membership == 1))
            
            print(f"\n[MALICIOUS] Risultati attacco MIA:")
            print(f"  - Accuracy attacco: {attack_acc:.4f}")
            print(f"  - Membri correttamente identificati: {true_positives}/{n_samples} ({true_positives/n_samples*100:.1f}%)")
            print(f"  - Non membri correttamente identificati: {true_negatives}/{n_samples} ({true_negatives/n_samples*100:.1f}%)")
            print(f"  - Falsi positivi: {false_positives}/{n_samples} ({false_positives/n_samples*100:.1f}%)")
            print(f"  - Falsi negativi: {false_negatives}/{n_samples} ({false_negatives/n_samples*100:.1f}%)")
            
            if attack_acc > 0.5:
                print(f"\n[MALICIOUS] ⚠️ ATTACCO RIUSCITO!")
                print(f"  Il modello rivela informazioni sui dati di training con accuracy {attack_acc:.2%}")
            else:
                print(f"\n[MALICIOUS] Attacco non riuscito")
                print(f"  Il modello sembra resistente all'inferenza di membership")
            
            return attack_acc
            
        except Exception as e:
            print(f"[MALICIOUS] Errore durante l'attacco MIA: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def fit(self, parameters, config):
        print(f"\n[MALICIOUS] Client {self.client_id} - Avvio training locale...")
    
        self.model.set_weights(parameters)
    
        history = self.model.fit(
        self.X_train, self.y_train,
        epochs=1,
        batch_size=32,
        verbose=0
        )
    
          # Esegui l'attacco MIA
        mia_accuracy = self.perform_mia_attack()
    
         # Assicurati che tutte le metriche siano valori scalari
        results = {
        'loss': float(history.history['loss'][-1]),
        'accuracy': float(history.history['accuracy'][-1]),
        'precision': float(history.history['precision'][-1]),
        'recall': float(history.history['recall'][-1]),
        'mia_accuracy': float(mia_accuracy)
        }
    
        print(f"[MALICIOUS] Training completato:")
        print(f"  - Loss: {results['loss']:.4f}")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
        print(f"  - MIA Accuracy: {results['mia_accuracy']:.4f}")
    
        return self.model.get_weights(), len(self.X_train), results
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        print(f"\n[MALICIOUS] Valutazione locale:")
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def main():
    if len(sys.argv) != 2:
        print("Uso: python client_mal.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)
    
    print(f"\n=== AVVIO CLIENT MALEVOLO {client_id} ===")
    print("Questo client eseguirà attacchi Membership Inference")
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MaliciousClient(client_id)
    )

if __name__ == "__main__":
    main()