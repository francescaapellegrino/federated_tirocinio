import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Nuova dipendenza per attacco MIA
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import KerasClassifier

def load_client_smartgrid_data(client_id):
    # ... [CODICE IDENTICO ALLA VERSIONE ORIGINALE] ...
    # (vedi file originale per dettagli)
    # [Funzione invariata, non la ripeto qui per chiarezza]

    # --- INIZIO COPIA DELLA FUNZIONE ORIGINALE ---
    print(f"=== CARICAMENTO DATI CLIENT {client_id} SMARTGRID ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")
    df = pd.read_csv(file_path)
    print(f"Dataset del client {client_id}:")
    print(f"  - Totale campioni: {len(df)}")
    print(f"  - Feature: {df.shape[1] - 1}")
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    print(f"Pulizia dei dati:")
    initial_samples = len(X)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isnull().sum().sum()
    print(f"  - Valori NaN trovati: {nan_count}")
    X.dropna(inplace=True)
    y = y.loc[X.index]
    final_samples = len(X)
    removed_samples = initial_samples - final_samples
    print(f"  - Campioni rimossi: {removed_samples}")
    print(f"  - Campioni finali: {final_samples}")
    if final_samples == 0:
        raise ValueError(f"Nessun campione valido rimasto per il client {client_id}")
    print(f"Normalizzazione feature locali...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"Suddivisione locale train/test:")
    print(f"  - Training set: {len(X_train)} campioni")
    print(f"  - Test set: {len(X_test)} campioni")
    if len(X_train) > 0:
        train_attack_ratio = y_train.mean() if len(y_train) > 0 else 0
        print(f"  - Proporzione attacchi training: {train_attack_ratio*100:.2f}%")
    if len(X_test) > 0:
        test_attack_ratio = y_test.mean() if len(y_test) > 0 else 0
        print(f"  - Proporzione attacchi test: {test_attack_ratio*100:.2f}%")
    dataset_info = {
        'client_id': client_id,
        'total_samples': final_samples,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'attack_samples': attack_samples,
        'natural_samples': natural_samples,
        'attack_ratio': attack_ratio,
        'features': X.shape[1]
    }
    print("=" * 60)
    return X_train, y_train, X_test, y_test, scaler, dataset_info
    # --- FINE COPIA DELLA FUNZIONE ORIGINALE ---

def create_smartgrid_client_model(input_shape):
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

client_id = None
model = None
X_train = None
y_train = None
X_test = None
y_test = None
dataset_info = None

class SmartGridMaliciousClient(fl.client.NumPyClient):
    """
    Client Flower malevolo: esegue un attacco Membership Inference dopo la ricezione dei pesi globali.
    """
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        global model, X_train, y_train, dataset_info

        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO ===")
        print(f"[Client {client_id}] Ricevuti parametri dal server, avvio addestramento locale...")
        sys.stdout.flush()

        # Imposta pesi globali
        model.set_weights(parameters)

        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}

        print(f"[Client {client_id}] Addestramento su {len(X_train)} campioni per 1 epoca...")
        history = model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        train_precision = history.history.get('precision', [0])[0]
        train_recall = history.history.get('recall', [0])[0]

        print(f"[Client {client_id}] Addestramento completato:")
        print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")

        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'client_id': int(dataset_info['client_id']),
            'total_samples': int(dataset_info['total_samples']),
            'train_samples': int(dataset_info['train_samples']),
            'test_samples': int(dataset_info['test_samples']),
            'attack_samples': int(dataset_info['attack_samples']),
            'natural_samples': int(dataset_info['natural_samples']),
            'attack_ratio': float(dataset_info['attack_ratio']),
            'features': int(dataset_info['features']),
        }

        # === ATTACCO MALEVOLo: Membership Inference BlackBox ===
        print(f"[Client {client_id}] INIZIO ATTACCO MEMBERSHIP INFERENCE (ART)")
        try:
            # 1. Wrap del modello Keras in un KerasClassifier di ART
            classifier = KerasClassifier(model=model, clip_values=(0, 1))
            # 2. Crea l'attacco MIA (black-box)
            mia_attack = MembershipInferenceBlackBox(classifier, input_type='tabular')

            # 3. Prepara i dati (X_train: membri, X_test: non membri)
            X_mia = np.concatenate([X_train, X_test])
            y_mia = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])  # 1 = membro, 0 = non membro

            # 4. Addestra l'attaccante sui dati locali
            mia_attack.fit(X_mia, y_mia)

            # 5. Prova l'attacco su un sottoinsieme dei dati
            n_samples = min(100, len(X_train), len(X_test))
            if n_samples == 0:
                print(f"[Client {client_id}] (MIA) Troppi pochi dati per testare l'attacco.")
            else:
                X_attack = np.concatenate([X_train[:n_samples], X_test[:n_samples]])
                y_attack = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
                y_pred = mia_attack.infer(X_attack)

                mia_accuracy = np.mean(y_pred == y_attack)
                print(f"[Client {client_id}] Risultato Membership Inference Attack:")
                print(f"[Client {client_id}]   - Accuracy attaccante: {mia_accuracy:.4f}")
                metrics['mia_accuracy'] = float(mia_accuracy)
        except Exception as e:
            print(f"[Client {client_id}] ERRORE durante l'attacco MIA: {e}")
            metrics['mia_accuracy'] = -1.0

        print(f"[Client {client_id}] Invio pesi aggiornati al server...")
        sys.stdout.flush()

        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        global model, X_test, y_test

        print(f"[Client {client_id}] === VALUTAZIONE LOCALE ===")
        model.set_weights(parameters)

        if len(X_test) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di test disponibile!")
            return 0.0, 0, {"accuracy": 0.0}

        results = model.evaluate(X_test, y_test, verbose=0)
        loss, accuracy, precision, recall = results
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"[Client {client_id}] Valutazione locale completata:")
        print(f"[Client {client_id}]   - Loss: {loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
        print(f"[Client {client_id}]   - Campioni test: {len(X_test)}")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "test_samples": len(X_test)
        }

        return loss, len(X_test), metrics

def main():
    global client_id, model, X_train, y_train, X_test, y_test, dataset_info

    if len(sys.argv) != 2:
        print("Uso: python client_sg.py <client_id>")
        print("Esempio: python client_sg.py 1")
        print("Client ID validi: 1-15 (corrispondenti ai file data1.csv - data15.csv)")
        sys.exit(1)

    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: Client ID non valido. {e}")
        sys.exit(1)

    print(f"AVVIO CLIENT SMARTGRID MALEVOLo {client_id}")
    print("=" * 50)

    try:
        X_train, y_train, X_test, y_test, scaler, dataset_info = load_client_smartgrid_data(client_id)
        print(f"[Client {client_id}] Creazione modello locale...")
        model = create_smartgrid_client_model(X_train.shape[1])
        print(f"[Client {client_id}] Modello creato con {X_train.shape[1]} feature di input")
        print(f"[Client {client_id}] === RIASSUNTO CLIENT ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            print(f"[Client {client_id}]   - {key}: {value}")
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()

        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridMaliciousClient()
        )

    except Exception as e:
        print(f"[Client {client_id}] Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()