import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

def load_client_smartgrid_data_with_validation(client_id, n_components=30):
    """
    Carica e pre-processa i dati SmartGrid per un client con suddivisione train/validation/test.
    
    Args:
        client_id: ID del client (1-15)
        n_components: Numero di componenti principali per PCA
    
    Returns:
        Tuple con (X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, dataset_info)
    """
    print(f"=== CARICAMENTO DATI CLIENT {client_id} SMARTGRID CON TRAIN/VAL/TEST ===")
    
    # Caricamento del file CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    df = pd.read_csv(file_path)
    print(f"Dataset del client {client_id}: {len(df)} campioni, {df.shape[1]-1} feature")
    
    # Separazione feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
    
    # Stampa distribuzione originale delle classi
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"Distribuzione originale:")
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # STEP 1: Pulizia dei dati
    print(f"STEP 1: Pulizia dati...")
    initial_samples = len(X)
    
    # Gestisce valori infiniti e NaN con imputazione
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isnull().sum().sum()
    
    if nan_count > 0:
        print(f"  - Valori NaN trovati: {nan_count}, applicazione imputazione con mediana...")
        X.fillna(X.median(), inplace=True)
    else:
        print(f"  - Nessun valore NaN trovato")
    
    print(f"  - Campioni dopo pulizia: {len(X)}")
    
    # STEP 2: PCA (prima della normalizzazione)
    print(f"STEP 2: Applicazione PCA...")
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X)
    print(f"  - Riduzione da {X.shape[1]} a {X_pca.shape[1]} feature")
    print(f"  - Varianza spiegata: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # STEP 3: Prima split - separa test set (15%)
    print(f"STEP 3: Prima split (train+val / test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_pca, y,
        test_size=0.15,  # 15% per test finale
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"  - Train+Validation: {len(X_temp)} campioni ({len(X_temp)/len(X_pca)*100:.1f}%)")
    print(f"  - Test: {len(X_test)} campioni ({len(X_test)/len(X_pca)*100:.1f}%)")
    
    # STEP 4: Seconda split - separa train e validation
    print(f"STEP 4: Seconda split (train / validation)...")
    # 10% del totale = 10/85 ≈ 0.118 del temp set
    validation_size = 0.118  # Questo ci darà circa 10% del dataset totale
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_size,
        random_state=42,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    
    # Verifica percentuali finali
    total_samples = len(X_pca)
    train_pct = len(X_train) / total_samples * 100
    val_pct = len(X_val) / total_samples * 100
    test_pct = len(X_test) / total_samples * 100
    
    print(f"SUDDIVISIONE FINALE:")
    print(f"  - Training: {len(X_train)} campioni ({train_pct:.1f}%)")
    print(f"  - Validation: {len(X_val)} campioni ({val_pct:.1f}%)")
    print(f"  - Test: {len(X_test)} campioni ({test_pct:.1f}%)")
    
    # STEP 5: Normalizzazione (IMPORTANTE: fit solo su training set)
    print(f"STEP 5: Normalizzazione...")
    scaler = StandardScaler()
    
    # Fit dello scaler SOLO sui dati di training
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform (non fit_transform!) su validation e test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  - Training set normalizzato: {X_train_scaled.shape}")
    print(f"  - Validation set normalizzato: {X_val_scaled.shape}")
    print(f"  - Test set normalizzato: {X_test_scaled.shape}")
    
    # STEP 6: Statistiche per set
    print(f"STEP 6: Statistiche distribuzione classi per set...")
    
    train_attack_ratio = y_train.mean() if len(y_train) > 0 else 0
    val_attack_ratio = y_val.mean() if len(y_val) > 0 else 0
    test_attack_ratio = y_test.mean() if len(y_test) > 0 else 0
    
    print(f"  - Training set - Attacchi: {train_attack_ratio*100:.2f}%")
    print(f"  - Validation set - Attacchi: {val_attack_ratio*100:.2f}%")
    print(f"  - Test set - Attacchi: {test_attack_ratio*100:.2f}%")
    
    # Informazioni finali del dataset
    dataset_info = {
        'client_id': client_id,
        'total_samples': initial_samples,
        'train_samples': len(X_train_scaled),
        'val_samples': len(X_val_scaled),
        'test_samples': len(X_test_scaled),
        'attack_samples_original': attack_samples,
        'natural_samples_original': natural_samples,
        'attack_ratio_original': attack_ratio,
        'train_attack_ratio': train_attack_ratio,
        'val_attack_ratio': val_attack_ratio,
        'test_attack_ratio': test_attack_ratio,
        'features': X_train_scaled.shape[1],
        'pca_variance_explained': pca.explained_variance_ratio_.sum(),
        'train_percentage': train_pct,
        'val_percentage': val_pct,
        'test_percentage': test_pct
    }
    
    print(f"=== RIEPILOGO CLIENT {client_id} ===")
    print(f"  - Feature finali: {dataset_info['features']}")
    print(f"  - Varianza PCA spiegata: {dataset_info['pca_variance_explained']*100:.2f}%")
    print(f"  - Suddivisione: {train_pct:.1f}% train, {val_pct:.1f}% val, {test_pct:.1f}% test")
    print("=" * 60)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, pca, dataset_info

def create_smartgrid_client_model_with_validation(input_shape):
    """
    Crea il modello SmartGrid per il client ottimizzato per train/validation/test.
    
    Args:
        input_shape: Numero di feature in input
    
    Returns:
        Modello Keras compilato
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve='ROC')
        ]
    )
    
    return model

# Variabili globali per il client
client_id = None
model = None
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None
scaler = None
pca = None
dataset_info = None

class SmartGridClientWithValidation(fl.client.NumPyClient):
    """
    Client Flower per SmartGrid con supporto train/validation/test split.
    """
    
    def get_parameters(self, config):
        """
        Restituisce i pesi attuali del modello locale.
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Addestra il modello sui dati di training con monitoraggio su validation set.
        """
        global model, X_train, y_train, X_val, y_val, dataset_info
        
        print(f"[Client {client_id}] === ROUND DI ADDESTRAMENTO CON VALIDATION ===")
        print(f"[Client {client_id}] Ricevuti parametri dal server, avvio addestramento locale...")
        sys.stdout.flush()
        
        # Imposta i pesi ricevuti dal server
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di training
        if len(X_train) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di training disponibile!")
            return model.get_weights(), 0, {}
        
        # Addestra il modello con validation monitoring
        print(f"[Client {client_id}] Addestramento su {len(X_train)} campioni, validation su {len(X_val)} campioni")
        print(f"[Client {client_id}] Epoche per round: 5")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),  # ⚠️ NUOVO: monitoring su validation set
            epochs=5,
            batch_size=32,
            verbose=0  # Usa 1 se vuoi vedere il progresso dettagliato
        )
        
        # Estrai le metriche dall'addestramento (ultima epoca)
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        train_precision = history.history['precision'][-1]
        train_recall = history.history['recall'][-1]
        train_auc = history.history['auc'][-1]
        
        # ⚠️ NUOVO: Estrai metriche di validation
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        val_precision = history.history['val_precision'][-1]
        val_recall = history.history['val_recall'][-1]
        val_auc = history.history['val_auc'][-1]
        
        # Calcola F1-score per training e validation
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        print(f"[Client {client_id}] === RISULTATI ADDESTRAMENTO ===")
        print(f"[Client {client_id}] TRAINING:")
        print(f"[Client {client_id}]   - Loss: {train_loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {train_accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {train_precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {train_recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {train_f1:.4f}")
        print(f"[Client {client_id}]   - AUC-ROC: {train_auc:.4f}")
        
        print(f"[Client {client_id}] VALIDATION:")
        print(f"[Client {client_id}]   - Loss: {val_loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {val_accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {val_precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {val_recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {val_f1:.4f}")
        print(f"[Client {client_id}]   - AUC-ROC: {val_auc:.4f}")
        
        # ⚠️ NUOVO: Controlla overfitting
        overfitting_warning = ""
        if train_accuracy - val_accuracy > 0.1:  # Differenza > 10%
            overfitting_warning = " ⚠️ POSSIBILE OVERFITTING"
        elif val_accuracy > train_accuracy:
            overfitting_warning = " ✅ BUONA GENERALIZZAZIONE"
        
        print(f"[Client {client_id}] Gap Train-Val Accuracy: {train_accuracy - val_accuracy:.4f}{overfitting_warning}")
        
        # Metriche da inviare al server (include sia training che validation)
        metrics = {
            # Training metrics
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            'train_auc': float(train_auc),
            
            # ⚠️ NUOVO: Validation metrics
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1),
            'val_auc': float(val_auc),
            
            # Dataset info
            'client_id': int(dataset_info['client_id']),
            'train_samples': int(dataset_info['train_samples']),
            'val_samples': int(dataset_info['val_samples']),
            'train_attack_ratio': float(dataset_info['train_attack_ratio']),
            'val_attack_ratio': float(dataset_info['val_attack_ratio']),
            'pca_variance_explained': float(dataset_info['pca_variance_explained']),
            
            # ⚠️ NUOVO: Overfitting indicator
            'overfitting_score': float(train_accuracy - val_accuracy)
        }
        
        print(f"[Client {client_id}] Invio pesi aggiornati al server...")
        sys.stdout.flush()
        
        return model.get_weights(), len(X_train), metrics

    def evaluate(self, parameters, config):
        """
        Valuta il modello sui dati di TEST locali del client (non validation, non training).
        """
        global model, X_test, y_test
        
        print(f"[Client {client_id}] === VALUTAZIONE FINALE SU TEST SET ===")
        
        # Imposta i pesi da valutare
        model.set_weights(parameters)
        
        # Verifica che ci siano dati di test
        if len(X_test) == 0:
            print(f"[Client {client_id}] ATTENZIONE: Nessun dato di test disponibile!")
            return 0.0, 0, {"accuracy": 0.0}
        
        print(f"[Client {client_id}] Valutazione su {len(X_test)} campioni di test...")
        
        # Valuta sui dati di TEST (mai visti durante training)
        results = model.evaluate(X_test, y_test, verbose=0)
        loss, accuracy, precision, recall, auc = results
        
        # Calcola F1-score e AUC-ROC manualmente per maggiore precisione
        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred_bin = (y_pred_prob > 0.5).astype(int)
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc_roc = roc_auc_score(y_test, y_pred_prob)
        except Exception:
            auc_roc = 0.0
        
        print(f"[Client {client_id}] === RISULTATI TEST FINALE ===")
        print(f"[Client {client_id}]   - Loss: {loss:.4f}")
        print(f"[Client {client_id}]   - Accuracy: {accuracy:.4f}")
        print(f"[Client {client_id}]   - Precision: {precision:.4f}")
        print(f"[Client {client_id}]   - Recall: {recall:.4f}")
        print(f"[Client {client_id}]   - F1-Score: {f1_score:.4f}")
        print(f"[Client {client_id}]   - AUC-ROC: {auc_roc:.4f}")
        print(f"[Client {client_id}]   - Campioni test: {len(X_test)}")
        
        # Metriche da restituire al server
        metrics = {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1_score": float(f1_score),
            "test_auc_roc": float(auc_roc),
            "test_samples": int(len(X_test)),
            "test_attack_ratio": float(dataset_info['test_attack_ratio'])
        }
        
        return loss, len(X_test), metrics

def main():
    """
    Funzione principale per avviare il client SmartGrid con train/validation/test split.
    """
    global client_id, model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, dataset_info
    
    # Verifica argomenti della riga di comando
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
    
    print(f"AVVIO CLIENT SMARTGRID {client_id} CON TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 50)
    
    try:
        # 1. Carica i dati locali del client con suddivisione train/val/test
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, dataset_info = load_client_smartgrid_data_with_validation(client_id, n_components=30)
        
        # 2. Crea il modello locale
        print(f"[Client {client_id}] Creazione modello locale...")
        model = create_smartgrid_client_model_with_validation(X_train.shape[1])
        print(f"[Client {client_id}] Modello creato con {X_train.shape[1]} feature di input")
        
        # 3. Stampa riassunto del client
        print(f"[Client {client_id}] === RIASSUNTO CLIENT ===")
        print(f"[Client {client_id}] Dataset info:")
        for key, value in dataset_info.items():
            if isinstance(value, float):
                print(f"[Client {client_id}]   - {key}: {value:.4f}")
            else:
                print(f"[Client {client_id}]   - {key}: {value}")
        
        print(f"[Client {client_id}] Tentativo di connessione al server su localhost:8080...")
        sys.stdout.flush()
        
        # 4. Avvia il client Flower con la nuova classe
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridClientWithValidation()  # ⚠️ NUOVO: classe aggiornata
        )
        
    except Exception as e:
        print(f"[Client {client_id}] Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()