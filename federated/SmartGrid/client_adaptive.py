#!/usr/bin/env python3
"""
Adaptive Client for SmartGrid Federated Learning

Client adattivo che implementa:
- Learning rate personalizzati basati sulla performance
- Reporting statistiche dettagliate per il server
- Supporto per weighted updates
- Monitoring locale avanzato
- Adattamento dinamico ai parametri del server

Author: francescaapellegrino
Date: 2025-08-17
"""

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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import time
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


def load_adaptive_client_data(client_id, n_components=20, validation_split=0.15, test_split=0.15):
    """
    Carica e pre-processa i dati SmartGrid per un client adattivo con monitoring avanzato.
    """
    print(f"=== CARICAMENTO DATI ADATTIVI CLIENT {client_id} ===")
    
    # Caricamento del file CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato per il client {client_id}")

    df = pd.read_csv(file_path)
    print(f"Dataset originale: {len(df)} campioni, {df.shape[1]-1} feature")
    
    # Separazione feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
    
    # Statistiche originali
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    print(f"Distribuzione originale:")
    print(f"  - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"  - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # STEP 1: Pulizia avanzata dei dati
    print(f"STEP 1: Pulizia avanzata dati...")
    initial_samples = len(X)
    
    # Gestione NaN
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"  🔧 Sostituiti {nan_count} valori NaN con mediana")
        X.fillna(X.median(), inplace=True)
    
    # Gestione valori infiniti
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        print(f"  🔧 Sostituiti {inf_count} valori infiniti")
        X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()], inplace=True)
    
    # Limitazione valori estremi
    max_val = np.finfo(np.float64).max / 100
    large_val_mask = np.abs(X) > max_val
    large_count = large_val_mask.sum().sum()
    if large_count > 0:
        print(f"  🔧 Limitati {large_count} valori estremi")
        X = X.clip(-max_val, max_val)
    
    # Controllo finale per NaN
    if np.any(np.isnan(X)):
        print(f"  🔧 Controllo finale: rimozione NaN residui")
        X = np.nan_to_num(X.values, nan=0.0, posinf=1e10, neginf=-1e10)
        X = pd.DataFrame(X, columns=df.columns[:-1])
    
    print(f"  ✅ Pulizia completata: {len(X)} campioni mantenuti")
    
    # STEP 2: Split train/validation/test
    print(f"STEP 2: Suddivisione dataset...")
    
    # Prima split: train+val vs test
    test_size = test_split
    val_size = validation_split / (1 - test_split)  # Aggiusta per riflettere la dimensione finale
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Seconda split: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    print(f"  📊 Suddivisione completata:")
    print(f"    - Training: {len(X_train)} campioni ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Validation: {len(X_val)} campioni ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Test: {len(X_test)} campioni ({len(X_test)/len(X)*100:.1f}%)")
    
    # STEP 3: Normalizzazione
    print(f"STEP 3: Normalizzazione feature...")
    scaler = StandardScaler()
    
    # Controllo finale per NaN prima della normalizzazione
    if np.any(np.isnan(X_train)):
        print(f"  🔧 Controllo pre-normalizzazione: rimozione NaN residui")
        X_train = np.nan_to_num(X_train.values, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val = np.nan_to_num(X_val.values, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test.values, nan=0.0, posinf=1e10, neginf=-1e10)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # STEP 4: PCA
    print(f"STEP 4: Riduzione dimensionalità (PCA)...")
    pca = PCA(n_components=n_components, random_state=42)
    
    # Controllo finale per NaN prima del PCA
    if np.any(np.isnan(X_train_scaled)):
        print(f"  🔧 Controllo pre-PCA: rimozione NaN residui")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  📊 PCA applicata: {X.shape[1]} -> {n_components} feature")
    print(f"  📊 Varianza spiegata: {variance_explained*100:.2f}%")
    
    # STEP 5: Statistiche finali per set
    train_attack_ratio = y_train.mean()
    val_attack_ratio = y_val.mean()
    test_attack_ratio = y_test.mean()
    
    print(f"STEP 5: Distribuzione classi finale:")
    print(f"  - Training: {train_attack_ratio*100:.2f}% attacchi")
    print(f"  - Validation: {val_attack_ratio*100:.2f}% attacchi")
    print(f"  - Test: {test_attack_ratio*100:.2f}% attacchi")
    
    # Informazioni dettagliate del dataset
    dataset_info = {
        'client_id': client_id,
        'original_samples': initial_samples,
        'final_samples': len(X),
        'original_features': X.shape[1],
        'final_features': n_components,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_attack_ratio': train_attack_ratio,
        'val_attack_ratio': val_attack_ratio,
        'test_attack_ratio': test_attack_ratio,
        'pca_variance_explained': variance_explained,
        'data_quality': {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'extreme_values_count': large_count
        }
    }
    
    print(f"✅ Dataset adattivo pronto per client {client_id}")
    print("=" * 60)
    
    return (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, 
            scaler, pca, dataset_info)


def create_adaptive_model(input_shape, initial_lr=0.001):
    """
    Crea modello adattivo per SmartGrid con configurazione ottimizzata.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,), name='dense_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.2, name='dropout_2'),
        tf.keras.layers.Dense(16, activation='relu', name='dense_3'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Optimizer adattivo
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


class SmartGridAdaptiveClient(fl.client.NumPyClient):
    """
    Client adattivo Flower per SmartGrid con learning rate personalizzati,
    monitoring avanzato e reporting dettagliato.
    """
    
    def __init__(self, client_id, n_components=20, verbose=True, save_logs=True):
        self.client_id = client_id
        self.n_components = n_components
        self.verbose = verbose
        self.save_logs = save_logs
        
        # Carica dati
        (self.X_train, self.y_train, self.X_val, self.y_val, 
         self.X_test, self.y_test, self.scaler, self.pca, 
         self.dataset_info) = load_adaptive_client_data(client_id, n_components)
        
        # Crea modello
        self.model = create_adaptive_model(self.X_train.shape[1])
        
        # Metriche di tracking
        self.training_history = []
        self.performance_history = []
        self.learning_rates = []
        self.round_count = 0
        self.adaptive_config = {
            'base_lr': 0.001,
            'lr_decay_factor': 0.95,
            'min_lr': 0.0001,
            'max_lr': 0.01,
            'performance_window': 5,
            'improvement_threshold': 0.001
        }
        
        # Inizializza logs
        if self.save_logs:
            self.log_dir = f"client_{client_id}_logs"
            os.makedirs(self.log_dir, exist_ok=True)
        
        if self.verbose:
            print(f"🤖 CLIENT ADATTIVO {client_id} INIZIALIZZATO")
            print(f"   • Training samples: {len(self.X_train)}")
            print(f"   • Validation samples: {len(self.X_val)}")
            print(f"   • Test samples: {len(self.X_test)}")
            print(f"   • Attack ratio: {self.dataset_info['train_attack_ratio']*100:.1f}%")
    
    def get_parameters(self, config):
        """Restituisce i parametri del modello."""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """
        Addestra il modello con learning rate adattivo e monitoring avanzato.
        """
        self.round_count += 1
        
        if self.verbose:
            print(f"\n🔄 [Client {self.client_id}] ROUND {self.round_count} - Training...")
        
        # Aggiorna parametri del modello
        self.model.set_weights(parameters)
        
        # Estrai configurazione dal server
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", 32))
        
        # Calcola learning rate adattivo
        adaptive_lr = self._calculate_adaptive_learning_rate(config)
        
        # Aggiorna learning rate del modello
        self.model.optimizer.learning_rate.assign(adaptive_lr)
        self.learning_rates.append(adaptive_lr)
        
        if self.verbose:
            print(f"   📊 Configurazione training:")
            print(f"      • Epochs: {epochs}")
            print(f"      • Batch size: {batch_size}")
            print(f"      • Learning rate: {adaptive_lr:.6f}")
        
        # Training con monitoring
        start_time = time.time()
        
        # Callback per monitoring
        monitor_callback = AdaptiveMonitoringCallback(
            client_id=self.client_id,
            verbose=self.verbose
        )
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,  # Gestito dal callback
            callbacks=[monitor_callback]
        )
        
        training_duration = time.time() - start_time
        
        # Calcola metriche dettagliate
        metrics = self._calculate_detailed_metrics(history, training_duration, adaptive_lr)
        
        # Aggiorna storia performance
        self._update_performance_history(metrics)
        
        # Salva logs se abilitato
        if self.save_logs:
            self._save_round_logs(metrics, history)
        
        if self.verbose:
            self._print_training_summary(metrics)
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        """
        Valuta il modello con metriche complete.
        """
        if self.verbose:
            print(f"📊 [Client {self.client_id}] Valutazione locale...")
        
        # Aggiorna parametri
        self.model.set_weights(parameters)
        
        # Valutazione su test set
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        # Metriche aggiuntive
        y_test_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, y_test_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        evaluation_metrics = {
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1),
            'test_auc': float(test_auc),
            'test_loss': float(test_loss),
            'test_samples': len(self.X_test),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'client_id': self.client_id,
            'current_round': self.round_count
        }
        
        if self.verbose:
            print(f"   📈 Risultati test:")
            print(f"      • Accuracy: {test_accuracy:.4f}")
            print(f"      • F1-Score: {test_f1:.4f}")
            print(f"      • AUC-ROC: {test_auc:.4f}")
            print(f"      • Loss: {test_loss:.4f}")
        
        return test_loss, len(self.X_test), evaluation_metrics
    
    def _calculate_adaptive_learning_rate(self, config):
        """
        Calcola learning rate adattivo basato sulla performance storica.
        """
        # Learning rate dal server (se fornito)
        server_lr = config.get("lr", self.adaptive_config['base_lr'])
        
        # Se è il primo round, usa il LR dal server
        if self.round_count <= 1 or len(self.performance_history) < 2:
            return float(server_lr)
        
        # Calcola trend performance
        recent_performances = self.performance_history[-self.adaptive_config['performance_window']:]
        if len(recent_performances) < 2:
            return float(server_lr)
        
        # Estrai accuracy di validazione
        val_accuracies = [p['val_accuracy'] for p in recent_performances]
        
        # Calcola improvement trend
        recent_improvement = val_accuracies[-1] - val_accuracies[0]
        
        # Adatta learning rate
        current_lr = self.learning_rates[-1] if self.learning_rates else server_lr
        
        if recent_improvement > self.adaptive_config['improvement_threshold']:
            # Performance migliorano -> mantieni o aumenta leggermente LR
            adaptive_lr = current_lr * 1.05
        elif recent_improvement < -self.adaptive_config['improvement_threshold']:
            # Performance peggiorano -> riduci LR
            adaptive_lr = current_lr * self.adaptive_config['lr_decay_factor']
        else:
            # Performance stabili -> riduci leggermente LR per convergenza fine
            adaptive_lr = current_lr * 0.98
        
        # Applica vincoli
        adaptive_lr = max(self.adaptive_config['min_lr'], 
                         min(self.adaptive_config['max_lr'], adaptive_lr))
        
        # Combina con LR del server (peso 30% server, 70% adattivo)
        final_lr = 0.3 * server_lr + 0.7 * adaptive_lr
        
        return float(final_lr)
    
    def _calculate_detailed_metrics(self, history, training_duration, learning_rate):
        """
        Calcola metriche dettagliate del training.
        """
        # Estrai ultime metriche di training
        train_loss = float(history.history['loss'][-1])
        train_accuracy = float(history.history['accuracy'][-1])
        train_precision = float(history.history['precision'][-1])
        train_recall = float(history.history['recall'][-1])
        
        # Metriche di validazione
        val_loss = float(history.history['val_loss'][-1])
        val_accuracy = float(history.history['val_accuracy'][-1])
        val_precision = float(history.history['val_precision'][-1])
        val_recall = float(history.history['val_recall'][-1])
        
        # Calcola F1 scores
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        # Overfitting indicator
        overfitting_score = train_accuracy - val_accuracy
        
        # Stability indicator (varianza nelle ultime epoch)
        if len(history.history['val_accuracy']) >= 3:
            recent_val_acc = history.history['val_accuracy'][-3:]
            stability_score = 1.0 / (1.0 + np.std(recent_val_acc))
        else:
            stability_score = 0.5
        
        return {
            # Training metrics
            'loss': train_loss,
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1,
            
            # Validation metrics
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_score': val_f1,
            
            # Dataset info
            'client_id': int(self.client_id),
            'train_samples': int(len(self.X_train)),
            'val_samples': int(len(self.X_val)),
            'train_attack_ratio': float(self.dataset_info['train_attack_ratio']),
            'val_attack_ratio': float(self.dataset_info['val_attack_ratio']),
            'pca_variance_explained': float(self.dataset_info['pca_variance_explained']),
            
            # Adaptive metrics
            'learning_rate': float(learning_rate),
            'training_duration': float(training_duration),
            'overfitting_score': float(overfitting_score),
            'stability_score': float(stability_score),
            'round_number': int(self.round_count),
            
            # Trend indicators
            'performance_trend': self._calculate_performance_trend(),
            'lr_adaptation_factor': float(learning_rate / self.adaptive_config['base_lr'])
        }
    
    def _calculate_performance_trend(self):
        """
        Calcola trend di performance recente.
        """
        if len(self.performance_history) < 2:
            return 0.0
        
        recent_performances = self.performance_history[-3:] if len(self.performance_history) >= 3 else self.performance_history
        val_accuracies = [p['val_accuracy'] for p in recent_performances]
        
        if len(val_accuracies) >= 2:
            return val_accuracies[-1] - val_accuracies[0]
        return 0.0
    
    def _update_performance_history(self, metrics):
        """
        Aggiorna la storia delle performance.
        """
        self.performance_history.append({
            'round': self.round_count,
            'val_accuracy': metrics['val_accuracy'],
            'val_f1_score': metrics['val_f1_score'],
            'overfitting_score': metrics['overfitting_score'],
            'learning_rate': metrics['learning_rate'],
            'timestamp': time.time()
        })
        
        # Mantieni solo la finestra di storia necessaria
        max_history = self.adaptive_config['performance_window'] * 2
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def _save_round_logs(self, metrics, history):
        """
        Salva logs dettagliati del round.
        """
        log_data = {
            'round': self.round_count,
            'metrics': metrics,
            'training_history': {k: [float(v) for v in values] for k, values in history.history.items()},
            'dataset_info': self.dataset_info,
            'adaptive_config': self.adaptive_config,
            'timestamp': time.time()
        }
        
        log_file = os.path.join(self.log_dir, f"round_{self.round_count:03d}.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
    
    def _print_training_summary(self, metrics):
        """
        Stampa riepilogo del training.
        """
        print(f"   📊 Risultati training:")
        print(f"      • Train Accuracy: {metrics['accuracy']:.4f}")
        print(f"      • Val Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"      • Val F1-Score: {metrics['val_f1_score']:.4f}")
        print(f"      • Overfitting: {metrics['overfitting_score']:.4f}")
        print(f"      • Stability: {metrics['stability_score']:.4f}")
        print(f"      • Duration: {metrics['training_duration']:.2f}s")
        
        # Trend indicator
        trend = metrics['performance_trend']
        trend_icon = "📈" if trend > 0.001 else "📉" if trend < -0.001 else "➡️"
        print(f"      • Trend: {trend_icon} {trend:+.4f}")
    
    def get_client_stats(self):
        """
        Restituisce statistiche complete del client per analisi.
        """
        return {
            'client_id': self.client_id,
            'dataset_info': self.dataset_info,
            'performance_history': self.performance_history,
            'learning_rates': self.learning_rates,
            'current_round': self.round_count,
            'adaptive_config': self.adaptive_config,
            'training_history_summary': {
                'total_rounds': self.round_count,
                'avg_val_accuracy': np.mean([p['val_accuracy'] for p in self.performance_history]) if self.performance_history else 0,
                'best_val_accuracy': max([p['val_accuracy'] for p in self.performance_history]) if self.performance_history else 0,
                'performance_variance': np.var([p['val_accuracy'] for p in self.performance_history]) if self.performance_history else 0
            }
        }


class AdaptiveMonitoringCallback(tf.keras.callbacks.Callback):
    """
    Callback per monitoring avanzato durante il training.
    """
    
    def __init__(self, client_id, verbose=True):
        super().__init__()
        self.client_id = client_id
        self.verbose = verbose
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        if self.verbose:
            print(f"      Epoch {epoch + 1}...", end=" ")
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose and logs:
            duration = time.time() - self.epoch_start_time
            print(f"Loss: {logs.get('loss', 0):.4f}, "
                  f"Acc: {logs.get('accuracy', 0):.4f}, "
                  f"Val_Acc: {logs.get('val_accuracy', 0):.4f} "
                  f"({duration:.1f}s)")


def start_adaptive_client(client_id, server_address="localhost:8080", n_components=20):
    """
    Avvia un client adattivo SmartGrid.
    """
    print(f"🚀 AVVIO CLIENT ADATTIVO SMARTGRID {client_id}")
    print("=" * 60)
    print(f"🌐 Server: {server_address}")
    print(f"🔧 Componenti PCA: {n_components}")
    print("=" * 60)
    
    try:
        # Crea client adattivo
        client = SmartGridAdaptiveClient(
            client_id=client_id,
            n_components=n_components,
            verbose=True,
            save_logs=True
        )
        
        print(f"✅ Client {client_id} creato con successo")
        print(f"🔗 Connessione al server {server_address}...")
        
        # Avvia client
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        print(f"🎉 Client {client_id} completato con successo!")
        
        # Stampa statistiche finali
        stats = client.get_client_stats()
        print(f"\n📊 STATISTICHE FINALI CLIENT {client_id}:")
        print(f"   • Round completati: {stats['current_round']}")
        print(f"   • Accuracy media: {stats['training_history_summary']['avg_val_accuracy']:.4f}")
        print(f"   • Miglior accuracy: {stats['training_history_summary']['best_val_accuracy']:.4f}")
        print(f"   • Varianza performance: {stats['training_history_summary']['performance_variance']:.6f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Client {client_id} interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore client {client_id}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Funzione principale per avviare un client adattivo.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartGrid Adaptive Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True, help="ID del client (1-13)")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Indirizzo del server")
    parser.add_argument("--components", type=int, default=20, help="Numero componenti PCA")
    
    args = parser.parse_args()
    
    print("🤖 SMARTGRID ADAPTIVE CLIENT")
    print("=" * 50)
    print(f"📅 Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Autore: francescaapellegrino")
    print(f"🎯 Client ID: {args.client_id}")
    print("=" * 50)
    
    # Validazione client ID
    if args.client_id < 1 or args.client_id > 13:
        print(f"❌ ERRORE: Client ID deve essere tra 1 e 13 (training clients)")
        sys.exit(1)
    
    # Avvia client
    start_adaptive_client(
        client_id=args.client_id,
        server_address=args.server,
        n_components=args.components
    )


if __name__ == "__main__":
    main()