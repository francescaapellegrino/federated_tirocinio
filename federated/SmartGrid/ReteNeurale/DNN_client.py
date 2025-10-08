"""
Client federato SmartGrid con Rete Neurale
Francesca Pellegrino
"""

import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from scipy import stats
from typing import Dict, Any, Tuple
import warnings
from federated.SmartGrid.RandomForestFederatoIncrementale.preprocessing import load_improved_client_data
from federated.SmartGrid.ReteNeurale.DNNmodel import create_improved_model, create_advanced_callbacks
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# CONFIGURAZIONE (in caso di errore modello)
class ClientConfig:
    RANDOM_SEED = 42

# CLIENT
class SmartGridClient(fl.client.NumPyClient):

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.config = ClientConfig()  # Usa la tua config esistente
        
        print(f" CLIENT {client_id}")
        
        # Carica dati con preprocessing migliorato
        from federated.SmartGrid.RandomForestFederatoIncrementale.preprocessing import load_improved_client_data
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.dataset_info = load_improved_client_data(client_id, self.config)
        
        # Calcola class weights per dataset sbilanciato
        self.class_weights = self._compute_class_weights()
        
        # Crea modello migliorato
        from federated.SmartGrid.ReteNeurale.DNNmodel import create_improved_model
        self.model = create_improved_model(self.X_train.shape[1], self.config)
        
        print(f"Client {client_id} migliorato inizializzato")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Train: {len(self.X_train)} samples")
        print(f"Attack ratio: {self.y_train.mean()*100:.1f}%")
    
    def _compute_class_weights(self):
        """Calcola pesi per bilanciare le classi"""
        if len(np.unique(self.y_train)) == 2:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            print(f"Class weights: Normal={class_weights[0]:.2f}, Attack={class_weights[1]:.2f}")
            return weight_dict
        else:
            return None
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        print(f"\n[IMPROVED Client {self.client_id}] Training...")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Callbacks avanzati
        callbacks = create_advanced_callbacks(self.config)

        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Training con class weights e callback
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=25,  # Aumentato per convergenza migliore
            batch_size=64,  # Batch size più grande
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Estrai metriche finali con conversione
        final_epoch = len(history.history['loss']) - 1
        
        # Conversione per tutte le metriche
        def safe_extract(metric_history, epoch_idx):
            """Estrae valore da history"""
            try:
                value = metric_history[epoch_idx]
                # Se è un array, prendi il primo elemento; altrimenti usa direttamente
                if hasattr(value, '__len__') and len(value) > 0:
                    return float(value[0])
                else:
                    return float(value)
            except (IndexError, TypeError, ValueError):
                return 0.0
        
        # Estrazione di tutte le metriche di training
        train_loss = safe_extract(history.history['loss'], final_epoch)
        train_acc = safe_extract(history.history['accuracy'], final_epoch)
        train_precision = safe_extract(history.history['precision'], final_epoch)
        train_recall = safe_extract(history.history['recall'], final_epoch)
        train_f1 = safe_extract(history.history['f1_score'], final_epoch)
        
        # Estrazione di tutte le metriche di validation
        val_loss = safe_extract(history.history['val_loss'], final_epoch)
        val_acc = safe_extract(history.history['val_accuracy'], final_epoch)
        val_precision = safe_extract(history.history['val_precision'], final_epoch)
        val_recall = safe_extract(history.history['val_recall'], final_epoch)
        val_f1 = safe_extract(history.history['val_f1_score'], final_epoch)
        
        # Calcolo AUC ROC, Specificity, Sensitivity sui dati di validation
        try:
            y_val_pred_prob = self.model.predict(self.X_val, verbose=0).flatten()
            y_val_pred_binary = (y_val_pred_prob > 0.5).astype(int)
            
            # AUC ROC
            from sklearn.metrics import roc_auc_score
            val_auc_roc = roc_auc_score(self.y_val, y_val_pred_prob)
            
            # Matrice di confusione per Specificity e Sensitivity
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_val, y_val_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                val_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:
                val_specificity = 0.0
                val_sensitivity = val_recall
                
        except Exception as e:
            print(f"Errore calcolo metriche aggiuntive: {e}")
            val_auc_roc = 0.5
            val_specificity = 0.0
            val_sensitivity = val_recall
        
        print(f"[IMPROVED Client {self.client_id}] Training completato:")
        print(f"Train - Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Val - AUC: {val_auc_roc:.4f}, Spec: {val_specificity:.4f}, Sens: {val_sensitivity:.4f}")
        
        # Metriche enhanced con valori garantiti scalari
        metrics = {
            # Training metrics
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            
            # Validation metrics (COMPLETE)
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1),
            'val_auc_roc': float(val_auc_roc),
            'val_specificity': float(val_specificity),
            'val_sensitivity': float(val_sensitivity),

            # Client info
            'client_id': int(self.client_id),
            'total_features': int(self.dataset_info['features']),
            'epochs_trained': final_epoch + 1,
            'architecture_type': 'improved_kaggle_style',
            'preprocessing': 'advanced_feature_engineering',
            'class_weights_used': self.class_weights is not None,
            
            # Target tracking
            'target_accuracy_90': float(val_acc >= 0.90),
            'target_precision_90': float(val_precision >= 0.90),
            'target_recall_90': float(val_recall >= 0.90),
            'target_f1_90': float(val_f1 >= 0.90),
            'all_targets_met': float(all([
                val_acc >= 0.90,
                val_precision >= 0.90,
                val_recall >= 0.90,
                val_f1 >= 0.90
            ]))
        }
        
        return self.model.get_weights(), len(self.X_train), metrics
    
    def evaluate(self, parameters, config):
        print(f"\n[Client {self.client_id}] Evaluation...")

        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Valutazione standard
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Estrazione sicura dei valori
        def safe_extract_result(results_list, index, default=0.0):
            """Estrae valore sicuro da results"""
            try:
                if len(results_list) > index:
                    value = results_list[index]
                    # Se è un array, prendi il primo elemento; altrimenti usa direttamente
                    if hasattr(value, '__len__') and len(value) > 0:
                        return float(value[0])
                    else:
                        return float(value)
                else:
                    return default
            except (IndexError, TypeError, ValueError):
                return default
        
        # Estrazione sicura delle metriche
        loss = safe_extract_result(results, 0, 1.0)
        accuracy = safe_extract_result(results, 1, 0.0)
        precision = safe_extract_result(results, 2, 0.0)
        recall = safe_extract_result(results, 3, 0.0)
        f1_score = safe_extract_result(results, 4, 0.0)
        auc_roc = safe_extract_result(results, 5, 0.5)
        auc_pr = safe_extract_result(results, 6, 0.5)
        
        # Resto del codice rimane identico...
        y_pred_prob = self.model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Matrice di confusione
        from sklearn.metrics import confusion_matrix, classification_report
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Metriche aggiuntive
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        print(f"[Client {self.client_id}] Results:")
        print(f"Accuracy: {accuracy:.4f} ({'OK' if accuracy >= 0.90 else 'NO'} target: >90%)")
        print(f"Precision: {precision:.4f} ({'OK' if precision >= 0.90 else 'NO'} target: >90%)")
        print(f"Recall: {recall:.4f} ({'OK' if recall >= 0.90 else 'NO'} target: >90%)")
        print(f"F1-Score: {f1_score:.4f} ({'OK' if f1_score >= 0.90 else 'NO'} target: >90%)")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Balanced Acc: {balanced_accuracy:.4f}")
        print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Verifica target raggiunti
        targets_met = {
            'accuracy_90': accuracy >= 0.90,
            'precision_90': precision >= 0.90,
            'recall_90': recall >= 0.90,
            'f1_90': f1_score >= 0.90
        }
        
        all_targets = all(targets_met.values())
        
        if all_targets:
            print(f"Target raggiunti!")
        else:
            missed = [k for k, v in targets_met.items() if not v]
            print(f"Target mancati: {missed}")

        metrics = {
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1_score),
            'test_auc_roc': float(auc_roc),
            'test_auc_pr': float(auc_pr),
            'test_specificity': float(specificity),
            'test_sensitivity': float(sensitivity),
            'test_balanced_accuracy': float(balanced_accuracy),
            
            # Confusion matrix
            'test_tn': int(tn),
            'test_fp': int(fp),
            'test_fn': int(fn),
            'test_tp': int(tp),
            
            # Target tracking
            'target_accuracy_90': float(targets_met['accuracy_90']),
            'target_precision_90': float(targets_met['precision_90']),
            'target_recall_90': float(targets_met['recall_90']),
            'target_f1_90': float(targets_met['f1_90']),
            'all_targets_met': float(all_targets),
            
            # Client info
            'client_id': int(self.client_id),
            'test_samples': int(len(self.X_test)),
            'model_type': 'improved_kaggle_style',
            'version': 'v2_improved'
        }
        
        return loss, len(self.X_test), metrics


# MAIN FUNCTION
def main():

    if len(sys.argv) != 2:
        print("Uso: python client.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)

    print(f"AVVIO CLIENT {client_id}")
    print("=" * 70)
    
    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=SmartGridClient(client_id)
        )
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()