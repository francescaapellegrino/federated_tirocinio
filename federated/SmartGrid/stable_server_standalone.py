import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

class ServerMetricsTracker:
    """
    Classe per tracciare le metriche del server in memoria e salvare JSON solo al round finale.
    Ottimizza le operazioni I/O per migliori prestazioni.
    """
    
    def __init__(self, total_rounds=10):
        self.total_rounds = total_rounds
        self.metrics_history = []
        self.start_time = time.time()
        self.round_times = {}
        
    def add_round_metrics(self, server_round, fit_results, global_metrics=None):
        """
        Aggiunge le metriche di un round alla cronologia in memoria.
        
        Args:
            server_round: Numero del round
            fit_results: Risultati dell'aggregazione dai client
            global_metrics: Metriche di valutazione globale (opzionali)
        """
        round_start_time = time.time()
        
        # Calcola metriche aggregate dai client
        total_samples = 0
        total_weighted_train_acc = 0
        total_weighted_val_acc = 0
        client_metrics_list = []
        
        for i, (client_proxy, fit_res) in enumerate(fit_results):
            client_samples = fit_res.num_examples
            client_metrics = fit_res.metrics
            total_samples += client_samples
            
            if 'train_accuracy' in client_metrics:
                total_weighted_train_acc += client_metrics['train_accuracy'] * client_samples
            if 'val_accuracy' in client_metrics:
                total_weighted_val_acc += client_metrics['val_accuracy'] * client_samples
                
            # Raccoglie metriche dettagliate del client
            client_data = {
                'client_id': i + 1,
                'samples': client_samples,
                **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in client_metrics.items()}
            }
            client_metrics_list.append(client_data)
        
        # Calcola metriche aggregate
        round_metrics = {
            'round': server_round,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            'total_clients': len(fit_results),
            'total_samples': total_samples,
            'avg_weighted_train_accuracy': total_weighted_train_acc / total_samples if total_samples > 0 else 0,
            'avg_weighted_val_accuracy': total_weighted_val_acc / total_samples if total_samples > 0 else 0,
            'train_val_gap': (total_weighted_train_acc - total_weighted_val_acc) / total_samples if total_samples > 0 else 0,
            'client_metrics': client_metrics_list
        }
        
        # Aggiunge metriche globali se disponibili
        if global_metrics:
            round_metrics['global_validation'] = global_metrics
            
        self.metrics_history.append(round_metrics)
        self.round_times[server_round] = time.time() - round_start_time
        
        print(f"[MetricsTracker] Round {server_round}/{self.total_rounds} metrics stored in memory")
        
    def should_save_convergence_report(self, server_round):
        """
        Determina se salvare un report di convergenza per questo round.
        """
        return server_round % 5 == 0  # Ogni 5 rounds
        
    def print_convergence_report(self, server_round):
        """
        Stampa un report di convergenza senza salvare file.
        """
        if not self.should_save_convergence_report(server_round):
            return
            
        print(f"\n=== REPORT DI CONVERGENZA - ROUND {server_round}/{self.total_rounds} ===")
        
        if len(self.metrics_history) >= 2:
            current = self.metrics_history[-1]
            previous = self.metrics_history[-2]
            
            acc_change = current['avg_weighted_train_accuracy'] - previous['avg_weighted_train_accuracy']
            val_acc_change = current['avg_weighted_val_accuracy'] - previous['avg_weighted_val_accuracy']
            
            print(f"Cambiamento Training Accuracy: {acc_change:+.4f}")
            print(f"Cambiamento Validation Accuracy: {val_acc_change:+.4f}")
            print(f"Gap Train-Validation corrente: {current['train_val_gap']:.4f}")
            
            if abs(acc_change) < 0.001 and abs(val_acc_change) < 0.001:
                print("🔄 Possibile convergenza rilevata (cambiamenti minimi)")
            elif acc_change > 0 and val_acc_change > 0:
                print("📈 Miglioramento su entrambi train e validation")
            elif acc_change > 0 and val_acc_change < 0:
                print("⚠️ Possibile overfitting (train migliora, validation peggiora)")
                
        elapsed = time.time() - self.start_time
        print(f"Tempo trascorso: {elapsed:.1f}s")
        print(f"Tempo medio per round: {elapsed/server_round:.1f}s")
        print("=" * 60)
        
    def save_final_json(self):
        """
        Salva il file JSON finale con tutte le metriche dei 10 rounds.
        """
        if len(self.metrics_history) == 0:
            print("[MetricsTracker] ATTENZIONE: Nessuna metrica da salvare")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"server_metrics_tracking_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Prepara i dati finali
        final_data = {
            'experiment_info': {
                'total_rounds': self.total_rounds,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - self.start_time,
                'avg_round_time': sum(self.round_times.values()) / len(self.round_times) if self.round_times else 0
            },
            'rounds_data': self.metrics_history,
            'performance_summary': self._generate_performance_summary()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ METRICHE FINALI SALVATE: {filename}")
            print(f"   Percorso completo: {filepath}")
            print(f"   Rounds salvati: {len(self.metrics_history)}")
            print(f"   Dimensione file: {os.path.getsize(filepath)} bytes")
        except Exception as e:
            print(f"❌ Errore nel salvataggio finale: {e}")
            
    def _generate_performance_summary(self):
        """
        Genera un riassunto delle performance dell'esperimento.
        """
        if not self.metrics_history:
            return {}
            
        first_round = self.metrics_history[0]
        last_round = self.metrics_history[-1]
        
        train_improvement = last_round['avg_weighted_train_accuracy'] - first_round['avg_weighted_train_accuracy']
        val_improvement = last_round['avg_weighted_val_accuracy'] - first_round['avg_weighted_val_accuracy']
        
        return {
            'initial_train_accuracy': first_round['avg_weighted_train_accuracy'],
            'final_train_accuracy': last_round['avg_weighted_train_accuracy'],
            'train_improvement': train_improvement,
            'initial_val_accuracy': first_round['avg_weighted_val_accuracy'],
            'final_val_accuracy': last_round['avg_weighted_val_accuracy'],
            'val_improvement': val_improvement,
            'final_train_val_gap': last_round['train_val_gap'],
            'convergence_achieved': abs(train_improvement) < 0.01 and abs(val_improvement) < 0.01
        }

def get_smartgrid_global_validation_fn(n_components=30):
    """
    Crea una funzione di valutazione globale per il server SmartGrid.
    Usa client 14-15 come validation set globale (mai usati per training).
    
    Args:
        n_components: Numero di componenti principali per PCA
    
    Returns:
        Funzione di valutazione globale
    """
    
    def load_global_validation_data():
        """
        Carica un dataset globale di validazione per il server.
        Applica lo stesso preprocessing dei client (PCA + normalizzazione).
        """
        print("=== CARICAMENTO DATASET GLOBALE DI VALIDAZIONE ===")
        
        # Directory in cui si trova questo script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Usa client 14-15 per validazione globale (mai usati nel training federato)
        validation_clients = [14, 15]
        df_list = []

        for client_id in validation_clients:
            file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
                print(f"  - Caricato data{client_id}.csv: {len(df)} campioni")
            except FileNotFoundError:
                print(f"  - File data{client_id}.csv non trovato, saltato")
                continue

        # Fallback se non trova i file di validazione
        if not df_list:
            print("  - ATTENZIONE: Nessun file di validazione globale trovato!")
            print("  - Usando il primo client disponibile come fallback...")
            fallback_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", "data1.csv")
            try:
                df_fallback = pd.read_csv(fallback_path)
                df_list = [df_fallback.sample(n=min(500, len(df_fallback)), random_state=42)]
            except FileNotFoundError:
                raise FileNotFoundError("Impossibile caricare dati per validazione globale")
        
        # Combina i dataframe
        df_global = pd.concat(df_list, ignore_index=True)
        
        # Prepara X e y
        X_global = df_global.drop(columns=["marker"])
        y_global = (df_global["marker"] != "Natural").astype(int)
        
        print(f"  - Dataset globale grezzo: {len(X_global)} campioni, {X_global.shape[1]} feature")
        
        # PREPROCESSING IDENTICO AI CLIENT
        
        # STEP 1: Pulizia dati
        X_global.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count = X_global.isnull().sum().sum()
        
        if nan_count > 0:
            print(f"  - Valori NaN trovati: {nan_count}, applicazione imputazione...")
            X_global.fillna(X_global.median(), inplace=True)
        
        # STEP 2: PCA (stesso numero di componenti dei client)
        pca_global = PCA(n_components=min(n_components, X_global.shape[1]))
        X_global_pca = pca_global.fit_transform(X_global)
        print(f"  - PCA applicata: da {X_global.shape[1]} a {X_global_pca.shape[1]} feature")
        print(f"  - Varianza spiegata: {pca_global.explained_variance_ratio_.sum()*100:.2f}%")
        
        # STEP 3: Normalizzazione
        scaler_global = StandardScaler()
        X_global_scaled = scaler_global.fit_transform(X_global_pca)
        
        # Statistiche dataset globale
        attack_count = y_global.sum()
        natural_count = (y_global == 0).sum()
        attack_ratio = y_global.mean()
        
        print(f"  - Dataset globale preparato: {len(X_global_scaled)} campioni")
        print(f"  - Attacchi: {attack_count} ({attack_ratio*100:.2f}%)")
        print(f"  - Naturali: {natural_count} ({(1-attack_ratio)*100:.2f}%)")
        print(f"  - NOTA: Questo è il set di validazione globale (mai usato per training)")
        print("=" * 60)
        
        return X_global_scaled, y_global
    
    # Carica i dati globali una sola volta
    try:
        X_global, y_global = load_global_validation_data()
        input_shape = X_global.shape[1]
    except Exception as e:
        print(f"Errore nel caricamento dati globali: {e}")
        # Fallback: crea dati fittizi per evitare crash
        X_global = np.random.random((100, 20))  # 20 componenti PCA
        y_global = np.random.randint(0, 2, 100)
        input_shape = 20
        print("Usando dati fittizi per validazione globale")
    
    def evaluate(server_round, parameters, config):
        """
        Funzione di valutazione globale chiamata ad ogni round.
        
        Args:
            server_round: Numero del round corrente
            parameters: Pesi del modello aggregato
            config: Configurazione
        
        Returns:
            Tuple con (loss, metriche)
        """
        print(f"\n=== VALUTAZIONE GLOBALE ROUND {server_round} ===")
        
        try:
            # Crea il modello per la valutazione (identico ai client)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            
            # Compila il modello
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.AUC(name="auc", curve='ROC')
                ]
            )
            
            # Imposta i pesi aggregati
            model.set_weights(parameters)
            
            # Valutazione sul dataset globale
            results = model.evaluate(X_global, y_global, verbose=0)
            loss, accuracy, precision, recall, auc = results
            
            # Calcola F1-score e AUC-ROC manualmente per maggiore precisione
            y_pred_prob = model.predict(X_global, verbose=0).flatten()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            try:
                auc_roc = roc_auc_score(y_global, y_pred_prob)
            except Exception:
                auc_roc = 0.0
            
            print(f"RISULTATI VALIDAZIONE GLOBALE:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  - F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"  - AUC-ROC: {auc_roc:.4f} ({auc_roc*100:.2f}%)")
            print(f"  - Campioni utilizzati: {len(X_global)}")
            
            # Analisi dettagliata predizioni
            predictions_prob = model.predict(X_global, verbose=0).flatten()
            predictions_binary = (predictions_prob > 0.5).astype(int)
            
            # Matrice di confusione
            tn = np.sum((y_global == 0) & (predictions_binary == 0))
            fp = np.sum((y_global == 0) & (predictions_binary == 1))
            fn = np.sum((y_global == 1) & (predictions_binary == 0))
            tp = np.sum((y_global == 1) & (predictions_binary == 1))
            
            print(f"  - Matrice confusione: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # Calcolo specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"  - Specificity (TNR): {specificity:.4f} ({specificity*100:.2f}%)")
            
            print("=" * 60)
            sys.stdout.flush()
            
            return float(loss), {
                "global_accuracy": float(accuracy),
                "global_precision": float(precision),
                "global_recall": float(recall),
                "global_f1_score": float(f1_score),
                "global_auc_roc": float(auc_roc),
                "global_specificity": float(specificity),
                "global_samples": int(len(X_global))
            }
            
        except Exception as e:
            print(f"Errore durante la valutazione globale: {e}")
            import traceback
            traceback.print_exc()
            return 1.0, {"global_accuracy": 0.0, "error": str(e), "global_samples": 0}
    
    return evaluate

def print_client_metrics_with_validation(fit_results):
    """
    Stampa le metriche dei client dopo ogni round, includendo train e validation.
    
    Args:
        fit_results: Risultati dell'addestramento dai client
    """
    if not fit_results:
        return
    
    print(f"\n=== METRICHE CLIENT ROUND (TRAIN/VALIDATION) ===")
    
    total_samples = 0
    total_weighted_train_acc = 0
    total_weighted_val_acc = 0
    overfitting_clients = 0
    
    for i, (client_proxy, fit_res) in enumerate(fit_results):
        client_samples = fit_res.num_examples
        client_metrics = fit_res.metrics
        
        total_samples += client_samples
        
        print(f"Client {i+1}:")
        print(f"  - Campioni training: {client_samples}")
        
        # Training metrics
        if 'train_accuracy' in client_metrics:
            train_acc = client_metrics['train_accuracy']
            total_weighted_train_acc += train_acc * client_samples
            print(f"  - Train Accuracy: {train_acc:.4f}")
        
        if 'train_loss' in client_metrics:
            print(f"  - Train Loss: {client_metrics['train_loss']:.4f}")
        
        if 'train_f1_score' in client_metrics:
            print(f"  - Train F1-Score: {client_metrics['train_f1_score']:.4f}")
            
        if 'train_auc' in client_metrics:
            print(f"  - Train AUC-ROC: {client_metrics['train_auc']:.4f}")
        
        # ⚠️ NUOVO: Validation metrics
        if 'val_accuracy' in client_metrics:
            val_acc = client_metrics['val_accuracy']
            total_weighted_val_acc += val_acc * client_samples
            print(f"  - Val Accuracy: {val_acc:.4f}")
        
        if 'val_loss' in client_metrics:
            print(f"  - Val Loss: {client_metrics['val_loss']:.4f}")
            
        if 'val_f1_score' in client_metrics:
            print(f"  - Val F1-Score: {client_metrics['val_f1_score']:.4f}")
            
        if 'val_auc' in client_metrics:
            print(f"  - Val AUC-ROC: {client_metrics['val_auc']:.4f}")
        
        # ⚠️ NUOVO: Overfitting check
        if 'overfitting_score' in client_metrics:
            overfitting_score = client_metrics['overfitting_score']
            if overfitting_score > 0.1:  # Gap > 10%
                overfitting_clients += 1
                print(f"  - Overfitting Score: {overfitting_score:.4f} ⚠️ ALTO")
            else:
                print(f"  - Overfitting Score: {overfitting_score:.4f} ✅ OK")
        
        # Dataset info
        if 'val_samples' in client_metrics:
            print(f"  - Validation samples: {client_metrics['val_samples']}")
        
        if 'pca_variance_explained' in client_metrics:
            print(f"  - PCA varianza spiegata: {client_metrics['pca_variance_explained']*100:.1f}%")
    
    # ⚠️ NUOVO: Statistiche aggregate
    if total_samples > 0:
        avg_weighted_train_acc = total_weighted_train_acc / total_samples
        avg_weighted_val_acc = total_weighted_val_acc / total_samples
        
        print(f"\n=== STATISTICHE AGGREGATE ===")
        print(f"Media pesata Train Accuracy: {avg_weighted_train_acc:.4f}")
        print(f"Media pesata Val Accuracy: {avg_weighted_val_acc:.4f}")
        print(f"Gap Train-Val globale: {avg_weighted_train_acc - avg_weighted_val_acc:.4f}")
        print(f"Client con possibile overfitting: {overfitting_clients}/{len(fit_results)}")
        
        if overfitting_clients > len(fit_results) // 2:
            print("⚠️ ATTENZIONE: Oltre la metà dei client mostra segni di overfitting!")
        elif overfitting_clients == 0:
            print("✅ Nessun client mostra segni significativi di overfitting")
    
    print("=" * 50)

class SmartGridFedAvgWithValidationAndTracking(FedAvg):
    """
    Strategia FedAvg personalizzata per SmartGrid con supporto train/validation monitoring e metrics tracking.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_tracker = ServerMetricsTracker(total_rounds=10)
        
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati dell'addestramento e stampa metriche dettagliate incluso validation.
        """
        print(f"\n=== AGGREGAZIONE ROUND {server_round} ===")
        print(f"Client partecipanti: {len(results)}")
        print(f"Client falliti: {len(failures)}")
        
        if failures:
            print("Fallimenti:")
            for failure in failures:
                print(f"  - {failure}")
        
        # Stampa metriche dei client con informazioni train/validation
        print_client_metrics_with_validation(results)
        
        # Chiama l'aggregazione standard
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is not None:
            print(f"✅ Aggregazione completata per round {server_round}")
            
            # Traccia le metriche in memoria
            self.metrics_tracker.add_round_metrics(server_round, results)
            
            # Stampa report di convergenza ogni 5 rounds
            self.metrics_tracker.print_convergence_report(server_round)
            
            # Salva JSON solo al round finale
            if server_round == self.metrics_tracker.total_rounds:
                self.metrics_tracker.save_final_json()
                print(f"\n🎉 TRAINING FEDERATO COMPLETATO DOPO {server_round} ROUNDS!")
                
        else:
            print(f"❌ ATTENZIONE: Aggregazione fallita per round {server_round}")
        
        return aggregated_result
        
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggrega i risultati della valutazione e li include nel tracking.
        """
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            # Aggiunge le metriche globali al tracker se disponibili
            if hasattr(self.metrics_tracker, 'metrics_history') and self.metrics_tracker.metrics_history:
                last_round_idx = len(self.metrics_tracker.metrics_history) - 1
                if last_round_idx >= 0:
                    self.metrics_tracker.metrics_history[last_round_idx]['global_evaluation'] = {
                        'loss': float(loss),
                        **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in metrics.items()}
                    }
        
        return aggregated_result

def main():
    """
    Funzione principale per avviare il server SmartGrid federato con 10 rounds e tracking ottimizzato.
    """
    print("=== AVVIO SERVER FEDERATO SMARTGRID STANDALONE (10 ROUNDS) ===")
    print("Configurazione:")
    print("  - Numero di round: 10")  # ⚠️ CAMBIATO: da 100 a 10
    print("  - Epoche per round: 5")
    print("  - Suddivisione client: 75% train, 10% validation, 15% test")
    print("  - Client minimi per training: 2")
    print("  - Client minimi per valutazione: 2")
    print("  - Client minimi disponibili: 2")
    print("  - Strategia: FedAvg con validation monitoring e metrics tracking")
    print("  - Preprocessing: PCA + StandardScaler")
    print("  - Validazione globale: Client 14-15 (mai usati per training)")
    print("  - Metriche: accuracy, loss, precision, recall, f1-score, AUC-ROC")
    print("  - Overfitting detection: Train vs Validation gap monitoring")
    print("  - ⚠️ OTTIMIZZAZIONE: JSON salvato solo al round finale per ridurre I/O")
    print("  - ⚠️ CONVERGENZA: Report ogni 5 rounds (round 5 e 10)")
    print("=" * 70)
    
    # ⚠️ CONFIGURAZIONE PRINCIPALE: 10 rounds
    config = fl.server.ServerConfig(num_rounds=10)
    
    # Strategia Federated Averaging personalizzata con validation e tracking
    strategy = SmartGridFedAvgWithValidationAndTracking(
        fraction_fit=1.0,                    # Usa tutti i client disponibili per training
        fraction_evaluate=1.0,               # Usa tutti i client disponibili per valutazione
        min_fit_clients=2,                   # Numero minimo di client per iniziare training
        min_evaluate_clients=2,              # Numero minimo di client per valutazione
        min_available_clients=2,             # Numero minimo di client connessi
        evaluate_fn=get_smartgrid_global_validation_fn(n_components=30)  # Validazione globale
    )
    
    print("Server in attesa di client...")
    print("Per connettere i client, esegui in terminali separati:")
    print("  python stable_client_standalone.py 1")
    print("  python stable_client_standalone.py 2")
    print("  python stable_client_standalone.py 3")
    print("  ...")
    print("  python stable_client_standalone.py 13")
    print("\n📋 NOTA IMPORTANTE:")
    print("  - Usa client ID 1-13 per training federato")
    print("  - Client 14-15 sono riservati per validazione globale del server")
    print("  - Ogni client esegue 5 epoche per round")
    print("  - Il server monitora overfitting e performance validation")
    print("  - Metriche tracciate in memoria, JSON salvato solo al round 10")
    print("\nIl training inizierà quando almeno 2 client saranno connessi.")
    print("=" * 70)
    sys.stdout.flush()
    
    try:
        # Avvia il server
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"Errore durante l'avvio del server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()