#!/usr/bin/env python3
"""
Optimized Server for SmartGrid Federated Learning

Server ottimizzato che integra:
- Strategie di aggregazione bilanciate
- Monitoring distribuzione real-time
- Adaptive client selection
- Metriche avanzate per ottimizzazioni
- Logging completo per analisi

Author: francescaapellegrino
Date: 2025-08-17
"""

import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

# Import strategia bilanciata
from strategies import create_smartgrid_optimized_strategy, BalancedFedAvg


def load_smartgrid_global_validation_data(validation_clients=[14, 15], n_components=20):
    """
    Carica dataset di validazione globale per il server usando client 14-15.
    Applica lo stesso preprocessing dei client (PCA + normalizzazione).
    """
    print("=== CARICAMENTO DATASET GLOBALE DI VALIDAZIONE ===")
    
    # Directory dei dati
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
    
    df_list = []
    for client_id in validation_clients:
        file_path = os.path.join(data_dir, f"data{client_id}.csv")
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"  ✅ Caricato data{client_id}.csv: {len(df)} campioni")
        except FileNotFoundError:
            print(f"  ❌ File data{client_id}.csv non trovato")
            continue
    
    if not df_list:
        print("  ⚠️  Fallback: uso data1.csv per validazione")
        fallback_path = os.path.join(data_dir, "data1.csv")
        df_combined = pd.read_csv(fallback_path)
    else:
        df_combined = pd.concat(df_list, ignore_index=True)
    
    print(f"📊 Dataset validazione globale: {len(df_combined)} campioni, {df_combined.shape[1]-1} feature")
    
    # Preprocessing identico ai client
    X = df_combined.drop(columns=["marker"])
    y = (df_combined["marker"] != "Natural").astype(int)
    
    # Pulizia dati
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"🔧 Sostituiti {nan_count} valori NaN con mediana")
        X.fillna(X.median(), inplace=True)
    
    # Gestione valori infiniti
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        print(f"🔧 Sostituiti {inf_count} valori infiniti")
        X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()], inplace=True)
    
    # Limitazione valori estremi
    max_val = np.finfo(np.float64).max / 100
    large_val_mask = np.abs(X) > max_val
    large_count = large_val_mask.sum().sum()
    if large_count > 0:
        print(f"🔧 Limitati {large_count} valori estremi")
        X = X.clip(-max_val, max_val)
    
    # Normalizzazione
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Controllo finale per NaN dopo tutti i preprocessing
    if np.any(np.isnan(X)):
        print(f"🔧 Controllo finale: rimozione NaN residui")
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    
    # Controllo finale per NaN prima del PCA
    if np.any(np.isnan(X_scaled)):
        print(f"🔧 Controllo pre-PCA: rimozione NaN residui")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
    
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"📊 Preprocessing completato:")
    print(f"  - Feature ridotte da {X.shape[1]} a {X_pca.shape[1]}")
    print(f"  - Varianza spiegata: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  - Distribuzione classi: {y.mean()*100:.1f}% attacchi")
    
    return X_pca, y


def create_smartgrid_model(input_shape):
    """
    Crea modello ottimizzato per SmartGrid (identico a quello dei client).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def get_smartgrid_optimized_evaluation_fn(n_components=20, verbose=True):
    """
    Crea funzione di valutazione globale ottimizzata per SmartGrid.
    """
    
    # Carica dati di validazione una sola volta
    X_val, y_val = load_smartgrid_global_validation_data(
        validation_clients=[14, 15], 
        n_components=n_components
    )
    
    # Crea modello di valutazione
    model = create_smartgrid_model(X_val.shape[1])
    
    def evaluate(server_round: int, parameters, config: Dict[str, fl.common.Scalar]):
        """
        Funzione di valutazione globale per ogni round.
        """
        # Aggiorna pesi del modello
        model.set_weights(parameters)
        
        # Valutazione
        loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
        
        # Calcola F1-score e AUC
        from sklearn.metrics import f1_score, roc_auc_score
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        # Metriche dettagliate
        metrics = {
            "global_accuracy": accuracy,
            "global_precision": precision,
            "global_recall": recall,
            "global_f1_score": f1,
            "global_auc": auc,
            "global_loss": loss,
            "validation_samples": len(y_val),
            "attack_ratio": float(y_val.mean())
        }
        
        if verbose:
            print(f"📊 [Round {server_round}] Valutazione Globale:")
            print(f"   🎯 Accuracy: {accuracy:.4f}")
            print(f"   🎯 F1-Score: {f1:.4f}")
            print(f"   🎯 AUC-ROC: {auc:.4f}")
            print(f"   🎯 Loss: {loss:.4f}")
        
        return loss, metrics
    
    return evaluate


def create_optimized_metrics_aggregation_fn():
    """
    Crea funzione di aggregazione metriche ottimizzata.
    """
    def aggregate_metrics(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
        """
        Aggrega metriche di valutazione con statistiche avanzate.
        """
        if not metrics:
            return {}
        
        # Raccoglie tutte le metriche
        all_metrics = {}
        total_samples = 0
        
        for num_examples, client_metrics in metrics:
            total_samples += num_examples
            for key, value in client_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append((float(value), num_examples))
        
        # Calcola aggregate
        aggregated = {}
        
        for metric_name, values_and_samples in all_metrics.items():
            values = [v for v, s in values_and_samples]
            weights = [s for v, s in values_and_samples]
            
            # Media pesata
            weighted_avg = np.average(values, weights=weights)
            aggregated[f"avg_{metric_name}"] = weighted_avg
            
            # Statistiche addizionali
            aggregated[f"std_{metric_name}"] = np.std(values)
            aggregated[f"min_{metric_name}"] = np.min(values)
            aggregated[f"max_{metric_name}"] = np.max(values)
            
            # Coefficiente di variazione (importante per fairness)
            if weighted_avg > 0:
                aggregated[f"cv_{metric_name}"] = np.std(values) / weighted_avg
        
        # Metriche di fairness
        if "accuracy" in [k.replace("avg_", "") for k in aggregated.keys()]:
            accuracy_values = [v for v, s in all_metrics.get("accuracy", [])]
            aggregated["fairness_accuracy_variance"] = np.var(accuracy_values)
            aggregated["fairness_accuracy_range"] = np.max(accuracy_values) - np.min(accuracy_values)
        
        # Informazioni generali
        aggregated["total_clients"] = len(metrics)
        aggregated["total_samples"] = total_samples
        
        return aggregated
    
    return aggregate_metrics


class SmartGridOptimizedServer:
    """
    Server ottimizzato per SmartGrid con monitoring avanzato e strategie bilanciate.
    """
    
    def __init__(
        self,
        strategy_type: str = "smartgrid_optimized",
        n_components: int = 20,
        num_rounds: int = 50,
        verbose: bool = True,
        save_logs: bool = True,
        log_dir: str = "server_logs"
    ):
        self.strategy_type = strategy_type
        self.n_components = n_components
        self.num_rounds = num_rounds
        self.verbose = verbose
        self.save_logs = save_logs
        self.log_dir = log_dir
        
        # Crea directory logs
        if self.save_logs:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Inizializza metriche di monitoring
        self.server_metrics = {
            'start_time': time.time(),
            'rounds_completed': 0,
            'total_clients_seen': set(),
            'round_history': [],
            'strategy_metrics': {},
            'convergence_metrics': []
        }
        
        if self.verbose:
            print(f"🚀 SMARTGRID OPTIMIZED SERVER INIZIALIZZATO")
            print(f"   • Strategia: {self.strategy_type}")
            print(f"   • Componenti PCA: {self.n_components}")
            print(f"   • Round totali: {self.num_rounds}")
            print(f"   • Logging: {'Abilitato' if self.save_logs else 'Disabilitato'}")
    
    def create_strategy(self) -> BalancedFedAvg:
        """
        Crea la strategia di aggregazione ottimizzata.
        """
        # Funzione di valutazione globale
        evaluation_fn = get_smartgrid_optimized_evaluation_fn(
            n_components=self.n_components,
            verbose=self.verbose
        )
        
        # Funzione aggregazione metriche
        metrics_aggregation_fn = create_optimized_metrics_aggregation_fn()
        
        # Strategia bilanciata
        if self.strategy_type == "smartgrid_optimized":
            strategy = create_smartgrid_optimized_strategy(
                evaluate_fn=evaluation_fn,
                evaluate_metrics_aggregation_fn=metrics_aggregation_fn
            )
        else:
            # Fallback a strategia standard
            strategy = FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                evaluate_fn=evaluation_fn,
                evaluate_metrics_aggregation_fn=metrics_aggregation_fn
            )
        
        # Hook per monitoring
        self._add_monitoring_hooks(strategy)
        
        return strategy
    
    def _add_monitoring_hooks(self, strategy):
        """
        Aggiunge hook di monitoring alla strategia.
        """
        # Salva riferimento ai metodi originali
        original_aggregate_fit = strategy.aggregate_fit
        original_aggregate_evaluate = strategy.aggregate_evaluate
        
        def monitored_aggregate_fit(server_round, results, failures):
            """Wrapper per aggregate_fit con monitoring."""
            start_time = time.time()
            
            # Chiama aggregazione originale
            aggregated_result = original_aggregate_fit(server_round, results, failures)
            
            # Raccoglie metriche di monitoring
            round_duration = time.time() - start_time
            client_ids = set()
            total_samples = 0
            
            for client, fit_res in results:
                if hasattr(client, 'cid'):
                    client_ids.add(str(client.cid))
                total_samples += fit_res.num_examples
            
            # Aggiorna metriche server
            self.server_metrics['rounds_completed'] = server_round
            self.server_metrics['total_clients_seen'].update(client_ids)
            
            round_info = {
                'round': server_round,
                'duration': round_duration,
                'num_clients': len(results),
                'num_failures': len(failures),
                'total_samples': total_samples,
                'client_ids': list(client_ids),
                'timestamp': time.time()
            }
            
            # Aggiungi metriche di strategia se disponibili
            if hasattr(strategy, 'get_strategy_metrics'):
                try:
                    round_info['strategy_metrics'] = strategy.get_strategy_metrics()
                except:
                    pass
            
            self.server_metrics['round_history'].append(round_info)
            
            # Salva logs se abilitato
            if self.save_logs:
                self._save_round_logs(round_info)
            
            return aggregated_result
        
        def monitored_aggregate_evaluate(server_round, results, failures):
            """Wrapper per aggregate_evaluate con monitoring."""
            aggregated_result = original_aggregate_evaluate(server_round, results, failures)
            
            # Analizza convergenza
            if aggregated_result and len(aggregated_result) > 1:
                self._analyze_convergence(server_round, aggregated_result[1])
            
            return aggregated_result
        
        # Sostituisce i metodi con le versioni monitorate
        strategy.aggregate_fit = monitored_aggregate_fit
        strategy.aggregate_evaluate = monitored_aggregate_evaluate
    
    def _save_round_logs(self, round_info):
        """
        Salva logs dettagliati per round.
        """
        log_file = os.path.join(self.log_dir, f"round_{round_info['round']:03d}.json")
        
        # Converti set in lista per JSON serialization
        round_info_serializable = round_info.copy()
        if 'strategy_metrics' in round_info_serializable:
            strategy_metrics = round_info_serializable['strategy_metrics']
            if 'total_clients_seen' in strategy_metrics:
                strategy_metrics['total_clients_seen'] = list(strategy_metrics['total_clients_seen'])
        
        with open(log_file, 'w') as f:
            json.dump(round_info_serializable, f, indent=2, default=str)
    
    def _analyze_convergence(self, server_round, metrics):
        """
        Analizza la convergenza del training.
        """
        if 'global_accuracy' in metrics:
            accuracy = float(metrics['global_accuracy'])
            
            convergence_info = {
                'round': server_round,
                'global_accuracy': accuracy,
                'timestamp': time.time()
            }
            
            # Calcola trend di convergenza
            if len(self.server_metrics['convergence_metrics']) >= 2:
                recent_accuracies = [m['global_accuracy'] for m in self.server_metrics['convergence_metrics'][-5:]]
                recent_accuracies.append(accuracy)
                
                # Calcola improvement rate
                if len(recent_accuracies) >= 2:
                    improvement = recent_accuracies[-1] - recent_accuracies[0]
                    convergence_info['improvement_rate'] = improvement / len(recent_accuracies)
                    
                    # Detecta stagnazione
                    if len(recent_accuracies) >= 5 and abs(improvement) < 0.001:
                        convergence_info['stagnation_detected'] = True
                        if self.verbose:
                            print(f"⚠️  Possibile stagnazione rilevata al round {server_round}")
            
            self.server_metrics['convergence_metrics'].append(convergence_info)
    
    def run_server(self, server_address: str = "0.0.0.0:8080"):
        """
        Avvia il server federato ottimizzato.
        """
        print(f"\n🌟 AVVIO SERVER SMARTGRID FEDERATO OTTIMIZZATO")
        print("=" * 70)
        print(f"🏗️  Configurazione:")
        print(f"   • Indirizzo server: {server_address}")
        print(f"   • Strategia: {self.strategy_type}")
        print(f"   • Round totali: {self.num_rounds}")
        print(f"   • Componenti PCA: {self.n_components}")
        print("=" * 70)
        
        # Crea strategia
        strategy = self.create_strategy()
        
        # Configurazione server
        config = fl.server.ServerConfig(num_rounds=self.num_rounds)
        
        # Avvia server
        try:
            print(f"🚀 Server in ascolto su {server_address}")
            print(f"   Attendendo connessione client...")
            
            fl.server.start_server(
                server_address=server_address,
                config=config,
                strategy=strategy
            )
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Server interrotto dall'utente")
        except Exception as e:
            print(f"\n❌ Errore del server: {e}")
        finally:
            self._finalize_server()
    
    def _finalize_server(self):
        """
        Finalizza il server e salva report completo.
        """
        end_time = time.time()
        total_duration = end_time - self.server_metrics['start_time']
        
        print(f"\n📊 RIEPILOGO ESECUZIONE SERVER")
        print("=" * 50)
        print(f"⏱️  Durata totale: {total_duration:.1f} secondi")
        print(f"🔄 Round completati: {self.server_metrics['rounds_completed']}")
        print(f"👥 Client unici visti: {len(self.server_metrics['total_clients_seen'])}")
        
        if self.server_metrics['convergence_metrics']:
            final_accuracy = self.server_metrics['convergence_metrics'][-1]['global_accuracy']
            print(f"🎯 Accuracy finale: {final_accuracy:.4f}")
        
        # Salva report completo
        if self.save_logs:
            self._save_final_report(total_duration)
        
        print("=" * 50)
        print("✅ Server federato terminato con successo!")
    
    def _save_final_report(self, total_duration):
        """
        Salva report finale dell'esecuzione.
        """
        final_report = {
            'execution_summary': {
                'total_duration': total_duration,
                'rounds_completed': self.server_metrics['rounds_completed'],
                'unique_clients': len(self.server_metrics['total_clients_seen']),
                'strategy_type': self.strategy_type,
                'n_components': self.n_components
            },
            'performance_summary': {},
            'convergence_analysis': {},
            'round_history': self.server_metrics['round_history'],
            'convergence_metrics': self.server_metrics['convergence_metrics']
        }
        
        # Analisi performance
        if self.server_metrics['convergence_metrics']:
            accuracies = [m['global_accuracy'] for m in self.server_metrics['convergence_metrics']]
            final_report['performance_summary'] = {
                'final_accuracy': accuracies[-1] if accuracies else 0,
                'max_accuracy': max(accuracies) if accuracies else 0,
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'accuracy_std': np.std(accuracies) if accuracies else 0
            }
        
        # Analisi convergenza
        if len(self.server_metrics['convergence_metrics']) >= 2:
            improvement_rates = [m.get('improvement_rate', 0) for m in self.server_metrics['convergence_metrics'] 
                               if 'improvement_rate' in m]
            if improvement_rates:
                final_report['convergence_analysis'] = {
                    'avg_improvement_rate': np.mean(improvement_rates),
                    'stagnation_rounds': len([m for m in self.server_metrics['convergence_metrics'] 
                                            if m.get('stagnation_detected', False)])
                }
        
        # Salva report
        report_file = os.path.join(self.log_dir, "final_server_report.json")
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"💾 Report finale salvato: {report_file}")


def main():
    """
    Funzione principale per avviare il server ottimizzato.
    """
    print("🌟 SMARTGRID FEDERATED LEARNING - SERVER OTTIMIZZATO")
    print("=" * 70)
    print("📅 Data:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("👤 Autore: francescaapellegrino")
    print("🎯 Obiettivo: Server FL ottimizzato per SmartGrid con aggregazione bilanciata")
    print("=" * 70)
    
    # Configurazione server
    server = SmartGridOptimizedServer(
        strategy_type="smartgrid_optimized",
        n_components=20,  # Standard dal sistema esistente
        num_rounds=30,    # Ridotto per test, aumenta per training completo
        verbose=True,
        save_logs=True,
        log_dir="server_logs"
    )
    
    # Avvia server
    server.run_server(server_address="0.0.0.0:8080")


if __name__ == "__main__":
    main()