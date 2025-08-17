#!/usr/bin/env python3
"""
Benchmark Script for SmartGrid Federated Learning Optimizations

Confronta performance tra:
- Baseline: FedAvg standard con client originali
- Optimized: Balanced strategy con client adattivi
- Ibrido: Combinazioni di diverse ottimizzazioni

Metriche analizzate:
- Performance (accuracy, F1-score, AUC-ROC)
- Convergenza (round per convergenza, stabilità)
- Fairness (varianza tra client, bilanciamento)
- Efficienza (tempo per round, comunicazione)

Author: francescaapellegrino
Date: 2025-08-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import subprocess
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Per simulazione locale (senza server reale)
import sys
sys.path.append('.')
from analyze_client_distribution import SmartGridClientAnalyzer
from strategies import (
    create_smartgrid_optimized_strategy, 
    create_class_weighted_strategy,
    create_outlier_penalty_strategy,
    create_adaptive_strategy,
    create_hybrid_strategy
)


class SmartGridBenchmark:
    """
    Sistema di benchmark completo per confrontare strategie di federated learning.
    """
    
    def __init__(
        self,
        client_ids: List[int] = None,
        n_components: int = 20,
        output_dir: str = "benchmark_results",
        verbose: bool = True
    ):
        self.client_ids = client_ids if client_ids else list(range(1, 14))  # Client 1-13 per training
        self.n_components = n_components
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Crea directory output
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurazioni benchmark
        self.benchmark_configs = {
            'baseline': {
                'name': 'Baseline FedAvg',
                'strategy': 'standard',
                'client_type': 'standard',
                'description': 'FedAvg standard con client originali'
            },
            'class_weighted': {
                'name': 'Class Weighted',
                'strategy': 'class_weighted',
                'client_type': 'standard',
                'description': 'Aggregazione pesata per bilanciamento classi'
            },
            'outlier_penalty': {
                'name': 'Outlier Penalty',
                'strategy': 'outlier_penalty',
                'client_type': 'standard',
                'description': 'Penalizzazione client outlier'
            },
            'adaptive_lr': {
                'name': 'Adaptive Learning',
                'strategy': 'adaptive',
                'client_type': 'adaptive',
                'description': 'Learning rates adattivi'
            },
            'hybrid_optimized': {
                'name': 'Hybrid Optimized',
                'strategy': 'hybrid',
                'client_type': 'adaptive',
                'description': 'Strategia ibrida completa'
            },
            'smartgrid_optimized': {
                'name': 'SmartGrid Optimized',
                'strategy': 'smartgrid_optimized',
                'client_type': 'adaptive',
                'description': 'Configurazione ottimizzata per SmartGrid'
            }
        }
        
        # Risultati benchmark
        self.benchmark_results = {}
        self.comparison_metrics = {}
        
        if self.verbose:
            print(f"🏁 SMARTGRID FEDERATED LEARNING BENCHMARK")
            print(f"   • Client testati: {self.client_ids}")
            print(f"   • Configurazioni: {len(self.benchmark_configs)}")
            print(f"   • Output directory: {self.output_dir}")
    
    def run_analysis_benchmark(self):
        """
        Esegue benchmark dell'analisi di distribuzione client.
        """
        print(f"\n📊 BENCHMARK ANALISI DISTRIBUZIONE CLIENT")
        print("=" * 60)
        
        # Esegui analisi distribuzione
        analyzer = SmartGridClientAnalyzer(
            client_ids=self.client_ids,
            n_components=self.n_components,
            output_dir=os.path.join(self.output_dir, "client_analysis")
        )
        
        analysis_results = analyzer.run_complete_analysis()
        
        if analysis_results:
            # Salva risultati analisi
            analysis_summary = {
                'heterogeneity_index': analysis_results['noniid_metrics']['heterogeneity_index'],
                'outlier_clients': len(analysis_results['outlier_results']['outlier_clients']),
                'similarity_metrics': analysis_results['similarity_results'],
                'clustering_quality': analysis_results['clustering_results']['silhouette_score'],
                'recommendations': self._generate_recommendations(analysis_results)
            }
            
            with open(os.path.join(self.output_dir, "analysis_summary.json"), 'w') as f:
                json.dump(analysis_summary, f, indent=2, default=str)
            
            print(f"✅ Analisi distribuzione completata")
            print(f"   • Eterogeneità: {analysis_results['noniid_metrics']['heterogeneity_index']:.3f}")
            print(f"   • Client outlier: {len(analysis_results['outlier_results']['outlier_clients'])}")
            print(f"   • Qualità clustering: {analysis_results['clustering_results']['silhouette_score']:.3f}")
            
            return analysis_summary
        else:
            print(f"❌ Analisi distribuzione fallita")
            return None
    
    def run_simulated_benchmark(self, rounds_per_config: int = 20):
        """
        Esegue benchmark simulato delle diverse strategie di aggregazione.
        (Simulazione senza server reale per test rapido)
        """
        print(f"\n🎮 BENCHMARK SIMULATO STRATEGIE FL")
        print("=" * 60)
        print(f"Round per configurazione: {rounds_per_config}")
        print("")
        
        # Carica dati client per simulazione
        client_data = self._load_all_client_data()
        
        # Testa ogni configurazione
        for config_name, config in self.benchmark_configs.items():
            print(f"🔄 Testando configurazione: {config['name']}")
            
            start_time = time.time()
            
            # Simula training federato
            simulation_results = self._simulate_federated_training(
                config, client_data, rounds_per_config
            )
            
            simulation_duration = time.time() - start_time
            simulation_results['total_duration'] = simulation_duration
            
            self.benchmark_results[config_name] = simulation_results
            
            print(f"   ✅ Completato in {simulation_duration:.2f}s")
            print(f"      Accuracy finale: {simulation_results['final_accuracy']:.4f}")
            print(f"      Convergenza: round {simulation_results['convergence_round']}")
            
            # Salva risultati intermedi
            self._save_intermediate_results(config_name, simulation_results)
        
        print(f"\n✅ Benchmark simulato completato per {len(self.benchmark_configs)} configurazioni")
        return self.benchmark_results
    
    def _load_all_client_data(self):
        """
        Carica dati di tutti i client per la simulazione.
        """
        print(f"📂 Caricamento dati client per simulazione...")
        
        from client_adaptive import load_adaptive_client_data
        
        client_data = {}
        for client_id in self.client_ids:
            try:
                (X_train, y_train, X_val, y_val, X_test, y_test, 
                 scaler, pca, dataset_info) = load_adaptive_client_data(
                    client_id, self.n_components
                )
                
                client_data[client_id] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test,
                    'info': dataset_info
                }
                
                if self.verbose:
                    print(f"   ✅ Client {client_id}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
                    
            except Exception as e:
                print(f"   ❌ Errore caricamento client {client_id}: {e}")
                continue
        
        print(f"📊 Caricati {len(client_data)} client su {len(self.client_ids)} richiesti")
        return client_data
    
    def _simulate_federated_training(self, config, client_data, num_rounds):
        """
        Simula training federato per una configurazione specifica.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import tensorflow as tf
        
        # Crea modello globale
        global_model = self._create_simulation_model()
        
        # Metriche tracking
        round_metrics = []
        client_performances = {cid: [] for cid in client_data.keys()}
        
        # Parametri globali iniziali
        global_weights = global_model.get_weights()
        
        # Simula ogni round
        for round_num in range(1, num_rounds + 1):
            round_start = time.time()
            
            # Client selection (semplificato)
            participating_clients = list(client_data.keys())
            if config['strategy'] in ['outlier_penalty', 'hybrid', 'smartgrid_optimized']:
                # Simula client selection intelligente escludendo alcuni client
                participating_clients = participating_clients[:-2] if len(participating_clients) > 3 else participating_clients
            
            # Training locale per ogni client
            client_updates = []
            client_weights_list = []
            
            for client_id in participating_clients:
                client_info = client_data[client_id]
                
                # Crea modello client e applica pesi globali
                client_model = self._create_simulation_model()
                client_model.set_weights(global_weights)
                
                # Calcola learning rate adattivo
                if config['client_type'] == 'adaptive':
                    lr = self._calculate_adaptive_lr(client_id, round_num, client_performances)
                else:
                    lr = 0.001
                
                # Training locale simulato
                client_model.optimizer.learning_rate.assign(lr)
                
                # Simula epochs (ridotte per velocità)
                epochs = 2 if config['client_type'] == 'adaptive' else 1
                history = client_model.fit(
                    client_info['X_train'], client_info['y_train'],
                    validation_data=(client_info['X_val'], client_info['y_val']),
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                # Calcola metriche client
                client_weights = client_model.get_weights()
                val_accuracy = history.history['val_accuracy'][-1]
                
                client_performances[client_id].append(val_accuracy)
                
                # Peso per aggregazione
                if config['strategy'] == 'class_weighted':
                    weight = self._calculate_class_weight(client_info['info'])
                elif config['strategy'] in ['outlier_penalty', 'hybrid', 'smartgrid_optimized']:
                    weight = self._calculate_outlier_weight(client_id, client_info['info'])
                else:
                    weight = len(client_info['X_train'])  # Standard FedAvg
                
                client_updates.append((client_weights, weight, val_accuracy))
                client_weights_list.append(client_weights)
            
            # Aggregazione
            if config['strategy'] in ['class_weighted', 'outlier_penalty', 'hybrid', 'smartgrid_optimized']:
                global_weights = self._weighted_aggregate(client_updates)
            else:
                global_weights = self._standard_aggregate(client_weights_list)
            
            # Aggiorna modello globale
            global_model.set_weights(global_weights)
            
            # Valutazione globale (su validation set combinato)
            val_accuracy, val_f1, val_auc = self._evaluate_global_model(global_model, client_data)
            
            round_duration = time.time() - round_start
            
            round_metrics.append({
                'round': round_num,
                'val_accuracy': val_accuracy,
                'val_f1_score': val_f1,
                'val_auc': val_auc,
                'participating_clients': len(participating_clients),
                'round_duration': round_duration,
                'client_variance': np.var([perf[-1] for perf in client_performances.values() if perf])
            })
            
            # Early stopping per convergenza
            if round_num >= 5:
                recent_accuracies = [m['val_accuracy'] for m in round_metrics[-5:]]
                if max(recent_accuracies) - min(recent_accuracies) < 0.001:
                    convergence_round = round_num
                    break
        else:
            convergence_round = num_rounds
        
        # Test finale
        final_test_accuracy, final_test_f1, final_test_auc = self._evaluate_global_model(
            global_model, client_data, test_set=True
        )
        
        return {
            'config_name': config['name'],
            'strategy': config['strategy'],
            'client_type': config['client_type'],
            'round_metrics': round_metrics,
            'client_performances': client_performances,
            'final_accuracy': final_test_accuracy,
            'final_f1_score': final_test_f1,
            'final_auc': final_test_auc,
            'convergence_round': convergence_round,
            'total_rounds': len(round_metrics),
            'avg_round_duration': np.mean([m['round_duration'] for m in round_metrics]),
            'final_client_variance': np.var([perf[-1] for perf in client_performances.values() if perf])
        }
    
    def _create_simulation_model(self):
        """Crea modello per simulazione."""
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.n_components,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _calculate_adaptive_lr(self, client_id, round_num, client_performances):
        """Calcola learning rate adattivo per simulazione."""
        base_lr = 0.001
        
        if round_num <= 1 or not client_performances[client_id]:
            return base_lr
        
        recent_perf = client_performances[client_id][-3:] if len(client_performances[client_id]) >= 3 else client_performances[client_id]
        
        if len(recent_perf) >= 2:
            improvement = recent_perf[-1] - recent_perf[0]
            if improvement > 0.01:
                return base_lr * 1.1
            elif improvement < -0.01:
                return base_lr * 0.8
        
        return base_lr
    
    def _calculate_class_weight(self, client_info):
        """Calcola peso basato su bilanciamento classi."""
        attack_ratio = client_info['train_attack_ratio']
        # Penalizza client con distribuzione molto sbilanciata
        balance_score = 1.0 - abs(attack_ratio - 0.5) * 2
        return max(0.1, balance_score)
    
    def _calculate_outlier_weight(self, client_id, client_info):
        """Calcola peso con penalizzazione outlier."""
        # Identifica outlier basandosi su attack_ratio estremi
        attack_ratio = client_info['train_attack_ratio']
        
        # Client con ratio < 0.6 o > 0.8 sono considerati outlier
        if attack_ratio < 0.6 or attack_ratio > 0.8:
            return 0.5  # Peso ridotto per outlier
        else:
            return 1.0  # Peso normale
    
    def _weighted_aggregate(self, client_updates):
        """Aggregazione pesata."""
        total_weight = sum(weight for _, weight, _ in client_updates)
        
        if total_weight == 0:
            return client_updates[0][0]  # Fallback
        
        # Inizializza con pesi del primo client
        aggregated_weights = [np.zeros_like(w) for w in client_updates[0][0]]
        
        for client_weights, weight, _ in client_updates:
            normalized_weight = weight / total_weight
            for i, w in enumerate(client_weights):
                aggregated_weights[i] += w * normalized_weight
        
        return aggregated_weights
    
    def _standard_aggregate(self, client_weights_list):
        """Aggregazione standard (media)."""
        if not client_weights_list:
            return None
        
        aggregated_weights = [np.zeros_like(w) for w in client_weights_list[0]]
        num_clients = len(client_weights_list)
        
        for client_weights in client_weights_list:
            for i, w in enumerate(client_weights):
                aggregated_weights[i] += w / num_clients
        
        return aggregated_weights
    
    def _evaluate_global_model(self, model, client_data, test_set=False):
        """Valuta modello globale su dati combinati."""
        from sklearn.metrics import f1_score, roc_auc_score
        
        # Combina dati di tutti i client
        if test_set:
            all_X = np.vstack([data['X_test'] for data in client_data.values()])
            all_y = np.hstack([data['y_test'] for data in client_data.values()])
        else:
            all_X = np.vstack([data['X_val'] for data in client_data.values()])
            all_y = np.hstack([data['y_val'] for data in client_data.values()])
        
        # Predizioni
        y_pred_proba = model.predict(all_X, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metriche
        accuracy = np.mean(y_pred == all_y)
        f1 = f1_score(all_y, y_pred)
        auc = roc_auc_score(all_y, y_pred_proba)
        
        return accuracy, f1, auc
    
    def _save_intermediate_results(self, config_name, results):
        """Salva risultati intermedi."""
        result_file = os.path.join(self.output_dir, f"{config_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _generate_recommendations(self, analysis_results):
        """Genera raccomandazioni basate sui risultati di analisi."""
        het_index = analysis_results['noniid_metrics']['heterogeneity_index']
        n_outliers = len(analysis_results['outlier_results']['outlier_clients'])
        
        recommendations = []
        
        if het_index > 0.4:
            recommendations.append("Strategia bilanciata necessaria")
        elif het_index > 0.2:
            recommendations.append("Strategia bilanciata consigliata")
        else:
            recommendations.append("FedAvg standard sufficiente")
        
        if n_outliers > 2:
            recommendations.append("Client selection adattiva necessaria")
        elif n_outliers > 0:
            recommendations.append("Monitoraggio outlier consigliato")
        
        return recommendations
    
    def generate_comparison_report(self):
        """
        Genera report completo di confronto tra le strategie.
        """
        print(f"\n📊 GENERAZIONE REPORT COMPARATIVO")
        print("=" * 50)
        
        if not self.benchmark_results:
            print(f"❌ Nessun risultato di benchmark disponibile")
            return None
        
        # Calcola metriche comparative
        comparison_data = []
        
        for config_name, results in self.benchmark_results.items():
            comparison_data.append({
                'Configuration': results['config_name'],
                'Strategy': results['strategy'],
                'Final_Accuracy': results['final_accuracy'],
                'Final_F1_Score': results['final_f1_score'],
                'Final_AUC': results['final_auc'],
                'Convergence_Round': results['convergence_round'],
                'Avg_Round_Duration': results['avg_round_duration'],
                'Client_Variance': results['final_client_variance'],
                'Total_Rounds': results['total_rounds']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Salva tabella comparativa
        comparison_file = os.path.join(self.output_dir, "strategy_comparison.csv")
        df_comparison.to_csv(comparison_file, index=False)
        
        # Trova migliore configurazione per diverse metriche
        best_accuracy = df_comparison.loc[df_comparison['Final_Accuracy'].idxmax()]
        best_f1 = df_comparison.loc[df_comparison['Final_F1_Score'].idxmax()]
        best_convergence = df_comparison.loc[df_comparison['Convergence_Round'].idxmin()]
        best_fairness = df_comparison.loc[df_comparison['Client_Variance'].idxmin()]
        
        # Report testuale
        report = {
            'summary': {
                'total_configurations': len(self.benchmark_results),
                'best_accuracy': {
                    'config': best_accuracy['Configuration'],
                    'value': best_accuracy['Final_Accuracy']
                },
                'best_f1': {
                    'config': best_f1['Configuration'],
                    'value': best_f1['Final_F1_Score']
                },
                'fastest_convergence': {
                    'config': best_convergence['Configuration'],
                    'rounds': best_convergence['Convergence_Round']
                },
                'best_fairness': {
                    'config': best_fairness['Configuration'],
                    'variance': best_fairness['Client_Variance']
                }
            },
            'detailed_comparison': df_comparison.to_dict('records'),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Salva report JSON
        report_file = os.path.join(self.output_dir, "comparison_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Genera visualizzazioni
        self._create_comparison_visualizations(df_comparison)
        
        print(f"✅ Report comparativo generato:")
        print(f"   📊 Tabella: {comparison_file}")
        print(f"   📝 Report: {report_file}")
        print(f"   📈 Visualizzazioni: {self.output_dir}/comparison_plots.png")
        
        # Stampa riepilogo
        print(f"\n🏆 RISULTATI MIGLIORI:")
        print(f"   • Accuracy: {best_accuracy['Configuration']} ({best_accuracy['Final_Accuracy']:.4f})")
        print(f"   • F1-Score: {best_f1['Configuration']} ({best_f1['Final_F1_Score']:.4f})")
        print(f"   • Convergenza: {best_convergence['Configuration']} ({best_convergence['Convergence_Round']} round)")
        print(f"   • Fairness: {best_fairness['Configuration']} (var: {best_fairness['Client_Variance']:.6f})")
        
        return report
    
    def _create_comparison_visualizations(self, df_comparison):
        """
        Crea visualizzazioni comparative delle strategie.
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SmartGrid Federated Learning - Confronto Strategie', fontsize=16, fontweight='bold')
        
        # 1. Accuracy finale
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df_comparison['Configuration'], df_comparison['Final_Accuracy'], 
                       color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy Finale')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Aggiungi valori sopra le barre
        for bar, val in zip(bars1, df_comparison['Final_Accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. F1-Score finale
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df_comparison['Configuration'], df_comparison['Final_F1_Score'], 
                       color='lightgreen', alpha=0.7)
        ax2.set_title('F1-Score Finale')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, df_comparison['Final_F1_Score']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Round di convergenza
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df_comparison['Configuration'], df_comparison['Convergence_Round'], 
                       color='lightcoral', alpha=0.7)
        ax3.set_title('Round per Convergenza')
        ax3.set_ylabel('Round')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, df_comparison['Convergence_Round']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 4. Tempo per round
        ax4 = axes[1, 0]
        bars4 = ax4.bar(df_comparison['Configuration'], df_comparison['Avg_Round_Duration'], 
                       color='orange', alpha=0.7)
        ax4.set_title('Tempo Medio per Round')
        ax4.set_ylabel('Secondi')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, val in zip(bars4, df_comparison['Avg_Round_Duration']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Varianza client (fairness)
        ax5 = axes[1, 1]
        bars5 = ax5.bar(df_comparison['Configuration'], df_comparison['Client_Variance'], 
                       color='purple', alpha=0.7)
        ax5.set_title('Varianza Client (Fairness)')
        ax5.set_ylabel('Varianza')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        for bar, val in zip(bars5, df_comparison['Client_Variance']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.05,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Radar chart comparativo
        ax6 = axes[1, 2]
        
        # Normalizza metriche per radar chart
        metrics = ['Final_Accuracy', 'Final_F1_Score', 'Final_AUC']
        normalized_data = {}
        
        for metric in metrics:
            min_val = df_comparison[metric].min()
            max_val = df_comparison[metric].max()
            normalized_data[metric] = (df_comparison[metric] - min_val) / (max_val - min_val + 1e-8)
        
        # Prende top 3 configurazioni per il radar
        top_configs = df_comparison.nlargest(3, 'Final_Accuracy')
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Completa il cerchio
        
        colors = ['red', 'blue', 'green']
        for i, (_, config) in enumerate(top_configs.iterrows()):
            if i >= 3:
                break
            
            values = [normalized_data[metric].iloc[_] for metric in metrics]
            values += values[:1]  # Completa il cerchio
            
            ax6.plot(angles, values, 'o-', linewidth=2, label=config['Configuration'], color=colors[i])
            ax6.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(['Accuracy', 'F1-Score', 'AUC'])
        ax6.set_ylim(0, 1)
        ax6.set_title('Confronto Top 3 Configurazioni')
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Salva figura
        plot_file = os.path.join(self.output_dir, "comparison_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualizzazioni salvate: {plot_file}")
    
    def run_complete_benchmark(self, rounds_per_config: int = 20):
        """
        Esegue benchmark completo: analisi + simulazione + report.
        """
        print(f"🚀 BENCHMARK COMPLETO SMARTGRID FEDERATED LEARNING")
        print("=" * 70)
        print(f"📅 Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 Autore: francescaapellegrino")
        print(f"🎯 Obiettivo: Confronto strategie ottimizzate vs baseline")
        print("=" * 70)
        
        total_start_time = time.time()
        
        try:
            # 1. Analisi distribuzione client
            analysis_results = self.run_analysis_benchmark()
            
            # 2. Benchmark simulato strategie
            simulation_results = self.run_simulated_benchmark(rounds_per_config)
            
            # 3. Generazione report comparativo
            comparison_report = self.generate_comparison_report()
            
            total_duration = time.time() - total_start_time
            
            print(f"\n🎉 BENCHMARK COMPLETO TERMINATO!")
            print("=" * 50)
            print(f"⏱️  Durata totale: {total_duration:.1f} secondi")
            print(f"📊 Configurazioni testate: {len(self.benchmark_results)}")
            print(f"📁 Risultati salvati in: {self.output_dir}")
            
            # Riepilogo finale
            if comparison_report:
                best_overall = max(self.benchmark_results.items(), 
                                 key=lambda x: x[1]['final_accuracy'])
                print(f"🏆 Migliore configurazione: {best_overall[1]['config_name']}")
                print(f"   Accuracy: {best_overall[1]['final_accuracy']:.4f}")
                print(f"   F1-Score: {best_overall[1]['final_f1_score']:.4f}")
                print(f"   Convergenza: {best_overall[1]['convergence_round']} round")
            
            return {
                'analysis_results': analysis_results,
                'simulation_results': simulation_results,
                'comparison_report': comparison_report,
                'total_duration': total_duration
            }
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE IL BENCHMARK: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Funzione principale per eseguire il benchmark.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartGrid Federated Learning Benchmark")
    parser.add_argument("--rounds", type=int, default=15, help="Round per configurazione")
    parser.add_argument("--components", type=int, default=20, help="Componenti PCA")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Directory output")
    parser.add_argument("--clients", type=str, default="1-13", help="Range client (es: 1-13)")
    
    args = parser.parse_args()
    
    # Parse client range
    if '-' in args.clients:
        start, end = map(int, args.clients.split('-'))
        client_ids = list(range(start, end + 1))
    else:
        client_ids = [int(x) for x in args.clients.split(',')]
    
    # Crea e esegui benchmark
    benchmark = SmartGridBenchmark(
        client_ids=client_ids,
        n_components=args.components,
        output_dir=args.output,
        verbose=True
    )
    
    results = benchmark.run_complete_benchmark(rounds_per_config=args.rounds)
    
    if results:
        print(f"\n✨ Benchmark completato con successo!")
        print(f"📁 Controlla i risultati in: {args.output}")
    else:
        print(f"\n💥 Benchmark fallito!")


if __name__ == "__main__":
    main()