#!/usr/bin/env python3
"""
Balanced Aggregation Strategy for SmartGrid Federated Learning

Implementa strategie di aggregazione bilanciate per migliorare le performance
del federated learning in presenza di dati eterogenei (Non-IID).

Features:
- Weighted aggregation basata su bilanciamento classi
- Riduzione peso client dominanti/outlier
- Aumento peso client sottorappresentati
- Client selection intelligente per round
- Adaptive learning rates per client

Author: francescaapellegrino
Date: 2025-08-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import pickle
import os
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BalancedFedAvg(FedAvg):
    """
    Strategia FedAvg bilanciata per SmartGrid con gestione dell'eterogeneità.
    
    Questa strategia implementa diverse ottimizzazioni:
    1. Weighted aggregation basata su distribuzione classi
    2. Penalizzazione client outlier
    3. Client selection intelligente
    4. Adaptive learning rates
    5. Monitoring real-time della distribuzione
    """
    
    def __init__(
        self,
        *,
        # Parametri standard FedAvg
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        
        # Parametri balanced strategy
        balance_strategy: str = "class_weighted",  # "class_weighted", "outlier_penalty", "adaptive", "hybrid"
        outlier_penalty: float = 0.5,              # Fattore penalizzazione outlier (0-1)
        adaptive_lr_factor: float = 0.8,           # Fattore per LR adattivi
        min_client_weight: float = 0.1,            # Peso minimo per client
        max_client_weight: float = 3.0,            # Peso massimo per client
        diversity_threshold: float = 0.3,          # Soglia per client selection diversificata
        use_client_selection: bool = True,         # Abilita client selection intelligente
        history_window: int = 10,                  # Finestra per media mobile performance
        verbose: bool = True                       # Logging dettagliato
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        
        # Configurazione balanced strategy
        self.balance_strategy = balance_strategy
        self.outlier_penalty = outlier_penalty
        self.adaptive_lr_factor = adaptive_lr_factor
        self.min_client_weight = min_client_weight
        self.max_client_weight = max_client_weight
        self.diversity_threshold = diversity_threshold
        self.use_client_selection = use_client_selection
        self.history_window = history_window
        self.verbose = verbose
        
        # Stato interno per tracking
        self.client_stats = {}          # Statistiche per client
        self.client_weights = {}        # Pesi calcolati per client
        self.round_history = []         # Storia performance per round
        self.client_performance = defaultdict(list)  # Performance storica per client
        self.outlier_clients = set()    # Client identificati come outlier
        self.cluster_assignments = {}   # Assegnazioni cluster per client
        
        # Metriche per monitoraggio
        self.aggregation_metrics = {
            'total_rounds': 0,
            'weight_adjustments': 0,
            'outlier_penalties': 0,
            'diversity_selections': 0
        }
        
        if self.verbose:
            print(f"🎯 BALANCED FEDAVG STRATEGY INIZIALIZZATA")
            print(f"   • Balance strategy: {self.balance_strategy}")
            print(f"   • Outlier penalty: {self.outlier_penalty}")
            print(f"   • Adaptive LR factor: {self.adaptive_lr_factor}")
            print(f"   • Client selection: {'Abilitata' if self.use_client_selection else 'Disabilitata'}")
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """
        Configura il training con client selection intelligente e adaptive LR.
        """
        config = {}
        
        # Calcola learning rate adattivo se abilitato
        if self.balance_strategy in ["adaptive", "hybrid"]:
            config["lr"] = self._get_adaptive_learning_rate(server_round)
        
        # Client selection standard
        clients = super().configure_fit(server_round, parameters, client_manager)
        
        # Client selection intelligente se abilitato
        if self.use_client_selection and len(self.client_stats) > 0:
            clients = self._intelligent_client_selection(clients, server_round)
        
        # Applica configurazioni personalizzate per client
        configured_clients = []
        for client, base_config in clients:
            client_id = self._get_client_id(client)
            client_config = base_config.copy()
            client_config.update(config)
            
            # Learning rate personalizzato per client se disponibile
            if client_id in self.client_stats:
                client_config["lr"] = self._get_client_specific_lr(client_id, server_round)
            
            configured_clients.append((client, client_config))
        
        if self.verbose:
            print(f"📋 Round {server_round}: {len(configured_clients)} client configurati per training")
        
        return configured_clients
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggrega i risultati del training con weighted aggregation bilanciata.
        """
        if not results:
            return None, {}
        
        if self.verbose:
            print(f"\n🔄 AGGREGAZIONE BILANCIATA ROUND {server_round}")
            print(f"   • Client partecipanti: {len(results)}")
            print(f"   • Client falliti: {len(failures)}")
        
        # Aggiorna statistiche client
        self._update_client_statistics(results)
        
        # Calcola pesi bilanciati
        balanced_weights = self._calculate_balanced_weights(results, server_round)
        
        # Aggregazione pesata
        aggregated_parameters = self._weighted_aggregate_parameters(results, balanced_weights)
        
        # Calcola metriche aggregate
        aggregated_metrics = self._aggregate_fit_metrics(results, balanced_weights)
        
        # Aggiorna storia e metriche
        self._update_round_history(server_round, results, balanced_weights, aggregated_metrics)
        
        if self.verbose:
            self._print_aggregation_summary(server_round, balanced_weights, aggregated_metrics)
        
        self.aggregation_metrics['total_rounds'] += 1
        
        return aggregated_parameters, aggregated_metrics
    
    def _update_client_statistics(self, results: List[Tuple[ClientProxy, FitRes]]):
        """
        Aggiorna le statistiche dei client basate sui risultati del training.
        """
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                # Aggiorna statistiche di base
                if client_id not in self.client_stats:
                    self.client_stats[client_id] = {
                        'total_samples': 0,
                        'attack_ratio': 0.5,  # Default neutro
                        'performance_history': [],
                        'weight_history': [],
                        'is_outlier': False
                    }
                
                # Estrai metriche dal training
                metrics = fit_res.metrics
                
                # Campioni e distribuzione classi
                if 'train_samples' in metrics:
                    self.client_stats[client_id]['total_samples'] = int(metrics['train_samples'])
                
                if 'train_attack_ratio' in metrics:
                    self.client_stats[client_id]['attack_ratio'] = float(metrics['train_attack_ratio'])
                
                # Performance metrics
                performance_score = self._calculate_performance_score(metrics)
                self.client_stats[client_id]['performance_history'].append(performance_score)
                self.client_performance[client_id].append(performance_score)
                
                # Mantieni solo la finestra di storia richiesta
                if len(self.client_stats[client_id]['performance_history']) > self.history_window:
                    self.client_stats[client_id]['performance_history'] = \
                        self.client_stats[client_id]['performance_history'][-self.history_window:]
                
                if len(self.client_performance[client_id]) > self.history_window:
                    self.client_performance[client_id] = \
                        self.client_performance[client_id][-self.history_window:]
        
        # Aggiorna identificazione outlier
        self._update_outlier_detection()
    
    def _calculate_performance_score(self, metrics: Dict[str, Scalar]) -> float:
        """
        Calcola un punteggio di performance composito dalle metriche del client.
        """
        score = 0.0
        weights = {'accuracy': 0.4, 'f1_score': 0.3, 'val_accuracy': 0.2, 'val_f1_score': 0.1}
        
        total_weight = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                score += float(metrics[metric_name]) * weight
                total_weight += weight
        
        # Normalizza se non tutti i pesi sono disponibili
        if total_weight > 0:
            score = score / total_weight
        else:
            score = 0.5  # Default neutro
        
        return score
    
    def _update_outlier_detection(self):
        """
        Aggiorna l'identificazione dei client outlier basata su statistiche correnti.
        """
        if len(self.client_stats) < 3:
            return  # Insufficienti client per outlier detection
        
        # Raccoglie metriche per analisi outlier
        attack_ratios = []
        sample_counts = []
        performance_scores = []
        client_ids = []
        
        for client_id, stats in self.client_stats.items():
            if stats['performance_history']:
                attack_ratios.append(stats['attack_ratio'])
                sample_counts.append(stats['total_samples'])
                performance_scores.append(np.mean(stats['performance_history']))
                client_ids.append(client_id)
        
        if len(client_ids) < 3:
            return
        
        # Calcola Z-scores per identificare outlier
        metrics_matrix = np.array([attack_ratios, sample_counts, performance_scores]).T
        
        # Standardizza
        means = np.mean(metrics_matrix, axis=0)
        stds = np.std(metrics_matrix, axis=0)
        z_scores = np.abs((metrics_matrix - means) / (stds + 1e-8))
        
        # Client con Z-score > 2 in qualsiasi metrica sono outlier
        outlier_threshold = 2.0
        outliers = np.any(z_scores > outlier_threshold, axis=1)
        
        # Aggiorna outlier set
        self.outlier_clients.clear()
        for i, is_outlier in enumerate(outliers):
            client_id = client_ids[i]
            self.client_stats[client_id]['is_outlier'] = is_outlier
            if is_outlier:
                self.outlier_clients.add(client_id)
    
    def _calculate_balanced_weights(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        server_round: int
    ) -> Dict[str, float]:
        """
        Calcola i pesi bilanciati per l'aggregazione basata sulla strategia scelta.
        """
        balanced_weights = {}
        
        # Pesi standard basati su numero di campioni
        standard_weights = self._calculate_standard_weights(results)
        
        if self.balance_strategy == "standard":
            return standard_weights
        
        elif self.balance_strategy == "class_weighted":
            balanced_weights = self._calculate_class_weighted(results, standard_weights)
            
        elif self.balance_strategy == "outlier_penalty":
            balanced_weights = self._calculate_outlier_penalty_weights(results, standard_weights)
            
        elif self.balance_strategy == "adaptive":
            balanced_weights = self._calculate_adaptive_weights(results, standard_weights, server_round)
            
        elif self.balance_strategy == "hybrid":
            balanced_weights = self._calculate_hybrid_weights(results, standard_weights, server_round)
            
        else:
            balanced_weights = standard_weights
        
        # Applica vincoli min/max sui pesi
        balanced_weights = self._apply_weight_constraints(balanced_weights)
        
        # Normalizza i pesi per sommare a 1.0
        total_weight = sum(balanced_weights.values())
        if total_weight > 0:
            balanced_weights = {k: v/total_weight for k, v in balanced_weights.items()}
        
        # Salva pesi nella storia dei client
        for client_id, weight in balanced_weights.items():
            if client_id in self.client_stats:
                self.client_stats[client_id]['weight_history'].append(weight)
        
        return balanced_weights
    
    def _calculate_standard_weights(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, float]:
        """Calcola pesi standard basati su numero di campioni."""
        weights = {}
        total_samples = 0
        
        # Calcola pesi proporzionali al numero di campioni
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            num_samples = fit_res.num_examples
            weights[client_id] = num_samples
            total_samples += num_samples
        
        # Normalizza
        if total_samples > 0:
            weights = {k: v/total_samples for k, v in weights.items()}
        
        return weights
    
    def _calculate_class_weighted(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        standard_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcola pesi bilanciati per compensare lo sbilanciamento delle classi.
        """
        balanced_weights = standard_weights.copy()
        
        # Calcola bilanciamento medio dataset
        attack_ratios = []
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            if client_id in self.client_stats:
                attack_ratios.append(self.client_stats[client_id]['attack_ratio'])
        
        if not attack_ratios:
            return balanced_weights
        
        mean_attack_ratio = np.mean(attack_ratios)
        
        # Aggiusta pesi basati su quanto il client si discosta dalla media
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            if client_id in self.client_stats:
                client_ratio = self.client_stats[client_id]['attack_ratio']
                
                # Calcola fattore di correzione
                # Client con distribuzione più vicina alla media ottengono peso maggiore
                deviation = abs(client_ratio - mean_attack_ratio)
                correction_factor = 1.0 / (1.0 + deviation * 2.0)  # Penalizza deviazioni
                
                balanced_weights[client_id] *= correction_factor
                
                if deviation > 0.1:  # Soglia significativa
                    self.aggregation_metrics['weight_adjustments'] += 1
        
        return balanced_weights
    
    def _calculate_outlier_penalty_weights(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        standard_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Applica penalizzazione ai client outlier.
        """
        penalized_weights = standard_weights.copy()
        
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            if client_id in self.outlier_clients:
                # Applica penalizzazione
                penalized_weights[client_id] *= self.outlier_penalty
                self.aggregation_metrics['outlier_penalties'] += 1
                
                if self.verbose:
                    print(f"   ⚠️  Client {client_id} (outlier): peso ridotto a {penalized_weights[client_id]:.3f}")
        
        return penalized_weights
    
    def _calculate_adaptive_weights(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        standard_weights: Dict[str, float], 
        server_round: int
    ) -> Dict[str, float]:
        """
        Calcola pesi adattivi basati sulla performance storica dei client.
        """
        adaptive_weights = standard_weights.copy()
        
        # Calcola performance media per client
        performance_scores = {}
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            if client_id in self.client_stats and self.client_stats[client_id]['performance_history']:
                performance_scores[client_id] = np.mean(self.client_stats[client_id]['performance_history'])
        
        if not performance_scores:
            return adaptive_weights
        
        # Normalizza performance scores
        min_perf = min(performance_scores.values())
        max_perf = max(performance_scores.values())
        perf_range = max_perf - min_perf
        
        if perf_range > 0:
            for client_id in performance_scores:
                normalized_perf = (performance_scores[client_id] - min_perf) / perf_range
                # Client con performance migliore ottengono peso maggiore
                performance_multiplier = 0.5 + normalized_perf * 0.5  # Range [0.5, 1.0]
                adaptive_weights[client_id] *= performance_multiplier
        
        return adaptive_weights
    
    def _calculate_hybrid_weights(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        standard_weights: Dict[str, float], 
        server_round: int
    ) -> Dict[str, float]:
        """
        Strategia ibrida che combina tutte le tecniche.
        """
        # Applica tutte le strategie in sequenza
        weights = standard_weights.copy()
        
        # 1. Class weighting
        weights = self._calculate_class_weighted(results, weights)
        
        # 2. Outlier penalty
        weights = self._calculate_outlier_penalty_weights(results, weights)
        
        # 3. Adaptive weighting (con fattore ridotto per evitare over-correction)
        adaptive_factor = 0.5  # Riduce l'impatto della performance history
        adaptive_weights = self._calculate_adaptive_weights(results, weights, server_round)
        for client_id in weights:
            if client_id in adaptive_weights:
                # Media pesata tra peso attuale e peso adattivo
                weights[client_id] = weights[client_id] * (1 - adaptive_factor) + \
                                   adaptive_weights[client_id] * adaptive_factor
        
        return weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Applica vincoli min/max sui pesi calcolati.
        """
        constrained_weights = {}
        
        for client_id, weight in weights.items():
            constrained_weight = max(self.min_client_weight, 
                                   min(self.max_client_weight, weight))
            constrained_weights[client_id] = constrained_weight
        
        return constrained_weights
    
    def _weighted_aggregate_parameters(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        weights: Dict[str, float]
    ) -> Optional[Parameters]:
        """
        Aggrega i parametri usando i pesi bilanciati calcolati.
        """
        if not results:
            return None
        
        # Converte i pesi in formato per l'aggregazione standard
        weighted_results = []
        for client, fit_res in results:
            client_id = self._get_client_id(client)
            weight = weights.get(client_id, 1.0 / len(results))
            
            # Crea un nuovo FitRes con numero di esempi modificato per riflettere il peso
            effective_samples = int(weight * 10000)  # Scala per preservare precisione
            weighted_fit_res = FitRes(
                status=fit_res.status,
                parameters=fit_res.parameters,
                num_examples=effective_samples,
                metrics=fit_res.metrics
            )
            weighted_results.append((client, weighted_fit_res))
        
        # Usa l'aggregazione standard di FedAvg con pesi modificati
        return super().aggregate_fit(0, weighted_results, [])[0]  # server_round non usato nell'aggregazione
    
    def _aggregate_fit_metrics(
        self, 
        results: List[Tuple[ClientProxy, FitRes]], 
        weights: Dict[str, float]
    ) -> Dict[str, Scalar]:
        """
        Aggrega le metriche di training usando i pesi bilanciati.
        """
        aggregated_metrics = {}
        
        # Raccoglie tutte le metriche disponibili
        all_metrics = set()
        for _, fit_res in results:
            if fit_res.metrics:
                all_metrics.update(fit_res.metrics.keys())
        
        # Calcola media pesata per ogni metrica
        for metric_name in all_metrics:
            metric_values = []
            metric_weights = []
            
            for client, fit_res in results:
                if fit_res.metrics and metric_name in fit_res.metrics:
                    client_id = self._get_client_id(client)
                    weight = weights.get(client_id, 1.0 / len(results))
                    
                    metric_values.append(float(fit_res.metrics[metric_name]))
                    metric_weights.append(weight)
            
            if metric_values:
                # Media pesata
                weighted_avg = np.average(metric_values, weights=metric_weights)
                aggregated_metrics[f'weighted_{metric_name}'] = weighted_avg
                
                # Include anche media non pesata per confronto
                aggregated_metrics[f'unweighted_{metric_name}'] = np.mean(metric_values)
        
        # Aggiunge metriche di bilanciamento
        aggregated_metrics['weight_variance'] = np.var(list(weights.values()))
        aggregated_metrics['num_outlier_clients'] = len(self.outlier_clients)
        aggregated_metrics['effective_clients'] = len([w for w in weights.values() if w > self.min_client_weight])
        
        return aggregated_metrics
    
    def _update_round_history(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        weights: Dict[str, float], 
        metrics: Dict[str, Scalar]
    ):
        """
        Aggiorna la storia delle performance per round.
        """
        round_info = {
            'round': server_round,
            'num_clients': len(results),
            'weights': weights.copy(),
            'metrics': metrics.copy(),
            'outlier_clients': list(self.outlier_clients),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.round_history.append(round_info)
        
        # Mantieni solo la finestra di storia richiesta
        if len(self.round_history) > self.history_window * 2:  # Mantieni storia più lunga per analisi
            self.round_history = self.round_history[-self.history_window * 2:]
    
    def _intelligent_client_selection(
        self, 
        clients: List[Tuple[ClientProxy, Dict[str, Scalar]]], 
        server_round: int
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """
        Implementa client selection intelligente per massimizzare la diversità.
        """
        if len(clients) <= self.min_fit_clients:
            return clients  # Non abbastanza client per selezione intelligente
        
        # Calcola diversità dei client disponibili
        client_diversity_scores = {}
        
        for client, config in clients:
            client_id = self._get_client_id(client)
            diversity_score = self._calculate_client_diversity_score(client_id)
            client_diversity_scores[client_id] = diversity_score
        
        # Ordina per diversità (punteggio più alto = più diverso)
        sorted_clients = sorted(
            clients, 
            key=lambda x: client_diversity_scores.get(self._get_client_id(x[0]), 0.0), 
            reverse=True
        )
        
        # Seleziona un mix di client diversi e performanti
        selected_clients = []
        target_num_clients = min(len(clients), max(self.min_fit_clients, int(len(clients) * self.fraction_fit)))
        
        # Prendi i più diversi fino alla soglia di diversità
        diversity_clients = 0
        for client, config in sorted_clients:
            client_id = self._get_client_id(client)
            if client_diversity_scores[client_id] > self.diversity_threshold:
                selected_clients.append((client, config))
                diversity_clients += 1
                if len(selected_clients) >= target_num_clients:
                    break
        
        # Riempi il resto con client rimanenti (migliori performance)
        if len(selected_clients) < target_num_clients:
            remaining_clients = [c for c in sorted_clients if c not in selected_clients]
            
            # Ordina per performance se disponibile
            if self.client_performance:
                remaining_clients.sort(
                    key=lambda x: np.mean(self.client_performance.get(
                        self._get_client_id(x[0]), [0.5]
                    )), 
                    reverse=True
                )
            
            needed_clients = target_num_clients - len(selected_clients)
            selected_clients.extend(remaining_clients[:needed_clients])
        
        if diversity_clients > 0:
            self.aggregation_metrics['diversity_selections'] += 1
            
        if self.verbose and diversity_clients > 0:
            print(f"   🎯 Client selection intelligente: {diversity_clients} client diversi, "
                  f"{len(selected_clients) - diversity_clients} client performanti")
        
        return selected_clients
    
    def _calculate_client_diversity_score(self, client_id: str) -> float:
        """
        Calcola un punteggio di diversità per un client.
        """
        if client_id not in self.client_stats:
            return 0.5  # Diversità neutrale per client sconosciuti
        
        diversity_score = 0.0
        
        # Fattore 1: Quanto il client è diverso dalla media in termini di attack ratio
        all_ratios = [stats['attack_ratio'] for stats in self.client_stats.values()]
        if all_ratios:
            mean_ratio = np.mean(all_ratios)
            client_ratio = self.client_stats[client_id]['attack_ratio']
            ratio_diversity = abs(client_ratio - mean_ratio)
            diversity_score += ratio_diversity * 0.4
        
        # Fattore 2: Quanto il client è diverso in termini di performance
        all_performances = []
        for stats in self.client_stats.values():
            if stats['performance_history']:
                all_performances.append(np.mean(stats['performance_history']))
        
        if all_performances and self.client_stats[client_id]['performance_history']:
            mean_performance = np.mean(all_performances)
            client_performance = np.mean(self.client_stats[client_id]['performance_history'])
            performance_diversity = abs(client_performance - mean_performance)
            diversity_score += performance_diversity * 0.3
        
        # Fattore 3: È un outlier? (diversità alta)
        if self.client_stats[client_id]['is_outlier']:
            diversity_score += 0.2
        
        # Fattore 4: Frequenza di selezione recente (diversità alta se poco selezionato)
        recent_selections = sum(1 for round_info in self.round_history[-5:] 
                               if client_id in round_info.get('weights', {}))
        selection_diversity = max(0, (5 - recent_selections) / 5.0)
        diversity_score += selection_diversity * 0.1
        
        return min(1.0, diversity_score)  # Normalizza tra 0 e 1
    
    def _get_adaptive_learning_rate(self, server_round: int) -> float:
        """
        Calcola learning rate globale adattivo basato sul progresso del training.
        """
        base_lr = 0.001  # Learning rate di base
        
        if len(self.round_history) < 2:
            return base_lr
        
        # Calcola trend della performance globale
        recent_rounds = self.round_history[-5:]  # Ultimi 5 round
        if len(recent_rounds) >= 2:
            performances = []
            for round_info in recent_rounds:
                if 'weighted_accuracy' in round_info['metrics']:
                    performances.append(round_info['metrics']['weighted_accuracy'])
                elif 'weighted_val_accuracy' in round_info['metrics']:
                    performances.append(round_info['metrics']['weighted_val_accuracy'])
            
            if len(performances) >= 2:
                # Se la performance sta migliorando, mantieni LR
                # Se sta peggiorando o stagnando, riduci LR
                recent_trend = performances[-1] - performances[0]
                if recent_trend < 0.001:  # Stagnazione o peggioramento
                    return base_lr * self.adaptive_lr_factor
        
        return base_lr
    
    def _get_client_specific_lr(self, client_id: str, server_round: int) -> float:
        """
        Calcola learning rate specifico per un client.
        """
        base_lr = self._get_adaptive_learning_rate(server_round)
        
        if client_id not in self.client_stats:
            return base_lr
        
        # Fattore di aggiustamento basato su:
        # 1. Se è outlier (LR più basso)
        # 2. Performance trend (LR più basso se performance in declino)
        
        lr_factor = 1.0
        
        # Aggiustamento per outlier
        if self.client_stats[client_id]['is_outlier']:
            lr_factor *= 0.8  # LR ridotto per outlier
        
        # Aggiustamento basato su performance trend
        performance_history = self.client_stats[client_id]['performance_history']
        if len(performance_history) >= 3:
            recent_trend = performance_history[-1] - performance_history[-3]
            if recent_trend < -0.05:  # Performance in declino
                lr_factor *= 0.7
            elif recent_trend > 0.05:  # Performance in miglioramento
                lr_factor *= 1.1
        
        return base_lr * lr_factor
    
    def _get_client_id(self, client: ClientProxy) -> str:
        """
        Estrae l'ID del client dal ClientProxy.
        """
        # Assume che l'ID del client sia nel campo cid
        if hasattr(client, 'cid'):
            return str(client.cid)
        else:
            # Fallback: usa rappresentazione string
            return str(client)
    
    def _print_aggregation_summary(
        self, 
        server_round: int, 
        weights: Dict[str, float], 
        metrics: Dict[str, Scalar]
    ):
        """
        Stampa un riepilogo dell'aggregazione bilanciata.
        """
        print(f"   📊 Pesi client bilanciati:")
        for client_id, weight in sorted(weights.items()):
            outlier_marker = " [OUTLIER]" if client_id in self.outlier_clients else ""
            print(f"      Client {client_id}: {weight:.3f}{outlier_marker}")
        
        print(f"   📈 Metriche aggregate:")
        key_metrics = ['weighted_accuracy', 'weighted_f1_score', 'weight_variance']
        for metric_name in key_metrics:
            if metric_name in metrics:
                print(f"      {metric_name}: {metrics[metric_name]:.4f}")
        
        print(f"   🎯 Statistiche bilanciamento:")
        print(f"      Outlier attivi: {len(self.outlier_clients)}")
        print(f"      Varianza pesi: {metrics.get('weight_variance', 0):.4f}")
        print(f"      Client effettivi: {metrics.get('effective_clients', len(weights))}")
    
    def get_strategy_metrics(self) -> Dict:
        """
        Restituisce metriche complete sulla strategia bilanciata.
        """
        return {
            'strategy_type': self.balance_strategy,
            'aggregation_metrics': self.aggregation_metrics,
            'client_stats': self.client_stats,
            'outlier_clients': list(self.outlier_clients),
            'round_history': self.round_history[-10:],  # Ultimi 10 round
            'total_rounds': len(self.round_history),
            'avg_weight_variance': np.mean([r['metrics'].get('weight_variance', 0) 
                                          for r in self.round_history[-5:]]) if self.round_history else 0
        }
    
    def save_strategy_state(self, filepath: str):
        """
        Salva lo stato della strategia su file per analisi successive.
        """
        strategy_state = {
            'config': {
                'balance_strategy': self.balance_strategy,
                'outlier_penalty': self.outlier_penalty,
                'adaptive_lr_factor': self.adaptive_lr_factor,
                'min_client_weight': self.min_client_weight,
                'max_client_weight': self.max_client_weight,
                'diversity_threshold': self.diversity_threshold,
                'use_client_selection': self.use_client_selection,
                'history_window': self.history_window
            },
            'metrics': self.get_strategy_metrics(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(strategy_state, f, indent=2, default=str)
        
        if self.verbose:
            print(f"💾 Stato strategia salvato: {filepath}")


# Factory functions per creare strategie pre-configurate

def create_class_weighted_strategy(**kwargs) -> BalancedFedAvg:
    """Crea strategia bilanciata basata su distribuzione classi."""
    return BalancedFedAvg(
        balance_strategy="class_weighted",
        use_client_selection=False,
        **kwargs
    )

def create_outlier_penalty_strategy(**kwargs) -> BalancedFedAvg:
    """Crea strategia con penalizzazione outlier."""
    return BalancedFedAvg(
        balance_strategy="outlier_penalty",
        outlier_penalty=0.5,
        use_client_selection=True,
        **kwargs
    )

def create_adaptive_strategy(**kwargs) -> BalancedFedAvg:
    """Crea strategia adattiva basata su performance."""
    return BalancedFedAvg(
        balance_strategy="adaptive",
        adaptive_lr_factor=0.8,
        use_client_selection=True,
        **kwargs
    )

def create_hybrid_strategy(**kwargs) -> BalancedFedAvg:
    """Crea strategia ibrida completa (raccomandato)."""
    return BalancedFedAvg(
        balance_strategy="hybrid",
        outlier_penalty=0.6,
        adaptive_lr_factor=0.8,
        use_client_selection=True,
        diversity_threshold=0.3,
        **kwargs
    )

def create_smartgrid_optimized_strategy(**kwargs) -> BalancedFedAvg:
    """
    Crea strategia ottimizzata specificatamente per SmartGrid basata 
    sui risultati dell'analisi di distribuzione.
    """
    return BalancedFedAvg(
        balance_strategy="hybrid",
        fraction_fit=0.8,              # Usa 80% dei client per round
        min_fit_clients=3,             # Minimo 3 client per diversità
        outlier_penalty=0.7,           # Penalizzazione moderata outlier
        adaptive_lr_factor=0.85,       # LR adattivo conservativo  
        min_client_weight=0.05,        # Peso minimo più basso per inclusività
        max_client_weight=2.5,         # Peso massimo moderato
        diversity_threshold=0.25,      # Soglia diversità adatta a SmartGrid
        use_client_selection=True,     # Abilita selezione intelligente
        history_window=8,              # Finestra storia appropriata
        verbose=True,                  # Logging dettagliato per debug
        **kwargs
    )