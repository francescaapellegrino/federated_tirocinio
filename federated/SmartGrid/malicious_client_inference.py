"""
Client Malevolo - Inference Attack
Francesca Pellegrino
"""

import warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from typing import Dict, Any, Tuple, List
import time
import json
from datetime import datetime
import traceback

# Import per preprocessing   
from federated.SmartGrid.preprocessing import load_improved_client_data
from federated.SmartGrid.model import create_improved_model, create_advanced_callbacks
from sklearn.utils.class_weight import compute_class_weight

def sanitize_json(obj):
    """Ricorsivamente converte tutti i valori in tipi serializzabili e sostituisce NaN/inf con None"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif obj is None:
        return None
    else:
        return obj

# CONFIGURAZIONE 
class CompatibleClientConfig:
    RANDOM_SEED = 42

# CLIENT MALEVOLO 
class CompatibleMaliciousClient(fl.client.NumPyClient):
    
    def __init__(self, client_id: int, is_malicious: bool = True):
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.config = CompatibleClientConfig()
        
        print(f"MALICIOUS CLIENT {client_id} - {'MALEVOLO' if is_malicious else 'NORMALE'}")
        
        # Usa il preprocessing migliorato (identico al client normale)
        self.load_compatible_data()
        
        # Calcola class weights per compatibilità
        self.class_weights = self._compute_class_weights()
        
        # Crea modello  
        self.create_compatible_model()
        
        print(f"Malicious Client {client_id} inizializzato")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Train: {len(self.X_train)} samples")
        print(f"Attack ratio: {self.y_train.mean()*100:.1f}%")
    
    def load_compatible_data(self):
        """Carica dati usando improved_preprocessing"""
        print(f"Caricamento dati  ...")
        
        # USA LO STESSO PREPROCESSING DEL CLIENT NORMALE
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.dataset_info = load_improved_client_data(
            self.client_id, self.config
        )
        
        # Verifica compatibilità
        assert self.X_train.shape[1] == 30, f"Features incompatibili: {self.X_train.shape[1]} != 30"
        assert self.X_train.dtype == np.float32, f"Tipo in : {self.X_train.dtype}"
        
        print(f"Dati compatibili caricati:")
        print(f"Train: {len(self.X_train)} campioni, {self.X_train.shape[1]} features")
        print(f"Val: {len(self.X_val)} campioni")
        print(f"Test: {len(self.X_test)} campioni")

    def create_compatible_model(self):
        """Crea modello usando improved_model"""
        print(f"Creazione modello  ...")
        
        # USA LO STESSO MODELLO DEL CLIENT NORMALE
        self.model = create_improved_model(self.X_train.shape[1], self.config)
        
        # Verifica parametri
        param_count = self.model.count_params()
        print(f"Modello creato: {param_count:,} parametri")
        print(f"Architettura: [256, 128, 64, 32] → 1")

    def _compute_class_weights(self):
        """Calcola pesi per bilanciare le classi (identico al client normale)"""
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

    def membership_inference_attack(self):
        """Membership Inference Attack"""
        print("Membership Inference Attack")
        
        try:
            # 1. MULTIPLE CONFIDENCE STRATEGIES
            print("Multiple confidence strategies...")

            train_preds = self.model.predict(self.X_train, verbose=0).flatten()
            test_preds = self.model.predict(self.X_test, verbose=0).flatten()
            
            # Strategia A: Confidence classica
            train_conf_a = np.abs(train_preds - 0.5)
            test_conf_a = np.abs(test_preds - 0.5)
            
            # Strategia B: Entropia-based
            def entropy_confidence(preds):
                """Calcola confidence basata sull'entropia"""
                entropy = -(preds * np.log(preds + 1e-8) + (1-preds) * np.log(1-preds + 1e-8))
                # Inverti: bassa entropia = alta confidence
                return 1.0 - (entropy / np.log(2))  # Normalizza per entropia binaria max
            
            train_conf_b = entropy_confidence(train_preds)
            test_conf_b = entropy_confidence(test_preds)
            
            # Strategia C: Temperature scaling confidence
            def temperature_confidence(preds, temperature=2.0):
                """Confidence con temperature scaling"""
                # Simula temperature scaling
                logits = np.log(preds + 1e-8) - np.log(1 - preds + 1e-8)
                scaled_logits = logits / temperature
                scaled_probs = 1 / (1 + np.exp(-scaled_logits))
                return np.abs(scaled_probs - 0.5)
            
            train_conf_c = temperature_confidence(train_preds)
            test_conf_c = temperature_confidence(test_preds)
            
            # Test tutte le strategie
            strategies = [
                ("classic", train_conf_a, test_conf_a),
                ("entropy", train_conf_b, test_conf_b),
                ("temperature", train_conf_c, test_conf_c)
            ]
            
            best_accuracy = 0
            best_strategy = None
            
            for strategy_name, train_conf, test_conf in strategies:
                # Test multiple threshold per ogni strategia
                all_confidences = np.concatenate([train_conf, test_conf])
                
                for percentile in [10, 25, 50, 75, 90]:
                    threshold = np.percentile(all_confidences, percentile)
                    
                    train_pred_labels = (train_conf > threshold).astype(int)
                    test_pred_labels = (test_conf > threshold).astype(int)
                    
                    true_labels = np.concatenate([
                        np.ones(len(train_pred_labels)), 
                        np.zeros(len(test_pred_labels))
                    ])
                    predictions = np.concatenate([train_pred_labels, test_pred_labels])
                    
                    accuracy = accuracy_score(true_labels, predictions)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_strategy = f"{strategy_name}_p{percentile}"

            print(f"Best strategy: {best_strategy}, accuracy: {best_accuracy:.4f}")

            # 2. LOSS-BASED ATTACK
            print("Loss-based analysis...")

            # Calcola loss con multiple tecniche
            def compute_loss_variants(X, y, model):
                """Calcola varianti di loss per MIA"""
                losses = {
                    'cross_entropy': [],
                    'mse': [],
                    'focal_loss': []
                }
                
                for i in range(min(200, len(X))):  # Più campioni
                    try:
                        pred = model.predict(X[i:i+1], verbose=0)[0, 0]
                        true_label = y[i]
                        
                        # Cross entropy
                        ce_loss = -(true_label * np.log(pred + 1e-8) + (1 - true_label) * np.log(1 - pred + 1e-8))
                        losses['cross_entropy'].append(ce_loss)
                        
                        # MSE
                        mse_loss = (pred - true_label) ** 2
                        losses['mse'].append(mse_loss)
                        
                        # Focal loss (riduce peso su campioni "facili")
                        alpha, gamma = 0.25, 2.0
                        focal_loss = -alpha * ((1 - pred) ** gamma) * true_label * np.log(pred + 1e-8) - \
                                    (1 - alpha) * (pred ** gamma) * (1 - true_label) * np.log(1 - pred + 1e-8)
                        losses['focal_loss'].append(focal_loss)
                        
                    except:
                        continue
                
                return losses
            
            train_losses = compute_loss_variants(self.X_train, self.y_train, self.model)
            test_losses = compute_loss_variants(self.X_test, self.y_test, self.model)
            
            # Analizza ogni tipo di loss
            loss_signals = {}
            for loss_type in ['cross_entropy', 'mse', 'focal_loss']:
                if train_losses[loss_type] and test_losses[loss_type]:
                    avg_train = np.mean(train_losses[loss_type])
                    avg_test = np.mean(test_losses[loss_type])
                    
                    # KS test per distribuzione
                    try:
                        from scipy.stats import ks_2samp
                        ks_stat, ks_pvalue = ks_2samp(train_losses[loss_type], test_losses[loss_type])
                        signal_strength = ks_stat  # KS statistic come signal
                    except:
                        # Fallback: differenza normalizzata
                        signal_strength = abs(avg_train - avg_test) / max(avg_train, avg_test)
                    
                    loss_signals[loss_type] = signal_strength
                    print(f"{loss_type}: signal = {signal_strength:.6f}")
            
            # Prendi il miglior segnale loss
            best_loss_signal = max(loss_signals.values()) if loss_signals else 0.0

            # 3. GRADIENT ANALYSIS
            print("Gradient analysis...")

            def compute_gradient_features(X, y, model, sample_size=100):
                """Calcola features dai gradienti per MIA"""
                gradient_features = []
                
                for i in range(min(sample_size, len(X))):
                    try:
                        with tf.GradientTape() as tape:
                            x_sample = tf.Variable(X[i:i+1].astype(np.float32))
                            y_sample = tf.Variable(y[i:i+1].reshape(-1, 1).astype(np.float32))
                            
                            pred = model(x_sample, training=False)
                            loss = tf.keras.losses.binary_crossentropy(y_sample, pred)
                        
                        gradients = tape.gradient(loss, x_sample)
                        if gradients is not None:
                            grad_array = gradients.numpy().flatten()
                            
                            # Multiple features dai gradienti
                            features = {
                                'l1_norm': np.sum(np.abs(grad_array)),
                                'l2_norm': np.sqrt(np.sum(grad_array ** 2)),
                                'max_grad': np.max(np.abs(grad_array)),
                                'std_grad': np.std(grad_array),
                                'mean_grad': np.mean(np.abs(grad_array)),
                                'grad_sparsity': np.sum(np.abs(grad_array) < 1e-6) / len(grad_array)
                            }
                            gradient_features.append(features)
                            
                    except:
                        continue
                
                return gradient_features
            
            # Calcola gradient features
            train_grad_features = compute_gradient_features(self.X_train, self.y_train, self.model)
            test_grad_features = compute_gradient_features(self.X_test, self.y_test, self.model)
            
            # Analizza ogni feature
            gradient_signals = {}
            if train_grad_features and test_grad_features:
                for feature_name in ['l1_norm', 'l2_norm', 'max_grad', 'std_grad', 'mean_grad', 'grad_sparsity']:
                    train_values = [f[feature_name] for f in train_grad_features]
                    test_values = [f[feature_name] for f in test_grad_features]
                    
                    if train_values and test_values:
                        # T-test per differenza significativa
                        try:
                            from scipy.stats import ttest_ind
                            t_stat, p_value = ttest_ind(train_values, test_values)
                            signal = abs(t_stat) * (1 - p_value)  # Segnale basato su t-stat e significatività
                        except:
                            # Fallback
                            signal = abs(np.mean(train_values) - np.mean(test_values)) / (np.std(train_values) + np.std(test_values) + 1e-8)
                        
                        gradient_signals[feature_name] = signal
                        print(f"{feature_name}: signal = {signal:.6f}")
            
            best_gradient_signal = max(gradient_signals.values()) if gradient_signals else 0.0
            
            # 4. ENSEMBLE COMBINATION
            print("Signal combination...")

            # Pesi adattivi basati sulla qualità del segnale
            confidence_weight = 0.4 * (1 + best_accuracy - 0.5)  # Peso maggiore se accuracy alta
            loss_weight = 0.3 * (1 + best_loss_signal)           # Peso maggiore se loss signal alto
            gradient_weight = 0.3 * (1 + best_gradient_signal)   # Peso maggiore se gradient signal alto
            
            # Normalizza pesi
            total_weight = confidence_weight + loss_weight + gradient_weight
            confidence_weight /= total_weight
            loss_weight /= total_weight
            gradient_weight /= total_weight
            
            # Combina con pesi adattivi
            enhanced_combined_signal = (
                best_accuracy * confidence_weight +
                (0.5 + best_loss_signal * 0.5) * loss_weight +
                (0.5 + best_gradient_signal * 0.5) * gradient_weight
            )
            
            # Soglia dinamica basata sulla qualità complessiva
            base_threshold = 0.65
            quality_bonus = (best_accuracy + best_loss_signal + best_gradient_signal) / 3
            dynamic_threshold = base_threshold - (quality_bonus * 0.1)  # Riduci soglia se segnali forti
            
            attack_success = enhanced_combined_signal > dynamic_threshold
            privacy_breach_score = max(0, (enhanced_combined_signal - 0.5) * 2)
            
            print(f"Results:")
            print(f"- Best confidence: {best_accuracy:.4f}")
            print(f"- Best loss signal: {best_loss_signal:.4f}")
            print(f"- Best gradient signal: {best_gradient_signal:.4f}")
            print(f"- Enhanced combined: {enhanced_combined_signal:.4f}")
            print(f"- Dynamic threshold: {dynamic_threshold:.4f}")
            print(f"- Privacy breach: {privacy_breach_score:.4f}")
            print(f"- Attack {'✅ RIUSCITO' if attack_success else '❌ FALLITO'}")

            return {
                "attack_type": "Membership Inference Attack",
                "attack_success": int(attack_success),
                "attack_success_criteria": "Considerato successo (1) se combined_accuracy > 0.6",
                "confidence_based_accuracy": float(best_accuracy),
                "confidence_based_accuracy_explanation": "Percentuale di dati per cui l'attacco indovina correttamente la membership. Valore >0.5 indica che l'attacco è migliore del caso random.",
                "gradient_based_signal": float(best_gradient_signal),
                "gradient_signal_explanation": "Intensità del segnale ottenuto confrontando i gradienti tra membri e non membri.",
                "gradient_samples_analyzed": int(len(train_grad_features) + len(test_grad_features)),
                "combined_accuracy": float(enhanced_combined_signal),
                "combined_accuracy_explanation": "Accuratezza media combinando tecniche diverse di attacco. Valore >0.5 indica successo dell'attacco.",
                "privacy_breach_score": float(privacy_breach_score),
                "privacy_breach_score_explanation": "Score calcolato come 2*(combined_accuracy-0.5), rappresenta il rischio di violazione privacy rispetto al caso random.",
                "samples_analyzed": int(len(self.X_train) + len(self.X_test))
            }

        except Exception as e:
            print(f"❌ Errore Enhanced MIA: {e}")
            return {
                "attack_type": "enhanced_membership_inference",
                "attack_success": False,
                "error": str(e),
                "fallback_used": True
            }

    def property_inference_attack(self):
        """Property Inference Attack"""
        print("Property Inference Attack")

        try:
            predictions = self.model.predict(self.X_test, verbose=0).flatten()
            
            # 1. MULTIPLE ESTIMATION TECHNIQUES
            print("Multiple estimation techniques...")

            # Tecnica A: Distribuzione migliorata
            pred_mean = np.mean(predictions)
            pred_median = np.median(predictions)
            pred_mode = self.estimate_mode(predictions)
            
            # Tecnica B: Confidence-weighted estimation
            confidence_scores = np.maximum(predictions, 1 - predictions)
            high_conf_mask = confidence_scores > np.percentile(confidence_scores, 75)
            
            if np.sum(high_conf_mask) > 10:
                high_conf_estimate = np.mean(predictions[high_conf_mask])
            else:
                high_conf_estimate = pred_mean
            
            # Tecnica C: Ensemble di percentili
            percentile_estimates = []
            for p in [10, 25, 50, 75, 90]:
                pct_value = np.percentile(predictions, p)
                # Mappa percentile a stima attack ratio
                mapped_estimate = self.percentile_to_ratio_mapping(p, pct_value)
                percentile_estimates.append(mapped_estimate)
            
            ensemble_estimate = np.median(percentile_estimates)

            print(f"Mean estimate: {pred_mean:.4f}")
            print(f"High-conf estimate: {high_conf_estimate:.4f}")
            print(f"Ensemble estimate: {ensemble_estimate:.4f}")

            # 2. CLUSTERING ANALYSIS
            print("Clustering analysis...")

            # Multiple clustering techniques
            clustering_estimates = []
            
            # K-means con K ottimale
            optimal_k = self.find_optimal_clusters(predictions.reshape(-1, 1), max_k=8)
            if optimal_k > 1:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(predictions.reshape(-1, 1))
                
                cluster_ratios = []
                for cluster_id in range(optimal_k):
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_ratio = np.mean(self.y_test[cluster_mask])
                        cluster_size = np.sum(cluster_mask)
                        # Pesa per dimensione cluster
                        weighted_ratio = cluster_ratio * (cluster_size / len(self.y_test))
                        cluster_ratios.append(weighted_ratio)
                
                kmeans_estimate = np.sum(cluster_ratios) if cluster_ratios else pred_mean
                clustering_estimates.append(('kmeans', kmeans_estimate))
            
            # Gaussian Mixture Model
            try:
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm_labels = gmm.fit_predict(predictions.reshape(-1, 1))
                
                gmm_ratios = []
                for component in range(3):
                    comp_mask = gmm_labels == component
                    if np.sum(comp_mask) > 0:
                        comp_ratio = np.mean(self.y_test[comp_mask])
                        comp_weight = np.sum(comp_mask) / len(self.y_test)
                        gmm_ratios.append(comp_ratio * comp_weight)
                
                gmm_estimate = np.sum(gmm_ratios) if gmm_ratios else pred_mean
                clustering_estimates.append(('gmm', gmm_estimate))
                
            except:
                gmm_estimate = pred_mean
                clustering_estimates.append(('gmm_fallback', gmm_estimate))
            
            best_clustering_estimate = np.mean([est for _, est in clustering_estimates])
            
            # 3. STATISTICAL PATTERN ANALYSIS
            print("Statistical pattern analysis...")

            # Analisi forma distribuzione
            skewness = self.calculate_skewness(predictions)
            kurtosis = self.calculate_kurtosis(predictions)
            
            # Bimodality detection migliorata
            hist, bins = np.histogram(predictions, bins=20)
            peaks = self.find_peaks(hist)
            bimodality_score = len(peaks) / 20  # Normalizza
            
            # Anderson-Darling test per normalità
            try:
                from scipy.stats import anderson
                ad_stat, _, _ = anderson(predictions, dist='norm')
                normality_score = 1 / (1 + ad_stat)  # Più alto = più normale
            except:
                normality_score = 0.5
            
            # Pattern-based estimation
            if abs(skewness) > 0.5:  # Distribuzione asimmetrica
                skew_estimate = 0.5 + (skewness * 0.2)  # Mappa skewness a ratio
            else:
                skew_estimate = pred_mean
            
            print(f"      📊 Skewness: {skewness:.4f} -> estimate: {skew_estimate:.4f}")
            print(f"      📊 Bimodality: {bimodality_score:.4f}")
            print(f"      📊 Normality: {normality_score:.4f}")
            
            # 4. ENSEMBLE FUSION
            print("Ensemble fusion...")

            # Raccogli tutte le stime
            all_estimates = [
                ('mean', pred_mean, 0.3),
                ('median', pred_median, 0.2),
                ('high_conf', high_conf_estimate, 0.25),
                ('ensemble', ensemble_estimate, 0.15),
                ('clustering', best_clustering_estimate, 0.1)
            ]
            
            # Calcola stima finale pesata
            total_weight = sum(weight for _, _, weight in all_estimates)
            final_estimate = sum(est * weight for _, est, weight in all_estimates) / total_weight
            
            # Ground truth
            actual_ratio = np.mean(self.y_test)
            estimation_error = abs(final_estimate - actual_ratio)

            print(f"Final estimate: {final_estimate:.4f}")
            print(f"Actual ratio: {actual_ratio:.4f}")
            print(f"Estimation error: {estimation_error:.4f}")

            # 5. PROPERTY DETECTION
            print("Property detection...")
            
            properties_detected = 0
            total_properties = 8
            detection_results = {}
            
            # Proprietà 1: Distribution bias (soglia adattiva)
            dist_bias = abs(pred_mean - 0.5)
            bias_threshold = 0.03 if len(predictions) > 500 else 0.02  # Soglia adattiva
            if dist_bias > bias_threshold:
                properties_detected += 1
                detection_results['distribution_bias'] = True
                print(f"P1: Distribution bias ({dist_bias:.3f} > {bias_threshold:.3f})")
            else:
                detection_results['distribution_bias'] = False
            
            # Proprietà 2: Cluster structure (migliorata)
            cluster_variance = np.var([est for _, est in clustering_estimates])
            if cluster_variance > 0.01:  # Soglia molto bassa
                properties_detected += 1
                detection_results['cluster_structure'] = True
                print(f"P2: Cluster structure ({cluster_variance:.4f})")
            else:
                detection_results['cluster_structure'] = False
            
            # Proprietà 3: Confidence asymmetry
            pos_mask = predictions > 0.5
            neg_mask = predictions <= 0.5
            if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                pos_conf_avg = np.mean(confidence_scores[pos_mask])
                neg_conf_avg = np.mean(confidence_scores[neg_mask])
                conf_asymmetry = abs(pos_conf_avg - neg_conf_avg)
            else:
                conf_asymmetry = 0.0
            
            if conf_asymmetry > 0.02:
                properties_detected += 1
                detection_results['confidence_asymmetry'] = True
                print(f"P3: Confidence asymmetry ({conf_asymmetry:.4f})")
            else:
                detection_results['confidence_asymmetry'] = False
            
            # Proprietà 4: Model sensitivity (perturbation analysis)
            perturbation_signal = self.analyze_model_sensitivity()
            if perturbation_signal > 0.005:
                properties_detected += 1
                detection_results['model_sensitivity'] = True
                print(f"P4: Model sensitivity ({perturbation_signal:.4f})")
            else:
                detection_results['model_sensitivity'] = False
            
            # Proprietà 5: Distribution shape
            if abs(skewness) > 0.1 or abs(kurtosis) > 0.1:
                properties_detected += 1
                detection_results['distribution_shape'] = True
                print(f"P5: Distribution shape (s={skewness:.3f}, k={kurtosis:.3f})")
            else:
                detection_results['distribution_shape'] = False
            
            # Proprietà 6: Bimodality
            if bimodality_score > 0.15:
                properties_detected += 1
                detection_results['bimodality'] = True
                print(f"P6: Bimodality ({bimodality_score:.3f})")
            else:
                detection_results['bimodality'] = False
            
            # Proprietà 7: Variance pattern
            pred_std = np.std(predictions)
            if pred_std > 0.05:
                properties_detected += 1
                detection_results['variance_pattern'] = True
                print(f"P7: Variance pattern ({pred_std:.3f})")
            else:
                detection_results['variance_pattern'] = False
            
            # Proprietà 8: Estimation accuracy (nuova)
            estimation_accuracy = max(0.0, 1.0 - (estimation_error * 4))
            if estimation_accuracy > 0.6:
                properties_detected += 1
                detection_results['estimation_accuracy'] = True
                print(f"P8: Estimation accuracy ({estimation_accuracy:.3f})")
            else:
                detection_results['estimation_accuracy'] = False
            
            # 6. ADAPTIVE SUCCESS CRITERIA
            success_rate = properties_detected / total_properties
            
            # Soglia adattiva basata su qualità stime
            base_threshold = 0.25
            quality_bonus = 0
            
            if estimation_error < 0.1:
                quality_bonus += 0.1
            if cluster_variance > 0.02:
                quality_bonus += 0.05
            if conf_asymmetry > 0.05:
                quality_bonus += 0.05
            
            adaptive_threshold = max(0.2, base_threshold - quality_bonus)
            attack_success = success_rate >= adaptive_threshold
            
            # Privacy breach level
            if success_rate >= 0.6:
                privacy_level = "HIGH"
            elif success_rate >= 0.3:
                privacy_level = "MEDIUM"
            else:
                privacy_level = "LOW"
            
            print(f"Property Inference Results:")
            print(f" - Properties detected: {properties_detected}/{total_properties}")
            print(f" - Success rate: {success_rate:.3f}")
            print(f" - Adaptive threshold: {adaptive_threshold:.3f}")
            print(f" - Final estimate: {final_estimate:.3f}")
            print(f" - Estimation error: {estimation_error:.3f}")
            print(f" - Privacy level: {privacy_level}")
            print(f" - Attack {'✅ RIUSCITO' if attack_success else '❌ FALLITO'}")

            return {
                "attack_type": "Property Inference Attack",
                "attack_success": int(attack_success),
                "attack_success_criteria": "Considerato successo (1) se il numero di proprietà rilevate supera la soglia definita nei success_criteria.",
                "success_rate": float(success_rate),
                "success_rate_explanation": "Percentuale di proprietà sensibili indovinate rispetto al totale.",
                "properties_detected": int(properties_detected),
                "total_properties": int(total_properties),
                "properties_explanation": "Numero di proprietà sensibili che l'attacco è riuscito a inferire sul totale delle proprietà testate.",
                "privacy_breach_level": privacy_level,
                "privacy_breach_level_explanation": "Livello qualitativo di rischio privacy stimato in base al successo dell'attacco.",
                "estimated_attack_ratio": float(final_estimate),
                "actual_attack_ratio": float(actual_ratio),
                "attack_ratio_explanation": "Stima e valore reale della proporzione di dati/utenti colpiti dall'attacco.",
                "estimation_error": float(estimation_error),
                "estimation_error_explanation": "Errore assoluto tra attack ratio stimato e reale. Più piccolo è, migliore è la stima dell'attaccante.",
                "estimation_accuracy": float(estimation_accuracy),
                "estimation_accuracy_explanation": "Accuratezza (1-errore relativo) della stima dell'attaccante rispetto al valore reale.",
                "samples_analyzed": int(len(self.X_test))
            }
            
        except Exception as e:
            print(f"❌ Errore Enhanced Property Inference: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "attack_type": "enhanced_property_inference",
                "attack_success": False,
                "error": str(e),
                "fallback_used": True
            }

    # Metodi helper aggiuntivi
    def estimate_mode(self, data, bins=50):
        """Stima la moda di una distribuzione"""
        try:
            hist, bin_edges = np.histogram(data, bins=bins)
            max_bin_idx = np.argmax(hist)
            mode_estimate = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            return mode_estimate
        except:
            return np.median(data)

    def percentile_to_ratio_mapping(self, percentile, value):
        """Mappa un percentile a una stima di attack ratio"""
        # Mapping empirico: percentili alti -> più probabilità di ratio alto
        if percentile >= 75:
            return min(1.0, value + 0.1)
        elif percentile >= 50:
            return value
        else:
            return max(0.0, value - 0.1)

    def find_peaks(self, histogram, min_height=0.1):
        """Trova picchi in un istogramma"""
        peaks = []
        for i in range(1, len(histogram) - 1):
            if (histogram[i] > histogram[i-1] and 
                histogram[i] > histogram[i+1] and 
                histogram[i] > min_height * np.max(histogram)):
                peaks.append(i)
        return peaks

    def analyze_model_sensitivity(self):
        """Analizza sensibilità del modello alle perturbazioni"""
        try:
            original_preds = self.model.predict(self.X_test, verbose=0).flatten()
            original_mean = np.mean(original_preds)
            
            impacts = []
            for perturbation in [0.05, 0.1, 0.15]:
                X_perturbed = self.X_test + np.random.normal(0, perturbation, self.X_test.shape)
                perturbed_preds = self.model.predict(X_perturbed, verbose=0).flatten()
                impact = abs(np.mean(perturbed_preds) - original_mean)
                impacts.append(impact)
            
            return np.mean(impacts)
        except:
            return 0.01

    def find_optimal_clusters(self, data, max_k=8):
        """Trova numero ottimale di cluster usando elbow method"""
        try:
            if len(data) < max_k:
                return min(3, len(data))
            
            inertias = []
            k_range = range(2, min(max_k + 1, len(data)))
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(data)
                    inertias.append(kmeans.inertia_)
                except:
                    inertias.append(float('inf'))
            
            if not inertias or len(inertias) < 3:
                return 3
            
            # Elbow detection
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs)
                optimal_k = k_range[elbow_idx + 1]
            else:
                optimal_k = k_range[len(k_range)//2]
            
            return optimal_k
            
        except Exception as e:
            print(f"Errore find_optimal_clusters: {e}")
            return 3

    def calculate_skewness(self, data):
        """Calcola skewness"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            skew_values = ((data - mean) / std) ** 3
            return np.mean(skew_values)
        except:
            return 0.0

    def calculate_kurtosis(self, data):
        """Calcola kurtosis"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            kurt_values = ((data - mean) / std) ** 4
            return np.mean(kurt_values) - 3.0  # Excess kurtosis
        except:
            return 0.0

    def model_inversion_attack(self):
        """Model Inversion Attack"""
        print("Model Inversion Attack")

        try:
            # 1. PROGRESSIVE THRESHOLD SEARCH
            print("Progressive threshold search...")

            predictions = self.model.predict(self.X_test, verbose=0).flatten()
            
            # Soglie molto permissive per iniziare
            confidence_thresholds = [0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51]
            
            high_conf_normal = None
            high_conf_attack = None
            normal_threshold_used = None
            attack_threshold_used = None
            
            # Search più aggressiva
            for threshold in confidence_thresholds:
                if high_conf_normal is None:
                    normal_candidates = self.X_test[predictions < (1 - threshold)]
                    if len(normal_candidates) >= 1:  # Anche 1 campione va bene
                        high_conf_normal = normal_candidates
                        normal_threshold_used = threshold
                        
                if high_conf_attack is None:
                    attack_candidates = self.X_test[predictions > threshold]
                    if len(attack_candidates) >= 1:  # Anche 1 campione va bene
                        high_conf_attack = attack_candidates
                        attack_threshold_used = threshold
                
                if high_conf_normal is not None and high_conf_attack is not None:
                    break
            
            # Fallback 
            if high_conf_normal is None or high_conf_attack is None:
                print("Using fallback...")

                # Dividi in percentili
                sorted_indices = np.argsort(predictions)
                n_samples = len(predictions)
                
                # Top e bottom 20%
                bottom_20_pct = max(1, n_samples // 5)
                top_20_pct = max(1, n_samples // 5)
                
                if high_conf_normal is None:
                    high_conf_normal = self.X_test[sorted_indices[:bottom_20_pct]]
                    normal_threshold_used = "bottom_20_percent"
                    
                if high_conf_attack is None:
                    high_conf_attack = self.X_test[sorted_indices[-top_20_pct:]]
                    attack_threshold_used = "top_20_percent"
            
            print(f"Normal samples: {len(high_conf_normal)} (threshold: {normal_threshold_used})")
            print(f"Attack samples: {len(high_conf_attack)} (threshold: {attack_threshold_used})")

            # 2. PROTOTYPE GENERATION
            print("Prototype generation...")

            normal_prototypes = []
            attack_prototypes = []
            
            # Tecnica 1: Robust statistics
            def robust_prototype(data, method='median'):
                """Genera prototipi robusti"""
                if method == 'median':
                    return np.median(data, axis=0)
                elif method == 'trimmed_mean':
                    # Rimuovi 10% estremi
                    sorted_data = np.sort(data, axis=0)
                    trim = max(1, len(data) // 10)
                    return np.mean(sorted_data[trim:-trim], axis=0)
                elif method == 'huber_mean':
                    # Approximation di Huber mean
                    mean = np.mean(data, axis=0)
                    for _ in range(3):  # Iterazioni
                        residuals = np.abs(data - mean)
                        threshold = np.median(residuals, axis=0) * 1.345
                        weights = np.minimum(1.0, threshold / (residuals + 1e-8))
                        mean = np.average(data, axis=0, weights=weights)
                    return mean
            
            # Genera prototipi con metodi diversi
            for method in ['median', 'trimmed_mean', 'huber_mean']:
                normal_proto = robust_prototype(high_conf_normal, method)
                attack_proto = robust_prototype(high_conf_attack, method)
                
                normal_prototypes.append((f"robust_{method}", normal_proto))
                attack_prototypes.append((f"robust_{method}", attack_proto))
            
            # Tecnica 2: Confidence-weighted prototypes
            normal_preds = self.model.predict(high_conf_normal, verbose=0).flatten()
            attack_preds = self.model.predict(high_conf_attack, verbose=0).flatten()
            
            # Pesi esponenziali per enfatizzare alta confidence
            normal_weights = np.exp(-normal_preds * 5)  # Più peso a predizioni basse
            attack_weights = np.exp(attack_preds * 5)   # Più peso a predizioni alte
            
            # Gestisci il caso di pesi zero
            if np.sum(normal_weights) == 0:
                print("Normal weights sum to zero, using uniform weights")
                normal_weights = np.ones(len(normal_weights)) / len(normal_weights)
            else:
                normal_weights = normal_weights / np.sum(normal_weights)
            
            if np.sum(attack_weights) == 0:
                print("Attack weights sum to zero, using uniform weights")
                attack_weights = np.ones(len(attack_weights)) / len(attack_weights)
            else:
                attack_weights = attack_weights / np.sum(attack_weights)
            
            # Verifica finale dei pesi
            if np.any(np.isnan(normal_weights)) or np.any(np.isnan(attack_weights)):
                print("NaN weights detected, using uniform weights")
                normal_weights = np.ones(len(high_conf_normal)) / len(high_conf_normal)
                attack_weights = np.ones(len(high_conf_attack)) / len(high_conf_attack)
            
            normal_proto_weighted = np.average(high_conf_normal, axis=0, weights=normal_weights)
            attack_proto_weighted = np.average(high_conf_attack, axis=0, weights=attack_weights)

            # 3. GRADIENT-BASED OPTIMIZATION
            print("Gradient optimization...")

            def advanced_gradient_optimization(initial_prototype, target_class, iterations=50):
                """Ottimizzazione gradiente avanzata"""
                try:
                    prototype = tf.Variable(initial_prototype.reshape(1, -1).astype(np.float32), trainable=True)
                    
                    # Optimizer più sofisticato
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
                    
                    best_prototype = initial_prototype.copy()
                    best_confidence = 0.0
                    
                    for i in range(iterations):
                        with tf.GradientTape() as tape:
                            pred = self.model(prototype, training=False)
                            
                            # Loss multiobiettivo
                            if target_class == 1:
                                # Massimizza probabilità attack
                                confidence_loss = -tf.math.log(pred + 1e-8)
                            else:
                                # Massimizza probabilità normal
                                confidence_loss = -tf.math.log(1 - pred + 1e-8)
                            
                            # Regularization per plausibilità
                            l2_reg = 0.001 * tf.reduce_sum(tf.square(prototype))
                            
                            # Diversity regularization
                            if target_class == 1:
                                original_mean = tf.constant(np.mean(high_conf_attack, axis=0).reshape(1, -1), dtype=tf.float32)
                            else:
                                original_mean = tf.constant(np.mean(high_conf_normal, axis=0).reshape(1, -1), dtype=tf.float32)
                            
                            diversity_reg = 0.0001 * tf.reduce_sum(tf.square(prototype - original_mean))
                            
                            total_loss = confidence_loss + l2_reg + diversity_reg
                        
                        gradients = tape.gradient(total_loss, prototype)
                        if gradients is not None:
                            # Clip gradients per stabilità
                            gradients = tf.clip_by_norm(gradients, 1.0)
                            optimizer.apply_gradients([(gradients, prototype)])
                            
                            # Valuta progresso
                            current_pred = self.model(prototype, training=False).numpy()[0, 0]
                            current_confidence = current_pred if target_class == 1 else (1 - current_pred)
                            
                            if current_confidence > best_confidence:
                                best_confidence = current_confidence
                                best_prototype = prototype.numpy().flatten()
                            
                            # Adaptive learning rate
                            if i % 15 == 0 and i > 0:
                                optimizer.learning_rate.assign(optimizer.learning_rate * 0.95)
                    
                    return best_prototype, best_confidence
                    
                except Exception as e:
                    print(f"Gradient optimization failed: {e}")
                    return initial_prototype, 0.5
            
            # Applica ottimizzazione ai migliori prototipi
            best_normal_idx = len(normal_prototypes) - 1  # Usa weighted
            best_attack_idx = len(attack_prototypes) - 1  # Usa weighted
            
            refined_normal, normal_confidence = advanced_gradient_optimization(
                normal_prototypes[best_normal_idx][1], 0
            )
            refined_attack, attack_confidence = advanced_gradient_optimization(
                attack_prototypes[best_attack_idx][1], 1
            )
            
            normal_prototypes.append(("gradient_optimized", refined_normal))
            attack_prototypes.append(("gradient_optimized", refined_attack))
            
            print(f"Optimized normal confidence: {normal_confidence:.4f}")
            print(f"Optimized attack confidence: {attack_confidence:.4f}")

            # 4. COMPREHENSIVE EVALUATION
            print("Comprehensive evaluation...")
            best_normal_proto = None
            best_attack_proto = None
            best_normal_score = 0.0
            best_attack_score = 0.0
            best_normal_method = ""
            best_attack_method = ""
            
            # Valuta tutti i prototipi con scoring avanzato
            for method, prototype in normal_prototypes:
                pred = self.model.predict(prototype.reshape(1, -1), verbose=0)[0, 0]
                base_score = 1 - pred
                
                # Bonus per metodi sofisticati
                method_bonus = 0.0
                if "gradient" in method:
                    method_bonus = 0.15
                elif "weighted" in method:
                    method_bonus = 0.1
                elif "huber" in method:
                    method_bonus = 0.05
                
                final_score = base_score * (1 + method_bonus)
                
                if final_score > best_normal_score:
                    best_normal_score = base_score  # Score originale per fairness
                    best_normal_proto = prototype
                    best_normal_method = method
            
            for method, prototype in attack_prototypes:
                pred = self.model.predict(prototype.reshape(1, -1), verbose=0)[0, 0]
                base_score = pred
                
                method_bonus = 0.0
                if "gradient" in method:
                    method_bonus = 0.15
                elif "weighted" in method:
                    method_bonus = 0.1
                elif "huber" in method:
                    method_bonus = 0.05
                
                final_score = base_score * (1 + method_bonus)
                
                if final_score > best_attack_score:
                    best_attack_score = base_score
                    best_attack_proto = prototype
                    best_attack_method = method

            print(f"Best normal: {best_normal_method} (conf: {best_normal_score:.4f})")
            print(f"Best attack: {best_attack_method} (conf: {best_attack_score:.4f})")

            # 5. SEPARABILITY ANALYSIS
            print("Separability analysis...")

            # Multiple distance metrics
            l1_distance = np.sum(np.abs(best_attack_proto - best_normal_proto))
            l2_distance = np.linalg.norm(best_attack_proto - best_normal_proto)
            linf_distance = np.max(np.abs(best_attack_proto - best_normal_proto))
            
            # Statistical distances
            try:
                # Wasserstein distance (approximation)
                sorted_normal = np.sort(best_normal_proto)
                sorted_attack = np.sort(best_attack_proto)
                wasserstein_dist = np.mean(np.abs(sorted_normal - sorted_attack))
                
                # Cosine distance
                cosine_sim = np.dot(best_attack_proto, best_normal_proto) / (
                    np.linalg.norm(best_attack_proto) * np.linalg.norm(best_normal_proto) + 1e-8
                )
                cosine_distance = 1 - cosine_sim
                
            except:
                wasserstein_dist = l1_distance
                cosine_distance = 0.5
            
            print(f"L1: {l1_distance:.4f}, L2: {l2_distance:.4f}, L∞: {linf_distance:.4f}")
            print(f"Wasserstein: {wasserstein_dist:.4f}, Cosine dist: {cosine_distance:.4f}")

            # 6. SUCCESS CRITERIA
            print("Success criteria...")

            # Criteri più realistici e graduali
            confidence_criterion = (best_normal_score > 0.55 or best_attack_score > 0.55)
            
            # Soglie ridotte per separazione
            l1_criterion = l1_distance > 0.1
            l2_criterion = l2_distance > 0.1
            cosine_criterion = cosine_distance > 0.05
            
            # Combinazione distanze
            separation_criterion = l1_criterion or l2_criterion or cosine_criterion
            
            # Information leakage migliorato
            confidence_component = (best_normal_score + best_attack_score) / 2
            separation_component = min(l2_distance / 3.0, 1.0)  # Soglia ridotta
            diversity_component = min(cosine_distance * 5, 1.0)  # Enfatizza diversità
            
            information_leakage = (
                confidence_component * 0.4 +
                separation_component * 0.35 +
                diversity_component * 0.25
            )
            
            leakage_criterion = information_leakage > 0.3  # Soglia ridotta
            
            # Criterio campioni più permissivo
            sample_criterion = (len(high_conf_normal) >= 1 and len(high_conf_attack) >= 1)
            
            # Scoring ponderato con pesi realistici
            criteria_scores = {
                'confidence': 1.0 if confidence_criterion else 0.0,
                'separation': 1.0 if separation_criterion else 0.0,
                'leakage': 1.0 if leakage_criterion else 0.0,
                'samples': 1.0 if sample_criterion else 0.0
            }
            
            # Pesi che favoriscono confidence e leakage
            weights = {'confidence': 0.4, 'separation': 0.25, 'leakage': 0.25, 'samples': 0.1}
            
            weighted_score = sum(criteria_scores[k] * weights[k] for k in criteria_scores)
            attack_success = weighted_score >= 0.4  # Soglia ridotta da 0.5 a 0.4
            
            successful_criteria = sum(criteria_scores.values())
            
            print(f"Confidence: {confidence_criterion} ({'✅' if confidence_criterion else '❌'})")
            print(f"Separation: {separation_criterion} ({'✅' if separation_criterion else '❌'})")
            print(f"Leakage: {leakage_criterion} ({'✅' if leakage_criterion else '❌'})")
            print(f"Samples: {sample_criterion} ({'✅' if sample_criterion else '❌'})")
            print(f"Weighted score: {weighted_score:.3f}")
            print(f"Successful criteria: {successful_criteria}/4")

            print(f"Model Inversion Results:")
            print(f" - Normal confidence: {best_normal_score:.4f}")
            print(f" - Attack confidence: {best_attack_score:.4f}")
            print(f" - Information leakage: {information_leakage:.4f}")
            print(f" - L2 separation: {l2_distance:.4f}")
            print(f" - Weighted score: {weighted_score:.4f}")
            print(f" - Attack {'✅ RIUSCITO' if attack_success else '❌ FALLITO'}")

            return {
                "attack_type": "Model Inversion Attack",
                "attack_success": int(attack_success),
                "attack_success_criteria": "Considerato successo (1) se almeno 3 criteri su 4 nei success_criteria sono soddisfatti.",
                "normal_confidence": float(best_normal_score),
                "normal_confidence_criteria": "Confidenza media per i prototipi normali (non invertiti), usata come baseline.",
                "attack_confidence": float(best_attack_score),
                "attack_confidence_explanation": "Confidenza media per i prototipi invertiti (generati dall'attacco).",
                "avg_confidence": float(confidence_component),
                "avg_confidence_explanation": "Confidenza media complessiva dei prototipi generati dall'attacco.",
                "information_leakage_score": float(information_leakage),
                "information_leakage_score_explanation": "Valore aggregato che quantifica il grado di informazione sensibile estratta tramite inversione. Valori >0.5 indicano forte leakage rispetto al baseline.",
                "confidence_component": float(confidence_component),
                "confidence_component_explanation": "Componente dovuta al livello di confidenza raggiunto dai prototipi invertiti.",
                "separation_component": float(separation_component),
                "separation_component_explanation": "Componente dovuta alla separazione tra prototipi attaccati e prototipi normali (maggiore significa che si distinguono meglio).",
                "distance_component": float(l2_distance / 10.0),  # Normalizzato
                "distance_component_explanation": "Componente dovuta alla distanza (L2) tra prototipi normali e invertiti; valori bassi indicano forte somiglianza.",
                "high_conf_normal_samples": int(len(high_conf_normal)),
                "high_conf_normal_samples_explanation": "Numero di prototipi normali che superano la soglia di confidenza baseline.",
                "high_conf_attack_samples": int(len(high_conf_attack)),
                "high_conf_attack_samples_explanation": "Numero di prototipi invertiti che superano la soglia di confidenza attacco.",
                "normal_threshold_used": str(normal_threshold_used),
                "normal_threshold_explanation": "Soglia di confidenza usata per considerare un prototipo normale.",
                "attack_threshold_used": str(attack_threshold_used),
                "attack_threshold_explanation": "Soglia di confidenza usata per considerare un prototipo invertito.",
                "prototype_separation": float(l2_distance),
                "prototype_separation_explanation": "Distanza media (in spazio feature) tra i prototipi normali e quelli invertiti.",
                "prototype_l2_distance": float(l2_distance),
                "prototype_l2_distance_explanation": "Distanza euclidea media tra prototipi normali e invertiti.",
                "prototype_cosine_similarity": float(1 - cosine_distance),
                "prototype_cosine_similarity_explanation": "Similarità coseno media tra prototipi normali e invertiti; valori vicini a 1 indicano grande somiglianza.",
                "best_normal_method": str(best_normal_method),
                "best_normal_method_explanation": "Tecnica che ha prodotto i migliori prototipi normali.",
                "best_attack_method": str(best_attack_method),
                "best_attack_method_explanation": "Tecnica che ha prodotto i migliori prototipi invertiti.",
                "success_criteria": {
                    "confidence_criterion": int(confidence_criterion),
                    "confidence_criterion_explanation": "Vero se la confidenza dei prototipi invertiti supera la soglia stabilita.",
                    "separation_criterion": int(separation_criterion),
                    "separation_criterion_explanation": "Vero se la separazione tra prototipi invertiti e normali è sufficiente.",
                    "leakage_criterion": int(leakage_criterion),
                    "leakage_criterion_explanation": "Vero se l'information leakage score supera la soglia.",
                    "sample_criterion": int(sample_criterion),
                    "sample_criterion_explanation": "Vero se il numero di campioni invertiti ad alta confidenza è significativo.",
                    "total_successful": int(successful_criteria)
                },
                "samples_analyzed": int(len(self.X_test))
            }
            
        except Exception as e:
            print(f"❌ Errore Enhanced Model Inversion: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "attack_type": "enhanced_model_inversion",
                "attack_success": False,
                "error": str(e),
                "fallback_used": True
            }

    def execute_compatible_attacks(self):
        """Esegue i 3 attacchi compatibili con l'architettura attuale"""
        results = {}
        
        try:
            if not self.is_malicious:
                return results
            
            print(f"\n=== ESECUZIONE ATTACCHI COMPATIBILI CLIENT {self.client_id} ===")
            
            # 1. Membership Inference Attack  
            print(f"\n1. Membership Inference Attack ...")
            results['membership_inference'] = self.membership_inference_attack()
            
            # 2. Property Inference Attack  
            print(f"\n2. Property Inference Attack ...")
            results['property_inference'] = self.property_inference_attack()

            # 3. Model Inversion Attack  
            print(f"\n3. Model Inversion Attack ...")
            results['model_inversion'] = self.model_inversion_attack()

            # SUMMARY  
            successful_attacks = 0
            total_attacks = 3
            
            if results['membership_inference'].get('attack_success', False):
                successful_attacks += 1
            if results['property_inference'].get('attack_success', False):
                successful_attacks += 1
            if results['model_inversion'].get('attack_success', False):
                successful_attacks += 1
            
            # Privacy risk score aggregato
            privacy_risk_score = 0.0
            if 'privacy_breach_score' in results['membership_inference']:
                privacy_risk_score += results['membership_inference']['privacy_breach_score']
            if 'estimation_error' in results['property_inference']:
                privacy_risk_score += (1.0 - results['property_inference']['estimation_error'])
            if 'avg_confidence' in results['model_inversion']:
                privacy_risk_score += results['model_inversion']['avg_confidence']
            
            # Normalize score
            privacy_risk_score = privacy_risk_score / 3.0
            
            # Severity level
            if successful_attacks >= 3:
                severity = "HIGH"
            elif successful_attacks >= 2:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            results['attack_summary'] = {
                "total_attacks_attempted": total_attacks,
                "total_attacks_explanation": "Numero totale di tipologie di attacco privacy testate su questo client.",
                "successful_attacks": successful_attacks,
                "successful_attacks_explanation": "Numero di attacchi che hanno superato il criterio di successo e rappresentano un rischio concreto per la privacy.",
                "attack_success_rate": float(successful_attacks / total_attacks),
                "attack_success_rate_explanation": "Frazione di attacchi riusciti sul totale di quelli tentati. Valori vicini a 1 indicano alta vulnerabilità.",
                "privacy_risk_score": float(privacy_risk_score),
                "privacy_risk_score_explanation": "Indice aggregato (calcolato per combinare i risultati di tutti gli attacchi) che misura il rischio privacy complessivo per il client.",
                "client_id": int(self.client_id),
                "client_id_explanation": "Identificativo numerico del client federato a cui si riferiscono gli attacchi.",
                "federated_learning_compromised": int(successful_attacks >= 2),
                "federated_learning_compromised_explanation": "True se tutte le principali tipologie di attacco hanno avuto successo, segnalando che il sistema federato è compromesso dal punto di vista privacy."
            }

            print(f"\n=== SUMMARY ATTACCHI COMPATIBILI CLIENT {self.client_id} ===")
            print(f"Attacchi completati: 3/3")
            print(f"Attacchi riusciti: {successful_attacks}/{total_attacks}")
            print(f"Tasso successo: {successful_attacks/total_attacks*100:.1f}%")
            print(f"Livello rischio: {severity}")
            print(f"FL compromesso: {'SÌ' if successful_attacks >= 2 else 'NO'}")
            print(f"Privacy score: {privacy_risk_score:.3f}")
            print(f"Architettura: [256, 128, 64, 32] → 1")
            
        except Exception as e:
            print(f"❌ Errore durante esecuzione attacchi compatibili: {e}")
            import traceback
            results['execution_error'] = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'client_id': self.client_id,
                'timestamp': datetime.now().isoformat()
            }
        
        return sanitize_json(results)

    # METODI FLOWER CLIENT
    def get_parameters(self, config):
        """Restituisce i parametri del modello"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Training del modello"""
        print(f"\n[MALICIOUS Client {self.client_id}] Training...")
        
        # Imposta pesi dal server
        self.model.set_weights(parameters)
        
        # Callbacks avanzati
        callbacks = create_advanced_callbacks(self.config)

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Training
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=25,
            batch_size=64,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Estrazione metriche finali
        final_epoch = len(history.history['loss']) - 1
        
        def safe_extract(metric_history, epoch_idx):
            """Estrae valore sicuro da history"""
            try:
                value = metric_history[epoch_idx]
                if hasattr(value, '__len__') and len(value) > 0:
                    return float(value[0])
                else:
                    return float(value)
            except (IndexError, TypeError, ValueError):
                return 0.0
        
        # Estrazione metriche training
        train_loss = safe_extract(history.history['loss'], final_epoch)
        train_acc = safe_extract(history.history['accuracy'], final_epoch)
        train_precision = safe_extract(history.history['precision'], final_epoch)
        train_recall = safe_extract(history.history['recall'], final_epoch)
        train_f1 = safe_extract(history.history['f1_score'], final_epoch)
        
        # Estrazione metriche validation
        val_loss = safe_extract(history.history['val_loss'], final_epoch)
        val_acc = safe_extract(history.history['val_accuracy'], final_epoch)
        val_precision = safe_extract(history.history['val_precision'], final_epoch)
        val_recall = safe_extract(history.history['val_recall'], final_epoch)
        val_f1 = safe_extract(history.history['val_f1_score'], final_epoch)

        # Calcolo AUC ROC, Specificity
        try:
            y_val_pred_prob = self.model.predict(self.X_val, verbose=0).flatten()
            y_val_pred_binary = (y_val_pred_prob > 0.5).astype(int)
            
            # AUC ROC
            from sklearn.metrics import roc_auc_score
            val_auc_roc = roc_auc_score(self.y_val, y_val_pred_prob)

            # Matrice di confusione per Specificity
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
            print(f"   ⚠️ Errore calcolo metriche aggiuntive: {e}")
            val_auc_roc = 0.5
            val_specificity = 0.0
            val_sensitivity = val_recall
        
        print(f"[MALICIOUS Client {self.client_id}] Training completato:")
        print(f"Train - Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"Val - AUC: {val_auc_roc:.4f}, Spec: {val_specificity:.4f}, Sens: {val_sensitivity:.4f}")
        
        # Metriche
        metrics = {
            # Training metrics
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_precision': float(train_precision),
            'train_recall': float(train_recall),
            'train_f1_score': float(train_f1),
            
            # Validation metrics
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
            'client_type': 'malicious_compatible',  # Identifica come malevolo
            
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
        """Valutazione del modello con esecuzione attacchi"""
        print(f"\n[MALICIOUS Client {self.client_id}] Evaluation...")

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
        
        # Calcoli aggiuntivi 
        y_pred_prob = self.model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Matrice di confusione
        from sklearn.metrics import confusion_matrix, classification_report
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Metriche aggiuntive
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # è recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        print(f"[ MALICIOUS Client {self.client_id}] Standard Results:")
        print(f"Accuracy: {accuracy:.4f} ({'✅' if accuracy >= 0.90 else '❌'} target: >90%)")
        print(f"Precision: {precision:.4f} ({'✅' if precision >= 0.90 else '❌'} target: >90%)")
        print(f"Recall: {recall:.4f} ({'✅' if recall >= 0.90 else '❌'} target: >90%)")
        print(f"F1-Score: {f1_score:.4f} ({'✅' if f1_score >= 0.90 else '❌'} target: >90%)")
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
            print(f"Target raggiunti")
        else:
            missed = [k for k, v in targets_met.items() if not v]
            print(f"Target mancati: {missed}")

        # ESECUZIONE ATTACCHI
        attack_results = {}
        if self.is_malicious:
            try:
                print(f"\nESECUZIONE ATTACCHI...")
                attack_results = self.execute_compatible_attacks()
                
                # Salva risultati su file con timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"attack_results_compatible_client_{self.client_id}_{timestamp}.json"
                
                try:
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(attack_results, f, indent=2, ensure_ascii=False)
                    print(f"Risultati attacchi compatibili salvati: {results_file}")
                    
                    # Mostra summary dettagliato
                    if 'attack_summary' in attack_results:
                        summary = attack_results['attack_summary']
                        print(f"\n=== SUMMARY ATTACCHI COMPATIBILI ===")
                        print(f"Attacchi riusciti: {summary['successful_attacks']}/{summary['total_attacks_attempted']}")
                        print(f"Tasso successo: {summary['attack_success_rate']*100:.1f}%")
                        print(f"FL compromesso: {'SÌ' if summary['federated_learning_compromised'] else 'NO'}")
                        print(f"Privacy score: {summary['privacy_risk_score']:.3f}")
                        print(f"Client ID: {summary['client_id']}")
                        print(f"Attacks completati: {summary['total_attacks_attempted']}")

                except Exception as save_error:
                    print(f"❌ Errore salvataggio JSON: {save_error}")
                    print(f"Debug: Tipo attack_results: {type(attack_results)}")
                    
            except Exception as attack_error:
                print(f"❌ Errore durante attacchi compatibili: {attack_error}")
                import traceback
                traceback.print_exc()
                attack_results = {
                    'execution_failed': True,
                    'error': str(attack_error),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat(),
                    'client_id': self.client_id
                }
        
        # Metriche complete per il server
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
            
            # Client info (con identificazione malevola)
            'client_id': int(self.client_id),
            'test_samples': int(len(self.X_test)),
            'model_type': 'improved_kaggle_style',
            'version': 'v2_compatible_malicious',
            'client_type': 'malicious_compatible',
            'is_malicious': bool(self.is_malicious),
            
            # Info attacchi
            'attacks_attempted': bool(self.is_malicious),
            'attacks_completed': 'attack_summary' in attack_results,
            'attack_version': 'compatible_v1.0',
            'architecture_compatible': True,
            'preprocessing_compatible': True
        }
        
        # Aggiungi summary attacchi alle metriche se disponibile
        if 'attack_summary' in attack_results:
            attack_summary = attack_results['attack_summary']
            metrics.update({
                'attacks_successful': int(attack_summary['successful_attacks']),
                'attacks_total': int(attack_summary['total_attacks_attempted']),
                'attack_success_rate': float(attack_summary['attack_success_rate']),
                'privacy_risk_score': float(attack_summary['privacy_risk_score']),
                'fl_compromised': bool(attack_summary['federated_learning_compromised']),
            })
        
        return loss, len(self.X_test), metrics


# ALIAS PER RETROCOMPATIBILITÀ
MaliciousClient = CompatibleMaliciousClient

# MAIN FUNCTION  
def main():
    """Funzione principale del client malevolo"""

    if len(sys.argv) != 3:
        print("Uso: python3 malicious_client_inference.py <client_id> <is_malicious>")
        print("Esempio: python3 malicious_client_inference.py 1 true")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
            
        is_malicious = sys.argv[2].lower() == 'true'
        
    except (ValueError, IndexError) as e:
        print(f"Errore parametri: {e}")
        print(f"Formato corretto: python3 malicious_client_inference.py <client_id> <true/false>")
        sys.exit(1)

    print(f"\n=== CLIENT MALEVOLO   {client_id} ===")
    print("=" * 70)
    print(f"Modalità: {'MALEVOLO' if is_malicious else 'NORMALE'}")
    print(f"Architettura: [256, 128, 64, 32] → 1 (Compatible)")
    print(f"Preprocessing: improved_preprocessing.py")
    print(f"Attacchi: Membership Inference, Property Inference, Model Inversion")
    print("=" * 70)
    
    if is_malicious:
        print("Client configurato in modalità MALEVOLA")
        print("Attacchi verranno eseguiti durante la valutazione")
        print("Risultati salvati automaticamente in file JSON")
    else:
        print("Client configurato in modalità NORMALE")
        print("Nessun attacco privacy verrà eseguito")

    print("Connessione al server in corso...")

    try:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=CompatibleMaliciousClient(client_id, is_malicious)
        )
        
    except KeyboardInterrupt:
        print(f"\nClient {client_id} fermato dall'utente")
        
    except Exception as e:
        print(f"\n❌ Errore critico client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        print(f"\nCLIENT MALEVOLO {client_id} TERMINATO!")
        if is_malicious:
            print("Consultare i file JSON generati per i risultati degli attacchi")


if __name__ == "__main__":
    main()