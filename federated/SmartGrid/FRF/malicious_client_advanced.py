"""
Client malevolo avanzato che partecipa al training federato e esegue attacchi di inferenza.
Si integra perfettamente con il tuo sistema serverRFtmp.py esistente.

Questo client:
1. Appare come un client normale agli altri partecipanti
2. Partecipa completamente al training federato
3. Raccoglie intelligence sui modelli globali ricevuti
4. Esegue attacchi di inferenza durante e dopo il training
5. Salva tutti i dati raccolti per analisi post-training
"""

import flwr as fl
import numpy as np
import pandas as pd
import sys
import os
import warnings
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean, cosine
import matplotlib.pyplot as plt

# Importa il client normale del tuo sistema
sys.path.append(os.path.dirname(__file__))
from clientRFtmp import SmartGridRandomForestClient, load_client_smartgrid_data, create_random_forest_model

# Importazioni ART per attacchi avanzati
try:
    from art.estimators.classification import SklearnClassifier
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    ART_AVAILABLE = True
    print("✅ [MALICIOUS] ART disponibile per attacchi avanzati")
except ImportError:
    ART_AVAILABLE = False
    print("⚠️ [MALICIOUS] ART non disponibile - usando implementazioni custom")

warnings.filterwarnings('ignore')

class MaliciousSmartGridClient(SmartGridRandomForestClient):
    """
    Client malevolo che estende il tuo client normale per eseguire attacchi di inferenza.
    
    Caratteristiche:
    - Mantiene TUTTA la funzionalità del client normale
    - Appare identico agli altri client nel protocollo Flower
    - Raccoglie intelligence sui modelli globali ricevuti
    - Esegue attacchi di inferenza in background
    - Salva tutti i dati per analisi post-training
    """
    
    def __init__(self, client_id, intelligence_dir="malicious_intelligence"):
        """
        Inizializza il client malevolo.
        
        Args:
            client_id: ID del client malevolo (deve essere tra 1-13)
            intelligence_dir: Directory per salvare i dati raccolti
        """
        # Chiama il costruttore del client normale
        super().__init__()
        
        self.client_id = client_id
        self.intelligence_dir = intelligence_dir
        
        # Strutture dati per raccogliere intelligence
        self.collected_models = []           # Modelli globali ricevuti ad ogni round
        self.round_metrics = []              # Metriche per ogni round
        self.attack_results = []             # Risultati degli attacchi eseguiti
        self.victim_data_cache = {}          # Cache dei dati delle vittime
        self.server_communications = []      # Log delle comunicazioni col server
        
        # Configurazione attacchi
        self.enable_realtime_attacks = True  # Attacchi durante il training
        self.enable_intelligence_gathering = True  # Raccolta dati
        self.victim_clients = [14, 15]       # Client "vittima" da attaccare
        
        # Crea directory intelligence
        os.makedirs(intelligence_dir, exist_ok=True)
        
        print(f"🕵️ [MALICIOUS CLIENT {client_id}] Inizializzato con successo")
        print(f"🎭 [MALICIOUS CLIENT {client_id}] Appare come un client NORMALE nel protocollo")
        print(f"📡 [MALICIOUS CLIENT {client_id}] Intelligence gathering: {'ATTIVO' if self.enable_intelligence_gathering else 'DISATTIVO'}")
        print(f"🎯 [MALICIOUS CLIENT {client_id}] Attacchi real-time: {'ATTIVI' if self.enable_realtime_attacks else 'DISATTIVI'}")
        print(f"👥 [MALICIOUS CLIENT {client_id}] Client vittima: {self.victim_clients}")
        
        # Carica dati delle vittime per attacchi
        self._preload_victim_data()
    
    def _preload_victim_data(self):
        """
        Pre-carica i dati dei client vittima per attacchi più veloci.
        """
        print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Pre-caricamento dati vittime...")
        
        for victim_id in self.victim_clients:
            try:
                # Usa le TUE funzioni per caricare i dati delle vittime
                X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(victim_id)
                
                # Combina tutti i dati della vittima
                X_victim = np.vstack([X_train, X_val])
                y_victim = np.hstack([y_train, y_val])
                
                self.victim_data_cache[victim_id] = {
                    'X': X_victim,
                    'y': y_victim,
                    'info': dataset_info,
                    'samples': len(X_victim),
                    'features': X_victim.shape[1],
                    'attack_ratio': np.mean(y_victim)
                }
                
                print(f"📊 [MALICIOUS CLIENT {self.client_id}] Vittima {victim_id}: {len(X_victim)} campioni, {X_victim.shape[1]} features")
                
            except Exception as e:
                print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Errore caricamento vittima {victim_id}: {e}")
    
    def set_parameters(self, parameters):
        """
        Riceve il modello globale dal server e LO ANALIZZA per attacchi.
        Mantiene la funzionalità normale + aggiunge intelligence gathering.
        """
        round_number = len(self.collected_models) + 1
        
        print(f"🕵️ [MALICIOUS CLIENT {self.client_id}] ROUND {round_number} - Modello globale ricevuto")
        print(f"🎭 [MALICIOUS CLIENT {self.client_id}] Processamento NORMALE + intelligence gathering...")
        
        # IMPORTANTE: Chiama il metodo normale per aggiornare il modello
        super().set_parameters(parameters)
        
        # ATTACCO: Analizza e salva il modello globale
        if self.enable_intelligence_gathering and parameters and len(parameters) > 0:
            self._collect_global_model_intelligence(parameters, round_number)
        
        # ATTACCO: Esegui attacchi real-time se abilitati
        if self.enable_realtime_attacks and hasattr(self, 'target_model') and self.target_model is not None:
            self._perform_realtime_inference_attacks(round_number)
    
    def _collect_global_model_intelligence(self, parameters, round_number):
        """
        Raccoglie intelligence sul modello globale ricevuto.
        """
        try:
            print(f"📡 [MALICIOUS CLIENT {self.client_id}] Raccolta intelligence round {round_number}...")
            
            # Deserializza il modello globale
            model_array = parameters[0]
            
            if hasattr(model_array, 'tobytes'):
                model_bytes = model_array.tobytes()
            elif hasattr(model_array, 'data'):
                model_bytes = model_array.data.tobytes()
            else:
                model_bytes = bytes(model_array)
            
            # Deserializza il modello Random Forest
            global_model = pickle.loads(model_bytes)
            
            # Analizza il modello globale
            model_analysis = {
                'round': round_number,
                'timestamp': datetime.now().isoformat(),
                'model_type': type(global_model).__name__,
                'n_estimators': getattr(global_model, 'n_estimators', 0),
                'n_features': getattr(global_model, 'n_features_in_', 0),
                'n_classes': getattr(global_model, 'n_classes_', 0),
                'max_depth': getattr(global_model, 'max_depth', None),
                'model_size_bytes': len(model_bytes),
                'has_estimators': hasattr(global_model, 'estimators_'),
                'estimators_count': len(global_model.estimators_) if hasattr(global_model, 'estimators_') else 0
            }
            
            # Salva il modello per analisi posteriori
            model_info = {
                'analysis': model_analysis,
                'model': global_model,  # Modello completo per attacchi
                'serialized_size': len(model_bytes)
            }
            
            self.collected_models.append(model_info)
            
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Intelligence round {round_number}:")
            print(f"    N. alberi: {model_analysis['estimators_count']}")
            print(f"    N. features: {model_analysis['n_features']}")
            print(f"    Dimensione: {model_analysis['model_size_bytes']} bytes")
            
            # Salva immediatamente per sicurezza
            self._save_intelligence_checkpoint(round_number)
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore intelligence gathering: {e}")
    
    def _perform_realtime_inference_attacks(self, round_number):
        """
        Esegue attacchi di inferenza in tempo reale durante il training.
        """
        try:
            # Usa il modello globale corrente (aggiornato da set_parameters)
            global model
            if model is None or not hasattr(model, 'estimators_'):
                print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Modello non disponibile per attacchi round {round_number}")
                return
            
            print(f"🎯 [MALICIOUS CLIENT {self.client_id}] Attacchi real-time round {round_number}...")
            
            attack_results = {
                'round': round_number,
                'timestamp': datetime.now().isoformat(),
                'attacks': {}
            }
            
            # ATTACCO 1: Membership Inference sui dati delle vittime
            for victim_id in self.victim_clients:
                if victim_id in self.victim_data_cache:
                    mia_result = self._realtime_membership_inference(victim_id, model)
                    attack_results['attacks'][f'membership_inference_victim_{victim_id}'] = mia_result
            
            # ATTACCO 2: Model Inversion per estrarre pattern
            inversion_result = self._realtime_model_inversion(model)
            attack_results['attacks']['model_inversion'] = inversion_result
            
            # ATTACCO 3: Attribute Inference su dati vittima
            for victim_id in self.victim_clients:
                if victim_id in self.victim_data_cache:
                    attr_result = self._realtime_attribute_inference(victim_id, model)
                    attack_results['attacks'][f'attribute_inference_victim_{victim_id}'] = attr_result
            
            self.attack_results.append(attack_results)
            
            print(f"✅ [MALICIOUS CLIENT {self.client_id}] Attacchi round {round_number} completati")
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore attacchi real-time: {e}")
    
    def _realtime_membership_inference(self, victim_id, target_model):
        """
        Esegue Membership Inference Attack real-time su una vittima specifica.
        """
        try:
            victim_data = self.victim_data_cache[victim_id]
            X_victim = victim_data['X']
            y_victim = victim_data['y']
            
            # Campiona dati per efficienza
            sample_size = min(200, len(X_victim))
            indices = np.random.choice(len(X_victim), sample_size, replace=False)
            X_sample = X_victim[indices]
            y_sample = y_victim[indices]
            
            # Ottieni predizioni del modello globale sui dati della vittima
            try:
                victim_predictions = target_model.predict_proba(X_sample)
                victim_confidences = np.max(victim_predictions, axis=1)
                
                # Calcola statistiche di membership leakage
                high_confidence_ratio = np.mean(victim_confidences > 0.9)
                avg_confidence = np.mean(victim_confidences)
                confidence_std = np.std(victim_confidences)
                
                # Euristica per rilevare potenziale membership leakage
                # Confidenze molto alte + bassa deviazione standard = possibile overfitting
                leakage_score = high_confidence_ratio + (1 - confidence_std)
                
                if leakage_score > 1.5:
                    risk_level = "ALTO"
                elif leakage_score > 1.0:
                    risk_level = "MEDIO"
                else:
                    risk_level = "BASSO"
                
                result = {
                    'victim_id': victim_id,
                    'samples_tested': sample_size,
                    'avg_confidence': float(avg_confidence),
                    'high_confidence_ratio': float(high_confidence_ratio),
                    'confidence_std': float(confidence_std),
                    'leakage_score': float(leakage_score),
                    'risk_level': risk_level,
                    'success': leakage_score > 1.0
                }
                
                print(f"🎯 [MALICIOUS CLIENT {self.client_id}] MIA vittima {victim_id}: rischio {risk_level} (score: {leakage_score:.3f})")
                
                return result
                
            except Exception as e:
                print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Errore predizione MIA vittima {victim_id}: {e}")
                return {'victim_id': victim_id, 'error': str(e), 'success': False}
                
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore MIA real-time vittima {victim_id}: {e}")
            return {'victim_id': victim_id, 'error': str(e), 'success': False}
    
    def _realtime_model_inversion(self, target_model):
        """
        Esegue Model Inversion Attack real-time per estrarre pattern del modello.
        """
        try:
            # Usa dati di una vittima come base per l'inversione
            if not self.victim_data_cache:
                return {'error': 'no_victim_data', 'success': False}
            
            # Prendi il primo client vittima disponibile
            victim_id = list(self.victim_data_cache.keys())[0]
            victim_data = self.victim_data_cache[victim_id]
            X_victim = victim_data['X']
            
            inversion_results = {}
            
            # Inverti per entrambe le classi (Natural=0, Attack=1)
            for target_class in [0, 1]:
                class_name = "Natural" if target_class == 0 else "Attack"
                
                try:
                    # Calcola statistiche della classe dai dati vittima
                    class_mask = (victim_data['y'] == target_class)
                    if np.sum(class_mask) == 0:
                        continue
                    
                    X_class = X_victim[class_mask]
                    class_mean = np.mean(X_class, axis=0)
                    class_std = np.std(X_class, axis=0)
                    
                    # Funzione obiettivo semplificata per velocità
                    def quick_objective(x):
                        try:
                            x_reshaped = x.reshape(1, -1)
                            proba = target_model.predict_proba(x_reshaped)[0]
                            confidence = proba[target_class]
                            
                            # Regolarizzazione leggera
                            reg_term = 0.01 * np.sum((x - class_mean) ** 2)
                            return -confidence + reg_term
                        except:
                            return 1e6
                    
                    # Ottimizzazione veloce (poche iterazioni per real-time)
                    x_init = class_mean + np.random.normal(0, 0.1 * class_std, len(class_mean))
                    
                    # Bounds ristretti per velocità
                    bounds = [(class_mean[i] - 2*class_std[i], class_mean[i] + 2*class_std[i]) 
                             for i in range(len(class_mean))]
                    
                    result = minimize(
                        quick_objective,
                        x_init,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 20}  # RIDOTTO per velocità real-time
                    )
                    
                    if result.success:
                        x_optimal = result.x
                        final_confidence = target_model.predict_proba(x_optimal.reshape(1, -1))[0, target_class]
                        
                        inversion_results[target_class] = {
                            'class_name': class_name,
                            'confidence': float(final_confidence),
                            'optimization_success': True
                        }
                        
                    else:
                        inversion_results[target_class] = {
                            'class_name': class_name,
                            'confidence': 0.0,
                            'optimization_success': False
                        }
                        
                except Exception as e:
                    print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Errore inversione classe {target_class}: {e}")
                    continue
            
            # Calcola risultati aggregati
            confidences = [r['confidence'] for r in inversion_results.values() if r['optimization_success']]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            max_confidence = np.max(confidences) if confidences else 0.0
            
            result = {
                'classes_inverted': len(inversion_results),
                'avg_confidence': float(avg_confidence),
                'max_confidence': float(max_confidence),
                'per_class_results': inversion_results,
                'success': len(confidences) > 0 and max_confidence > 0.6
            }
            
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Model Inversion: max_conf={max_confidence:.3f}, classes={len(inversion_results)}")
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore Model Inversion real-time: {e}")
            return {'error': str(e), 'success': False}
    
    def _realtime_attribute_inference(self, victim_id, target_model):
        """
        Esegue Attribute Inference Attack real-time veloce.
        """
        try:
            victim_data = self.victim_data_cache[victim_id]
            X_victim = victim_data['X']
            
            # Test solo su prime 3 feature per velocità
            target_attributes = [0, 1, 2]
            successful_inferences = 0
            
            for attr_idx in target_attributes:
                try:
                    # Campiona dati per velocità
                    sample_size = min(100, len(X_victim))
                    indices = np.random.choice(len(X_victim), sample_size, replace=False)
                    X_sample = X_victim[indices]
                    
                    # Rimuovi attributo target
                    X_partial = np.delete(X_sample, attr_idx, axis=1)
                    target_values = X_sample[:, attr_idx]
                    
                    # Discretizza attributo (veloce)
                    try:
                        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                        target_discrete = discretizer.fit_transform(target_values.reshape(-1, 1)).flatten().astype(int)
                    except:
                        # Fallback semplice
                        target_discrete = np.digitize(target_values, np.percentile(target_values, [33, 67]))
                    
                    # Attack model veloce
                    attack_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
                    
                    # Split veloce
                    split_idx = len(X_partial) // 2
                    X_train_attr = X_partial[:split_idx]
                    y_train_attr = target_discrete[:split_idx]
                    X_test_attr = X_partial[split_idx:]
                    y_test_attr = target_discrete[split_idx:]
                    
                    if len(np.unique(y_train_attr)) > 1:
                        attack_model.fit(X_train_attr, y_train_attr)
                        predictions = attack_model.predict(X_test_attr)
                        
                        accuracy = accuracy_score(y_test_attr, predictions)
                        baseline = 1.0 / len(np.unique(y_train_attr))
                        
                        if accuracy > baseline + 0.1:  # Soglia per successo
                            successful_inferences += 1
                        
                except:
                    continue
            
            success_rate = successful_inferences / len(target_attributes)
            
            result = {
                'victim_id': victim_id,
                'attributes_tested': len(target_attributes),
                'successful_inferences': successful_inferences,
                'success_rate': float(success_rate),
                'success': success_rate > 0.5
            }
            
            print(f"🔎 [MALICIOUS CLIENT {self.client_id}] Attribute Inference vittima {victim_id}: {successful_inferences}/{len(target_attributes)} successi")
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore Attribute Inference real-time: {e}")
            return {'victim_id': victim_id, 'error': str(e), 'success': False}
    
    # --- All'interno della classe MaliciousSmartGridClient: modifica il metodo fit così (sostituisci la versione esistente) ---

    def fit(self, parameters, config):
        """
        Addestra il modello normalmente ma raccoglie metriche aggiuntive.
        Prima di chiamare super().fit assicuriamo che il modulo clientRFtmp
        abbia tutte le variabili globali richieste dalla superclasse.
        """
        server_round = config.get("server_round", 0)

        print(f"🎭 [MALICIOUS CLIENT {self.client_id}] ROUND {server_round} - Training NORMALE + intelligence")

        # ----------------- SINCRONIZZA VARIABILI NEL MODULO clientRFtmp -----------------
        try:
            import clientRFtmp as client_module
            # Assegna/aggiorna le variabili globali nel modulo clientRFtmp
            client_module.client_id = self.client_id
            # Se il client malevolo ha un modello locale memorizzato, assegna anche quello
            if hasattr(self, 'model') and self.model is not None:
                client_module.model = self.model
            # assegna dati di training e validation se presenti nella cache del client malevolo
            if hasattr(self, 'X_train') and hasattr(self, 'y_train') and hasattr(self, 'X_val') and hasattr(self, 'y_val'):
                client_module.X_train = self.X_train
                client_module.y_train = self.y_train
                client_module.X_val = self.X_val
                client_module.y_val = self.y_val
            # dataset_info
            if hasattr(self, 'dataset_info'):
                client_module.dataset_info = self.dataset_info
        except Exception as ex:
            print(f"[MALICIOUS CLIENT {self.client_id}] ⚠️ Errore sincronizzazione client_module: {ex}")
            import traceback; traceback.print_exc()
        # -----------------------------------------------------------------------------

        # IMPORTANTE: chiama il fit della superclasse (SmartGridRandomForestClient)
        # che si aspetta di trovare nel modulo clientRFtmp le variabili globali
        result = super().fit(parameters, config)

        # Dopo il fit, puoi aggiornare la cache locale del client malevolo se necessario
        try:
            # Sincronizza eventuali modifiche al modello o ai dati dal modulo clientRFtmp al client malevolo
            import clientRFtmp as client_module
            self.model = getattr(client_module, 'model', getattr(self, 'model', None))
            self.X_train = getattr(client_module, 'X_train', getattr(self, 'X_train', None))
            self.y_train = getattr(client_module, 'y_train', getattr(self, 'y_train', None))
            self.X_val = getattr(client_module, 'X_val', getattr(self, 'X_val', None))
            self.y_val = getattr(client_module, 'y_val', getattr(self, 'y_val', None))
            self.dataset_info = getattr(client_module, 'dataset_info', getattr(self, 'dataset_info', None))
        except Exception:
            pass

        # ATTACCO: Raccoglie metriche del round (come prima)
        if self.enable_intelligence_gathering:
            round_metrics = {
                'round': server_round,
                'timestamp': datetime.now().isoformat(),
                'training_samples': result[1] if len(result) > 1 else 0,
                'client_metrics': result[2] if len(result) > 2 else {},
                'appears_normal': True
            }
            self.round_metrics.append(round_metrics)
            print(f"📊 [MALICIOUS CLIENT {self.client_id}] Metriche round {server_round} raccolte")

        return result
    
    def evaluate(self, parameters, config):
        """
        Valuta normalmente ma esegue attacchi post-training.
        """
        print(f"🎭 [MALICIOUS CLIENT {self.client_id}] Valutazione NORMALE + attacchi post-training")
        
        # IMPORTANTE: Chiama la valutazione normale
        result = super().evaluate(parameters, config)
        
        # ATTACCO: Esegui attacchi completi se siamo nell'ultimo round
        if self.enable_realtime_attacks and len(self.collected_models) > 0:
            self._perform_comprehensive_post_training_attacks()
        
        return result
    
    def _perform_comprehensive_post_training_attacks(self):
        """
        Esegue attacchi completi al termine del training con tutti i modelli raccolti.
        """
        print(f"🎯 [MALICIOUS CLIENT {self.client_id}] ATTACCHI COMPLETI POST-TRAINING...")
        print(f"📊 [MALICIOUS CLIENT {self.client_id}] Modelli raccolti: {len(self.collected_models)}")
        
        if not self.collected_models:
            print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Nessun modello raccolto per attacchi")
            return
        
        # Usa l'ultimo modello globale ricevuto
        final_model = self.collected_models[-1]['model']
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'models_analyzed': len(self.collected_models),
            'final_model_analysis': self.collected_models[-1]['analysis'],
            'attacks': {}
        }
        
        try:
            # ATTACCO COMPLETO 1: Membership Inference con ART se disponibile
            if ART_AVAILABLE:
                mia_results = self._comprehensive_membership_inference_art(final_model)
                comprehensive_results['attacks']['membership_inference_art'] = mia_results
            
            # ATTACCO COMPLETO 2: Membership Inference custom
            mia_custom_results = self._comprehensive_membership_inference_custom(final_model)
            comprehensive_results['attacks']['membership_inference_custom'] = mia_custom_results
            
            # ATTACCO COMPLETO 3: Attribute Inference completo
            attr_results = self._comprehensive_attribute_inference(final_model)
            comprehensive_results['attacks']['attribute_inference'] = attr_results
            
            # ATTACCO COMPLETO 4: Model Inversion completo
            inversion_results = self._comprehensive_model_inversion(final_model)
            comprehensive_results['attacks']['model_inversion'] = inversion_results
            
            # ATTACCO COMPLETO 5: Reconstruction Attack
            reconstruction_results = self._comprehensive_reconstruction_attack(final_model)
            comprehensive_results['attacks']['reconstruction'] = reconstruction_results
            
            # ATTACCO COMPLETO 6: Analisi evoluzione modelli
            evolution_results = self._analyze_model_evolution()
            comprehensive_results['attacks']['model_evolution_analysis'] = evolution_results
            
            # Salva risultati completi
            self.attack_results.append({
                'type': 'comprehensive_post_training',
                'results': comprehensive_results
            })
            
            print(f"✅ [MALICIOUS CLIENT {self.client_id}] Attacchi completi post-training completati")
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore attacchi completi: {e}")
    
    def _comprehensive_membership_inference_art(self, target_model):
        """
        Membership Inference Attack completo usando ART.
        """
        if not ART_AVAILABLE:
            return {'error': 'ART_not_available', 'success': False}
        
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] MIA completo con ART...")
            
            # Prepara dati membri (noi) vs non-membri (vittime)
            global X_train, y_train, X_val, y_val
            
            # Dati membri: nostri dati di training
            X_members = np.vstack([X_train, X_val])
            y_members = np.hstack([y_train, y_val])
            
            # Dati non-membri: dati delle vittime
            victim_data_list = []
            for victim_id in self.victim_clients:
                if victim_id in self.victim_data_cache:
                    victim_data_list.append(self.victim_data_cache[victim_id]['X'])
            
            if not victim_data_list:
                return {'error': 'no_victim_data', 'success': False}
            
            X_nonmembers = np.vstack(victim_data_list)
            y_nonmembers = np.zeros(len(X_nonmembers))  # Label dummy per non-membri
            
            # Configura ART
            clip_min = np.min(X_members, axis=0)
            clip_max = np.max(X_members, axis=0)
            
            # Correggi clip values
            mask_equal = clip_min >= clip_max
            if np.any(mask_equal):
                clip_min[mask_equal] -= 0.1
                clip_max[mask_equal] += 0.1
            
            art_classifier = SklearnClassifier(
                model=target_model,
                clip_values=(clip_min, clip_max)
            )
            
            # Crea e addestra MIA
            mia_attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
            mia_attack.fit(X_members, y_members, X_nonmembers, y_nonmembers)
            
            # Test su dati misti
            n_test = min(200, len(X_members) // 2, len(X_nonmembers) // 2)
            X_test = np.vstack([X_members[:n_test], X_nonmembers[:n_test]])
            y_test_labels = np.hstack([y_members[:n_test], y_nonmembers[:n_test]])
            membership_ground_truth = np.hstack([np.ones(n_test), np.zeros(n_test)])
            
            # Esegui attacco
            membership_predictions = mia_attack.infer(X_test, y_test_labels)
            
            # Calcola metriche
            accuracy = accuracy_score(membership_ground_truth, membership_predictions)
            precision = precision_score(membership_ground_truth, membership_predictions, zero_division=0)
            recall = recall_score(membership_ground_truth, membership_predictions, zero_division=0)
            f1 = f1_score(membership_ground_truth, membership_predictions, zero_division=0)
            
            # Attack advantage
            baseline = max(np.mean(membership_ground_truth), 1 - np.mean(membership_ground_truth))
            advantage = accuracy - baseline
            
            try:
                membership_probs = mia_attack.infer(X_test, y_test_labels, probabilities=True)
                auc = roc_auc_score(membership_ground_truth, membership_probs)
            except:
                auc = 0.5
            
            result = {
                'method': 'ART_MembershipInferenceBlackBox',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc),
                'attack_advantage': float(advantage),
                'baseline': float(baseline),
                'test_samples': len(X_test),
                'members_tested': n_test,
                'nonmembers_tested': n_test,
                'success': advantage > 0.1
            }
            
            print(f"🎯 [MALICIOUS CLIENT {self.client_id}] MIA ART: accuracy={accuracy:.3f}, advantage={advantage:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore MIA ART: {e}")
            return {'error': str(e), 'success': False}
    
    def _comprehensive_membership_inference_custom(self, target_model):
        """
        Membership Inference Attack custom completo.
        """
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] MIA custom completo...")
            
            # Usa le stesse configurazioni dell'attacco ART
            global X_train, y_train, X_val, y_val
            
            X_members = np.vstack([X_train, X_val])
            y_members = np.hstack([y_train, y_val])
            
            # Dati delle vittime come non-membri
            victim_data_list = []
            victim_labels_list = []
            for victim_id in self.victim_clients:
                if victim_id in self.victim_data_cache:
                    victim_data_list.append(self.victim_data_cache[victim_id]['X'])
                    victim_labels_list.append(self.victim_data_cache[victim_id]['y'])
            
            if not victim_data_list:
                return {'error': 'no_victim_data', 'success': False}
            
            X_nonmembers = np.vstack(victim_data_list)
            y_nonmembers = np.hstack(victim_labels_list)
            
            # Ottieni predizioni del modello target
            member_probs = target_model.predict_proba(X_members)
            nonmember_probs = target_model.predict_proba(X_nonmembers)
            
            # Estrai features dalle predizioni
            def extract_features(probs, labels):
                features = []
                for prob, label in zip(probs, labels):
                    confidence = np.max(prob)
                    entropy = -np.sum(prob * np.log(prob + 1e-8))
                    loss = -np.log(prob[label] + 1e-8)
                    features.append([prob[0], prob[1], confidence, entropy, loss])
                return np.array(features)
            
            member_features = extract_features(member_probs, y_members)
            nonmember_features = extract_features(nonmember_probs, y_nonmembers)
            
            # Training set per attack model
            X_attack = np.vstack([member_features, nonmember_features])
            y_attack = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])
            
            # Addestra attack model
            attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
            attack_model.fit(X_attack, y_attack)
            
            # Test su dati misti
            n_test = min(500, len(member_features) // 2, len(nonmember_features) // 2)
            X_test_features = np.vstack([member_features[:n_test], nonmember_features[:n_test]])
            y_test_membership = np.hstack([np.ones(n_test), np.zeros(n_test)])
            
            # Predizioni
            membership_pred = attack_model.predict(X_test_features)
            membership_prob = attack_model.predict_proba(X_test_features)[:, 1]
            
            # Metriche
            accuracy = accuracy_score(y_test_membership, membership_pred)
            precision = precision_score(y_test_membership, membership_pred, zero_division=0)
            recall = recall_score(y_test_membership, membership_pred, zero_division=0)
            f1 = f1_score(y_test_membership, membership_pred, zero_division=0)
            
            baseline = max(np.mean(y_test_membership), 1 - np.mean(y_test_membership))
            advantage = accuracy - baseline
            
            try:
                auc = roc_auc_score(y_test_membership, membership_prob)
            except:
                auc = 0.5
            
            result = {
                'method': 'Custom_RF_Attack',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc),
                'attack_advantage': float(advantage),
                'baseline': float(baseline),
                'test_samples': len(X_test_features),
                'success': advantage > 0.1
            }
            
            print(f"🎯 [MALICIOUS CLIENT {self.client_id}] MIA Custom: accuracy={accuracy:.3f}, advantage={advantage:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore MIA custom: {e}")
            return {'error': str(e), 'success': False}
    
    def _comprehensive_attribute_inference(self, target_model):
        """
        Attribute Inference Attack completo su tutte le vittime.
        """
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Attribute Inference completo...")
            
            all_results = {}
            
            for victim_id in self.victim_clients:
                if victim_id not in self.victim_data_cache:
                    continue
                
                victim_data = self.victim_data_cache[victim_id]
                X_victim = victim_data['X']
                
                # Test su prime 5 feature
                target_attributes = [0, 1, 2, 3, 4]
                victim_results = {}
                
                for attr_idx in target_attributes:
                    try:
                        # Prepara dati
                        X_partial = np.delete(X_victim, attr_idx, axis=1)
                        target_values = X_victim[:, attr_idx]
                        
                        # Discretizza
                        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                        target_discrete = discretizer.fit_transform(target_values.reshape(-1, 1)).flatten().astype(int)
                        
                        # Split train/test
                        split_idx = len(X_partial) // 2
                        X_train_attr = X_partial[:split_idx]
                        y_train_attr = target_discrete[:split_idx]
                        X_test_attr = X_partial[split_idx:]
                        y_test_attr = target_discrete[split_idx:]
                        
                        # Attack model
                        attack_model = RandomForestClassifier(n_estimators=50, random_state=42)
                        attack_model.fit(X_train_attr, y_train_attr)
                        
                        # Predizioni
                        predictions = attack_model.predict(X_test_attr)
                        accuracy = accuracy_score(y_test_attr, predictions)
                        
                        baseline = 1.0 / len(np.unique(y_train_attr))
                        advantage = accuracy - baseline
                        
                        victim_results[attr_idx] = {
                            'accuracy': float(accuracy),
                            'baseline': float(baseline),
                            'advantage': float(advantage),
                            'success': advantage > 0.1
                        }
                        
                    except Exception as e:
                        victim_results[attr_idx] = {'error': str(e), 'success': False}
                
                # Risultati aggregati per vittima
                successful = [r for r in victim_results.values() if r.get('success', False)]
                all_results[victim_id] = {
                    'attributes_tested': len(target_attributes),
                    'successful_attributes': len(successful),
                    'success_rate': len(successful) / len(target_attributes),
                    'per_attribute': victim_results
                }
                
                print(f"🎯 [MALICIOUS CLIENT {self.client_id}] Attr Inference vittima {victim_id}: {len(successful)}/{len(target_attributes)} successi")
            
            # Risultati complessivi
            total_success = sum(r['successful_attributes'] for r in all_results.values())
            total_tested = sum(r['attributes_tested'] for r in all_results.values())
            
            result = {
                'victims_tested': len(all_results),
                'total_attributes_tested': total_tested,
                'total_successful': total_success,
                'overall_success_rate': total_success / total_tested if total_tested > 0 else 0,
                'per_victim_results': all_results,
                'success': total_success > 0
            }
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore Attribute Inference: {e}")
            return {'error': str(e), 'success': False}
    
    def _comprehensive_model_inversion(self, target_model):
        """
        Model Inversion Attack completo con ottimizzazione avanzata.
        """
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Model Inversion completo...")
            
            # Usa dati delle vittime per calcolare statistiche
            all_victim_data = []
            all_victim_labels = []
            
            for victim_id in self.victim_clients:
                if victim_id in self.victim_data_cache:
                    all_victim_data.append(self.victim_data_cache[victim_id]['X'])
                    all_victim_labels.append(self.victim_data_cache[victim_id]['y'])
            
            if not all_victim_data:
                return {'error': 'no_victim_data', 'success': False}
            
            X_all = np.vstack(all_victim_data)
            y_all = np.hstack(all_victim_labels)
            
            inversion_results = {}
            
            # Inverti per entrambe le classi
            for target_class in [0, 1]:
                class_name = "Natural" if target_class == 0 else "Attack"
                
                # Statistiche della classe
                class_mask = (y_all == target_class)
                if np.sum(class_mask) == 0:
                    continue
                
                X_class = X_all[class_mask]
                class_stats = {
                    'mean': np.mean(X_class, axis=0),
                    'std': np.std(X_class, axis=0),
                    'min': np.min(X_class, axis=0),
                    'max': np.max(X_class, axis=0)
                }
                
                # Funzione obiettivo
                def objective_function(x):
                    try:
                        x_reshaped = x.reshape(1, -1)
                        proba = target_model.predict_proba(x_reshaped)[0]
                        confidence = proba[target_class]
                        
                        # Regolarizzazione per realismo
                        reg_term = 0.01 * np.sum((x - class_stats['mean']) ** 2)
                        return -confidence + reg_term
                    except:
                        return 1e6
                
                # Multiple restart optimization
                best_confidence = 0
                best_pattern = class_stats['mean'].copy()
                
                for restart in range(5):
                    try:
                        if restart == 0:
                            x_init = class_stats['mean']
                        else:
                            noise_scale = 0.1 * (1 + restart * 0.1)  # Incrementa noise
                            x_init = class_stats['mean'] + np.random.normal(0, noise_scale * class_stats['std'], len(class_stats['mean']))
                        
                        # Bounds adattivi
                        bounds = [(class_stats['min'][i] - 0.2 * class_stats['std'][i], 
                                  class_stats['max'][i] + 0.2 * class_stats['std'][i]) 
                                 for i in range(len(class_stats['mean']))]
                        
                        result = minimize(
                            objective_function,
                            x_init,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 150}
                        )
                        
                        if result.success:
                            x_optimal = result.x
                            confidence = target_model.predict_proba(x_optimal.reshape(1, -1))[0, target_class]
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_pattern = x_optimal
                                
                    except:
                        continue
                
                # Calcola similarità
                similarity = 1 - np.linalg.norm(best_pattern - class_stats['mean']) / np.linalg.norm(class_stats['mean'])
                
                inversion_results[target_class] = {
                    'class_name': class_name,
                    'best_confidence': float(best_confidence),
                    'similarity_to_mean': float(similarity),
                    'optimization_restarts': 5
                }
                
                print(f"🎯 [MALICIOUS CLIENT {self.client_id}] Inversion {class_name}: confidence={best_confidence:.3f}")
            
            # Risultati aggregati
            confidences = [r['best_confidence'] for r in inversion_results.values()]
            similarities = [r['similarity_to_mean'] for r in inversion_results.values()]
            
            result = {
                'classes_inverted': len(inversion_results),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'max_confidence': float(np.max(confidences)) if confidences else 0.0,
                'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,
                'per_class_results': inversion_results,
                'success': len(confidences) > 0 and np.max(confidences) > 0.7
            }
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore Model Inversion: {e}")
            return {'error': str(e), 'success': False}
    
    def _comprehensive_reconstruction_attack(self, target_model):
        """
        Reconstruction Attack completo per ricostruire campioni specifici.
        """
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Reconstruction Attack completo...")
            
            # Seleziona campioni target dalle vittime
            reconstruction_results = []
            
            for victim_id in self.victim_clients:
                if victim_id not in self.victim_data_cache:
                    continue
                
                victim_data = self.victim_data_cache[victim_id]
                X_victim = victim_data['X']
                y_victim = victim_data['y']
                
                # Seleziona 3 campioni target per vittima
                n_targets = min(3, len(X_victim))
                target_indices = np.random.choice(len(X_victim), n_targets, replace=False)
                
                for i, target_idx in enumerate(target_indices):
                    target_sample = X_victim[target_idx]
                    target_label = y_victim[target_idx]
                    
                    # Ottieni predizione target
                    target_prediction = target_model.predict_proba(target_sample.reshape(1, -1))[0]
                    
                    # Funzione obiettivo per ricostruzione
                    def reconstruction_objective(x):
                        try:
                            x_reshaped = x.reshape(1, -1)
                            current_prediction = target_model.predict_proba(x_reshaped)[0]
                            
                            # Distanza tra predizioni
                            prediction_loss = np.sum((current_prediction - target_prediction) ** 2)
                            
                            # Regolarizzazione
                            class_mask = (y_victim == target_label)
                            if np.sum(class_mask) > 0:
                                class_mean = np.mean(X_victim[class_mask], axis=0)
                                reg_term = 0.1 * np.sum((x - class_mean) ** 2)
                            else:
                                reg_term = 0.1 * np.sum(x ** 2)
                            
                            return prediction_loss + reg_term
                        except:
                            return 1e6
                    
                    # Ottimizzazione
                    best_reconstruction = None
                    best_distance = float('inf')
                    
                    for restart in range(3):
                        try:
                            if restart == 0:
                                # Inizializza con media della classe
                                class_mask = (y_victim == target_label)
                                if np.sum(class_mask) > 0:
                                    x_init = np.mean(X_victim[class_mask], axis=0)
                                else:
                                    x_init = np.mean(X_victim, axis=0)
                            else:
                                # Random initialization
                                x_init = np.random.normal(
                                    np.mean(X_victim, axis=0),
                                    np.std(X_victim, axis=0)
                                )
                            
                            # Bounds
                            bounds = [(np.min(X_victim[:, j]), np.max(X_victim[:, j])) 
                                     for j in range(X_victim.shape[1])]
                            
                            result_opt = minimize(
                                reconstruction_objective,
                                x_init,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 100}
                            )
                            
                            if result_opt.success and result_opt.fun < best_distance:
                                best_distance = result_opt.fun
                                best_reconstruction = result_opt.x
                                
                        except:
                            continue
                    
                    if best_reconstruction is not None:
                        # Calcola qualità ricostruzione
                        reconstruction_distance = euclidean(target_sample, best_reconstruction)
                        max_possible_distance = np.sqrt(np.sum((np.max(X_victim, axis=0) - np.min(X_victim, axis=0)) ** 2))
                        normalized_distance = reconstruction_distance / max_possible_distance
                        reconstruction_quality = max(0, 1 - normalized_distance)
                        
                        # Similarità coseno
                        try:
                            cosine_similarity = 1 - cosine(target_sample, best_reconstruction)
                        except:
                            cosine_similarity = 0.0
                        
                        reconstruction_results.append({
                            'victim_id': victim_id,
                            'target_index': i,
                            'reconstruction_quality': float(reconstruction_quality),
                            'cosine_similarity': float(cosine_similarity),
                            'euclidean_distance': float(reconstruction_distance),
                            'normalized_distance': float(normalized_distance),
                            'success': reconstruction_quality > 0.3
                        })
                    else:
                        reconstruction_results.append({
                            'victim_id': victim_id,
                            'target_index': i,
                            'reconstruction_quality': 0.0,
                            'success': False,
                            'error': 'optimization_failed'
                        })
            
            # Risultati aggregati
            successful = [r for r in reconstruction_results if r.get('success', False)]
            success_rate = len(successful) / len(reconstruction_results) if reconstruction_results else 0
            avg_quality = np.mean([r['reconstruction_quality'] for r in successful]) if successful else 0.0
            
            result = {
                'targets_attempted': len(reconstruction_results),
                'successful_reconstructions': len(successful),
                'success_rate': float(success_rate),
                'average_quality': float(avg_quality),
                'individual_results': reconstruction_results,
                'success': success_rate > 0.2
            }
            
            print(f"🎯 [MALICIOUS CLIENT {self.client_id}] Reconstruction: {len(successful)}/{len(reconstruction_results)} successi")
            
            return result
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore Reconstruction: {e}")
            return {'error': str(e), 'success': False}
    
    def _analyze_model_evolution(self):
        """
        Analizza l'evoluzione dei modelli globali raccolti durante il training.
        """
        try:
            print(f"🔍 [MALICIOUS CLIENT {self.client_id}] Analisi evoluzione modelli...")
            
            if len(self.collected_models) < 2:
                return {'error': 'insufficient_models', 'success': False}
            
            evolution_analysis = {
                'models_analyzed': len(self.collected_models),
                'rounds_covered': [m['analysis']['round'] for m in self.collected_models],
                'evolution_metrics': []
            }
            
            # Analizza cambiamenti tra modelli consecutivi
            for i in range(1, len(self.collected_models)):
                prev_model = self.collected_models[i-1]['model']
                curr_model = self.collected_models[i]['model']
                
                prev_round = self.collected_models[i-1]['analysis']['round']
                curr_round = self.collected_models[i]['analysis']['round']
                
                # Testa su dati vittima per rilevare cambiamenti
                evolution_metric = {'round_from': prev_round, 'round_to': curr_round}
                
                for victim_id in self.victim_clients:
                    if victim_id in self.victim_data_cache:
                        victim_data = self.victim_data_cache[victim_id]
                        X_sample = victim_data['X'][:100]  # Sample per velocità
                        
                        try:
                            # Confronta predizioni
                            prev_pred = prev_model.predict_proba(X_sample)
                            curr_pred = curr_model.predict_proba(X_sample)
                            
                            # Calcola differenza nelle predizioni
                            prediction_change = np.mean(np.abs(curr_pred - prev_pred))
                            
                            # Calcola stabilità delle confidenze
                            prev_conf = np.max(prev_pred, axis=1)
                            curr_conf = np.max(curr_pred, axis=1)
                            confidence_change = np.mean(np.abs(curr_conf - prev_conf))
                            
                            evolution_metric[f'victim_{victim_id}'] = {
                                'prediction_change': float(prediction_change),
                                'confidence_change': float(confidence_change),
                                'samples_tested': len(X_sample)
                            }
                            
                        except Exception as e:
                            evolution_metric[f'victim_{victim_id}'] = {'error': str(e)}
                
                evolution_analysis['evolution_metrics'].append(evolution_metric)
            
            # Analisi aggregata
            all_pred_changes = []
            all_conf_changes = []
            
            for metric in evolution_analysis['evolution_metrics']:
                for key, value in metric.items():
                    if key.startswith('victim_') and isinstance(value, dict) and 'prediction_change' in value:
                        all_pred_changes.append(value['prediction_change'])
                        all_conf_changes.append(value['confidence_change'])
            
            if all_pred_changes:
                evolution_analysis['aggregated'] = {
                    'avg_prediction_change': float(np.mean(all_pred_changes)),
                    'max_prediction_change': float(np.max(all_pred_changes)),
                    'avg_confidence_change': float(np.mean(all_conf_changes)),
                    'max_confidence_change': float(np.max(all_conf_changes)),
                    'stability_score': float(1 - np.mean(all_pred_changes))  # 1 = perfettamente stabile
                }
                
                # Interpretazione
                avg_change = np.mean(all_pred_changes)
                if avg_change > 0.1:
                    interpretation = "Modello instabile - possibile information leakage"
                elif avg_change > 0.05:
                    interpretation = "Modello moderatamente instabile"
                else:
                    interpretation = "Modello relativamente stabile"
                
                evolution_analysis['interpretation'] = interpretation
                evolution_analysis['success'] = True
                
                print(f"🎯 [MALICIOUS CLIENT {self.client_id}] Evoluzione: avg_change={avg_change:.4f} - {interpretation}")
            else:
                evolution_analysis['success'] = False
            
            return evolution_analysis
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore analisi evoluzione: {e}")
            return {'error': str(e), 'success': False}
    
    def _save_intelligence_checkpoint(self, round_number):
        """
        Salva checkpoint dei dati raccolti per sicurezza.
        """
        try:
            checkpoint_data = {
                'client_id': self.client_id,
                'round': round_number,
                'timestamp': datetime.now().isoformat(),
                'models_collected': len(self.collected_models),
                'attacks_performed': len(self.attack_results),
                'round_metrics': len(self.round_metrics)
            }
            
            checkpoint_file = os.path.join(self.intelligence_dir, f"checkpoint_client_{self.client_id}_round_{round_number}.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ [MALICIOUS CLIENT {self.client_id}] Errore checkpoint: {e}")
    
    def save_complete_intelligence(self):
        """
        Salva tutti i dati di intelligence raccolti al termine del training.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            print(f"💾 [MALICIOUS CLIENT {self.client_id}] Salvataggio intelligence completa...")
            
            # Prepara dati completi (senza i modelli per dimensioni)
            complete_intelligence = {
                'client_info': {
                    'client_id': self.client_id,
                    'timestamp': datetime.now().isoformat(),
                    'intelligence_gathering_enabled': self.enable_intelligence_gathering,
                    'realtime_attacks_enabled': self.enable_realtime_attacks,
                    'victim_clients': self.victim_clients
                },
                'models_intelligence': [
                    {
                        'round': model['analysis']['round'],
                        'timestamp': model['analysis']['timestamp'],
                        'model_analysis': model['analysis']
                    }
                    for model in self.collected_models
                ],
                'round_metrics': self.round_metrics,
                'attack_results': self.attack_results,
                'victim_data_info': {
                    victim_id: {
                        'samples': data['samples'],
                        'features': data['features'],
                        'attack_ratio': data['attack_ratio']
                    }
                    for victim_id, data in self.victim_data_cache.items()
                },
                'summary': {
                    'total_rounds_observed': len(self.collected_models),
                    'total_attacks_performed': len(self.attack_results),
                    'victims_analyzed': len(self.victim_data_cache),
                    'intelligence_complete': True
                }
            }
            
            # File JSON principale
            intelligence_file = os.path.join(self.intelligence_dir, f"complete_intelligence_client_{self.client_id}_{timestamp}.json")
            with open(intelligence_file, 'w') as f:
                json.dump(complete_intelligence, f, indent=2, default=str)
            
            # Salva modelli separatamente (per dimensioni)
            models_file = os.path.join(self.intelligence_dir, f"collected_models_client_{self.client_id}_{timestamp}.pkl")
            with open(models_file, 'wb') as f:
                pickle.dump(self.collected_models, f)

            # Genera report testuale
            report_file = os.path.join(self.intelligence_dir, f"intelligence_report_client_{self.client_id}_{timestamp}.txt")
            self._generate_intelligence_report(complete_intelligence, report_file)
            
            print(f"💾 [MALICIOUS CLIENT {self.client_id}] Intelligence salvata:")
            print(f"   📄 JSON completo: {intelligence_file}")
            print(f"   📊 Modelli: {models_file}")
            print(f"   📋 Report: {report_file}")
            
            return intelligence_file, models_file, report_file
            
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore salvataggio: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _generate_intelligence_report(self, intelligence_data, report_file):
        """
        Genera un report testuale dettagliato dell'intelligence raccolta.
        """
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("REPORT INTELLIGENCE CLIENTE MALEVOLO\n")
                f.write("Sistema Federated Random Forest SmartGrid\n")
                f.write("=" * 80 + "\n\n")
                
                # Informazioni client
                client_info = intelligence_data['client_info']
                f.write(f"CLIENT ID: {client_info['client_id']}\n")
                f.write(f"TIMESTAMP: {client_info['timestamp']}\n")
                f.write(f"INTELLIGENCE GATHERING: {'ATTIVO' if client_info['intelligence_gathering_enabled'] else 'DISATTIVO'}\n")
                f.write(f"REALTIME ATTACKS: {'ATTIVI' if client_info['realtime_attacks_enabled'] else 'DISATTIVI'}\n")
                f.write(f"CLIENT VITTIMA: {client_info['victim_clients']}\n\n")
                
                # Modelli osservati
                f.write("MODELLI GLOBALI OSSERVATI:\n")
                f.write("-" * 40 + "\n")
                for model_info in intelligence_data['models_intelligence']:
                    f.write(f"Round {model_info['round']}:\n")
                    f.write(f"  Timestamp: {model_info['timestamp']}\n")
                    analysis = model_info['model_analysis']
                    f.write(f"  N. Alberi: {analysis.get('estimators_count', 'N/A')}\n")
                    f.write(f"  N. Features: {analysis.get('n_features', 'N/A')}\n")
                    f.write(f"  Dimensione: {analysis.get('model_size_bytes', 0)} bytes\n\n")
                
                # Attacchi eseguiti
                f.write("ATTACCHI DI INFERENZA ESEGUITI:\n")
                f.write("-" * 40 + "\n")
                
                for attack in intelligence_data['attack_results']:
                    if attack.get('type') == 'comprehensive_post_training':
                        f.write("ATTACCHI COMPLETI POST-TRAINING:\n")
                        comp_results = attack['results']
                        
                        for attack_name, attack_result in comp_results.get('attacks', {}).items():
                            f.write(f"\n{attack_name.upper().replace('_', ' ')}:\n")
                            if attack_result.get('success', False):
                                f.write("  Status: SUCCESSO ✅\n")
                                
                                if 'accuracy' in attack_result:
                                    f.write(f"  Accuracy: {attack_result['accuracy']:.4f}\n")
                                if 'attack_advantage' in attack_result:
                                    f.write(f"  Attack Advantage: {attack_result['attack_advantage']:.4f}\n")
                                if 'success_rate' in attack_result:
                                    f.write(f"  Success Rate: {attack_result['success_rate']:.4f}\n")
                                if 'max_confidence' in attack_result:
                                    f.write(f"  Max Confidence: {attack_result['max_confidence']:.4f}\n")
                            else:
                                f.write("  Status: FALLITO ❌\n")
                                if 'error' in attack_result:
                                    f.write(f"  Errore: {attack_result['error']}\n")
                    else:
                        # Attacchi real-time
                        f.write(f"ROUND {attack.get('round', 'N/A')} - ATTACCHI REAL-TIME:\n")
                        for attack_name, attack_result in attack.get('attacks', {}).items():
                            if attack_result.get('success', False):
                                f.write(f"  {attack_name}: SUCCESSO\n")
                            else:
                                f.write(f"  {attack_name}: FALLITO\n")
                        f.write("\n")
                
                # Informazioni vittime
                f.write("ANALISI DATI VITTIME:\n")
                f.write("-" * 40 + "\n")
                for victim_id, victim_info in intelligence_data['victim_data_info'].items():
                    f.write(f"Client Vittima {victim_id}:\n")
                    f.write(f"  Campioni: {victim_info['samples']}\n")
                    f.write(f"  Features: {victim_info['features']}\n")
                    f.write(f"  Ratio Attacchi: {victim_info['attack_ratio']:.4f}\n\n")
                
                # Summary
                summary = intelligence_data['summary']
                f.write("RIASSUNTO INTELLIGENCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Rounds osservati: {summary['total_rounds_observed']}\n")
                f.write(f"Attacchi eseguiti: {summary['total_attacks_performed']}\n")
                f.write(f"Vittime analizzate: {summary['victims_analyzed']}\n")
                f.write(f"Intelligence completa: {'SÌ' if summary['intelligence_complete'] else 'NO'}\n")
                
        except Exception as e:
            print(f"❌ [MALICIOUS CLIENT {self.client_id}] Errore generazione report: {e}")

# Funzioni di utilità globali per il client malevolo
def create_malicious_client_for_training():
    """
    Crea e configura un client malevolo pronto per il training federato.
    Questo wrapper assicura che il modulo clientRFtmp (la superclasse) trovi
    le variabili globali che si aspetta (client_id, model, X_train, ...).
    """
    global client_id, model, X_train, y_train, X_val, y_val, dataset_info

    if len(sys.argv) != 2:
        print("❌ Uso: python malicious_client_advanced.py <client_id>")
        print("Esempio: python malicious_client_advanced.py 5")
        sys.exit(1)

    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 13:
            raise ValueError("Client ID deve essere tra 1 e 13")
    except ValueError as e:
        print(f"❌ Errore: {e}")
        sys.exit(1)

    print(f"🕵️ [MALICIOUS CLIENT {client_id}] === INIZIALIZZAZIONE CLIENT MALEVOLO ===")
    try:
        # Carica dati normalmente usando le funzioni esistenti
        print(f"🎭 [MALICIOUS CLIENT {client_id}] Caricamento dati con preprocessing normale...")
        X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)

        # Crea modello normalmente
        model = create_random_forest_model(client_id)

        # ------------------- NUOVA PARTE: sincronizza variabili sul modulo clientRFtmp -------------------
        # Importa il modulo clientRFtmp e assegna le variabili globali che la superclasse si aspetta.
        import clientRFtmp as client_module

        # Assicurati che le variabili esistano nel modulo clientRFtmp
        client_module.client_id = client_id
        client_module.model = model
        client_module.X_train = X_train
        client_module.y_train = y_train
        client_module.X_val = X_val
        client_module.y_val = y_val
        client_module.dataset_info = dataset_info

        # Nota didattica:
        # - client_module è il modulo che contiene la definizione della superclasse.
        # - Le funzioni in clientRFtmp.py si basano su variabili globali di modulo,
        #   quindi dobbiamo popolare quelle stesse variabili.
        # ---------------------------------------------------------------------------------------------

        print(f"🎭 [MALICIOUS CLIENT {client_id}] === CONFIGURAZIONE NORMALE COMPLETATA ===")
        print(f"🎭 [MALICIOUS CLIENT {client_id}] Dataset: {dataset_info['train_samples']} train, {dataset_info['val_samples']} val")
        print(f"🎭 [MALICIOUS CLIENT {client_id}] Features: {dataset_info['final_features']}")
        print(f"🎭 [MALICIOUS CLIENT {client_id}] Modello: Random Forest con {model.n_estimators} alberi")

        # Crea il client malevolo
        malicious_client = MaliciousSmartGridClient(client_id)

        return malicious_client

    except Exception as e:
        print(f"❌ [MALICIOUS CLIENT {client_id}] Errore inizializzazione: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

def main():
    """
    Funzione principale per avviare il client malevolo.
    """
    print("🕵️ =" * 80)
    print("🕵️ CLIENT MALEVOLO AVANZATO - FEDERATED RANDOM FOREST SMARTGRID")
    print("🕵️ =" * 80)
    print("🕵️ ATTENZIONE: Questo client:")
    print("🕵️   1. ✅ Partecipa NORMALMENTE al training federato")
    print("🕵️   2. 🕵️ Raccoglie INTELLIGENCE sui modelli globali ricevuti")
    print("🕵️   3. 🎯 Esegue ATTACCHI DI INFERENZA durante il training")
    print("🕵️   4. 📊 Salva tutti i dati raccolti per analisi post-training")
    print("🕵️   5. 🎭 Appare IDENTICO agli altri client nel protocollo Flower")
    print("🕵️")
    print("🕵️ Il client malevolo implementa:")
    print("🕵️   📡 Intelligence gathering su modelli globali")
    print("🕵️   🎯 4 tipi di attacchi di inferenza (real-time + completi)")
    print("🕵️   📊 Analisi evoluzione modelli tra round")
    print("🕵️   💾 Salvataggio completo dell'intelligence raccolta")
    print("🕵️")
    print("🕵️ Per una valutazione completa, questo client dovrebbe sostituire")
    print("🕵️ UNO dei 13 client normali durante il training federato.")
    print("🕵️ =" * 80)
    
    try:
        # Crea e configura il client malevolo
        malicious_client = create_malicious_client_for_training()
        
        print(f"\n🕵️ [MALICIOUS CLIENT {malicious_client.client_id}] === CONNESSIONE AL SERVER FEDERATO ===")
        print(f"🎭 [MALICIOUS CLIENT {malicious_client.client_id}] Connessione a localhost:8080...")
        print(f"🎭 [MALICIOUS CLIENT {malicious_client.client_id}] Il server NON rileverà comportamenti anomali")
        print(f"📡 [MALICIOUS CLIENT {malicious_client.client_id}] Intelligence gathering ATTIVO")
        print(f"🎯 [MALICIOUS CLIENT {malicious_client.client_id}] Attacchi real-time ATTIVI")
        
        # Connetti al server Flower (protocollo normale)
        import flwr as fl
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=malicious_client
        )
        
        # Al termine del training, salva l'intelligence completa
        print(f"\n🕵️ [MALICIOUS CLIENT {malicious_client.client_id}] === TRAINING FEDERATO COMPLETATO ===")
        print(f"💾 [MALICIOUS CLIENT {malicious_client.client_id}] Salvataggio intelligence completa...")
        
        malicious_client.save_complete_intelligence()
        
        print(f"\n✅ [MALICIOUS CLIENT {malicious_client.client_id}] === MISSIONE MALEVOLA COMPLETATA ===")
        print(f"📊 [MALICIOUS CLIENT {malicious_client.client_id}] Modelli osservati: {len(malicious_client.collected_models)}")
        print(f"🎯 [MALICIOUS CLIENT {malicious_client.client_id}] Attacchi eseguiti: {len(malicious_client.attack_results)}")
        print(f"📁 [MALICIOUS CLIENT {malicious_client.client_id}] Controlla la cartella 'malicious_intelligence' per i risultati")
        print(f"🕵️ [MALICIOUS CLIENT {malicious_client.client_id}] Intelligence raccolta CON SUCCESSO!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ [MALICIOUS CLIENT] Training interrotto dall'utente")
        try:
            malicious_client.save_complete_intelligence()
            print(f"💾 [MALICIOUS CLIENT] Intelligence parziale salvata")
        except:
            pass
    except Exception as e:
        print(f"❌ [MALICIOUS CLIENT] Errore durante training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()