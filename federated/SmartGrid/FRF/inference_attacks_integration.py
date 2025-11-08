"""
Framework per integrare attacchi di inferenza nel sistema Federated Random Forest esistente.

AUTORE: Framework didattico per tesi di laurea
SCOPO: Valutare vulnerabilità privacy dei modelli federati

Questo file implementa 4 tipi di attacchi di inferenza:
1. Membership Inference Attack - Determina se un dato era nel training
2. Attribute Inference Attack - Inferisce attributi sensibili
3. Model Inversion Attack - Ricostruisce pattern delle classi
4. Reconstruction Attack - Ricostruisce campioni del training

COMPATIBILITÀ: Si integra con clientRFtmp.py e serverRFtmp.py esistenti
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

# Importa il tuo sistema esistente
sys.path.append(os.path.dirname(__file__))
try:
    from clientRFtmp import load_client_smartgrid_data, create_random_forest_model
    SYSTEM_AVAILABLE = True
    print("✅ Sistema clientRFtmp.py importato correttamente")
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"⚠️ Impossibile importare clientRFtmp.py: {e}")
    print("   Assicurati che inference_attacks_integration.py sia nella stessa directory di clientRFtmp.py")

# Importazioni ART per attacchi avanzati (opzionale)
try:
    from art.estimators.classification import SklearnClassifier
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
    ART_AVAILABLE = True
    print("✅ ART (Adversarial Robustness Toolbox) disponibile per attacchi avanzati")
except ImportError:
    ART_AVAILABLE = False
    print("⚠️ ART non disponibile. Attacchi useranno implementazione custom")
    print("   Per attacchi avanzati, installa: pip install adversarial-robustness-toolbox")


class FederatedInferenceAttacker:
    """
    Classe principale per implementare attacchi di inferenza su modelli federati.
    
    Questa classe può:
    1. Caricare un modello federato già salvato (.pkl)
    2. Estrarre un modello federato addestrato dal sistema esistente
    3. Implementare i 4 tipi di attacchi di inferenza
    4. Valutare le vulnerabilità del modello
    
    UTILIZZO TIPICO:
        attacker = FederatedInferenceAttacker()
        
        # Opzione A: Usa modello già salvato
        attacker.target_model = load_model_from_pkl("model.pkl")
        
        # Opzione B: Estrai modello dal sistema
        attacker.extract_federated_model([1, 2, 3])
        
        # Carica dati e esegui attacchi
        attack_data = attacker.load_attack_data([1,2,3], [14,15])
        results = attacker.membership_inference_attack(attack_data)
    """
    
    def __init__(self, output_dir="attack_results"):
        """
        Inizializza l'attaccante.
        
        Args:
            output_dir: Directory dove salvare i risultati degli attacchi
        """
        self.output_dir = output_dir
        self.target_model = None  # Modello da attaccare (può essere caricato da .pkl)
        self.training_info = {}   # Info sul training del modello
        self.attack_results = {}  # Risultati degli attacchi
        
        # Crea directory risultati
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[ATTACKER] 🎯 Framework attacchi di inferenza inizializzato")
        print(f"[ATTACKER] 📁 Directory risultati: {output_dir}")
    
    def extract_federated_model(self, training_client_ids=[1, 2, 3, 4, 5]):
        """
        Estrae un modello federato simulato usando il sistema esistente.
        
        NOTA: Questa funzione addestra un nuovo modello. Se vuoi usare un modello
        già salvato, caricalo direttamente e assegnalo a self.target_model.
        
        Args:
            training_client_ids: Client che hanno partecipato al training federato
            
        Returns:
            RandomForestClassifier: Modello federato estratto
        """
        if not SYSTEM_AVAILABLE:
            raise ImportError("Sistema clientRFtmp.py non disponibile per estrazione modello")
        
        print(f"[ATTACKER] === ESTRAZIONE MODELLO FEDERATO DAL SISTEMA ===")
        print(f"[ATTACKER] Client di training: {training_client_ids}")
        
        # Usa le funzioni esistenti per caricare i dati
        all_X_train = []
        all_y_train = []
        
        for client_id in training_client_ids:
            try:
                print(f"[ATTACKER] Caricamento dati client {client_id} con preprocessing sistema...")
                
                # USA LA FUNZIONE load_client_smartgrid_data dal sistema
                X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
                
                # Combina train e validation per simulare dati completi del client
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                
                all_X_train.append(X_combined)
                all_y_train.append(y_combined)
                
                print(f"[ATTACKER] Client {client_id}: {len(X_combined)} campioni, {X_combined.shape[1]} features")
                
            except Exception as e:
                print(f"[ATTACKER] ⚠️ Errore caricamento client {client_id}: {e}")
                continue
        
        if not all_X_train:
            raise ValueError("Impossibile caricare dati dai client per estrazione modello")
        
        # Combina i dati (simula aggregazione federata)
        X_federated = np.vstack(all_X_train)
        y_federated = np.hstack(all_y_train)
        
        print(f"[ATTACKER] Dataset federato combinato: {len(X_federated)} campioni")
        print(f"[ATTACKER] Features dopo preprocessing: {X_federated.shape[1]}")
        print(f"[ATTACKER] Distribuzione attacchi: {np.mean(y_federated)*100:.1f}%")
        
        # Crea e addestra modello usando LA FUNZIONE create_random_forest_model
        print(f"[ATTACKER] Creazione modello con configurazione sistema...")
        federated_model = create_random_forest_model(client_id=0)  # Usa configurazione server
        
        # Addestra il modello federato simulato
        print(f"[ATTACKER] Addestramento modello federato simulato...")
        federated_model.fit(X_federated, y_federated)
        
        # Valuta il modello
        accuracy = federated_model.score(X_federated, y_federated)
        
        print(f"[ATTACKER] ✅ Modello federato estratto dal sistema:")
        print(f"[ATTACKER]   Accuracy: {accuracy:.4f}")
        print(f"[ATTACKER]   N. alberi: {len(federated_model.estimators_)}")
        print(f"[ATTACKER]   N. campioni training: {len(X_federated)}")
        
        self.target_model = federated_model
        
        # Salva info per gli attacchi
        self.training_info = {
            'training_clients': training_client_ids,
            'n_samples': len(X_federated),
            'n_features': X_federated.shape[1],
            'accuracy': accuracy,
            'attack_ratio': np.mean(y_federated)
        }
        
        return federated_model
    
    def load_attack_data(self, member_clients=[1, 2, 3], nonmember_clients=[14, 15]):
        """
        Carica i dati per gli attacchi usando il preprocessing del sistema.
        
        COMPATIBILE CON: run_attacks_on_saved_model.py
        
        Args:
            member_clients: Client che erano nel training (membri)
            nonmember_clients: Client NON nel training (non-membri)
            
        Returns:
            dict: Dati preparati per gli attacchi con chiavi:
                - X_members, y_members: Dati dei membri
                - X_nonmembers, y_nonmembers: Dati dei non-membri
                - X_test, y_test: Dati per test
                - membership_ground_truth: Ground truth membership (1=membro, 0=no)
                - member_clients, nonmember_clients: Info client
                - n_features: Numero di feature
        """
        if not SYSTEM_AVAILABLE:
            raise ImportError("Sistema clientRFtmp.py non disponibile per caricamento dati")
        
        print(f"[ATTACKER] === CARICAMENTO DATI ATTACCO CON PREPROCESSING SISTEMA ===")
        print(f"[ATTACKER] Client membri: {member_clients}")
        print(f"[ATTACKER] Client non-membri: {nonmember_clients}")
        
        # Carica dati membri usando LE FUNZIONI del sistema
        X_members_list, y_members_list = [], []
        for client_id in member_clients:
            try:
                X_train, y_train, X_val, y_val, _ = load_client_smartgrid_data(client_id)
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                X_members_list.append(X_combined)
                y_members_list.append(y_combined)
                print(f"[ATTACKER] Membro client {client_id}: {len(X_combined)} campioni")
            except Exception as e:
                print(f"[ATTACKER] ⚠️ Errore client membro {client_id}: {e}")
        
        # Carica dati non-membri usando LE FUNZIONI del sistema
        X_nonmembers_list, y_nonmembers_list = [], []
        for client_id in nonmember_clients:
            try:
                X_train, y_train, X_val, y_val, _ = load_client_smartgrid_data(client_id)
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                X_nonmembers_list.append(X_combined)
                y_nonmembers_list.append(y_combined)
                print(f"[ATTACKER] Non-membro client {client_id}: {len(X_combined)} campioni")
            except Exception as e:
                print(f"[ATTACKER] ⚠️ Errore client non-membro {client_id}: {e}")
        
        if not X_members_list or not X_nonmembers_list:
            raise ValueError("Impossibile caricare dati membri/non-membri")
        
        # Combina i dati
        X_members = np.vstack(X_members_list)
        y_members = np.hstack(y_members_list)
        X_nonmembers = np.vstack(X_nonmembers_list)
        y_nonmembers = np.hstack(y_nonmembers_list)
        
        # Prepara dati test (mix di membri e non-membri)
        n_test_members = min(500, len(X_members) // 2)
        n_test_nonmembers = min(500, len(X_nonmembers) // 2)
        
        X_test = np.vstack([
            X_members[:n_test_members],
            X_nonmembers[:n_test_nonmembers]
        ])
        y_test = np.hstack([
            y_members[:n_test_members],
            y_nonmembers[:n_test_nonmembers]
        ])
        
        # Ground truth per membership (1=membro, 0=non-membro)
        membership_ground_truth = np.hstack([
            np.ones(n_test_members),
            np.zeros(n_test_nonmembers)
        ])
        
        attack_data = {
            'X_members': X_members,
            'y_members': y_members,
            'X_nonmembers': X_nonmembers,
            'y_nonmembers': y_nonmembers,
            'X_test': X_test,
            'y_test': y_test,
            'membership_ground_truth': membership_ground_truth,
            'member_clients': member_clients,
            'nonmember_clients': nonmember_clients,
            'n_features': X_members.shape[1]
        }
        
        print(f"[ATTACKER] ✅ Dati attacco preparati con preprocessing sistema:")
        print(f"[ATTACKER]   Membri: {len(X_members)} campioni")
        print(f"[ATTACKER]   Non-membri: {len(X_nonmembers)} campioni")
        print(f"[ATTACKER]   Test: {len(X_test)} campioni")
        print(f"[ATTACKER]   Features: {X_members.shape[1]}")
        
        return attack_data
    
    def membership_inference_attack(self, attack_data):
        """
        Implementa Membership Inference Attack.
        
        COSA FA: Determina se un campione specifico era nel training set del modello.
        COME: Analizza le confidenze/predizioni del modello per distinguere membri da non-membri.
        
        INTERPRETAZIONE RISULTATI:
        - Attack Advantage > 0.15: ALTA vulnerabilità
        - Attack Advantage > 0.05: MEDIA vulnerabilità  
        - Attack Advantage <= 0.05: BASSA vulnerabilità
        
        Args:
            attack_data: Dizionario con dati preparati da load_attack_data()
            
        Returns:
            dict: Risultati dell'attacco con metriche e interpretazione
        """
        print(f"[ATTACKER] 🎯 === MEMBERSHIP INFERENCE ATTACK ===")
        
        if self.target_model is None:
            raise ValueError("Devi prima caricare un modello target (self.target_model)")
        
        try:
            X_members = attack_data['X_members']
            y_members = attack_data['y_members']
            X_nonmembers = attack_data['X_nonmembers']
            y_nonmembers = attack_data['y_nonmembers']
            X_test = attack_data['X_test']
            membership_ground_truth = attack_data['membership_ground_truth']
            
            print(f"[ATTACKER] Training attack model su {len(X_members)} membri, {len(X_nonmembers)} non-membri")
            
            if ART_AVAILABLE:
                # Usa ART per Membership Inference Attack avanzato
                print(f"[ATTACKER] Usando ART MembershipInferenceBlackBox...")
                
                try:
                    # Configura ART classifier
                    clip_min = np.min(X_test, axis=0)
                    clip_max = np.max(X_test, axis=0)
                    
                    # Correggi clip values se necessario
                    mask_equal = clip_min >= clip_max
                    if np.any(mask_equal):
                        clip_min[mask_equal] -= 0.1
                        clip_max[mask_equal] += 0.1
                    
                    art_classifier = SklearnClassifier(
                        model=self.target_model,
                        clip_values=(clip_min, clip_max)
                    )
                    
                    # Crea e addestra MIA attack
                    mia_attack = MembershipInferenceBlackBox(
                        art_classifier,
                        attack_model_type="rf"
                    )
                    
                    # Addestra l'attacco
                    mia_attack.fit(X_members, y_members, X_nonmembers, y_nonmembers)
                    
                    # Esegui attacco
                    membership_predictions = mia_attack.infer(X_test, attack_data['y_test'])
                    
                    try:
                        membership_probabilities = mia_attack.infer(X_test, attack_data['y_test'], probabilities=True)
                    except:
                        membership_probabilities = membership_predictions.astype(float)
                    
                    print(f"[ATTACKER] ✅ Attacco ART completato")
                    
                except Exception as e:
                    print(f"[ATTACKER] ⚠️ Errore ART, fallback a implementazione custom: {e}")
                    membership_predictions, membership_probabilities = self._custom_membership_inference(attack_data)
                
            else:
                # Implementazione custom se ART non disponibile
                print(f"[ATTACKER] Usando implementazione custom...")
                membership_predictions, membership_probabilities = self._custom_membership_inference(attack_data)
            
            # Calcola metriche
            accuracy = accuracy_score(membership_ground_truth, membership_predictions)
            precision = precision_score(membership_ground_truth, membership_predictions, zero_division=0)
            recall = recall_score(membership_ground_truth, membership_predictions, zero_division=0)
            f1 = f1_score(membership_ground_truth, membership_predictions, zero_division=0)
            
            # Attack advantage (quanto meglio del random guessing)
            baseline_accuracy = max(np.mean(membership_ground_truth), 1 - np.mean(membership_ground_truth))
            attack_advantage = accuracy - baseline_accuracy
            
            try:
                auc = roc_auc_score(membership_ground_truth, membership_probabilities)
            except:
                auc = 0.5
            
            results = {
                'attack_type': 'membership_inference',
                'success': True,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc),
                'attack_advantage': float(attack_advantage),
                'baseline_accuracy': float(baseline_accuracy),
                'test_samples': len(X_test),
                'art_used': ART_AVAILABLE
            }
            
            print(f"[ATTACKER] ✅ Membership Inference Attack completato:")
            print(f"[ATTACKER]   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"[ATTACKER]   Attack Advantage: {attack_advantage:.4f} ({attack_advantage*100:.2f}%)")
            print(f"[ATTACKER]   AUC: {auc:.4f}")
            
            # Interpretazione risultati
            if attack_advantage > 0.15:
                interpretation = "🔴 ALTA VULNERABILITÀ - Il modello rivela significativamente la membership. Raccomandato Differential Privacy."
            elif attack_advantage > 0.05:
                interpretation = "🟡 MEDIA VULNERABILITÀ - Il modello potrebbe rivelare membership. Considerare contromisure."
            else:
                interpretation = "🟢 BASSA VULNERABILITÀ - Il modello non rivela membership in modo significativo."
            
            results['interpretation'] = interpretation
            print(f"[ATTACKER]   Interpretazione: {interpretation}")
            
            return results
            
        except Exception as e:
            print(f"[ATTACKER] ❌ Errore Membership Inference Attack: {e}")
            import traceback
            traceback.print_exc()
            return {'attack_type': 'membership_inference', 'success': False, 'error': str(e)}
    
    def attribute_inference_attack(self, attack_data, target_attributes=[0, 1, 2]):
        """
        Implementa Attribute Inference Attack.
        
        COSA FA: Cerca di inferire attributi sensibili (feature) usando altri attributi.
        COME: Addestra modelli che predicono un attributo target dagli altri attributi.
        
        INTERPRETAZIONE:
        - Success rate > 60% e advantage > 0.1: ALTA vulnerabilità
        - Success rate > 30% o advantage > 0.05: MEDIA vulnerabilità
        - Altrimenti: BASSA vulnerabilità
        
        Args:
            attack_data: Dizionario con dati preparati
            target_attributes: Indici delle feature da provare a inferire
            
        Returns:
            dict: Risultati dell'attacco con metriche per ogni attributo
        """
        print(f"[ATTACKER] 🎯 === ATTRIBUTE INFERENCE ATTACK ===")
        print(f"[ATTACKER] Attributi target: {target_attributes}")
        
        if self.target_model is None:
            raise ValueError("Devi prima caricare un modello target")
        
        try:
            X_train = attack_data['X_members']  # Usa dati membri per training attack model
            X_test = attack_data['X_test']
            
            results_per_attribute = {}
            
            for attr_idx in target_attributes:
                print(f"[ATTACKER] Attaccando attributo {attr_idx}...")
                
                try:
                    # Prepara dati per questo attributo
                    X_partial_train = np.delete(X_train, attr_idx, axis=1)
                    X_partial_test = np.delete(X_test, attr_idx, axis=1)
                    
                    target_values_train = X_train[:, attr_idx]
                    target_values_test = X_test[:, attr_idx]
                    
                    # Discretizza l'attributo continuo per classificazione
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    try:
                        target_discrete_train = discretizer.fit_transform(target_values_train.reshape(-1, 1)).flatten().astype(int)
                        target_discrete_test = discretizer.transform(target_values_test.reshape(-1, 1)).flatten().astype(int)
                    except:
                        # Fallback: binning manuale
                        percentiles = np.percentile(target_values_train, [20, 40, 60, 80])
                        target_discrete_train = np.digitize(target_values_train, percentiles)
                        target_discrete_test = np.digitize(target_values_test, percentiles)
                    
                    # Addestra attack model per inferire questo attributo
                    attack_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
                    
                    # Usa subset per efficienza
                    subset_size = min(1000, len(X_partial_train))
                    indices = np.random.choice(len(X_partial_train), subset_size, replace=False)
                    
                    attack_model.fit(X_partial_train[indices], target_discrete_train[indices])
                    
                    # Predici attributo target
                    predicted_attr = attack_model.predict(X_partial_test)
                    
                    # Calcola accuracy
                    accuracy = accuracy_score(target_discrete_test, predicted_attr)
                    baseline = 1.0 / len(np.unique(target_discrete_train))
                    advantage = accuracy - baseline
                    
                    results_per_attribute[attr_idx] = {
                        'accuracy': float(accuracy),
                        'baseline': float(baseline),
                        'advantage': float(advantage),
                        'success': advantage > 0.05,
                        'n_classes': len(np.unique(target_discrete_train))
                    }
                    
                    print(f"[ATTACKER] Attributo {attr_idx}: accuracy={accuracy:.4f}, advantage={advantage:.4f}")
                    
                except Exception as e:
                    print(f"[ATTACKER] ⚠️ Errore attributo {attr_idx}: {e}")
                    results_per_attribute[attr_idx] = {'error': str(e), 'success': False}
            
            # Calcola risultati aggregati
            successful_attacks = [r for r in results_per_attribute.values() if r.get('success', False)]
            avg_accuracy = np.mean([r['accuracy'] for r in successful_attacks]) if successful_attacks else 0.0
            avg_advantage = np.mean([r['advantage'] for r in successful_attacks]) if successful_attacks else 0.0
            
            results = {
                'attack_type': 'attribute_inference',
                'success': len(successful_attacks) > 0,
                'attributes_tested': len(target_attributes),
                'successful_attributes': len(successful_attacks),
                'success_rate': len(successful_attacks) / len(target_attributes),
                'average_accuracy': float(avg_accuracy),
                'average_advantage': float(avg_advantage),
                'per_attribute_results': results_per_attribute
            }
            
            print(f"[ATTACKER] ✅ Attribute Inference Attack completato:")
            print(f"[ATTACKER]   Attributi attaccati con successo: {len(successful_attacks)}/{len(target_attributes)}")
            print(f"[ATTACKER]   Success rate: {results['success_rate']*100:.1f}%")
            print(f"[ATTACKER]   Average advantage: {avg_advantage:.4f}")
            
            # Interpretazione
            if results['success_rate'] > 0.6 and avg_advantage > 0.1:
                interpretation = "🔴 ALTA VULNERABILITÀ - Attributi sensibili facilmente inferibili. Protezione insufficiente."
            elif results['success_rate'] > 0.3 or avg_advantage > 0.05:
                interpretation = "🟡 MEDIA VULNERABILITÀ - Alcuni attributi possono essere inferiti. Considerare feature masking."
            else:
                interpretation = "🟢 BASSA VULNERABILITÀ - Attributi ben protetti dal modello federato."
            
            results['interpretation'] = interpretation
            print(f"[ATTACKER]   Interpretazione: {interpretation}")
            
            return results
            
        except Exception as e:
            print(f"[ATTACKER] ❌ Errore Attribute Inference Attack: {e}")
            import traceback
            traceback.print_exc()
            return {'attack_type': 'attribute_inference', 'success': False, 'error': str(e)}
    
    def model_inversion_attack(self, attack_data):
        """
        Implementa Model Inversion Attack.
        
        COSA FA: Cerca di ricostruire input rappresentativi che massimizzano la confidenza per ogni classe.
        COME: Usa ottimizzazione per trovare pattern che il modello classifica con alta confidenza.
        
        INTERPRETAZIONE:
        - Max confidence > 0.9: ALTA vulnerabilità
        - Max confidence > 0.7: MEDIA vulnerabilità
        - Altrimenti: BASSA vulnerabilità
        
        Args:
            attack_data: Dizionario con dati preparati
            
        Returns:
            dict: Risultati con pattern ricostruiti e confidenze
        """
        print(f"[ATTACKER] 🎯 === MODEL INVERSION ATTACK ===")
        
        if self.target_model is None:
            raise ValueError("Devi prima caricare un modello target")
        
        try:
            from scipy.optimize import minimize
            
            X_members = attack_data['X_members']
            y_members = attack_data['y_members']
            n_features = attack_data['n_features']
            
            # Calcola statistiche per ogni classe
            class_stats = {}
            for class_label in [0, 1]:  # Natural vs Attack
                mask = (y_members == class_label)
                if np.sum(mask) > 0:
                    X_class = X_members[mask]
                    class_stats[class_label] = {
                        'mean': np.mean(X_class, axis=0),
                        'std': np.std(X_class, axis=0),
                        'min': np.min(X_class, axis=0),
                        'max': np.max(X_class, axis=0)
                    }
            
            print(f"[ATTACKER] Inversione per {len(class_stats)} classi...")
            
            inversion_results = {}
            
            for target_class in class_stats.keys():
                class_name = "Natural" if target_class == 0 else "Attack"
                print(f"[ATTACKER] Inversione classe {target_class} ({class_name})...")
                
                stats = class_stats[target_class]
                
                # Funzione obiettivo: massimizza confidenza per la classe target
                def objective_function(x):
                    try:
                        x_reshaped = x.reshape(1, -1)
                        proba = self.target_model.predict_proba(x_reshaped)[0]
                        confidence = proba[target_class]
                        
                        # Regolarizzazione: mantieni x realistico
                        reg_term = 0.01 * np.sum((x - stats['mean']) ** 2)
                        
                        return -confidence + reg_term  # Minimizza (-confidence + regolarizzazione)
                    except:
                        return 1e6  # Penalità alta per errori
                
                # Ottimizzazione con multiple restart
                best_confidence = 0
                best_pattern = stats['mean'].copy()
                
                for restart in range(3):
                    try:
                        if restart == 0:
                            x_init = stats['mean']
                        else:
                            noise = np.random.normal(0, 0.1 * stats['std'], n_features)
                            x_init = stats['mean'] + noise
                        
                        # Bounds basati sui dati osservati
                        bounds = [(stats['min'][i] - 0.1 * stats['std'][i], 
                                  stats['max'][i] + 0.1 * stats['std'][i]) 
                                 for i in range(n_features)]
                        
                        result = minimize(
                            objective_function,
                            x_init,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100}
                        )
                        
                        if result.success:
                            x_optimal = result.x
                            confidence = self.target_model.predict_proba(x_optimal.reshape(1, -1))[0, target_class]
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_pattern = x_optimal
                                
                    except Exception as e:
                        print(f"[ATTACKER] ⚠️ Errore restart {restart}: {e}")
                        continue
                
                # Calcola similarità con pattern reali
                similarity = 1 - np.linalg.norm(best_pattern - stats['mean']) / np.linalg.norm(stats['mean'])
                
                inversion_results[target_class] = {
                    'best_confidence': float(best_confidence),
                    'similarity_to_mean': float(similarity),
                    'class_name': class_name
                }
                
                print(f"[ATTACKER] Classe {target_class}: confidenza={best_confidence:.4f}, similarità={similarity:.4f}")
            
            # Risultati aggregati
            confidences = [r['best_confidence'] for r in inversion_results.values()]
            similarities = [r['similarity_to_mean'] for r in inversion_results.values()]
            
            results = {
                'attack_type': 'model_inversion',
                'success': len(inversion_results) > 0,
                'classes_inverted': len(inversion_results),
                'average_confidence': float(np.mean(confidences)),
                'max_confidence': float(np.max(confidences)),
                'average_similarity': float(np.mean(similarities)),
                'per_class_results': inversion_results
            }
            
            print(f"[ATTACKER] ✅ Model Inversion Attack completato:")
            print(f"[ATTACKER]   Classi invertite: {len(inversion_results)}")
            print(f"[ATTACKER]   Confidenza media: {results['average_confidence']:.4f}")
            print(f"[ATTACKER]   Confidenza massima: {results['max_confidence']:.4f}")
            
            # Interpretazione
            if results['max_confidence'] > 0.9:
                interpretation = "🔴 ALTA VULNERABILITÀ - Pattern delle classi facilmente ricostruibili. Privacy compromessa."
            elif results['max_confidence'] > 0.7:
                interpretation = "🟡 MEDIA VULNERABILITÀ - Alcune informazioni sulle classi estraibili. Protezione parziale."
            else:
                interpretation = "🟢 BASSA VULNERABILITÀ - Pattern delle classi ben protetti dal federato."
            
            results['interpretation'] = interpretation
            print(f"[ATTACKER]   Interpretazione: {interpretation}")
            
            return results
            
        except Exception as e:
            print(f"[ATTACKER] ❌ Errore Model Inversion Attack: {e}")
            import traceback
            traceback.print_exc()
            return {'attack_type': 'model_inversion', 'success': False, 'error': str(e)}
    
    def reconstruction_attack(self, attack_data, n_targets=5):
        """
        Implementa Reconstruction Attack.
        
        COSA FA: Cerca di ricostruire campioni specifici del training set.
        COME: Usa ottimizzazione per trovare input che producono predizioni simili ai target.
        
        INTERPRETAZIONE:
        - Success rate > 50% e quality > 0.7: ALTA vulnerabilità
        - Success rate > 20% o quality > 0.4: MEDIA vulnerabilità
        - Altrimenti: BASSA vulnerabilità
        
        Args:
            attack_data: Dizionario con dati preparati
            n_targets: Numero di campioni da tentare di ricostruire
            
        Returns:
            dict: Risultati con qualità delle ricostruzioni
        """
        print(f"[ATTACKER] 🎯 === RECONSTRUCTION ATTACK ===")
        print(f"[ATTACKER] Ricostruzione di {n_targets} campioni target...")
        
        if self.target_model is None:
            raise ValueError("Devi prima caricare un modello target")
        
        try:
            from scipy.optimize import minimize
            from scipy.spatial.distance import euclidean
            
            X_members = attack_data['X_members']
            y_members = attack_data['y_members']
            X_test = attack_data['X_test'][:n_targets]
            y_test = attack_data['y_test'][:n_targets]
            
            reconstruction_results = []
            
            for i, (target_sample, target_label) in enumerate(zip(X_test, y_test)):
                print(f"[ATTACKER] Ricostruzione campione {i+1}/{n_targets}...")
                
                # Ottieni predizione del modello target sul campione originale
                target_prediction = self.target_model.predict_proba(target_sample.reshape(1, -1))[0]
                
                # Funzione obiettivo: minimizza distanza tra predizioni
                def reconstruction_objective(x):
                    try:
                        x_reshaped = x.reshape(1, -1)
                        current_prediction = self.target_model.predict_proba(x_reshaped)[0]
                        
                        # Distanza tra predizioni
                        prediction_loss = np.sum((current_prediction - target_prediction) ** 2)
                        
                        # Regolarizzazione per mantenere x realistico
                        class_mask = (y_members == target_label)
                        if np.sum(class_mask) > 0:
                            class_mean = np.mean(X_members[class_mask], axis=0)
                            class_std = np.std(X_members[class_mask], axis=0)
                            reg_term = 0.1 * np.sum(((x - class_mean) / (class_std + 1e-8)) ** 2)
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
                            class_mask = (y_members == target_label)
                            if np.sum(class_mask) > 0:
                                x_init = np.mean(X_members[class_mask], axis=0)
                            else:
                                x_init = np.mean(X_members, axis=0)
                        else:
                            # Random initialization
                            x_init = np.random.normal(
                                np.mean(X_members, axis=0),
                                np.std(X_members, axis=0)
                            )
                        
                        # Bounds basati sui dati osservati
                        bounds = [(np.min(X_members[:, j]), np.max(X_members[:, j])) 
                                 for j in range(X_members.shape[1])]
                        
                        result = minimize(
                            reconstruction_objective,
                            x_init,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100}
                        )
                        
                        if result.success and result.fun < best_distance:
                            best_distance = result.fun
                            best_reconstruction = result.x
                            
                    except Exception as e:
                        print(f"[ATTACKER] ⚠️ Errore restart {restart}: {e}")
                        continue
                
                if best_reconstruction is not None:
                    # Calcola metriche di qualità
                    reconstruction_distance = euclidean(target_sample, best_reconstruction)
                    max_possible_distance = np.sqrt(np.sum((np.max(X_members, axis=0) - np.min(X_members, axis=0)) ** 2))
                    normalized_distance = reconstruction_distance / max_possible_distance
                    reconstruction_quality = 1 - normalized_distance
                    
                    reconstruction_results.append({
                        'target_index': i,
                        'reconstruction_quality': float(reconstruction_quality),
                        'euclidean_distance': float(reconstruction_distance),
                        'normalized_distance': float(normalized_distance),
                        'success': reconstruction_quality > 0.5
                    })
                    
                    print(f"[ATTACKER] Campione {i+1}: qualità={reconstruction_quality:.4f}")
                else:
                    reconstruction_results.append({
                        'target_index': i,
                        'reconstruction_quality': 0.0,
                        'success': False
                    })
            
            # Risultati aggregati
            successful = [r for r in reconstruction_results if r.get('success', False)]
            success_rate = len(successful) / len(reconstruction_results)
            avg_quality = np.mean([r['reconstruction_quality'] for r in successful]) if successful else 0.0
            
            results = {
                'attack_type': 'reconstruction',
                'success': success_rate > 0,
                'targets_attempted': len(reconstruction_results),
                'successful_reconstructions': len(successful),
                'success_rate': float(success_rate),
                'average_quality': float(avg_quality),
                'individual_results': reconstruction_results
            }
            
            print(f"[ATTACKER] ✅ Reconstruction Attack completato:")
            print(f"[ATTACKER]   Success rate: {success_rate*100:.1f}%")
            print(f"[ATTACKER]   Qualità media: {avg_quality:.4f}")
            
            # Interpretazione
            if success_rate > 0.5 and avg_quality > 0.7:
                interpretation = "🔴 ALTA VULNERABILITÀ - Campioni training facilmente ricostruibili. Grave leak privacy."
            elif success_rate > 0.2 or avg_quality > 0.4:
                interpretation = "🟡 MEDIA VULNERABILITÀ - Alcune ricostruzioni possibili. Protezione parziale."
            else:
                interpretation = "🟢 BASSA VULNERABILITÀ - Dati training ben protetti dal federato."
            
            results['interpretation'] = interpretation
            print(f"[ATTACKER]   Interpretazione: {interpretation}")
            
            return results
            
        except Exception as e:
            print(f"[ATTACKER] ❌ Errore Reconstruction Attack: {e}")
            import traceback
            traceback.print_exc()
            return {'attack_type': 'reconstruction', 'success': False, 'error': str(e)}
    
    def _custom_membership_inference(self, attack_data):
        """
        Implementazione custom di membership inference quando ART non è disponibile.
        
        LOGICA: Usa le confidenze/predizioni del modello target come feature per un attack model
        che impara a distinguere membri da non-membri.
        """
        X_members = attack_data['X_members']
        X_nonmembers = attack_data['X_nonmembers']
        X_test = attack_data['X_test']
        
        # Ottieni predizioni
        member_probs = self.target_model.predict_proba(X_members)
        nonmember_probs = self.target_model.predict_proba(X_nonmembers)
        test_probs = self.target_model.predict_proba(X_test)
        
        # Estrai features dalle predizioni
        def extract_features(probs, labels):
            features = []
            for prob, label in zip(probs, labels):
                confidence = np.max(prob)
                entropy = -np.sum(prob * np.log(prob + 1e-8))
                loss = -np.log(prob[int(label)] + 1e-8)
                features.append([prob[0], prob[1], confidence, entropy, loss])
            return np.array(features)
        
        member_features = extract_features(member_probs, attack_data['y_members'])
        nonmember_features = extract_features(nonmember_probs, attack_data['y_nonmembers'])
        test_features = extract_features(test_probs, attack_data['y_test'])
        
        # Crea training set per attack model
        X_attack = np.vstack([member_features, nonmember_features])
        y_attack = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])
        
        # Addestra attack model
        attack_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        attack_model.fit(X_attack, y_attack)
        
        # Predici membership
        membership_pred = attack_model.predict(test_features)
        membership_prob = attack_model.predict_proba(test_features)[:, 1]
        
        return membership_pred, membership_prob
    
    def _save_results(self, results):
        """
        Salva i risultati degli attacchi in formato JSON e TXT.
        
        Args:
            results: Dizionario con risultati completi
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salva JSON dettagliato
        json_file = os.path.join(self.output_dir, f"inference_attacks_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Salva report testuale
        report_file = os.path.join(self.output_dir, f"attack_report_{timestamp}.txt")
        self._generate_text_report(results, report_file)
        
        print(f"\n💾 Risultati salvati:")
        print(f"   📄 JSON dettagliato: {json_file}")
        print(f"   📄 Report testuale: {report_file}")
    
    def _generate_text_report(self, results, filename):
        """
        Genera report testuale leggibile.
        
        Args:
            results: Dizionario risultati
            filename: Path del file di output
        """
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORT ATTACCHI DI INFERENZA\n")
            f.write("Sistema Federated Random Forest SmartGrid\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
            f.write(f"Sistema target: {results.get('target_system', 'N/A')}\n")
            
            if 'training_info' in results:
                info = results['training_info']
                f.write(f"Client di training: {info.get('training_clients', 'N/A')}\n")
                f.write(f"Accuracy modello: {info.get('accuracy', 0):.4f}\n")
                f.write(f"N. campioni: {info.get('n_samples', 'N/A')}\n")
                f.write(f"N. features: {info.get('n_features', 'N/A')}\n\n")
            
            f.write("RISULTATI ATTACCHI:\n")
            f.write("-"*40 + "\n")
            
            for attack_name, attack_result in results.get('attacks', {}).items():
                f.write(f"\n{attack_name.upper().replace('_', ' ')}:\n")
                
                if attack_result.get('success', False):
                    f.write("  Status: SUCCESSO ✅\n")
                    
                    if attack_name == 'membership_inference':
                        f.write(f"  Accuracy: {attack_result.get('accuracy', 0):.4f}\n")
                        f.write(f"  Attack Advantage: {attack_result.get('attack_advantage', 0):.4f}\n")
                        f.write(f"  AUC: {attack_result.get('auc', 0):.4f}\n")
                        
                    elif attack_name == 'attribute_inference':
                        f.write(f"  Success Rate: {attack_result.get('success_rate', 0):.4f}\n")
                        f.write(f"  Average Advantage: {attack_result.get('average_advantage', 0):.4f}\n")
                        
                    elif attack_name == 'model_inversion':
                        f.write(f"  Average Confidence: {attack_result.get('average_confidence', 0):.4f}\n")
                        f.write(f"  Max Confidence: {attack_result.get('max_confidence', 0):.4f}\n")
                        
                    elif attack_name == 'reconstruction':
                        f.write(f"  Success Rate: {attack_result.get('success_rate', 0):.4f}\n")
                        f.write(f"  Average Quality: {attack_result.get('average_quality', 0):.4f}\n")
                    
                    f.write(f"  Interpretazione: {attack_result.get('interpretation', 'N/A')}\n")
                else:
                    f.write("  Status: FALLITO ❌\n")
                    f.write(f"  Errore: {attack_result.get('error', 'Errore sconosciuto')}\n")
            
            if 'summary' in results:
                summary = results['summary']
                f.write(f"\nVULNERABILITÀ COMPLESSIVA: {summary.get('overall_vulnerability_score', 0):.1f}%\n")
                f.write(f"ATTACCHI RIUSCITI: {summary.get('successful_attacks', 0)}/4\n")
                f.write(f"RISK ASSESSMENT: {summary.get('risk_assessment', 'N/A')}\n")


# ====== FUNZIONE PRINCIPALE PER USO FACILE ======

def run_inference_attacks_on_your_system():
    """
    Funzione principale per eseguire attacchi di inferenza sul sistema esistente.
    
    NOTA: Questa funzione ADDESTRA un nuovo modello. Se vuoi testare un modello
    già salvato, usa invece lo script run_attacks_on_saved_model.py
    
    Returns:
        dict: Risultati completi degli attacchi
    """
    print("🎯 AVVIO ATTACCHI DI INFERENZA SUL SISTEMA FEDERATO")
    print("=" * 70)
    
    if not SYSTEM_AVAILABLE:
        print("❌ Sistema clientRFtmp.py non disponibile")
        print("   Assicurati che inference_attacks_integration.py sia nella stessa directory di clientRFtmp.py")
        return None
    
    try:
        # Inizializza l'attaccante
        attacker = FederatedInferenceAttacker()
        
        # Estrai modello federato dal sistema
        print("\n[STEP 1] Estrazione modello federato...")
        federated_model = attacker.extract_federated_model(
            training_client_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        )
        
        # Carica dati per attacchi
        print("\n[STEP 2] Caricamento dati per attacchi...")
        attack_data = attacker.load_attack_data(
            member_clients=[1, 2, 3],
            nonmember_clients=[14, 15]
        )
        
        # Esegui tutti gli attacchi
        print("\n[STEP 3] Esecuzione 4 attacchi di inferenza...")
        
        all_results = {
            'evaluation_id': f"attack_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'target_system': 'federated_random_forest_smartgrid',
            'training_info': attacker.training_info,
            'attacks': {}
        }
        
        # Membership Inference
        print(f"\n🎯 [1/4] MEMBERSHIP INFERENCE ATTACK")
        all_results['attacks']['membership_inference'] = attacker.membership_inference_attack(attack_data)
        
        # Attribute Inference
        print(f"\n🎯 [2/4] ATTRIBUTE INFERENCE ATTACK")
        all_results['attacks']['attribute_inference'] = attacker.attribute_inference_attack(attack_data)
        
        # Model Inversion
        print(f"\n🎯 [3/4] MODEL INVERSION ATTACK")
        all_results['attacks']['model_inversion'] = attacker.model_inversion_attack(attack_data)
        
        # Reconstruction
        print(f"\n🎯 [4/4] RECONSTRUCTION ATTACK")
        all_results['attacks']['reconstruction'] = attacker.reconstruction_attack(attack_data)
        
        # Summary
        successful_attacks = [name for name, result in all_results['attacks'].items() 
                            if result.get('success', False)]
        
        vulnerability_scores = []
        for attack_name, result in all_results['attacks'].items():
            if not result.get('success', False):
                continue
            
            if attack_name == 'membership_inference':
                score = result.get('attack_advantage', 0) * 100
            elif attack_name == 'attribute_inference':
                score = result.get('average_advantage', 0) * 100
            elif attack_name == 'model_inversion':
                score = result.get('max_confidence', 0) * 100
            elif attack_name == 'reconstruction':
                score = result.get('success_rate', 0) * 100
            else:
                score = 0
            
            vulnerability_scores.append(max(0, score))
        
        overall_vulnerability = np.mean(vulnerability_scores) if vulnerability_scores else 0.0
        
        # Risk assessment
        def assess_risk(vuln_score, n_successful):
            if vuln_score > 50 or n_successful >= 3:
                return "🔴 ALTO RISCHIO"
            elif vuln_score > 25 or n_successful >= 2:
                return "🟡 MEDIO RISCHIO"
            else:
                return "🟢 BASSO RISCHIO"
        
        all_results['summary'] = {
            'total_attacks': 4,
            'successful_attacks': len(successful_attacks),
            'success_rate': len(successful_attacks) / 4,
            'successful_attack_types': successful_attacks,
            'overall_vulnerability_score': float(overall_vulnerability),
            'risk_assessment': assess_risk(overall_vulnerability, len(successful_attacks))
        }
        
        # Salva risultati
        attacker._save_results(all_results)
        
        print("\n✅ Valutazione attacchi di inferenza completata!")
        print("📁 Controlla la cartella 'attack_results' per i risultati dettagliati")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Errore durante attacchi: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Esegui attacchi se il file viene lanciato direttamente
    print("="*80)
    print("FRAMEWORK ATTACCHI DI INFERENZA - FEDERATED RANDOM FOREST")
    print("="*80)
    print("\nQuesta è una libreria. Per eseguire attacchi, usa:")
    print("  - run_inference_attacks.py (per addestrare nuovo modello)")
    print("  - run_attacks_on_saved_model.py <model.pkl> (per modello già salvato)")
    print("\nO importa questa libreria nel tuo script Python.")
    print("="*80)