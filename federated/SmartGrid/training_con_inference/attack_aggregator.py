"""
Aggregatore Metriche Attacchi Privacy - Formato JSON Strutturato
Francesca Pellegrino

Questo script analizza tutti i file JSON generati dagli attacchi privacy
e calcola le medie per ogni metrica, mantenendo la struttura JSON originale.
"""

import json
import os
import glob
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import statistics

class StructuredAttackMetricsAggregator:
    """
    Classe per aggregare i risultati degli attacchi privacy mantenendo
    la struttura JSON originale con valori aggregati (medie).
    """
    
    def __init__(self):
        self.attack_files = []
        self.raw_results = []
        self.aggregated_json = {}
        
    def find_attack_files(self, directory: str = ".") -> List[str]:
        """
        Trova tutti i file JSON contenenti risultati degli attacchi.
        
        Args:
            directory: Directory dove cercare i file (default: directory corrente)
        
        Returns:
            Lista dei percorsi dei file trovati
        """
        # Pattern per trovare i file degli attacchi
        patterns = [
            "attack_results_compatible_client_*.json",
            "attack_results_*.json",
            "*attack*.json"
        ]
        
        found_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            found_files.extend(files)
        
        # Rimuovi duplicati
        found_files = list(set(found_files))
        
        print(f"🔍 Trovati {len(found_files)} file di attacchi:")
        for file in found_files:
            print(f"  - {file}")
        
        self.attack_files = found_files
        return found_files
    
    def load_attack_results(self) -> bool:
        """
        Carica tutti i risultati degli attacchi dai file JSON.
        
        Returns:
            True se almeno un file è stato caricato con successo
        """
        if not self.attack_files:
            print("❌ Nessun file di attacchi trovato. Usa find_attack_files() prima.")
            return False
        
        self.raw_results = []
        successful_loads = 0
        
        for file_path in self.attack_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Aggiungi metadati del file
                data['_metadata'] = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path)
                }
                
                self.raw_results.append(data)
                successful_loads += 1
                print(f"✅ Caricato: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ Errore caricamento {file_path}: {e}")
                continue
        
        print(f"\n📊 Caricati con successo {successful_loads}/{len(self.attack_files)} file")
        return successful_loads > 0
    
    def safe_numeric_value(self, value, default=0.0):
        """Converte un valore in numerico gestendo NaN, inf, e casi speciali."""
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return default
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except:
                    return default
            return default
        except:
            return default
    
    def aggregate_numeric_values(self, values: List, key_name: str = "unknown") -> float:
        """
        Aggrega una lista di valori numerici calcolando la media.
        
        Args:
            values: Lista di valori da aggregare
            key_name: Nome della chiave per debug
            
        Returns:
            Media dei valori validi
        """
        if not values:
            return 0.0
        
        # Converte tutti i valori in numerico
        numeric_values = [self.safe_numeric_value(v) for v in values]
        
        # Filtra valori validi (non zero se sono booleani/flags)
        if key_name in ['attack_success', 'federated_learning_compromised'] or 'criterion' in key_name:
            # Per i flag booleani, mantieni anche gli zeri
            valid_values = [v for v in numeric_values if v is not None]
        else:
            # Per le altre metriche, rimuovi zeri che potrebbero essere errori
            valid_values = [v for v in numeric_values if v != 0.0 or len(numeric_values) == 1]
        
        if not valid_values:
            return 0.0
        
        mean_value = np.mean(valid_values)
        
        # Debug per valori sospetti
        if len(valid_values) != len(values):
            print(f"   ⚠️ {key_name}: {len(valid_values)}/{len(values)} valori validi, media: {mean_value:.4f}")
        
        return float(mean_value)
    
    def aggregate_string_values(self, values: List) -> str:
        """
        Aggrega valori stringa prendendo il più comune.
        
        Args:
            values: Lista di stringhe
            
        Returns:
            Stringa più frequente
        """
        if not values:
            return ""
        
        # Conta frequenze
        from collections import Counter
        counter = Counter(str(v) for v in values if v is not None)
        
        if counter:
            most_common = counter.most_common(1)[0][0]
            return most_common
        
        return str(values[0]) if values else ""
    
    def calculate_structured_aggregation(self) -> Dict[str, Any]:
        """
        Calcola l'aggregazione mantenendo la struttura JSON originale.
        
        Returns:
            Dizionario JSON aggregato con la struttura originale
        """
        if not self.raw_results:
            print("❌ Nessun risultato caricato. Usa load_attack_results() prima.")
            return {}
        
        print(f"📈 Calcolando aggregazione strutturata per {len(self.raw_results)} client...")
        
        # Struttura del JSON aggregato
        aggregated = {
            "membership_inference": {},
            "property_inference": {},
            "model_inversion": {},
            "attack_summary": {},
            "metadata": {
                "aggregation_info": {
                    "total_clients_analyzed": len(self.raw_results),
                    "aggregation_timestamp": datetime.now().isoformat(),
                    "aggregation_method": "arithmetic_mean",
                    "files_processed": len(self.attack_files),
                    "explanation": "Tutti i valori numerici sono medie calcolate sui risultati di tutti i client analizzati. I valori stringa rappresentano il valore più comune."
                }
            }
        }
        
        # 1. MEMBERSHIP INFERENCE AGGREGATION
        print("📊 Aggregando Membership Inference Attack...")
        
        # Raccogli tutti i valori per membership inference
        mia_values = {}
        mia_keys = [
            'attack_success', 'confidence_based_accuracy', 'gradient_based_signal',
            'gradient_samples_analyzed', 'combined_accuracy', 'privacy_breach_score',
            'samples_analyzed'
        ]
        
        # Raccogli valori da tutti i client
        for key in mia_keys:
            values = []
            for result in self.raw_results:
                if 'membership_inference' in result and key in result['membership_inference']:
                    values.append(result['membership_inference'][key])
            mia_values[key] = values
        
        # Aggrega membership inference
        aggregated['membership_inference'] = {
            "attack_type": "Membership Inference Attack",
            "attack_success": int(round(self.aggregate_numeric_values(mia_values.get('attack_success', []), 'attack_success'))),
            "attack_success_criteria": "Considerato successo (1) se combined_accuracy > 0.6",
            "confidence_based_accuracy": self.aggregate_numeric_values(mia_values.get('confidence_based_accuracy', []), 'confidence_based_accuracy'),
            "confidence_based_accuracy_explanation": "Percentuale media di dati per cui l'attacco indovina correttamente la membership. Valore >0.5 indica che l'attacco è migliore del caso random.",
            "gradient_based_signal": self.aggregate_numeric_values(mia_values.get('gradient_based_signal', []), 'gradient_based_signal'),
            "gradient_signal_explanation": "Intensità media del segnale ottenuto confrontando i gradienti tra membri e non membri.",
            "gradient_samples_analyzed": int(self.aggregate_numeric_values(mia_values.get('gradient_samples_analyzed', []), 'gradient_samples_analyzed')),
            "combined_accuracy": self.aggregate_numeric_values(mia_values.get('combined_accuracy', []), 'combined_accuracy'),
            "combined_accuracy_explanation": "Accuratezza media combinando tecniche diverse di attacco. Valore >0.5 indica successo dell'attacco.",
            "privacy_breach_score": self.aggregate_numeric_values(mia_values.get('privacy_breach_score', []), 'privacy_breach_score'),
            "privacy_breach_score_explanation": "Score medio calcolato come 2*(combined_accuracy-0.5), rappresenta il rischio di violazione privacy rispetto al caso random.",
            "samples_analyzed": int(self.aggregate_numeric_values(mia_values.get('samples_analyzed', []), 'samples_analyzed'))
        }
        
        # 2. PROPERTY INFERENCE AGGREGATION
        print("📊 Aggregando Property Inference Attack...")
        
        # Raccogli valori per property inference
        pia_values = {}
        pia_keys = [
            'attack_success', 'success_rate', 'properties_detected', 'total_properties',
            'estimated_attack_ratio', 'actual_attack_ratio', 'estimation_error',
            'estimation_accuracy', 'samples_analyzed'
        ]
        
        for key in pia_keys:
            values = []
            for result in self.raw_results:
                if 'property_inference' in result and key in result['property_inference']:
                    values.append(result['property_inference'][key])
            pia_values[key] = values
        
        # Raccogli privacy_breach_level (stringa)
        privacy_levels = []
        for result in self.raw_results:
            if 'property_inference' in result and 'privacy_breach_level' in result['property_inference']:
                privacy_levels.append(result['property_inference']['privacy_breach_level'])
        
        aggregated['property_inference'] = {
            "attack_type": "Property Inference Attack",
            "attack_success": int(round(self.aggregate_numeric_values(pia_values.get('attack_success', []), 'attack_success'))),
            "attack_success_criteria": "Considerato successo (1) se il numero di proprietà rilevate supera la soglia definita nei success_criteria.",
            "success_rate": self.aggregate_numeric_values(pia_values.get('success_rate', []), 'success_rate'),
            "success_rate_explanation": "Percentuale media di proprietà sensibili indovinate rispetto al totale.",
            "properties_detected": int(self.aggregate_numeric_values(pia_values.get('properties_detected', []), 'properties_detected')),
            "total_properties": int(self.aggregate_numeric_values(pia_values.get('total_properties', []), 'total_properties')),
            "properties_explanation": "Numero medio di proprietà sensibili che l'attacco è riuscito a inferire sul totale delle proprietà testate.",
            "privacy_breach_level": self.aggregate_string_values(privacy_levels),
            "privacy_breach_level_explanation": "Livello qualitativo di rischio privacy più comune stimato in base al successo dell'attacco.",
            "estimated_attack_ratio": self.aggregate_numeric_values(pia_values.get('estimated_attack_ratio', []), 'estimated_attack_ratio'),
            "actual_attack_ratio": self.aggregate_numeric_values(pia_values.get('actual_attack_ratio', []), 'actual_attack_ratio'),
            "attack_ratio_explanation": "Stima media e valore reale medio della proporzione di dati/utenti colpiti dall'attacco.",
            "estimation_error": self.aggregate_numeric_values(pia_values.get('estimation_error', []), 'estimation_error'),
            "estimation_error_explanation": "Errore assoluto medio tra attack ratio stimato e reale. Più piccolo è, migliore è la stima dell'attaccante.",
            "estimation_accuracy": self.aggregate_numeric_values(pia_values.get('estimation_accuracy', []), 'estimation_accuracy'),
            "estimation_accuracy_explanation": "Accuratezza media (1-errore relativo) della stima dell'attaccante rispetto al valore reale.",
            "samples_analyzed": int(self.aggregate_numeric_values(pia_values.get('samples_analyzed', []), 'samples_analyzed'))
        }
        
        # 3. MODEL INVERSION AGGREGATION
        print("📊 Aggregando Model Inversion Attack...")
        
        # Raccogli valori per model inversion
        miva_values = {}
        miva_keys = [
            'attack_success', 'normal_confidence', 'attack_confidence', 'avg_confidence',
            'information_leakage_score', 'confidence_component', 'separation_component',
            'distance_component', 'high_conf_normal_samples', 'high_conf_attack_samples',
            'prototype_separation', 'prototype_l2_distance', 'prototype_cosine_similarity',
            'samples_analyzed'
        ]
        
        for key in miva_keys:
            values = []
            for result in self.raw_results:
                if 'model_inversion' in result and key in result['model_inversion']:
                    values.append(result['model_inversion'][key])
            miva_values[key] = values
        
        # Raccogli valori stringa
        normal_methods = []
        attack_methods = []
        normal_thresholds = []
        attack_thresholds = []
        
        for result in self.raw_results:
            if 'model_inversion' in result:
                mi = result['model_inversion']
                if 'best_normal_method' in mi:
                    normal_methods.append(mi['best_normal_method'])
                if 'best_attack_method' in mi:
                    attack_methods.append(mi['best_attack_method'])
                if 'normal_threshold_used' in mi:
                    normal_thresholds.append(mi['normal_threshold_used'])
                if 'attack_threshold_used' in mi:
                    attack_thresholds.append(mi['attack_threshold_used'])
        
        # Raccogli success_criteria
        success_criteria_values = {}
        criteria_keys = ['confidence_criterion', 'separation_criterion', 'leakage_criterion', 'sample_criterion', 'total_successful']
        
        for key in criteria_keys:
            values = []
            for result in self.raw_results:
                if 'model_inversion' in result and 'success_criteria' in result['model_inversion']:
                    if key in result['model_inversion']['success_criteria']:
                        values.append(result['model_inversion']['success_criteria'][key])
            success_criteria_values[key] = values
        
        aggregated['model_inversion'] = {
            "attack_type": "Model Inversion Attack",
            "attack_success": int(round(self.aggregate_numeric_values(miva_values.get('attack_success', []), 'attack_success'))),
            "attack_success_criteria": "Considerato successo (1) se almeno 3 criteri su 4 nei success_criteria sono soddisfatti.",
            "normal_confidence": self.aggregate_numeric_values(miva_values.get('normal_confidence', []), 'normal_confidence'),
            "normal_confidence_criteria": "Confidenza media per i prototipi normali (non invertiti), usata come baseline.",
            "attack_confidence": self.aggregate_numeric_values(miva_values.get('attack_confidence', []), 'attack_confidence'),
            "attack_confidence_explanation": "Confidenza media per i prototipi invertiti (generati dall'attacco).",
            "avg_confidence": self.aggregate_numeric_values(miva_values.get('avg_confidence', []), 'avg_confidence'),
            "avg_confidence_explanation": "Confidenza media complessiva dei prototipi generati dall'attacco.",
            "information_leakage_score": self.aggregate_numeric_values(miva_values.get('information_leakage_score', []), 'information_leakage_score'),
            "information_leakage_score_explanation": "Valore aggregato medio che quantifica il grado di informazione sensibile estratta tramite inversione. Valori >0.5 indicano forte leakage rispetto al baseline.",
            "confidence_component": self.aggregate_numeric_values(miva_values.get('confidence_component', []), 'confidence_component'),
            "confidence_component_explanation": "Componente media dovuta al livello di confidenza raggiunto dai prototipi invertiti.",
            "separation_component": self.aggregate_numeric_values(miva_values.get('separation_component', []), 'separation_component'),
            "separation_component_explanation": "Componente media dovuta alla separazione tra prototipi attaccati e prototipi normali (maggiore significa che si distinguono meglio).",
            "distance_component": self.aggregate_numeric_values(miva_values.get('distance_component', []), 'distance_component'),
            "distance_component_explanation": "Componente media dovuta alla distanza (L2) tra prototipi normali e invertiti; valori bassi indicano forte somiglianza.",
            "high_conf_normal_samples": int(self.aggregate_numeric_values(miva_values.get('high_conf_normal_samples', []), 'high_conf_normal_samples')),
            "high_conf_normal_samples_explanation": "Numero medio di prototipi normali che superano la soglia di confidenza baseline.",
            "high_conf_attack_samples": int(self.aggregate_numeric_values(miva_values.get('high_conf_attack_samples', []), 'high_conf_attack_samples')),
            "high_conf_attack_samples_explanation": "Numero medio di prototipi invertiti che superano la soglia di confidenza attacco.",
            "normal_threshold_used": self.aggregate_string_values(normal_thresholds),
            "normal_threshold_explanation": "Soglia di confidenza più comune usata per considerare un prototipo normale.",
            "attack_threshold_used": self.aggregate_string_values(attack_thresholds),
            "attack_threshold_explanation": "Soglia di confidenza più comune usata per considerare un prototipo invertito.",
            "prototype_separation": self.aggregate_numeric_values(miva_values.get('prototype_separation', []), 'prototype_separation'),
            "prototype_separation_explanation": "Distanza media (in spazio feature) tra i prototipi normali e quelli invertiti.",
            "prototype_l2_distance": self.aggregate_numeric_values(miva_values.get('prototype_l2_distance', []), 'prototype_l2_distance'),
            "prototype_l2_distance_explanation": "Distanza euclidea media tra prototipi normali e invertiti.",
            "prototype_cosine_similarity": self.aggregate_numeric_values(miva_values.get('prototype_cosine_similarity', []), 'prototype_cosine_similarity'),
            "prototype_cosine_similarity_explanation": "Similarità coseno media tra prototipi normali e invertiti; valori vicini a 1 indicano grande somiglianza.",
            "best_normal_method": self.aggregate_string_values(normal_methods),
            "best_normal_method_explanation": "Tecnica più comune che ha prodotto i migliori prototipi normali.",
            "best_attack_method": self.aggregate_string_values(attack_methods),
            "best_attack_method_explanation": "Tecnica più comune che ha prodotto i migliori prototipi invertiti.",
            "success_criteria": {
                "confidence_criterion": int(round(self.aggregate_numeric_values(success_criteria_values.get('confidence_criterion', []), 'confidence_criterion'))),
                "confidence_criterion_explanation": "Vero se la confidenza media dei prototipi invertiti supera la soglia stabilita.",
                "separation_criterion": int(round(self.aggregate_numeric_values(success_criteria_values.get('separation_criterion', []), 'separation_criterion'))),
                "separation_criterion_explanation": "Vero se la separazione media tra prototipi invertiti e normali è sufficiente.",
                "leakage_criterion": int(round(self.aggregate_numeric_values(success_criteria_values.get('leakage_criterion', []), 'leakage_criterion'))),
                "leakage_criterion_explanation": "Vero se l'information leakage score medio supera la soglia.",
                "sample_criterion": int(round(self.aggregate_numeric_values(success_criteria_values.get('sample_criterion', []), 'sample_criterion'))),
                "sample_criterion_explanation": "Vero se il numero medio di campioni invertiti ad alta confidenza è significativo.",
                "total_successful": int(self.aggregate_numeric_values(success_criteria_values.get('total_successful', []), 'total_successful'))
            },
            "samples_analyzed": int(self.aggregate_numeric_values(miva_values.get('samples_analyzed', []), 'samples_analyzed'))
        }
        
        # 4. ATTACK SUMMARY AGGREGATION
        print("📊 Aggregando Attack Summary...")
        
        # Raccogli valori per attack summary
        summary_values = {}
        summary_keys = [
            'total_attacks_attempted', 'successful_attacks', 'attack_success_rate',
            'privacy_risk_score', 'client_id', 'federated_learning_compromised'
        ]
        
        for key in summary_keys:
            values = []
            for result in self.raw_results:
                if 'attack_summary' in result and key in result['attack_summary']:
                    values.append(result['attack_summary'][key])
            summary_values[key] = values
        
        aggregated['attack_summary'] = {
            "total_attacks_attempted": int(self.aggregate_numeric_values(summary_values.get('total_attacks_attempted', []), 'total_attacks_attempted')),
            "total_attacks_explanation": "Numero medio totale di tipologie di attacco privacy testate sui client.",
            "successful_attacks": int(round(self.aggregate_numeric_values(summary_values.get('successful_attacks', []), 'successful_attacks'))),
            "successful_attacks_explanation": "Numero medio di attacchi che hanno superato il criterio di successo e rappresentano un rischio concreto per la privacy.",
            "attack_success_rate": self.aggregate_numeric_values(summary_values.get('attack_success_rate', []), 'attack_success_rate'),
            "attack_success_rate_explanation": "Frazione media di attacchi riusciti sul totale di quelli tentati. Valori vicini a 1 indicano alta vulnerabilità.",
            "privacy_risk_score": self.aggregate_numeric_values(summary_values.get('privacy_risk_score', []), 'privacy_risk_score'),
            "privacy_risk_score_explanation": "Indice aggregato medio (calcolato per combinare i risultati di tutti gli attacchi) che misura il rischio privacy complessivo.",
            "client_id": int(self.aggregate_numeric_values(summary_values.get('client_id', []), 'client_id')),
            "client_id_explanation": "Identificativo numerico medio dei client federati analizzati (valore indicativo).",
            "federated_learning_compromised": int(round(self.aggregate_numeric_values(summary_values.get('federated_learning_compromised', []), 'federated_learning_compromised'))),
            "federated_learning_compromised_explanation": "True se in media le principali tipologie di attacco hanno avuto successo, segnalando che il sistema federato è compromesso dal punto di vista privacy."
        }
        
        self.aggregated_json = aggregated
        return aggregated
    
    def save_aggregated_json(self, output_file: str = None) -> str:
        """
        Salva i risultati aggregati nel formato JSON strutturato.
        
        Args:
            output_file: Nome del file di output (se None, genera automaticamente)
            
        Returns:
            Nome del file creato
        """
        if not self.aggregated_json:
            print("❌ Nessuna aggregazione da salvare. Esegui calculate_structured_aggregation() prima.")
            return ""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aggregated_attacks_structured_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.aggregated_json, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON aggregato strutturato salvato: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Errore nel salvataggio: {e}")
            return ""
    
    def print_structured_summary(self):
        """Stampa un summary strutturato leggibile."""
        if not self.aggregated_json:
            print("❌ Nessuna aggregazione disponibile.")
            return
        
        agg = self.aggregated_json
        
        print("\n" + "="*80)
        print("📊 REPORT AGGREGATO ATTACCHI PRIVACY - FEDERATED LEARNING")
        print("=" * 80)
        
        # Metadata
        meta = agg['metadata']['aggregation_info']
        print(f"\n📋 INFORMAZIONI AGGREGAZIONE:")
        print(f"   Client analizzati: {meta['total_clients_analyzed']}")
        print(f"   File processati: {meta['files_processed']}")
        print(f"   Metodo: {meta['aggregation_method']}")
        print(f"   Timestamp: {meta['aggregation_timestamp']}")
        
        # Attack Summary
        summary = agg['attack_summary']
        print(f"\n🎯 SUMMARY GENERALE (MEDIE):")
        print(f"   Attacchi tentati (media): {summary['total_attacks_attempted']}")
        print(f"   Attacchi riusciti (media): {summary['successful_attacks']}")
        print(f"   Tasso successo: {summary['attack_success_rate']*100:.1f}%")
        print(f"   Privacy risk score: {summary['privacy_risk_score']:.3f}")
        print(f"   FL compromesso: {'SÌ' if summary['federated_learning_compromised'] else 'NO'}")
        
        # Membership Inference
        mia = agg['membership_inference']
        print(f"\n🔍 MEMBERSHIP INFERENCE (MEDIE):")
        print(f"   Successo: {'SÌ' if mia['attack_success'] else 'NO'}")
        print(f"   Confidence accuracy: {mia['confidence_based_accuracy']:.3f}")
        print(f"   Combined accuracy: {mia['combined_accuracy']:.3f}")
        print(f"   Privacy breach score: {mia['privacy_breach_score']:.3f}")
        print(f"   Campioni analizzati: {mia['samples_analyzed']}")
        
        # Property Inference
        pia = agg['property_inference']
        print(f"\n🔍 PROPERTY INFERENCE (MEDIE):")
        print(f"   Successo: {'SÌ' if pia['attack_success'] else 'NO'}")
        print(f"   Success rate: {pia['success_rate']:.3f}")
        print(f"   Proprietà rilevate: {pia['properties_detected']}/{pia['total_properties']}")
        print(f"   Errore stima: {pia['estimation_error']:.3f}")
        print(f"   Privacy level: {pia['privacy_breach_level']}")
        
        # Model Inversion
        miva = agg['model_inversion']
        print(f"\n🔍 MODEL INVERSION (MEDIE):")
        print(f"   Successo: {'SÌ' if miva['attack_success'] else 'NO'}")
        print(f"   Normal confidence: {miva['normal_confidence']:.3f}")
        print(f"   Attack confidence: {miva['attack_confidence']:.3f}")
        print(f"   Information leakage: {miva['information_leakage_score']:.3f}")
        print(f"   Separazione prototipi: {miva['prototype_separation']:.3f}")
        
        # Success Criteria
        sc = miva['success_criteria']
        print(f"   Criteri soddisfatti: {sc['total_successful']}/4")
        
        print("\n" + "="*80)
        print("📄 I risultati dettagliati sono stati salvati nel file JSON generato.")
        print("="*80)


def main():
    """
    Funzione principale per l'aggregazione strutturata delle metriche di attacco.
    """
    print("📊 AGGREGATORE METRICHE ATTACCHI PRIVACY - FORMATO STRUTTURATO")
    print("="*70)
    
    # Inizializza aggregator
    aggregator = StructuredAttackMetricsAggregator()
    
    # Trova file di attacchi
    print("\n1️⃣ Ricerca file attacchi...")
    found_files = aggregator.find_attack_files()
    
    if not found_files:
        print("❌ Nessun file di attacchi trovato nella directory corrente.")
        print("   Assicurati che i file JSON degli attacchi siano nella stessa directory.")
        return
    
    # Carica risultati
    print("\n2️⃣ Caricamento risultati...")
    if not aggregator.load_attack_results():
        print("❌ Impossibile caricare i risultati degli attacchi.")
        return
    
    # Calcola aggregazione strutturata
    print("\n3️⃣ Calcolo aggregazione strutturata...")
    aggregated = aggregator.calculate_structured_aggregation()
    
    if not aggregated:
        print("❌ Errore nel calcolo dell'aggregazione strutturata.")
        return
    
    # Salva risultati
    print("\n4️⃣ Salvataggio JSON strutturato...")
    output_file = aggregator.save_aggregated_json()
    
    if output_file:
        print(f"✅ File JSON aggregato creato: {output_file}")
    
    # Stampa summary
    print("\n5️⃣ Generazione summary...")
    aggregator.print_structured_summary()
    
    print(f"\n✅ Aggregazione strutturata completata!")
    print(f"📄 Il file JSON mantiene la struttura originale con valori aggregati.")
    print(f"📊 Consulta '{output_file}' per i dettagli completi.")


if __name__ == "__main__":
    main()