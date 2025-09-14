"""
Aggregatore e Analizzatore Risultati Attacchi Privacy
Francesca Pellegrino - Tirocinio Federated Learning
Analizza e aggrega tutti i file JSON generati dagli attacchi privacy
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class AttackResultsAggregator:
    """
    Classe per aggregare e analizzare i risultati degli attacchi privacy
    
    Funzionalità:
    - Carica tutti i file JSON degli attacchi
    - Aggrega i risultati per tipologia di attacco
    - Calcola statistiche descrittive
    - Genera report dettagliati
    - Crea visualizzazioni
    """
    
    def __init__(self, data_directory: str = "."):
        """
        Inizializza l'aggregatore
        
        Args:
            data_directory: Directory contenente i file JSON degli attacchi
        """
        self.data_directory = Path(data_directory)
        self.attack_files = []
        self.aggregated_data = {}
        self.statistics = {}
        
        print(f"🔍 AGGREGATORE RISULTATI ATTACCHI PRIVACY")
        print(f"📁 Directory di lavoro: {self.data_directory.absolute()}")
        
    def find_attack_files(self) -> List[str]:
        """
        Trova tutti i file JSON degli attacchi nella directory
        
        Returns:
            Lista dei percorsi dei file trovati
        """
        print(f"\n🔍 Ricerca file JSON degli attacchi...")
        
        # Pattern per trovare i file degli attacchi
        patterns = [
            "attack_results_*.json",
            "*attack_results*.json", 
            "malicious_client_*.json"
        ]
        
        found_files = []
        for pattern in patterns:
            files = list(self.data_directory.glob(pattern))
            found_files.extend(files)
        
        # Rimuovi duplicati mantenendo l'ordine
        unique_files = []
        for file in found_files:
            if file not in unique_files:
                unique_files.append(file)
        
        self.attack_files = unique_files
        
        print(f"   ✅ Trovati {len(self.attack_files)} file di attacchi:")
        for i, file in enumerate(self.attack_files, 1):
            print(f"      {i}. {file.name}")
        
        return [str(f) for f in self.attack_files]
    
    def load_attack_data(self) -> Dict[str, Any]:
        """
        Carica tutti i dati degli attacchi dai file JSON
        
        Returns:
            Dizionario contenente tutti i dati aggregati
        """
        print(f"\n📊 Caricamento dati degli attacchi...")
        
        all_attacks = []
        load_errors = []
        
        for file_path in self.attack_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Estrai metadati dal nome file
                file_info = self._extract_file_metadata(file_path.name)
                
                # Aggiungi metadati ai dati
                data['file_metadata'] = file_info
                data['file_path'] = str(file_path)
                
                all_attacks.append(data)
                
                print(f"   ✅ {file_path.name}: Caricato con successo")
                
            except Exception as e:
                error_info = {
                    'file': str(file_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                load_errors.append(error_info)
                print(f"   ❌ {file_path.name}: Errore - {e}")
        
        self.aggregated_data = {
            'attacks': all_attacks,
            'total_files': len(self.attack_files),
            'successful_loads': len(all_attacks),
            'failed_loads': len(load_errors),
            'load_errors': load_errors,
            'aggregation_timestamp': datetime.now().isoformat()
        }
        
        print(f"   📊 Risultati caricamento:")
        print(f"      ✅ File caricati con successo: {len(all_attacks)}")
        print(f"      ❌ File con errori: {len(load_errors)}")
        
        return self.aggregated_data
    
    def _extract_file_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Estrae metadati dal nome del file
        
        Args:
            filename: Nome del file
            
        Returns:
            Dizionario con i metadati estratti
        """
        metadata = {
            'filename': filename,
            'extracted_client_id': None,
            'extracted_timestamp': None,
            'file_type': 'unknown'
        }
        
        # Estrai client ID
        try:
            if 'client_' in filename:
                parts = filename.split('client_')
                if len(parts) > 1:
                    client_part = parts[1].split('_')[0]
                    metadata['extracted_client_id'] = int(client_part)
        except:
            pass
        
        # Estrai timestamp
        try:
            # Pattern: YYYYMMDD_HHMMSS
            import re
            timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                metadata['extracted_timestamp'] = datetime.strptime(
                    timestamp_str, '%Y%m%d_%H%M%S'
                ).isoformat()
        except:
            pass
        
        # Determina tipo file
        if 'attack_results' in filename:
            metadata['file_type'] = 'attack_results'
        elif 'malicious_client' in filename:
            metadata['file_type'] = 'malicious_client'
        
        return metadata
    
    def analyze_attack_statistics(self) -> Dict[str, Any]:
        """
        Calcola statistiche dettagliate sui risultati degli attacchi
        
        Returns:
            Dizionario con le statistiche calcolate
        """
        print(f"\n📈 Analisi statistiche degli attacchi...")
        
        if not self.aggregated_data.get('attacks'):
            print("   ⚠️ Nessun dato di attacco disponibile per l'analisi")
            return {}
        
        attacks = self.aggregated_data['attacks']
        
        # Inizializza strutture per le statistiche
        stats = {
            'overview': {},
            'by_attack_type': {},
            'by_client': {},
            'success_analysis': {},
            'privacy_risk_analysis': {},
            'temporal_analysis': {}
        }
        
        # 1. OVERVIEW GENERALE
        print("   📊 Analisi overview generale...")
        
        total_experiments = len(attacks)
        clients_analyzed = set()
        total_attacks_attempted = 0
        total_attacks_successful = 0
        
        for attack_data in attacks:
            if 'attack_summary' in attack_data:
                summary = attack_data['attack_summary']
                clients_analyzed.add(summary.get('client_id', 'unknown'))
                total_attacks_attempted += summary.get('total_attacks_attempted', 0)
                total_attacks_successful += summary.get('successful_attacks', 0)
        
        stats['overview'] = {
            'total_experiment_files': total_experiments,
            'unique_clients_analyzed': len(clients_analyzed),
            'client_ids': sorted(list(clients_analyzed)),
            'total_attacks_attempted': total_attacks_attempted,
            'total_attacks_successful': total_attacks_successful,
            'overall_success_rate': total_attacks_successful / max(total_attacks_attempted, 1),
            'avg_attacks_per_client': total_attacks_attempted / max(len(clients_analyzed), 1)
        }
        
        # 2. ANALISI PER TIPOLOGIA DI ATTACCO
        print("   📊 Analisi per tipologia di attacco...")
        
        attack_types = ['membership_inference', 'property_inference', 'model_inversion']
        
        for attack_type in attack_types:
            type_stats = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'success_rate': 0.0,
                'metrics': {}
            }
            
            metrics_collected = []
            
            for attack_data in attacks:
                if attack_type in attack_data:
                    attack_info = attack_data[attack_type]
                    type_stats['total_attempts'] += 1
                    
                    if attack_info.get('attack_success', False):
                        type_stats['successful_attempts'] += 1
                    
                    # Raccogli metriche specifiche per tipo
                    if attack_type == 'membership_inference':
                        if 'combined_accuracy' in attack_info:
                            metrics_collected.append(attack_info['combined_accuracy'])
                    elif attack_type == 'property_inference':
                        if 'estimation_error' in attack_info:
                            metrics_collected.append(attack_info['estimation_error'])
                    elif attack_type == 'model_inversion':
                        if 'information_leakage_score' in attack_info:
                            metrics_collected.append(attack_info['information_leakage_score'])
            
            # Calcola statistiche metriche
            if metrics_collected:
                type_stats['metrics'] = {
                    'count': len(metrics_collected),
                    'mean': np.mean(metrics_collected),
                    'std': np.std(metrics_collected),
                    'min': np.min(metrics_collected),
                    'max': np.max(metrics_collected),
                    'median': np.median(metrics_collected)
                }
            
            if type_stats['total_attempts'] > 0:
                type_stats['success_rate'] = type_stats['successful_attempts'] / type_stats['total_attempts']
            
            stats['by_attack_type'][attack_type] = type_stats
        
        # 3. ANALISI PER CLIENT
        print("   📊 Analisi per client...")
        
        for client_id in clients_analyzed:
            client_stats = {
                'experiments_count': 0,
                'total_attacks': 0,
                'successful_attacks': 0,
                'success_rate': 0.0,
                'privacy_risk_scores': [],
                'attack_breakdown': {
                    'membership_inference': 0,
                    'property_inference': 0,
                    'model_inversion': 0
                }
            }
            
            for attack_data in attacks:
                summary = attack_data.get('attack_summary', {})
                if summary.get('client_id') == client_id:
                    client_stats['experiments_count'] += 1
                    client_stats['total_attacks'] += summary.get('total_attacks_attempted', 0)
                    client_stats['successful_attacks'] += summary.get('successful_attacks', 0)
                    
                    if 'privacy_risk_score' in summary:
                        client_stats['privacy_risk_scores'].append(summary['privacy_risk_score'])
                    
                    # Breakdown per tipo di attacco
                    for attack_type in attack_types:
                        if attack_type in attack_data:
                            if attack_data[attack_type].get('attack_success', False):
                                client_stats['attack_breakdown'][attack_type] += 1
            
            if client_stats['total_attacks'] > 0:
                client_stats['success_rate'] = client_stats['successful_attacks'] / client_stats['total_attacks']
            
            # Statistiche privacy risk
            if client_stats['privacy_risk_scores']:
                client_stats['avg_privacy_risk'] = np.mean(client_stats['privacy_risk_scores'])
                client_stats['max_privacy_risk'] = np.max(client_stats['privacy_risk_scores'])
            
            stats['by_client'][str(client_id)] = client_stats
        
        # 4. ANALISI SUCCESSO COMPLESSIVA
        print("   📊 Analisi pattern di successo...")
        
        success_patterns = {
            'all_attacks_successful': 0,
            'partial_success': 0,
            'no_success': 0,
            'most_vulnerable_clients': [],
            'attack_effectiveness_ranking': {}
        }
        
        for attack_data in attacks:
            summary = attack_data.get('attack_summary', {})
            total = summary.get('total_attacks_attempted', 0)
            successful = summary.get('successful_attacks', 0)
            
            if successful == total and total > 0:
                success_patterns['all_attacks_successful'] += 1
            elif successful > 0:
                success_patterns['partial_success'] += 1
            else:
                success_patterns['no_success'] += 1
        
        # Trova client più vulnerabili
        client_vulnerabilities = []
        for client_id, client_data in stats['by_client'].items():
            vulnerability_score = client_data['success_rate']
            if 'avg_privacy_risk' in client_data:
                vulnerability_score = (vulnerability_score + client_data['avg_privacy_risk']) / 2
            
            client_vulnerabilities.append({
                'client_id': client_id,
                'vulnerability_score': vulnerability_score,
                'success_rate': client_data['success_rate']
            })
        
        # Ordina per vulnerabilità
        client_vulnerabilities.sort(key=lambda x: x['vulnerability_score'], reverse=True)
        success_patterns['most_vulnerable_clients'] = client_vulnerabilities[:5]
        
        # Ranking efficacia attacchi
        for attack_type, type_data in stats['by_attack_type'].items():
            success_patterns['attack_effectiveness_ranking'][attack_type] = type_data['success_rate']
        
        stats['success_analysis'] = success_patterns
        
        # 5. ANALISI RISCHIO PRIVACY
        print("   📊 Analisi rischio privacy...")
        
        all_privacy_scores = []
        for attack_data in attacks:
            summary = attack_data.get('attack_summary', {})
            if 'privacy_risk_score' in summary:
                all_privacy_scores.append(summary['privacy_risk_score'])
        
        if all_privacy_scores:
            stats['privacy_risk_analysis'] = {
                'total_scores': len(all_privacy_scores),
                'mean_risk': np.mean(all_privacy_scores),
                'std_risk': np.std(all_privacy_scores),
                'min_risk': np.min(all_privacy_scores),
                'max_risk': np.max(all_privacy_scores),
                'median_risk': np.median(all_privacy_scores),
                'high_risk_count': sum(1 for score in all_privacy_scores if score > 0.7),
                'medium_risk_count': sum(1 for score in all_privacy_scores if 0.3 <= score <= 0.7),
                'low_risk_count': sum(1 for score in all_privacy_scores if score < 0.3)
            }
        
        self.statistics = stats
        
        print(f"   ✅ Analisi completata:")
        print(f"      📊 Esperimenti analizzati: {total_experiments}")
        print(f"      👥 Client unici: {len(clients_analyzed)}")
        print(f"      📈 Tasso successo generale: {stats['overview']['overall_success_rate']*100:.1f}%")
        
        return stats
    
    def generate_comprehensive_report(self, output_file: str = None) -> str:
        """
        Genera un report completo in formato testo
        
        Args:
            output_file: Path del file di output (opzionale)
            
        Returns:
            Contenuto del report come stringa
        """
        print(f"\n📄 Generazione report completo...")
        
        if not self.statistics:
            print("   ⚠️ Eseguire prima analyze_attack_statistics()")
            return ""
        
        # Data e ora corrente
        now = datetime.now()
        
        # Costruisci il report
        report_lines = []
        
        # HEADER
        report_lines.extend([
            "=" * 100,
            "REPORT COMPLETO ANALISI ATTACCHI PRIVACY",
            "FEDERATED LEARNING - SMARTGRID SECURITY ANALYSIS",
            "=" * 100,
            f"📅 Data generazione: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"👩‍💻 Sviluppatore: Francesca Pellegrino",
            f"🎯 Progetto: Tirocinio Federated Learning - Privacy Attacks",
            f"📁 Directory analizzata: {self.data_directory.absolute()}",
            "=" * 100,
            ""
        ])
        
        # 1. EXECUTIVE SUMMARY
        overview = self.statistics['overview']
        report_lines.extend([
            "📊 EXECUTIVE SUMMARY",
            "-" * 50,
            f"📄 File di esperimenti analizzati: {overview['total_experiment_files']}",
            f"👥 Client federati testati: {overview['unique_clients_analyzed']}",
            f"🎯 Attacchi totali tentati: {overview['total_attacks_attempted']}",
            f"✅ Attacchi riusciti: {overview['total_attacks_successful']}",
            f"📈 Tasso di successo complessivo: {overview['overall_success_rate']*100:.2f}%",
            f"📊 Media attacchi per client: {overview['avg_attacks_per_client']:.1f}",
            f"👥 Client IDs analizzati: {', '.join(map(str, overview['client_ids']))}",
            ""
        ])
        
        # 2. ANALISI PER TIPOLOGIA DI ATTACCO
        report_lines.extend([
            "🔍 ANALISI DETTAGLIATA PER TIPOLOGIA DI ATTACCO",
            "-" * 80,
            ""
        ])
        
        attack_type_names = {
            'membership_inference': 'Membership Inference Attack',
            'property_inference': 'Property Inference Attack', 
            'model_inversion': 'Model Inversion Attack'
        }
        
        for attack_type, type_data in self.statistics['by_attack_type'].items():
            attack_name = attack_type_names.get(attack_type, attack_type)
            
            report_lines.extend([
                f"🎯 {attack_name.upper()}",
                f"   📊 Tentativi totali: {type_data['total_attempts']}",
                f"   ✅ Successi: {type_data['successful_attempts']}",
                f"   📈 Tasso successo: {type_data['success_rate']*100:.2f}%",
            ])
            
            if 'metrics' in type_data and type_data['metrics']:
                metrics = type_data['metrics']
                report_lines.extend([
                    f"   📊 Statistiche metriche ({metrics['count']} campioni):",
                    f"      📊 Media: {metrics['mean']:.4f}",
                    f"      📊 Deviazione std: {metrics['std']:.4f}",
                    f"      📊 Min-Max: {metrics['min']:.4f} - {metrics['max']:.4f}",
                    f"      📊 Mediana: {metrics['median']:.4f}",
                ])
            
            report_lines.append("")
        
        # 3. ANALISI PER CLIENT
        report_lines.extend([
            "👥 ANALISI DETTAGLIATA PER CLIENT",
            "-" * 50,
            ""
        ])
        
        for client_id, client_data in self.statistics['by_client'].items():
            report_lines.extend([
                f"🔹 CLIENT {client_id}:",
                f"   📄 Esperimenti eseguiti: {client_data['experiments_count']}",
                f"   🎯 Attacchi totali: {client_data['total_attacks']}",
                f"   ✅ Attacchi riusciti: {client_data['successful_attacks']}",
                f"   📈 Tasso successo: {client_data['success_rate']*100:.2f}%",
            ])
            
            if 'avg_privacy_risk' in client_data:
                report_lines.extend([
                    f"   ⚠️ Rischio privacy medio: {client_data['avg_privacy_risk']:.4f}",
                    f"   ⚠️ Rischio privacy massimo: {client_data['max_privacy_risk']:.4f}",
                ])
            
            # Breakdown attacchi
            breakdown = client_data['attack_breakdown']
            report_lines.extend([
                f"   📊 Breakdown successi:",
                f"      🔍 Membership Inference: {breakdown['membership_inference']}",
                f"      🔍 Property Inference: {breakdown['property_inference']}",
                f"      🔍 Model Inversion: {breakdown['model_inversion']}",
                ""
            ])
        
        # 4. ANALISI PATTERN DI SUCCESSO
        success_analysis = self.statistics['success_analysis']
        report_lines.extend([
            "📈 ANALISI PATTERN DI SUCCESSO",
            "-" * 50,
            f"🎯 Esperimenti con tutti gli attacchi riusciti: {success_analysis['all_attacks_successful']}",
            f"🎯 Esperimenti con successo parziale: {success_analysis['partial_success']}",
            f"🎯 Esperimenti senza successi: {success_analysis['no_success']}",
            ""
        ])
        
        # Client più vulnerabili
        if success_analysis['most_vulnerable_clients']:
            report_lines.extend([
                "⚠️ CLIENT PIÙ VULNERABILI (Top 5):",
                "-" * 40,
            ])
            
            for i, client_vuln in enumerate(success_analysis['most_vulnerable_clients'], 1):
                report_lines.append(
                    f"   {i}. Client {client_vuln['client_id']}: "
                    f"Vulnerabilità {client_vuln['vulnerability_score']:.4f} "
                    f"(Successo: {client_vuln['success_rate']*100:.1f}%)"
                )
            
            report_lines.append("")
        
        # Ranking efficacia attacchi
        report_lines.extend([
            "🏆 RANKING EFFICACIA ATTACCHI:",
            "-" * 40,
        ])
        
        # Ordina per efficacia
        effectiveness = sorted(
            success_analysis['attack_effectiveness_ranking'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (attack_type, success_rate) in enumerate(effectiveness, 1):
            attack_name = attack_type_names.get(attack_type, attack_type)
            report_lines.append(
                f"   {i}. {attack_name}: {success_rate*100:.2f}% successo"
            )
        
        report_lines.append("")
        
        # 5. ANALISI RISCHIO PRIVACY
        if 'privacy_risk_analysis' in self.statistics:
            privacy_analysis = self.statistics['privacy_risk_analysis']
            report_lines.extend([
                "⚠️ ANALISI RISCHIO PRIVACY",
                "-" * 40,
                f"📊 Campioni analizzati: {privacy_analysis['total_scores']}",
                f"📊 Rischio medio: {privacy_analysis['mean_risk']:.4f}",
                f"📊 Deviazione standard: {privacy_analysis['std_risk']:.4f}",
                f"📊 Range rischio: {privacy_analysis['min_risk']:.4f} - {privacy_analysis['max_risk']:.4f}",
                f"📊 Rischio mediano: {privacy_analysis['median_risk']:.4f}",
                "",
                "🚨 DISTRIBUZIONE LIVELLI DI RISCHIO:",
                f"   🔴 Alto rischio (>0.7): {privacy_analysis['high_risk_count']} esperimenti",
                f"   🟡 Medio rischio (0.3-0.7): {privacy_analysis['medium_risk_count']} esperimenti",
                f"   🟢 Basso rischio (<0.3): {privacy_analysis['low_risk_count']} esperimenti",
                ""
            ])
        
        # 6. CONCLUSIONI E RACCOMANDAZIONI
        report_lines.extend([
            "💡 CONCLUSIONI E RACCOMANDAZIONI",
            "-" * 50,
            ""
        ])
        
        # Conclusioni automatiche basate sui dati
        overall_success = overview['overall_success_rate']
        
        if overall_success > 0.8:
            risk_level = "CRITICO"
            recommendations = [
                "Il sistema federato presenta vulnerabilità critiche alla privacy",
                "È necessario implementare tecniche di privacy-preserving ML",
                "Considerare l'uso di Differential Privacy",
                "Implementare tecniche di Secure Aggregation",
                "Valutare l'uso di Homomorphic Encryption"
            ]
        elif overall_success > 0.5:
            risk_level = "ALTO"
            recommendations = [
                "Il sistema presenta vulnerabilità significative",
                "Implementare controlli di accesso più stringenti",
                "Aggiungere noise ai gradienti condivisi",
                "Limitare le informazioni condivise durante l'aggregazione",
                "Monitorare continuamente le performance di privacy"
            ]
        elif overall_success > 0.3:
            risk_level = "MEDIO"
            recommendations = [
                "Il sistema ha una sicurezza moderata ma migliorabile",
                "Implementare basic privacy techniques",
                "Aggiungere monitoring degli attacchi",
                "Valutare periodic privacy audits"
            ]
        else:
            risk_level = "BASSO"
            recommendations = [
                "Il sistema mostra buona resistenza agli attacchi",
                "Mantenere i controlli di sicurezza attuali",
                "Continuare il monitoring periodico"
            ]
        
        report_lines.extend([
            f"🚨 LIVELLO DI RISCHIO COMPLESSIVO: {risk_level}",
            f"📈 Tasso successo attacchi: {overall_success*100:.1f}%",
            "",
            "💡 RACCOMANDAZIONI:",
        ])
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"   {i}. {rec}")
        
        # 7. APPENDICE TECNICA
        report_lines.extend([
            "",
            "📋 APPENDICE TECNICA",
            "-" * 30,
            f"📄 File analizzati: {len(self.aggregated_data['attacks'])}",
            f"📄 File caricati con successo: {self.aggregated_data['successful_loads']}",
            f"📄 File con errori: {self.aggregated_data['failed_loads']}",
            f"🕐 Timestamp aggregazione: {self.aggregated_data['aggregation_timestamp']}",
            "",
            "📂 DETTAGLI FILE ANALIZZATI:",
        ])
        
        for i, attack_data in enumerate(self.aggregated_data['attacks'], 1):
            metadata = attack_data.get('file_metadata', {})
            filename = metadata.get('filename', 'unknown')
            client_id = metadata.get('extracted_client_id', 'N/A')
            timestamp = metadata.get('extracted_timestamp', 'N/A')
            
            report_lines.append(f"   {i}. {filename} (Client: {client_id}, Timestamp: {timestamp})")
        
        # Footer
        report_lines.extend([
            "",
            "=" * 100,
            "END OF REPORT",
            f"Generato da AttackResultsAggregator v1.0",
            f"Francesca Pellegrino - Tirocinio Federated Learning",
            "=" * 100
        ])
        
        # Unisci tutte le righe
        report_content = "\n".join(report_lines)
        
        # Salva su file se specificato
        if output_file is None:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            output_file = f"attack_analysis_comprehensive_report_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"   ✅ Report salvato: {output_file}")
            print(f"   📄 Righe totali: {len(report_lines)}")
            print(f"   📊 Caratteri totali: {len(report_content):,}")
            
        except Exception as e:
            print(f"   ❌ Errore salvataggio report: {e}")
        
        return report_content
    
    def export_to_excel(self, output_file: str = None) -> str:
        """
        Esporta i dati aggregati in formato Excel per analisi avanzate
        
        Args:
            output_file: Path del file Excel di output
            
        Returns:
            Path del file Excel generato
        """
        print(f"\n📊 Esportazione dati in Excel...")
        
        if not self.aggregated_data.get('attacks'):
            print("   ⚠️ Nessun dato disponibile per l'esportazione")
            return ""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"attack_analysis_data_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                
                # 1. SUMMARY SHEET
                if self.statistics:
                    overview_data = []
                    overview = self.statistics['overview']
                    
                    for key, value in overview.items():
                        if key != 'client_ids':  # Liste gestite separatamente
                            overview_data.append({
                                'Metrica': key,
                                'Valore': value,
                                'Descrizione': self._get_metric_description(key)
                            })
                    
                    df_overview = pd.DataFrame(overview_data)
                    df_overview.to_excel(writer, sheet_name='Overview', index=False)
                
                # 2. ATTACK TYPE ANALYSIS
                if 'by_attack_type' in self.statistics:
                    attack_type_data = []
                    
                    for attack_type, data in self.statistics['by_attack_type'].items():
                        row = {
                            'Attack_Type': attack_type,
                            'Total_Attempts': data['total_attempts'],
                            'Successful_Attempts': data['successful_attempts'],
                            'Success_Rate': data['success_rate']
                        }
                        
                        if 'metrics' in data and data['metrics']:
                            metrics = data['metrics']
                            row.update({
                                'Metrics_Count': metrics['count'],
                                'Metrics_Mean': metrics['mean'],
                                'Metrics_Std': metrics['std'],
                                'Metrics_Min': metrics['min'],
                                'Metrics_Max': metrics['max'],
                                'Metrics_Median': metrics['median']
                            })
                        
                        attack_type_data.append(row)
                    
                    df_attacks = pd.DataFrame(attack_type_data)
                    df_attacks.to_excel(writer, sheet_name='Attack_Types', index=False)
                
                # 3. CLIENT ANALYSIS
                if 'by_client' in self.statistics:
                    client_data = []
                    
                    for client_id, data in self.statistics['by_client'].items():
                        row = {
                            'Client_ID': client_id,
                            'Experiments_Count': data['experiments_count'],
                            'Total_Attacks': data['total_attacks'],
                            'Successful_Attacks': data['successful_attacks'],
                            'Success_Rate': data['success_rate'],
                            'Membership_Inference_Successes': data['attack_breakdown']['membership_inference'],
                            'Property_Inference_Successes': data['attack_breakdown']['property_inference'],
                            'Model_Inversion_Successes': data['attack_breakdown']['model_inversion']
                        }
                        
                        if 'avg_privacy_risk' in data:
                            row['Avg_Privacy_Risk'] = data['avg_privacy_risk']
                            row['Max_Privacy_Risk'] = data['max_privacy_risk']
                        
                        client_data.append(row)
                    
                    df_clients = pd.DataFrame(client_data)
                    df_clients.to_excel(writer, sheet_name='Clients', index=False)
                
                # 4. DETAILED RESULTS
                detailed_data = []
                
                for attack_data in self.aggregated_data['attacks']:
                    metadata = attack_data.get('file_metadata', {})
                    summary = attack_data.get('attack_summary', {})
                    
                    base_row = {
                        'File_Name': metadata.get('filename', ''),
                        'Client_ID': summary.get('client_id', metadata.get('extracted_client_id')),
                        'Timestamp': metadata.get('extracted_timestamp', ''),
                        'Total_Attacks_Attempted': summary.get('total_attacks_attempted', 0),
                        'Successful_Attacks': summary.get('successful_attacks', 0),
                        'Attack_Success_Rate': summary.get('attack_success_rate', 0),
                        'Privacy_Risk_Score': summary.get('privacy_risk_score', 0),
                        'FL_Compromised': summary.get('federated_learning_compromised', False)
                    }
                    
                    # Aggiungi dettagli per ogni tipo di attacco
                    for attack_type in ['membership_inference', 'property_inference', 'model_inversion']:
                        if attack_type in attack_data:
                            attack_info = attack_data[attack_type]
                            prefix = attack_type.title().replace('_', '')
                            
                            base_row[f'{prefix}_Success'] = attack_info.get('attack_success', False)
                            
                            # Metriche specifiche
                            if attack_type == 'membership_inference':
                                base_row[f'{prefix}_Combined_Accuracy'] = attack_info.get('combined_accuracy', 0)
                                base_row[f'{prefix}_Privacy_Breach_Score'] = attack_info.get('privacy_breach_score', 0)
                            elif attack_type == 'property_inference':
                                base_row[f'{prefix}_Estimation_Error'] = attack_info.get('estimation_error', 0)
                                base_row[f'{prefix}_Properties_Detected'] = attack_info.get('properties_detected', 0)
                            elif attack_type == 'model_inversion':
                                base_row[f'{prefix}_Information_Leakage'] = attack_info.get('information_leakage_score', 0)
                                base_row[f'{prefix}_Normal_Confidence'] = attack_info.get('normal_confidence', 0)
                                base_row[f'{prefix}_Attack_Confidence'] = attack_info.get('attack_confidence', 0)
                    
                    detailed_data.append(base_row)
                
                df_detailed = pd.DataFrame(detailed_data)
                df_detailed.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # 5. METADATA
                metadata_info = [
                    {'Campo': 'Data_Generazione', 'Valore': datetime.now().isoformat()},
                    {'Campo': 'Directory_Analizzata', 'Valore': str(self.data_directory.absolute())},
                    {'Campo': 'File_Totali_Trovati', 'Valore': len(self.attack_files)},
                    {'Campo': 'File_Caricati_Successo', 'Valore': self.aggregated_data['successful_loads']},
                    {'Campo': 'File_Errori', 'Valore': self.aggregated_data['failed_loads']},
                    {'Campo': 'Versione_Aggregatore', 'Valore': 'v1.0'},
                    {'Campo': 'Sviluppatore', 'Valore': 'Francesca Pellegrino'},
                    {'Campo': 'Progetto', 'Valore': 'Tirocinio Federated Learning'}
                ]
                
                df_metadata = pd.DataFrame(metadata_info)
                df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            print(f"   ✅ Dati esportati in Excel: {output_file}")
            print(f"   📊 Fogli creati: Overview, Attack_Types, Clients, Detailed_Results, Metadata")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Errore esportazione Excel: {e}")
            return ""
    
    def _get_metric_description(self, metric_name: str) -> str:
        """
        Restituisce una descrizione per una metrica
        
        Args:
            metric_name: Nome della metrica
            
        Returns:
            Descrizione della metrica
        """
        descriptions = {
            'total_experiment_files': 'Numero totale di file di esperimenti analizzati',
            'unique_clients_analyzed': 'Numero di client federati unici testati',
            'total_attacks_attempted': 'Numero totale di attacchi tentati',
            'total_attacks_successful': 'Numero totale di attacchi riusciti',
            'overall_success_rate': 'Tasso di successo complessivo degli attacchi',
            'avg_attacks_per_client': 'Numero medio di attacchi per client'
        }
        
        return descriptions.get(metric_name, 'Descrizione non disponibile')
    
    def create_summary_visualization(self, output_dir: str = ".") -> List[str]:
        """
        Crea visualizzazioni riassuntive dei risultati
        
        Args:
            output_dir: Directory per salvare le visualizzazioni
            
        Returns:
            Lista dei file di visualizzazione creati
        """
        print(f"\n📈 Creazione visualizzazioni...")
        
        if not self.statistics:
            print("   ⚠️ Eseguire prima analyze_attack_statistics()")
            return []
        
        output_dir = Path(output_dir)
        created_files = []
        
        # Configura matplotlib per alta qualità
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        try:
            # 1. GRAFICO SUCCESS RATE PER TIPO DI ATTACCO
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            attack_types = []
            success_rates = []
            
            for attack_type, data in self.statistics['by_attack_type'].items():
                attack_types.append(attack_type.replace('_', ' ').title())
                success_rates.append(data['success_rate'] * 100)
            
            bars = ax.bar(attack_types, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_ylabel('Tasso di Successo (%)')
            ax.set_title('Efficacia degli Attacchi per Tipologia')
            ax.set_ylim(0, 100)
            
            # Aggiungi valori sulle barre
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            file_path = output_dir / "attack_success_rates.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            created_files.append(str(file_path))
            plt.close()
            
            # 2. GRAFICO VULNERABILITÀ CLIENT
            if self.statistics['by_client']:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                client_ids = []
                success_rates = []
                privacy_risks = []
                
                for client_id, data in self.statistics['by_client'].items():
                    client_ids.append(f"Client {client_id}")
                    success_rates.append(data['success_rate'] * 100)
                    privacy_risks.append(data.get('avg_privacy_risk', 0) * 100)
                
                x = np.arange(len(client_ids))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, success_rates, width, label='Tasso Successo (%)', color='#FF6B6B')
                bars2 = ax.bar(x + width/2, privacy_risks, width, label='Rischio Privacy (%)', color='#FFA07A')
                
                ax.set_ylabel('Percentuale (%)')
                ax.set_title('Vulnerabilità per Client')
                ax.set_xticks(x)
                ax.set_xticklabels(client_ids)
                ax.legend()
                ax.set_ylim(0, 100)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                file_path = output_dir / "client_vulnerabilities.png"
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                created_files.append(str(file_path))
                plt.close()
            
            # 3. DISTRIBUZIONE RISCHIO PRIVACY
            if 'privacy_risk_analysis' in self.statistics:
                privacy_data = self.statistics['privacy_risk_analysis']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Pie chart distribuzione rischi
                sizes = [
                    privacy_data['high_risk_count'],
                    privacy_data['medium_risk_count'], 
                    privacy_data['low_risk_count']
                ]
                labels = ['Alto Rischio\n(>0.7)', 'Medio Rischio\n(0.3-0.7)', 'Basso Rischio\n(<0.3)']
                colors = ['#FF4444', '#FFAA44', '#44AA44']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Distribuzione Livelli di Rischio Privacy')
                
                # Istogramma privacy scores (simulato)
                all_privacy_scores = []
                for attack_data in self.aggregated_data['attacks']:
                    summary = attack_data.get('attack_summary', {})
                    if 'privacy_risk_score' in summary:
                        all_privacy_scores.append(summary['privacy_risk_score'])
                
                if all_privacy_scores:
                    ax2.hist(all_privacy_scores, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
                    ax2.axvline(privacy_data['mean_risk'], color='red', linestyle='--', 
                               label=f'Media: {privacy_data["mean_risk"]:.3f}')
                    ax2.set_xlabel('Privacy Risk Score')
                    ax2.set_ylabel('Frequenza')
                    ax2.set_title('Distribuzione Privacy Risk Scores')
                    ax2.legend()
                
                plt.tight_layout()
                
                file_path = output_dir / "privacy_risk_distribution.png"
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                created_files.append(str(file_path))
                plt.close()
            
            print(f"   ✅ Visualizzazioni create:")
            for file in created_files:
                print(f"      📊 {Path(file).name}")
            
        except Exception as e:
            print(f"   ❌ Errore creazione visualizzazioni: {e}")
        
        return created_files


# FUNZIONE PRINCIPALE PER L'USO STANDALONE
def main():
    """
    Funzione principale per eseguire l'aggregazione e analisi completa
    """
    print("🚀 AVVIO AGGREGATORE RISULTATI ATTACCHI PRIVACY")
    print("=" * 70)
    print("👩‍💻 Sviluppatore: Francesca Pellegrino")
    print("🎯 Progetto: Tirocinio Federated Learning")
    print("=" * 70)
    
    try:
        # 1. Inizializza aggregatore
        aggregator = AttackResultsAggregator(".")
        
        # 2. Trova file di attacchi
        found_files = aggregator.find_attack_files()
        
        if not found_files:
            print("\n❌ Nessun file di attacchi trovato nella directory corrente")
            print("💡 Assicurati che i file JSON degli attacchi siano nella stessa directory")
            return
        
        # 3. Carica dati
        aggregated_data = aggregator.load_attack_data()
        
        # 4. Analizza statistiche
        statistics = aggregator.analyze_attack_statistics()
        
        # 5. Genera report completo
        report_content = aggregator.generate_comprehensive_report()
        
        # 6. Esporta in Excel
        excel_file = aggregator.export_to_excel()
        
        # 7. Crea visualizzazioni
        visualizations = aggregator.create_summary_visualization()
        
        print(f"\n🎉 AGGREGAZIONE COMPLETATA CON SUCCESSO!")
        print(f"📄 File analizzati: {len(found_files)}")
        print(f"📊 Statistiche calcolate per {len(statistics.get('by_client', {}))} client")
        print(f"📋 Report generato: Vedere file .txt creato")
        if excel_file:
            print(f"📊 Dati Excel: {excel_file}")
        if visualizations:
            print(f"📈 Visualizzazioni: {len(visualizations)} grafici creati")
        
        print(f"\n💡 I file generati possono essere utilizzati per:")
        print(f"   📖 Analisi dettagliata dei risultati")
        print(f"   📊 Presentazioni e report")
        print(f"   📋 Documentazione della tesi")
        print(f"   🔬 Ricerca e sviluppo ulteriori")
        
    except Exception as e:
        print(f"\n❌ Errore durante l'aggregazione: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()