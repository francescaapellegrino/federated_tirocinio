"""
Launcher Interattivo per Client SmartGrid
Permette di scegliere quanti client buoni e quanti malevoli avviare
Francesca Pellegrino
"""

import subprocess
import time
import os
import signal
import sys
from datetime import datetime
from typing import List, Dict, Tuple

# Launcher interattivo per scegliere configurazione client.
class InteractiveClientLauncher:
    
    def __init__(self):
        self.processes = []
        self.privacy_clients = 0
        self.malicious_clients = 0
        self.total_clients = 0
        self.session_config = {}
        
        # File client supportati
        self.client_files = {
            'privacy': 'client.py',
            'malicious': 'malicious_client_inference.py'
        }
        
        print("LAUNCHER INTERATTIVO SMARTGRID - FEDERATED LEARNING")
    
    def check_environment(self):
        """Verifica che i file client necessari esistano"""
        print("\nVERIFICA AMBIENTE DI ESECUZIONE...")
        
        missing_files = []
        available_clients = []
        
        for client_type, filename in self.client_files.items():
            if os.path.exists(filename):
                available_clients.append(client_type)
                
                if client_type == 'privacy':
                    print(f"   ✅ Client Privacy-Preserving: {filename}")
                    print(f"      → Implementa Differential Privacy")
                    print(f"      → Usa Secure Aggregation")
                elif client_type == 'malicious':
                    print(f"   ✅ Client Malevolo: {filename}")
                    print(f"      → Esegue attacchi di inferenza")
                    print(f"      → Testa robustezza sistema privacy")
            else:
                missing_files.append(filename)
                print(f"   ❌ File mancante: {filename}")
        
        if missing_files:
            print(f"\n⚠️ FILE MANCANTI: {missing_files}")
            print(f"💡 Assicurati di essere nella directory federated/SmartGrid/")
            return False
        
        # Verifica Python3
        try:
            result = subprocess.run(['python3', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   ✅ Python: {result.stdout.strip()}")
            else:
                print(f"   ❌ Errore Python3")
                return False
        except Exception as e:
            print(f"   ❌ Python3 non disponibile: {e}")
            return False
        
        print(f"   ✅ Ambiente pronto per esecuzione")
        return True
    
    def get_user_configuration(self):
        """Ottiene la configurazione desiderata dall'utente"""
        print("=" * 60)
        print("🎯 OBIETTIVO: Testare sistema privacy-preserving vs attacchi")
        print("📊 METODOLOGIA: Federeted Learning con diversi tipi di client")
        print("=" * 60)
        
        # Input client privacy-preserving
        while True:
            try:
                print(f"\n🔒 CLIENT PRIVACY-PRESERVING:")
                
                privacy_input = input(f"\nQuanti client privacy-preserving vuoi avviare? (0-20): ").strip()
                
                if privacy_input == "":
                    self.privacy_clients = 0
                else:
                    self.privacy_clients = int(privacy_input)
                
                if self.privacy_clients < 0 or self.privacy_clients > 20:
                    raise ValueError("Numero deve essere tra 0 e 20")
                
                print(f"   ✅ Client privacy configurati: {self.privacy_clients}")
                break
                
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print(f"\n🛑 Configurazione annullata")
                    return False
                print(f"   ❌ Input non valido. Inserisci un numero tra 0 e 20")
        
        # Input client malevoli
        while True:
            try:
                print(f"\n🔴 CLIENT MALEVOLI:")
                print(f"   - Includono: Membership Inference Attack, Property Inference Attack, Model Inversion Attack")

                malicious_input = input(f"\nQuanti client malevoli vuoi avviare? (0-5): ").strip()
                
                if malicious_input == "":
                    self.malicious_clients = 0
                else:
                    self.malicious_clients = int(malicious_input)
                
                if self.malicious_clients < 0 or self.malicious_clients > 5:
                    raise ValueError("Numero deve essere tra 0 e 5")
                
                print(f"   ✅ Client malevoli configurati: {self.malicious_clients}")
                break
                
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print(f"\n🛑 Configurazione annullata")
                    return False
                print(f"   ❌ Input non valido. Inserisci un numero tra 0 e 5")
        
        # Calcola totale
        self.total_clients = self.privacy_clients + self.malicious_clients
        
        if self.total_clients == 0:
            print(f"\n⚠️ Nessun client configurato!")
            print(f"💡 Devi avviare almeno 1 client per il federated learning")
            return False
        
        # Mostra configurazione finale
        print(f"\n📊 CONFIGURAZIONE FINALE:")
        print(f"   🔒 Client Privacy-Preserving: {self.privacy_clients}")
        print(f"   🔴 Client Malevoli: {self.malicious_clients}")
        print(f"   📊 Totale Client: {self.total_clients}")
        
        # Assegnazione ID automatica
        print(f"\n🆔 ASSEGNAZIONE ID CLIENT:")
        current_id = 1
        
        # Client privacy: ID da 1 a N
        if self.privacy_clients > 0:
            privacy_ids = list(range(current_id, current_id + self.privacy_clients))
            print(f"   🔒 Privacy: ID {privacy_ids[0]}-{privacy_ids[-1]}")
            current_id += self.privacy_clients
        
        # Client malevoli: ID da N+1 in poi
        if self.malicious_clients > 0:
            malicious_ids = list(range(current_id, current_id + self.malicious_clients))
            print(f"   🔴 Malevoli: ID {malicious_ids[0]}-{malicious_ids[-1]}")
        
        # Salva configurazione sessione
        self.session_config = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'privacy_clients': self.privacy_clients,
            'malicious_clients': self.malicious_clients,
            'total_clients': self.total_clients,
            'privacy_ids': privacy_ids if self.privacy_clients > 0 else [],
            'malicious_ids': malicious_ids if self.malicious_clients > 0 else [],
            'user': 'francescaapellegrino',
            'purpose': 'privacy_preserving_vs_adversarial'
        }
        
        # Conferma finale
        try:
            print(f"\n🚀 PRONTO PER AVVIARE IL SISTEMA!")
            
            confirm = input(f"\nProcedere con questa configurazione? (s/N): ").strip().lower()
            if confirm not in ['s', 'si', 'y', 'yes']:
                print(f"🛑 Configurazione annullata")
                return False
                
        except KeyboardInterrupt:
            print(f"\n🛑 Configurazione annullata")
            return False
        
        return True
    
    def start_client(self, client_id: int, client_type: str):
        """Avvia un singolo client del tipo specificato. 
        Args:
            client_id: ID numerico del client (1, 2, 3, ...)
            client_type: 'privacy' o 'malicious' 
        Spiegazione:
        - Costruisce il comando appropriato per ogni tipo di client
        - Usa subprocess per compatibilità MacOS
        - Monitora l'avvio per debug"""
        try:
            if client_type == 'privacy':
                # Client privacy-preserving con Differential Privacy
                cmd = ['python3', 'client.py', str(client_id)]
                description = f"🔒 Client Privacy-Preserving {client_id}"
            elif client_type == 'malicious':
                # Client malevolo con attacchi di inferenza
                cmd = ['python3', 'malicious_client_inference.py', str(client_id), 'true']
                description = f"🔴 Client Malevolo {client_id}"
            else:
                print(f"❌ Tipo client sconosciuto: {client_type}")
                return None
            
            print(f"   {description}")
            
            # Avvia processo con subprocess (più stabile su MacOS)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None,
                cwd=os.getcwd(),
                env=os.environ.copy()
            )
            
            # Salva informazioni processo
            process_info = {
                'client_id': client_id,
                'client_type': client_type,
                'process': process,
                'description': description,
                'start_time': time.time(),
                'cmd': ' '.join(cmd)
            }
            
            self.processes.append(process_info)
            return process
            
        except Exception as e:
            print(f"   ❌ Errore avvio client {client_id} ({client_type}): {e}")
            return None
    
    def launch_all_clients(self):
        """Avvia tutti i client configurati in sequenza.
        Spiegazione:
        - Avvio sequenziale
        - Delay progressivo per stabilizzare le connessioni
        - Monitoring dell'avvio per debugging
        """
        print(f"\n🚀 AVVIO SEQUENZIALE CLIENT")
        print("=" * 70)
        
        # Contatori statistiche
        privacy_started = 0
        malicious_started = 0
        failed_count = 0
        
        # Lista client da avviare
        clients_to_start = []
        current_id = 1
        
        # Aggiungi client privacy
        for i in range(self.privacy_clients):
            clients_to_start.append((current_id, 'privacy'))
            current_id += 1
        
        # Aggiungi client malevoli
        for i in range(self.malicious_clients):
            clients_to_start.append((current_id, 'malicious'))
            current_id += 1
        
        # Avvia in sequenza
        for i, (client_id, client_type) in enumerate(clients_to_start):
            print(f"\n[{i+1}/{len(clients_to_start)}] Avvio client {client_id} ({client_type})")
            
            process = self.start_client(client_id, client_type)
            
            if process:
                print(f"   ✅ Avviato con PID: {process.pid}")
                
                if client_type == 'privacy':
                    privacy_started += 1
                elif client_type == 'malicious':
                    malicious_started += 1
            else:
                print(f"   ❌ Avvio fallito")
                failed_count += 1
            
            # Delay per evitare conflitti gRPC
            if i < len(clients_to_start) - 1:  # Non aspettare dopo l'ultimo
                delay = 2.5 if client_type == 'privacy' else 3.0  # Delay maggiore per malevoli
                print(f"   ⏱️ Attesa {delay}s prima del prossimo client...")
                time.sleep(delay)
        
        # Summary avvio
        print(f"\n✅ AVVIO COMPLETATO:")
        print(f"   🔒 Privacy-Preserving avviati: {privacy_started}/{self.privacy_clients}")
        print(f"   🔴 Malevoli avviati: {malicious_started}/{self.malicious_clients}")
        if failed_count > 0:
            print(f"   ❌ Fallimenti: {failed_count}")
        print(f"   📊 Processi attivi: {len(self.processes)}")
        
        # Salva configurazione sessione
        self.save_session_log()
        
        if len(self.processes) > 0:
            print(f"   🎯 Sistema federato privacy-preserving pronto!")
            print(f"   📊 Avvia ora il server per iniziare il training")
            return True
        else:
            print(f"   ❌ Nessun client avviato con successo!")
            return False
    
    def wait_for_completion(self):
        """Attende il completamento di tutti i client"""
        if not self.processes:
            return
        
        print(f"\n⏳ MONITORAGGIO TRAINING FEDERATO...")
        print(f"💡 Premi Ctrl+C per terminare tutti i client")
        print(f"📊 Attendere... Processi in esecuzione: {len(self.processes)}")
        
        completed_count = 0
        privacy_completed = 0
        malicious_completed = 0
        
        while self.processes:
            time.sleep(5)  # Check ogni 5 secondi
            
            # Controlla processi terminati
            completed = []
            for proc_info in self.processes:
                if proc_info['process'].poll() is not None:
                    completed.append(proc_info)
                    completed_count += 1
                    
                    elapsed_time = time.time() - proc_info['start_time']
                    
                    if proc_info['client_type'] == 'privacy':
                        privacy_completed += 1
                        print(f"✅ {proc_info['description']} completato ({elapsed_time:.1f}s)")
                    elif proc_info['client_type'] == 'malicious':
                        malicious_completed += 1
                        print(f"🔴 {proc_info['description']} completato ({elapsed_time:.1f}s)")
                        print(f"   📊 Attacchi di inferenza eseguiti e risultati salvati")
            
            # Rimuovi processi completati
            for proc_info in completed:
                self.processes.remove(proc_info)
            
            # Progress update
            if completed:
                remaining = len(self.processes)
                print(f"📊 Progress: {completed_count} completati, {remaining} in esecuzione")
        
        # Summary finale
        print(f"\n🎉 TRAINING FEDERATO COMPLETATO!")
        print(f"📊 Client completati:")
        print(f"   🔒 Privacy-Preserving: {privacy_completed}")
        print(f"   🔴 Malevoli: {malicious_completed}")
        print(f"   📊 Totale: {completed_count}")
    
    def cleanup_processes(self):
        """Termina tutti i processi attivi in modo sicuro"""
        if not self.processes:
            return
        
        print(f"\n🧹 TERMINAZIONE SICURA PROCESSI...")
        
        terminated_count = 0
        
        for proc_info in self.processes:
            try:
                process = proc_info['process']
                client_id = proc_info['client_id']
                client_type = proc_info['client_type']
                
                if process.poll() is None:  # Ancora in esecuzione
                    print(f"   ⏹️ Terminazione {proc_info['description']}...")
                    
                    if os.name != 'nt':
                        # Unix/MacOS: termina gruppo di processi
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # Processo già terminato
                    else:
                        # Windows
                        process.terminate()
                    
                    # Aspetta terminazione educata (3 secondi)
                    try:
                        process.wait(timeout=3)
                        terminated_count += 1
                        print(f"   ✅ Terminato correttamente")
                    except subprocess.TimeoutExpired:
                        # Terminazione forzata
                        print(f"   ⚡ Terminazione forzata necessaria")
                        if os.name != 'nt':
                            try:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        terminated_count += 1
                        
            except Exception as e:
                print(f"   ⚠️ Errore terminazione client {proc_info['client_id']}: {e}")
        
        print(f"✅ Terminazione completata: {terminated_count} processi")
    
    def save_session_log(self):
        """Salva un log della sessione"""
        try:
            os.makedirs('session_logs', exist_ok=True)
            
            log_file = f"session_logs/interactive_session_{self.session_config['timestamp']}.txt"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("LOG SESSIONE INTERATTIVA SMARTGRID FEDERATED LEARNING\n")
                f.write("=" * 60 + "\n")
                f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Utente: {self.session_config['user']}\n")
                f.write(f"Progetto: Privacy-Preserving vs Adversarial\n")
                f.write(f"Timestamp: {self.session_config['timestamp']}\n\n")
                
                f.write("CONFIGURAZIONE CLIENT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Client Privacy-Preserving: {self.session_config['privacy_clients']}\n")
                f.write(f"Client Malevoli: {self.session_config['malicious_clients']}\n")
                f.write(f"Totale Client: {self.session_config['total_clients']}\n\n")
                
                f.write("ASSEGNAZIONE ID CLIENT:\n")
                f.write("-" * 25 + "\n")
                if self.session_config['privacy_ids']:
                    f.write(f"Privacy IDs: {self.session_config['privacy_ids']}\n")
                if self.session_config['malicious_ids']:
                    f.write(f"Malicious IDs: {self.session_config['malicious_ids']}\n")
                
                f.write(f"\nPROCESSI AVVIATI:\n")
                f.write("-" * 20 + "\n")
                for proc_info in self.processes:
                    f.write(f"ID {proc_info['client_id']}: {proc_info['client_type']} - {proc_info['cmd']}\n")
                
                f.write(f"\nSESSIONE CONFIGURATA CON SUCCESSO!\n")
            
            print(f"   📁 Log sessione salvato: {log_file}")
            
        except Exception as e:
            print(f"   ⚠️ Errore salvataggio log: {e}")

def show_welcome_banner():
    """Mostra banner di benvenuto con info"""
    print("🎓 SMARTGRID FEDERATED LEARNING")
    print("=" * 80)
    print("🔧 ARCHITETTURA SISTEMA:")
    print("   🔒 Client Privacy")
    print("   🔴 Client Malevoli: Attacchi Membership/Property/Model Inversion")
    print("   🖥️ Server Federato: Aggregazione FedAvg")
    print("=" * 80)

# MAIN FUNCTION
def main():
    show_welcome_banner()
    
    launcher = InteractiveClientLauncher()
    
    try:
        # 1. Verifica ambiente
        print(f"\n📋 FASE 1: VERIFICA AMBIENTE")
        if not launcher.check_environment():
            print(f"\n❌ Ambiente non configurato correttamente")
            print(f"💡 Verifica di essere nella directory federated/SmartGrid/")
            return False
        
        # 2. Configurazione utente
        print(f"\n⚙️ FASE 2: CONFIGURAZIONE INTERATTIVA")
        if not launcher.get_user_configuration():
            print(f"\n🛑 Configurazione annullata o fallita")
            return False
        
        # 3. Avvio client
        print(f"\n🚀 FASE 3: AVVIO CLIENT")
        if not launcher.launch_all_clients():
            print(f"\n❌ Avvio client fallito")
            return False
        
        # 4. Monitoraggio esecuzione
        print(f"\n📊 FASE 4: MONITORAGGIO ESECUZIONE")
        try:
            launcher.wait_for_completion()
        except KeyboardInterrupt:
            print(f"\n🛑 Interruzione manuale dell'utente...")
            launcher.cleanup_processes()
        
        print(f"\n✨ ADDESTRAMENTO COMPLETATO CON SUCCESSO! ✨")
        return True
        
    except Exception as e:
        print(f"\n💥 Errore imprevisto durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup di emergenza
        try:
            launcher.cleanup_processes()
        except:
            pass
        
        print(f"\nSuggerimenti per il debug:")
        print(f"   1. Verifica che il server sia avviato (python3 server.py)")
        print(f"   2. Controlla i log dei client per errori specifici")
        print(f"   3. Assicurati che non ci siano conflitti di porta")
        
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        print(f"\nECCELLENTE! Addestramento terminato!")