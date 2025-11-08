"""
Script per eseguire esperimenti comparativi con diverse configurazioni privacy.

Esegue il sistema 4 volte con configurazioni diverse:
1. Baseline (no protezioni)
2. Regularized (solo regolarizzazione)
3. DP (solo differential privacy)
4. Combined (DP + regolarizzazione)

VERSIONE CORRETTA: Gestisce correttamente l'output e il completamento del training.
"""

import os
import sys
import subprocess
import time
import re
from datetime import datetime
import threading
import queue

# Configurazioni da testare
CONFIGS = [
    {
        'name': 'A_baseline',
        'privacy_mode': 'baseline',
        'description': 'Sistema originale senza protezioni'
    },
    {
        'name': 'B_regularized',
        'privacy_mode': 'regularized',
        'description': 'Solo regolarizzazione del modello'
    },
    {
        'name': 'C_dp_only',
        'privacy_mode': 'dp',
        'description': 'Solo Differential Privacy (ε=1.0)'
    },
    {
        'name': 'D_combined',
        'privacy_mode': 'combined',
        'description': 'DP + Regolarizzazione (soluzione completa)'
    }
]

def modify_config_files(privacy_mode):
    """
    Modifica i file per impostare la modalità privacy desiderata.
    CORRETTO: Gestisce tutti i possibili valori attuali.
    """
    print(f"\n{'='*80}")
    print(f"🔧 CONFIGURAZIONE: {privacy_mode.upper()}")
    print(f"{'='*80}")
    
    files_to_modify = ['clientRFtmp.py', 'serverRFtmp.py']
    
    for filename in files_to_modify:
        if not os.path.exists(filename):
            print(f"⚠️ File {filename} non trovato!")
            continue
        
        # Leggi file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Pattern per trovare la riga PRIVACY_MODE
        pattern = r"PRIVACY_MODE\s*=\s*['\"](\w+)['\"]"
        
        # Trova il valore attuale
        match = re.search(pattern, content)
        if match:
            current_mode = match.group(1)
            print(f"   {filename}: {current_mode} → {privacy_mode}")
            
            # Sostituisci con nuovo valore
            new_line = f"PRIVACY_MODE = '{privacy_mode}'"
            content = re.sub(pattern, new_line, content)
            
            # Scrivi file modificato
            with open(filename, 'w') as f:
                f.write(content)
        else:
            print(f"⚠️ PRIVACY_MODE non trovato in {filename}")
    
    print(f"✅ File configurati per modalità: {privacy_mode}")

def stream_output(pipe, prefix, output_queue):
    """Thread per leggere output di un processo in real-time."""
    try:
        for line in iter(pipe.readline, b''):
            decoded = line.decode('utf-8', errors='ignore').strip()
            if decoded:
                output_queue.put((prefix, decoded))
    except:
        pass

def run_federated_training_monitored(config_name, num_rounds=100, timeout_minutes=60):
    """
    Esegue un round completo di training federato CON MONITORAGGIO.
    
    Args:
        config_name: Nome della configurazione
        num_rounds: Numero di round da eseguire
        timeout_minutes: Timeout massimo in minuti
        
    Returns:
        bool: True se training completato con successo, False altrimenti
    """
    print(f"\n{'='*80}")
    print(f"🚀 AVVIO TRAINING: {config_name}")
    print(f"{'='*80}")
    print(f"Rounds configurati: {num_rounds}")
    print(f"Timeout: {timeout_minutes} minuti")
    
    # Code per output
    output_queue = queue.Queue()
    
    # Avvia server
    print("\n[1/2] Avvio server...")
    server_process = subprocess.Popen(
        ['python3', 'serverRFtmp.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1
    )
    
    # Thread per leggere output server
    server_thread = threading.Thread(
        target=stream_output,
        args=(server_process.stdout, '[SERVER]', output_queue),
        daemon=True
    )
    server_thread.start()
    
    # Aspetta che il server sia pronto
    print("Attesa avvio server (5 secondi)...")
    time.sleep(5)
    
    # Avvia client
    print("\n[2/2] Avvio client...")
    client_process = subprocess.Popen(
        ['python3', 'run_clientsRFtmp.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1
    )
    
    # Thread per leggere output client
    client_thread = threading.Thread(
        target=stream_output,
        args=(client_process.stdout, '[CLIENT]', output_queue),
        daemon=True
    )
    client_thread.start()
    
    print(f"\n{'='*80}")
    print(f"📊 MONITORAGGIO TRAINING - {config_name}")
    print(f"{'='*80}")
    
    # Variabili per tracking
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    last_output_time = time.time()
    rounds_completed = 0
    training_completed = False
    
    # Pattern per rilevare completamento
    completion_patterns = [
        r"Training Random Forest.*completato",
        r"Report.*salvato",
        r"ESPERIMENTO COMPLETATO"
    ]
    
    # Pattern per contare rounds
    round_pattern = r"ROUND (\d+)"
    
    try:
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"\n⏱️ Timeout raggiunto ({timeout_minutes} minuti)")
                break
            
            # Check se processi sono ancora attivi
            if client_process.poll() is not None and server_process.poll() is not None:
                print(f"\n✅ Processi terminati naturalmente")
                training_completed = True
                break
            
            # Leggi output (con timeout breve)
            try:
                prefix, line = output_queue.get(timeout=1.0)
                last_output_time = time.time()
                
                # Stampa output (filtrato per ridurre rumore)
                if any(keyword in line for keyword in [
                    'Round', 'Accuracy', 'completato', 'RISULTATI',
                    'salvato', 'ESPERIMENTO', 'Privacy', 'DP'
                ]):
                    print(f"{prefix} {line}")
                
                # Conta rounds completati
                match = re.search(round_pattern, line)
                if match:
                    current_round = int(match.group(1))
                    if current_round > rounds_completed:
                        rounds_completed = current_round
                        print(f"   [PROGRESS] Round {rounds_completed}/{num_rounds} completato")
                
                # Check completamento
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in completion_patterns):
                    print(f"\n✅ Training completato rilevato!")
                    training_completed = True
                    time.sleep(2)  # Aspetta che gli ultimi messaggi vengano stampati
                    break
                    
            except queue.Empty:
                # Nessun output recente, continua a monitorare
                pass
            
            # Check se non c'è output da troppo tempo (stallo?)
            if time.time() - last_output_time > 300:  # 5 minuti senza output
                print(f"\n⚠️ Nessun output da 5 minuti, possibile stallo")
                break
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrotto dall'utente")
        training_completed = False
    
    finally:
        # Termina processi
        print(f"\nTerminazione processi...")
        try:
            client_process.terminate()
            client_process.wait(timeout=10)
        except:
            client_process.kill()
        
        try:
            server_process.terminate()
            server_process.wait(timeout=10)
        except:
            server_process.kill()
    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    if training_completed:
        print(f"✅ Training {config_name} completato con successo")
    else:
        print(f"⚠️ Training {config_name} terminato prematuramente")
    print(f"Tempo: {elapsed_minutes:.1f} minuti")
    print(f"Rounds completati: {rounds_completed}/{num_rounds}")
    print(f"{'='*80}")
    
    return training_completed

def run_attacks_if_model_exists(config_name):
    """
    Esegue attacchi se il modello è stato salvato.
    """
    print(f"\n{'='*80}")
    print(f"🎯 VERIFICA MODELLO PER ATTACCHI: {config_name}")
    print(f"{'='*80}")
    
    # Cerca modelli salvati
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"⚠️ Directory {results_dir} non trovata")
        return False
    
    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"⚠️ Nessun modello .pkl trovato in {results_dir}")
        return False
    
    # Ordina per timestamp e prendi il più recente
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(results_dir, latest_model)
    
    print(f"✅ Modello trovato: {latest_model}")
    print(f"📊 Esecuzione attacchi di inferenza...")
    
    try:
        # Esegui attacchi
        result = subprocess.run(
            ['python3', 'run_attacks_on_saved_model.py', model_path],
            timeout=600,  # 10 minuti timeout per attacchi
            capture_output=False  # Mostra output direttamente
        )
        
        if result.returncode == 0:
            print(f"✅ Attacchi completati per {config_name}")
            return True
        else:
            print(f"⚠️ Attacchi terminati con errori per {config_name}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout attacchi per {config_name}")
        return False
    except Exception as e:
        print(f"❌ Errore durante attacchi: {e}")
        return False

def main():
    """
    Esegue l'esperimento completo per tutte le configurazioni.
    """
    print("="*80)
    print("🔬 ESPERIMENTO COMPARATIVO PRIVACY MECHANISMS")
    print("Sistema: Federated Random Forest SmartGrid")
    print("="*80)
    
    # Trova NUM_ROUNDS nel serverRFtmp.py
    try:
        with open('serverRFtmp.py', 'r') as f:
            content = f.read()
            match = re.search(r'NUM_ROUNDS\s*=\s*(\d+)', content)
            num_rounds = int(match.group(1)) if match else 100
    except:
        num_rounds = 100
    
    print(f"\nConfigurazione esperimento:")
    print(f"  - Numero configurazioni: {len(CONFIGS)}")
    print(f"  - Rounds per configurazione: {num_rounds}")
    print(f"  - Client: 13")
    print(f"  - Dataset: SmartGrid")
    
    print(f"\nConfiguazioni da testare:")
    for i, config in enumerate(CONFIGS, 1):
        print(f"  {i}. {config['name']}: {config['description']}")
    
    # Stima tempo
    estimated_minutes = len(CONFIGS) * 30  # ~30 min per config
    print(f"\n⏱️ Tempo stimato: ~{estimated_minutes} minuti ({estimated_minutes/60:.1f} ore)")
    
    response = input("\n⚠️ ATTENZIONE: Questo richiederà MOLTO tempo. Continuare? (s/n): ")
    if response.lower() != 's':
        print("Esperimento annullato")
        return
    
    start_time = datetime.now()
    results_summary = []
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'#'*80}")
        print(f"# CONFIGURAZIONE {i}/{len(CONFIGS)}: {config['name']}")
        print(f"# {config['description']}")
        print(f"{'#'*80}")
        
        config_start = time.time()
        
        try:
            # 1. Configura file
            modify_config_files(config['privacy_mode'])
            
            # 2. Esegui training MONITORATO
            training_success = run_federated_training_monitored(
                config['name'],
                num_rounds=num_rounds,
                timeout_minutes=60  # 1 ora timeout per config
            )
            
            # 3. Esegui attacchi se training riuscito
            attacks_success = False
            if training_success:
                attacks_success = run_attacks_if_model_exists(config['name'])
            
            config_time = (time.time() - config_start) / 60
            
            results_summary.append({
                'config': config['name'],
                'training_success': training_success,
                'attacks_success': attacks_success,
                'time_minutes': config_time
            })
            
            print(f"\n✅ Configurazione {config['name']} completata in {config_time:.1f} minuti")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Esperimento interrotto dall'utente alla config {config['name']}")
            break
        except Exception as e:
            print(f"\n❌ Errore configurazione {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            
            response = input("\nContinuare con la prossima configurazione? (s/n): ")
            if response.lower() != 's':
                break
    
    # Riepilogo finale
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ ESPERIMENTO COMPLETATO")
    print(f"{'='*80}")
    print(f"Durata totale: {duration}")
    print(f"Configurazioni eseguite: {len(results_summary)}/{len(CONFIGS)}")
    
    print(f"\n📊 RIEPILOGO RISULTATI:")
    print(f"{'Config':<15} {'Training':<10} {'Attacks':<10} {'Tempo (min)':<12}")
    print("-"*50)
    for result in results_summary:
        training_status = "✅" if result['training_success'] else "❌"
        attacks_status = "✅" if result['attacks_success'] else "❌"
        print(f"{result['config']:<15} {training_status:<10} {attacks_status:<10} {result['time_minutes']:<12.1f}")
    
    print(f"\n📁 RISULTATI SALVATI IN:")
    print(f"  - Training metrics: cartella 'results'")
    print(f"  - Attack results: cartella 'attack_results'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Esperimento interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore fatale: {e}")
        import traceback
        traceback.print_exc()