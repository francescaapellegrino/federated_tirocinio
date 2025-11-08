"""
Script per avviare client con privacy mechanisms.
VERSIONE CON OUTPUT VISIBILE per debugging.
"""

import multiprocessing
import subprocess
import sys
import time

def run_client(client_id):
    """
    Avvia un client e mostra il suo output.
    """
    print(f"[LAUNCHER] Avvio client {client_id}...")
    
    try:
        # Usa subprocess invece di os.system per catturare errori
        result = subprocess.run(
            [sys.executable, 'clientRFtmp_privacy.py', str(client_id)],
            capture_output=False,  # Mostra output direttamente
            text=True
        )
        
        if result.returncode != 0:
            print(f"[LAUNCHER] ❌ Client {client_id} terminato con errore (code {result.returncode})")
        else:
            print(f"[LAUNCHER] ✅ Client {client_id} completato")
            
    except Exception as e:
        print(f"[LAUNCHER] ❌ Errore avvio client {client_id}: {e}")

def main():
    print("="*70)
    print("🛡️ AVVIO FEDERATED LEARNING CON PRIVACY MECHANISMS")
    print("="*70)
    print(f"Avvio {13} client con configurazione privacy...")
    print("")
    
    num_clients = 12
    processes = []

    num_clients2 = 15

    for client_id in range(2, num_clients + 1):  # da 2 a 12 inclusi
        p = multiprocessing.Process(target=run_client, args=(client_id,))
        p.start()
        processes.append(p)

    for client_id in range(14, num_clients2 + 1):  # da 14 a 15 inclusi
         p = multiprocessing.Process(target=run_client, args=(client_id,))
         p.start()
         processes.append(p)

    for p in processes:
        p.join()
        print(f"[LAUNCHER] Client terminato")
    
    print(f"\n✅ Tutti i client hanno completato l'esecuzione")

if __name__ == "__main__":
    main()