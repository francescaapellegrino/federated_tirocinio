import multiprocessing
import os
import time

def run_normal_client(client_id):
    """Avvia un client normale."""
    print(f"Avvio client normale {client_id}")
    os.system(f"python3 client.py {client_id}")

def run_malicious_client(client_id):
    """Avvia un client malevolo."""
    print(f"Avvio client malevolo {client_id}")
    os.system(f"python3 client_mia.py {client_id}")

if __name__ == "__main__":
    processes = []
    
    print("=== AVVIO SISTEMA FEDERATO CON CLIENT MALEVOLO ===")
    print("Verifica che il server sia attivo...")
    time.sleep(2)  # Breve pausa per leggere il messaggio
    
    # Configurazione client
    malicious_clients = [13]  # Client malevolo
    normal_clients = list(range(1, 13))  # Client normali (1-12)
    
    print(f"Client normali: {len(normal_clients)} (ID: {normal_clients})")
    print(f"Client malevoli: {len(malicious_clients)} (ID: {malicious_clients})")
    print("\nAvvio client tra 3 secondi...")
    time.sleep(3)
    
    try:
        # Avvia prima il client malevolo
        print("\nAvvio client malevolo...")
        for client_id in malicious_clients:
            p = multiprocessing.Process(target=run_malicious_client, args=(client_id,))
            p.start()
            processes.append(p)
            print(f"  - Client malevolo {client_id} avviato")
            time.sleep(2)  # Attendi che il client malevolo si inizializzi
        
        # Poi avvia i client normali
        print("\nAvvio client normali...")
        for client_id in normal_clients:
            p = multiprocessing.Process(target=run_normal_client, args=(client_id,))
            p.start()
            processes.append(p)
            print(f"  - Client {client_id} avviato")
            time.sleep(0.5)  # Pausa breve tra i client normali
        
        print("\nTutti i client sono stati avviati.")
        print("Il sistema continuerà a funzionare finché il server è attivo.")
        print("Premi Ctrl+C per terminare tutti i client.")
        
        # Attendi che tutti i processi terminino
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\nTerminazione richiesta dall'utente...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("Tutti i client sono stati terminati.")
        
    except Exception as e:
        print(f"\nErrore durante l'esecuzione: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("Tutti i client sono stati terminati a causa dell'errore.")
