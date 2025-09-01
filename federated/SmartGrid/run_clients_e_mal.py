import multiprocessing
import os

def run_privacy_client(client_id):
    os.system(f"python3 client_with_privacy.py {client_id}")

def run_malicious_client(client_id):
    os.system(f"python3 malicious_client_inference.py {client_id} true")

if __name__ == "__main__":
    # Client privacy 1-11 + malevolo 12
    privacy_clients = list(range(1, 12))
    malicious_clients = [12]
    
    processes = []
    
    # Avvia client privacy
    for client_id in privacy_clients:
        p = multiprocessing.Process(target=run_privacy_client, args=(client_id,))
        p.start()
        processes.append(p)
    
    # Avvia client malevolo
    for client_id in malicious_clients:
        p = multiprocessing.Process(target=run_malicious_client, args=(client_id,))
        p.start()
        processes.append(p)
    
    print(f"Avviati {len(privacy_clients)} client privacy + {len(malicious_clients)} malevolo")
    
    for p in processes:
        p.join()