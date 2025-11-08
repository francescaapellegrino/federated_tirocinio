#!/usr/bin/env python3
"""
Launcher migliorato per training federato:
- Avvia 12 client normali (clientRFtmp.py) + 1 client malevolo (malicious_client_advanced.py)
- Salva stdout/stderr di ogni client in logs/client_{id}.log e logs/malicious_client_{id}.log
- Monitora i processi e segnala se un processo termina (stdout/stderr rimangono disponibili)
- Utile per debug perché permette di leggere eventuali tracebacks del client malevolo
"""

import subprocess
import os
import sys
import time
import signal
from datetime import datetime

# CONFIGURAZIONE
MALICIOUS_CLIENT_ID = 5   # client che sarà malevolo (sostituisce il normale)
TOTAL_CLIENTS = 13
LOG_DIR = "logs"
PYTHON = sys.executable  # usa l'interprete corrente (es. /usr/bin/python3 or venv)

# Comandi dei file necessari
CLIENT_SCRIPT = "clientRFtmp.py"
MALICIOUS_SCRIPT = "malicious_client_advanced.py"
SERVER_SCRIPT = "serverRFtmp.py"  # assicurati che il server sia già avviato

# Creazione cartelle log
os.makedirs(LOG_DIR, exist_ok=True)

# Gestore per terminazione graceful
running = True
def handle_sigint(sig, frame):
    global running
    print("\n[LAUNCHER] Ricevuto SIGINT/SIGTERM, terminerò i processi figlio...")
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def start_client_process(client_id, malicious=False):
    """
    Avvia un singolo processo client e redireziona stdout/stderr su file.
    Restituisce il Popen object e i percorsi dei file di log.
    """
    if malicious:
        script = MALICIOUS_SCRIPT
        log_stdout = os.path.join(LOG_DIR, f"malicious_client_{client_id}.out.log")
        log_stderr = os.path.join(LOG_DIR, f"malicious_client_{client_id}.err.log")
        cmd = [PYTHON, script, str(client_id)]
    else:
        script = CLIENT_SCRIPT
        log_stdout = os.path.join(LOG_DIR, f"client_{client_id}.out.log")
        log_stderr = os.path.join(LOG_DIR, f"client_{client_id}.err.log")
        cmd = [PYTHON, script, str(client_id)]

    # Apri i file in append mode (in modo da non sovrascrivere se riavvi)
    stdout_f = open(log_stdout, "ab")
    stderr_f = open(log_stderr, "ab")

    # Avvia il processo
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, cwd=os.getcwd())

    print(f"[LAUNCHER] Avviato {'MALEVOLO' if malicious else 'normale'} client {client_id} (PID={proc.pid})")
    print(f"         stdout -> {log_stdout}")
    print(f"         stderr -> {log_stderr}")
    return proc, stdout_f, stderr_f, log_stdout, log_stderr

def main():
    # Verifica file esistenti
    missing = []
    for f in [CLIENT_SCRIPT, MALICIOUS_SCRIPT]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print(f"[LAUNCHER] ERRORE: file mancanti: {missing}")
        print("[LAUNCHER] Assicurati di eseguire lo script dalla directory federated/SmartGrid/FRF/")
        sys.exit(1)

    # Lista clienti normali (esclude il malevolo)
    normal_client_ids = [i for i in range(1, TOTAL_CLIENTS + 1) if i != MALICIOUS_CLIENT_ID]

    processes = []  # lista di dict {type, id, proc, files...}

    try:
        # Avvia client normali
        for cid in normal_client_ids:
            proc, out_f, err_f, out_path, err_path = start_client_process(cid, malicious=False)
            processes.append({'type': 'normal', 'id': cid, 'proc': proc, 'out_f': out_f, 'err_f': err_f, 'out_path': out_path, 'err_path': err_path})
            time.sleep(0.3)  # piccolo delay tra avvii

        # Pausa breve
        time.sleep(1.0)

        # Avvia client malevolo
        proc, out_f, err_f, out_path, err_path = start_client_process(MALICIOUS_CLIENT_ID, malicious=True)
        processes.append({'type': 'malicious', 'id': MALICIOUS_CLIENT_ID, 'proc': proc, 'out_f': out_f, 'err_f': err_f, 'out_path': out_path, 'err_path': err_path})

        print("\n[LAUNCHER] Tutti i client avviati. Monitoraggio in corso. Premi Ctrl+C per terminare.\n")

        # Monitor loop: controlla lo stato dei processi
        while running and processes:
            alive = 0
            for p in processes:
                proc = p['proc']
                if proc.poll() is None:
                    alive += 1
                else:
                    # Processo terminato: stampa info e per i dettagli apri il file di log corrispondente
                    rc = proc.returncode
                    print(f"[LAUNCHER] Processo {p['type']} {p['id']} TERMINATO (returncode={rc}). Controlla {p['out_path']} e {p['err_path']}")
            print(f"[LAUNCHER] {datetime.now().strftime('%H:%M:%S')} - Client attivi: {alive}/{len(processes)}")
            if not running:
                break
            time.sleep(8)

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Interruzione da tastiera ricevuta")
    finally:
        # Termina tutti i processi ancora vivi e chiudi file log
        for p in processes:
            proc = p['proc']
            try:
                if proc.poll() is None:
                    print(f"[LAUNCHER] Terminazione processo {p['type']} {p['id']} (PID={proc.pid})")
                    proc.terminate()
                    time.sleep(1)
                    if proc.poll() is None:
                        proc.kill()
            except Exception as e:
                print(f"[LAUNCHER] Errore terminazione processo {p['id']}: {e}")
            # Chiudi i file di log
            try:
                p['out_f'].close()
                p['err_f'].close()
            except:
                pass

        print("[LAUNCHER] Tutti i processi terminati. Vedi i log in:", LOG_DIR)

if __name__ == "__main__":
    main()