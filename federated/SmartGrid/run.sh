#!/bin/bash

# Script di avvio per lanciare 15 client federati SmartGrid uno alla volta
# Usa python3, lancia ogni client con 1 secondo di intervallo

CLIENT_SCRIPT="fed_RF_client_incremental.py"

for i in $(seq 1 15)
do
    echo "Avvio client federato $i ..."
    python3 "$CLIENT_SCRIPT" $i &
    sleep 1  # Attendi 1 secondo prima di far partire il prossimo client
done

echo "Tutti i 15 client sono stati avviati!"