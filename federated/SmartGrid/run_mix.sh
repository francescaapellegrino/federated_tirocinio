#!/bin/bash

# Script per avviare 14 client federati "buoni" e 1 client "malevolo"

# 1. Avvia il server federato (in una tab separata o in background)
echo "Avvio server federato Random Forest..."
python3 federated_random_forest_server.py &

sleep 3

# 2. Avvia 14 client buoni (client_id da 1 a 14)
for i in $(seq 1 14); do
    echo "Avvio client buono $i ..."
    python3 federated_random_forest_client.py $i &
    sleep 1
done

# 3. Avvia il client malevolo (client_id 15)
echo "Avvio client malevolo 15 ..."
python3 federated_random_forest_malicious.py 15 true &

# 4. Attendi che tutti i processi terminino
wait

echo "Tutti i client hanno terminato!"