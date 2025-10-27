#!/bin/bash

# Script per avviare 14 client federati "buoni" e 1 client "malevolo"

# 2. Avvia 14 client buoni (client_id da 1 a 14)
for i in $(seq 1 14); do
    echo "Avvio client buono $i ..."
    python3 fed_RF_client_incremental.py $i &
    sleep 1
done

# 3. Avvia il client malevolo (client_id 99)
echo "Avvio client malevolo 99 ..."
python3 fed_RF_client_malicious_aia_art.py 99 &

# 4. Attendi che tutti i processi terminino
wait

echo "Tutti i client hanno terminato!"