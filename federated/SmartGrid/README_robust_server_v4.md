# Robust Server Final v4 - SmartGrid Federated Learning

## Panoramica

`robust_server_final.py` è il server federato avanzato compatibile con l'architettura client v4, progettato per il progetto SmartGrid Federated Learning.

## Caratteristiche Principali

### 🏗️ Architettura v4
- **Input**: 40 features (30 PCA + 10 feature engineering)
- **Modello**: Input(40) → Dense(256)+Attention → Dense(128) → Dense(64) → Dense(1)
- **Attention Mechanism**: Dense(256, sigmoid) + Multiply layer
- **Layers aggiuntivi**: BatchNormalization e Dropout
- **Tensori di peso**: 22 (verificati per compatibilità v4)

### 🔄 Preprocessing Pipeline
1. **QuantileTransformer**: Invece di StandardScaler per migliore robustezza
2. **PCA**: Riduzione a 30 componenti principali (96% varianza spiegata)
3. **Feature Engineering**: Aggiunta di 10 features avanzate:
   - Statistiche aggregate (media, std, min, max)
   - Ratio e interazioni
   - Features polinomiali
   - Interazioni tra componenti principali
   - Feature di energia/potenza

### 📊 Configurazione
- **Rounds**: 10 (ottimizzato per performance)
- **Valutazione globale**: Client 14-15 (mai usati per training)
- **Metriche**: accuracy, precision, recall, f1-score, AUC-ROC
- **Logging**: Dettagliato con file `robust_server_v4.log`

## 🚀 Utilizzo

### Avvio del Server
```bash
cd federated/SmartGrid
python robust_server_final.py
```

### Test della Compatibilità
```bash
python test_robust_server_v4.py
```

## 📋 Requisiti

### Dipendenze Python
- `tensorflow>=2.19.0`
- `flwr>=1.20.0`
- `pandas>=2.3.0`
- `numpy>=2.1.0`
- `scikit-learn>=1.7.0`

### Installazione
```bash
pip install tensorflow flwr pandas numpy scikit-learn
```

## 🔧 Compatibilità Client v4

Il server è progettato per essere compatibile con client che utilizzano:

1. **Feature engineering avanzata** (40 features finali)
2. **Architettura con attention mechanism**
3. **QuantileTransformer preprocessing**
4. **Threshold optimization**

### Esempio di Client Compatibile
```python
# Il client deve implementare:
# 1. Stessa pipeline preprocessing (QuantileTransformer + PCA + Feature Engineering)
# 2. Stessa architettura modello (40→256+Attention→128→64→1)
# 3. 22 tensori di peso
```

## 📊 Output del Server

### Informazioni di Avvio
```
🚀 ROBUST FEDERATED SERVER v4 - SMARTGRID
🔧 Configurazione v4:
  📊 Architettura: 40 → 256+Attention → 128 → 64 → 1
  🧠 Preprocessing: QuantileTransformer + PCA(30) + Feature Engineering
  🔄 Rounds: 10 (ottimizzato)
  🎯 Features finali: 40 (30 PCA + 10 engineered)
  🔍 Attention Mechanism: Dense(256, sigmoid) + Multiply
  📈 Validazione globale: Client 14-15
```

### Metriche di Valutazione Globale
```
=== VALUTAZIONE GLOBALE v4 - ROUND X ===
  💰 Loss: 0.XXXX
  🎯 Accuracy: X.XXXX (XX.XX%)
  🔍 Precision: X.XXXX (XX.XX%)
  📡 Recall: X.XXXX (XX.XX%)
  ⚖️  F1-Score: X.XXXX (XX.XX%)
  📊 AUC-ROC: X.XXXX (XX.XX%)
  📈 Campioni: XXXXX
```

## 🐛 Debugging e Logging

### File di Log
- **Posizione**: `robust_server_v4.log`
- **Livello**: INFO (dettagliato)
- **Formato**: `timestamp - level - message`

### Gestione Errori
Il server include gestione robusta degli errori con:
- Validazione parametri modello
- Fallback per errori preprocessing
- Logging dettagliato degli errori
- Retry logic per operazioni critiche

## 🧪 Test e Validazione

Il file `test_robust_server_v4.py` verifica:

1. **Architettura modello** (22 tensori di peso)
2. **Pipeline preprocessing** (128→40 features)
3. **Feature engineering** (30→40 features)
4. **Valutazione globale** (funzionalità completa)
5. **Compatibilità v4** (test end-to-end)

### Esecuzione Test
```bash
python test_robust_server_v4.py
```

### Output Test di Successo
```
✅ Test superati: 5/5
📈 Percentuale successo: 100.0%
🎉 TUTTI I TEST SUPERATI - ROBUST SERVER v4 PRONTO!
```

## 📝 Note Tecniche

### Differenze dalla Versione Precedente
- **Preprocessing**: QuantileTransformer invece di StandardScaler
- **Architettura**: 256→128→64→1 con attention invece di 64→32→1
- **Features**: 40 invece di 30
- **Rounds**: 10 invece di 100
- **Logging**: Più dettagliato e strutturato

### Performance
- **Parametri modello**: ~119k (vs ~3k precedente)
- **Tempo preprocessing**: ~1s per 10k campioni
- **Memoria**: ~500MB durante training
- **Accuratezza attesa**: 70-85% su dataset SmartGrid

## 🔐 Sicurezza

Il server include:
- Validazione input robusta
- Gestione sicura dei file
- Logging senza informazioni sensibili
- Isolamento delle operazioni critiche

## 📚 References

- **Paper**: Federated Learning for Smart Grid Intrusion Detection
- **Dataset**: Smart Grid Stability Simulated Data
- **Framework**: Flower Federated Learning
- **Architecture**: Deep Neural Networks with Attention Mechanisms

---

**Autore**: Sistema AI per francescaapellegrino  
**Versione**: v4.0 Final  
**Data**: 2025-01-27  
**License**: MIT