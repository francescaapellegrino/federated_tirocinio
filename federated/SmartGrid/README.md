# SmartGrid Federated Learning - Sistema di Ottimizzazione Completo

## 📊 Panoramica

Questo sistema implementa un framework completo di federated learning ottimizzato per SmartGrid con analisi della distribuzione client, strategie di aggregazione bilanciate, client adattivi e benchmarking avanzato.

## 🎯 Obiettivi Raggiunti

✅ **Analisi completa distribuzione client**
✅ **Strategie di aggregazione bilanciate**  
✅ **Server ottimizzato con monitoring avanzato**
✅ **Client adattivi con learning rate personalizzati**
✅ **Sistema di benchmark comparativo**
✅ **Documentazione e visualizzazioni per tesi**

## 📁 Struttura del Sistema

```
federated/SmartGrid/
├── analyze_client_distribution.py    # Analisi distribuzione client
├── strategies/
│   ├── __init__.py
│   └── balanced_strategy.py         # Strategie aggregazione bilanciate
├── server_optimized.py              # Server FL ottimizzato
├── client_adaptive.py               # Client adattivi
├── benchmark_optimizations.py       # Sistema benchmark
├── analysis_output/                 # Risultati analisi distribuzione
└── test_benchmark/                  # Risultati benchmark test
```

## 🔍 Componenti Implementati

### 1. Analisi Distribuzione Client (`analyze_client_distribution.py`)

**Funzionalità principali:**
- Analisi quantitativa campioni per client (4966-5570 campioni/client)
- Analisi bilanciamento classi (65-78% attacchi vs normali)
- Calcolo similarità tra client (cosine similarity)
- Identificazione client outlier (6 outlier su 13 client)
- Clustering client (3 cluster con silhouette score 0.110)
- Calcolo indice eterogeneità Non-IID (0.125 - relativamente omogeneo)
- Visualizzazioni complete (12 pannelli di analisi)
- Report dettagliato con raccomandazioni

**Utilizzo:**
```bash
cd federated/SmartGrid
python analyze_client_distribution.py
```

**Output:**
- `analysis_output/smartgrid_client_distribution_analysis.png`
- `analysis_output/smartgrid_distribution_report.txt`
- `analysis_output/client_statistics.csv`
- `analysis_output/client_similarity_matrix.csv`

### 2. Strategie Aggregazione Bilanciate (`strategies/balanced_strategy.py`)

**Strategie implementate:**
- **Standard**: FedAvg classico
- **Class Weighted**: Aggregazione pesata per bilanciamento classi
- **Outlier Penalty**: Penalizzazione client outlier
- **Adaptive**: Learning rates adattivi basati su performance
- **Hybrid**: Combinazione di tutte le tecniche
- **SmartGrid Optimized**: Configurazione ottimizzata per SmartGrid

**Caratteristiche:**
- Weighted aggregation intelligente
- Client selection diversificata
- Monitoring real-time distribuzione
- Adaptive learning rates per client
- Gestione outlier automatica

**Utilizzo:**
```python
from strategies import create_smartgrid_optimized_strategy

strategy = create_smartgrid_optimized_strategy(
    fraction_fit=0.8,
    outlier_penalty=0.7,
    use_client_selection=True
)
```

### 3. Server Ottimizzato (`server_optimized.py`)

**Funzionalità:**
- Integrazione strategie bilanciate
- Monitoring distribuzione real-time
- Valutazione globale su client 14-15
- Analisi convergenza automatica
- Logging completo per debugging
- Report finale automatico

**Utilizzo:**
```bash
python server_optimized.py
```

**Output:**
- Logs dettagliati per round
- Analisi convergenza
- Report finale JSON
- Metriche performance complete

### 4. Client Adattivi (`client_adaptive.py`)

**Caratteristiche:**
- Learning rate personalizzati basati su performance
- Preprocessing robusto (gestione NaN/infinity)
- Monitoring locale avanzato
- Reporting statistiche dettagliate
- Adattamento dinamico parametri server

**Utilizzo:**
```bash
python client_adaptive.py --client-id 1 --server localhost:8080
```

**Funzionalità adaptive:**
- Calcolo LR basato su trend performance
- Stabilità e overfitting detection
- Logging locale dettagliato
- Metriche complete per aggregazione

### 5. Sistema Benchmark (`benchmark_optimizations.py`)

**Confronti implementati:**
- 6 configurazioni diverse (baseline vs ottimizzate)
- Simulazione federated learning realistica
- Metriche comparative complete
- Analisi fairness tra client
- Visualizzazioni automatiche

**Metriche analizzate:**
- **Performance**: Accuracy, F1-score, AUC-ROC
- **Convergenza**: Round per convergenza, stabilità
- **Fairness**: Varianza tra client, bilanciamento
- **Efficienza**: Tempo per round, overhead comunicazione

**Utilizzo:**
```bash
python benchmark_optimizations.py --rounds 20 --clients 1-13
```

## 📊 Risultati Chiave

### Analisi Distribuzione
- **13 client** analizzati per training (client 1-13)
- **Eterogeneità**: Indice 0.125 (relativamente omogeneo)
- **Outlier**: 6 client identificati con caratteristiche anomale
- **Distribuzione classi**: 65-78% attacchi, variazione moderata
- **Raccomandazione**: FedAvg standard sufficiente, ma ottimizzazioni bilanciate possono migliorare fairness

### Performance Benchmark
- **Baseline accuracy**: ~72.5% (FedAvg standard)
- **Strategie ottimizzate**: Performance competitive con migliore fairness
- **Convergenza**: 3-20 round tipici per convergenza
- **Fairness**: Varianza client ridotta con strategie bilanciate
- **Migliore configurazione**: SmartGrid Optimized per fairness

### Ottimizzazioni Implementate
1. **Weighted Aggregation**: Pesi basati su bilanciamento classi
2. **Client Selection**: Selezione diversificata per round  
3. **Adaptive Learning Rates**: LR personalizzati per caratteristiche client
4. **Outlier Handling**: Gestione client con dati anomali
5. **Real-time Monitoring**: Tracking distribuzione e performance

## 🚀 Come Utilizzare il Sistema

### Setup Iniziale
```bash
# Install dependencies
pip install flwr tensorflow pandas numpy scikit-learn matplotlib seaborn

# Navigate to SmartGrid directory
cd federated/SmartGrid
```

### 1. Analisi Distribuzione (Raccomandato come primo step)
```bash
python analyze_client_distribution.py
```

### 2. Training Federato Standard
```bash
# Terminal 1: Start server
python server_optimized.py

# Terminal 2-4: Start clients
python client_adaptive.py --client-id 1
python client_adaptive.py --client-id 2
python client_adaptive.py --client-id 3
```

### 3. Benchmark Comparativo
```bash
python benchmark_optimizations.py --rounds 20 --clients 1-13 --output results
```

## 📈 Visualizzazioni per Tesi

Il sistema genera automaticamente:

1. **Analisi Distribuzione**: 12-panel dashboard con:
   - Distribuzione campioni per client
   - Bilanciamento classi
   - Heatmap similarità client
   - Clustering e outlier detection
   - Metriche eterogeneità

2. **Benchmark Comparativo**: Grafici con:
   - Accuracy e F1-score finale
   - Round per convergenza
   - Tempo per round
   - Varianza client (fairness)
   - Radar chart top configurazioni

3. **Report Dettagliati**: 
   - Statistiche complete CSV
   - Report testuali con raccomandazioni
   - Logs JSON per analisi approfondite

## 🔧 Configurazioni Personalizzate

### Strategia Custom
```python
from strategies import BalancedFedAvg

custom_strategy = BalancedFedAvg(
    balance_strategy="hybrid",
    outlier_penalty=0.6,
    adaptive_lr_factor=0.85,
    diversity_threshold=0.3,
    use_client_selection=True
)
```

### Server Custom
```python
from server_optimized import SmartGridOptimizedServer

server = SmartGridOptimizedServer(
    strategy_type="hybrid",
    num_rounds=50,
    n_components=20,
    save_logs=True
)
```

## 📊 Metriche di Valutazione

### Performance
- **Global Accuracy**: Accuracy su test set globale
- **F1-Score**: Importante per classi sbilanciate
- **AUC-ROC**: Area sotto curva ROC
- **Precision/Recall**: Per classe attacchi e normali

### Convergenza
- **Round per Convergenza**: Velocità raggiungimento stabilità
- **Stabilità Loss**: Varianza loss nelle ultime epoch
- **Improvement Rate**: Tasso miglioramento per round

### Fairness
- **Client Variance**: Varianza accuracy tra client
- **Performance Range**: Differenza min-max tra client
- **Outlier Impact**: Effetto client outlier su performance globale

### Efficienza
- **Round Duration**: Tempo medio per round
- **Communication Overhead**: Overhead comunicazione
- **Client Participation**: Numero client partecipanti per round

## 🎓 Utilizzo per Tesi

Il sistema fornisce materiali pronti per tesi:

1. **Grafici e Tabelle**: Tutti in alta risoluzione (300 DPI)
2. **Statistiche Dettagliate**: CSV esportabili per ulteriori analisi
3. **Report Testuali**: Sezioni pronte per inserimento tesi
4. **Risultati Comparativi**: Baseline vs ottimizzazioni
5. **Raccomandazioni**: Basate su analisi quantitativa

## 🔬 Lavori Futuri

Possibili estensioni del sistema:
- Differential Privacy per privacy-preserving FL
- Algoritmi di consensus Byzantine-fault tolerant
- Federated learning cross-silo per multiple organizzazioni
- Ottimizzazioni per edge computing
- Integration con IoT devices reali

## 📝 Citazioni e Riferimenti

Questo sistema implementa e integra tecniche da:
- FedAvg (McMahan et al.)
- Non-IID federated learning (Li et al.)
- Adaptive federated optimization (Reddi et al.)
- Client selection strategies (Nishio & Yonetani)

---

**Autore**: francescaapellegrino  
**Data**: 2025-08-17  
**Versione**: 1.0  
**Repository**: federated_tirocinio  

Per domande o supporto, vedere la documentazione nei singoli file o contattare l'autore.