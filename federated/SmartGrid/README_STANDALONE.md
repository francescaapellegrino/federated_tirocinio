# Federated Learning SmartGrid - Standalone Version

This directory contains the standalone version of the federated learning system for SmartGrid with optimized configurations and performance improvements.

## Files

### Core Files
- `stable_server_standalone.py` - Server federato standalone (10 rounds)
- `stable_client_standalone.py` - Client federato standalone (5 epoche per round)

### Original Files (for reference)
- `server.py` - Server originale (100 rounds)
- `client.py` - Client originale (5 epoche per round)

### Test and Validation
- `test_standalone_system.py` - Test di validazione completo del sistema

## Key Features

### 🔧 Performance Optimizations
- **Reduced I/O Operations**: Metrics stored in memory during training
- **JSON Saving**: Only at round 10 (final round) instead of every round
- **10 Rounds Total**: Reduced from 100 for faster experimentation

### 📊 Enhanced Tracking
- **ServerMetricsTracker**: In-memory metrics collection
- **Convergence Reports**: Every 5 rounds (round 5 and 10)
- **Performance Summary**: Comprehensive analysis at completion
- **JSON Output**: Complete experiment data with metadata

### ⚙️ Configuration
- **Server**: 10 rounds total
- **Client**: 5 epochs per round
- **Validation Split**: 75% train, 10% validation, 15% test
- **PCA Components**: 30 features
- **Monitoring**: Overfitting detection and convergence analysis

## Usage

### Starting the Server
```bash
cd federated/SmartGrid
python stable_server_standalone.py
```

The server will:
- Configure for 10 rounds
- Wait for at least 2 clients
- Track metrics in memory
- Save JSON only at round 10

### Starting Clients
In separate terminals:
```bash
cd federated/SmartGrid
python stable_client_standalone.py 1
python stable_client_standalone.py 2
python stable_client_standalone.py 3
# ... up to client 13
```

Each client will:
- Load and preprocess its data
- Train for 5 epochs per round
- Send metrics to server
- Monitor overfitting

### Expected Output

#### Server Output
```
=== AVVIO SERVER FEDERATO SMARTGRID STANDALONE (10 ROUNDS) ===
Configurazione:
  - Numero di round: 10
  - Epoche per round: 5
  - Strategia: FedAvg con validation monitoring e metrics tracking
  - ⚠️ OTTIMIZZAZIONE: JSON salvato solo al round finale per ridurre I/O
  - ⚠️ CONVERGENZA: Report ogni 5 rounds (round 5 e 10)
```

#### Client Output
```
[Client 1] === ROUND DI ADDESTRAMENTO STANDALONE ===
[Client 1] ⚠️ CONFIGURAZIONE: 5 epoche per round
[Client 1] Addestramento su 3722 campioni, validation su 499 campioni
[Client 1] === RISULTATI ADDESTRAMENTO STANDALONE ===
```

#### Final JSON Output
At round 10, a JSON file will be saved:
```
server_metrics_tracking_YYYYMMDD_HHMMSS.json
```

Structure:
```json
{
  "experiment_info": {
    "total_rounds": 10,
    "start_time": "...",
    "end_time": "...",
    "total_duration": 123.45,
    "avg_round_time": 12.34
  },
  "rounds_data": [
    {
      "round": 1,
      "timestamp": "...",
      "total_clients": 2,
      "avg_weighted_train_accuracy": 0.85,
      "avg_weighted_val_accuracy": 0.82,
      "train_val_gap": 0.03,
      "client_metrics": [...]
    }
  ],
  "performance_summary": {
    "initial_train_accuracy": 0.80,
    "final_train_accuracy": 0.90,
    "train_improvement": 0.10,
    "convergence_achieved": true
  }
}
```

## Validation

Run the comprehensive test suite:
```bash
python test_standalone_system.py
```

This validates:
- ✅ Import functionality
- ✅ Data availability
- ✅ Client data processing
- ✅ Server configuration
- ✅ Metrics tracking
- ✅ System integration

## Differences from Original

| Feature | Original | Standalone |
|---------|----------|------------|
| Rounds | 100 | **10** |
| Epochs per round | 5 | **5** (same) |
| JSON saving | Every round | **Only round 10** |
| Metrics tracking | Basic | **Enhanced with ServerMetricsTracker** |
| Convergence reports | Manual | **Every 5 rounds** |
| Performance analysis | Limited | **Complete summary** |
| I/O operations | High | **Optimized** |

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Flower (flwr)
- scikit-learn
- pandas
- numpy

## Data Files

Requires SmartGrid data files:
- `data/SmartGrid/data1.csv` - `data/SmartGrid/data15.csv`
- Client 1-13: Used for training
- Client 14-15: Reserved for global validation

## Notes

- **Compatibility**: Server and client standalone versions are designed to work together
- **Monitoring**: All original monitoring and overfitting detection features are maintained
- **Performance**: Optimized for faster experimentation with reduced I/O
- **Extensibility**: Easy to modify rounds and epochs in configuration classes