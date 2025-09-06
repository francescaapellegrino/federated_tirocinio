"""
Ottimizzatore Optuna per SmartGrid Federated Learning
Francesca Pellegrino
"""

import optuna
import tensorflow as tf
import keras
from keras import layers, regularizers
from keras.models import Sequential
from keras.optimizers import Adam, AdamW, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import Precision, Recall, F1Score, AUC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# CONFIGURAZIONE OPTUNA
class OptimizationConfig:
    
    # Dataset
    CLIENT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Client per training optimization
    VALIDATION_SPLIT = 0.3
    
    # Optuna
    N_TRIALS = 50              # Numero trial da eseguire
    TIMEOUT_HOURS = 6             # Timeout massimo
    N_JOBS = 1                    # Parallelismo
    
    # Evaluation
    CV_FOLDS = 3                  # Cross-validation folds
    OBJECTIVE_METRIC = 'val_loss'  # Metrica da ottimizzare
    
    # Search space bounds
    LR_MIN = 0.0001
    LR_MAX = 0.01
    L2_MIN = 0.00001  
    L2_MAX = 0.01
    DROPOUT_MIN = 0.1
    DROPOUT_MAX = 0.5
    NEURONS_MIN = 8
    NEURONS_MAX = 256
    
    # Fixed parameters
    PCA_COMPONENTS = 30 
    RANDOM_SEED = 42
    EPOCHS = 15                   # Fixed per trial speed
    BATCH_SIZE = 32               # Fixed per trial speed

# DATA LOADER
class SmartGridDataLoader:
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def load_combined_data(self):
        """Carica dati da multipli client per robustezza"""
        print("📂Caricamento dati SmartGrid per ottimizzazione...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
        
        df_list = []
        loaded_clients = []
        
        for client_id in self.config.CLIENT_IDS:
            file_path = os.path.join(data_dir, f"data{client_id}.csv")
            
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0 and 'marker' in df.columns:
                    df_list.append(df)
                    loaded_clients.append(client_id)
                    print(f"   ✅ Client {client_id}: {len(df)} campioni")
            except Exception as e:
                print(f"   ❌ Client {client_id}: {e}")
        
        if not df_list:
            raise FileNotFoundError("Nessun dato valido trovato!")
        
        # Combina dati
        df_combined = pd.concat(df_list, ignore_index=True)
        X = df_combined.drop(columns=["marker"])
        y = (df_combined["marker"] != "Natural").astype(int)
        
        print(f"Dataset combinato: {len(df_combined)} campioni")
        print(f"   - Attack ratio: {y.mean()*100:.1f}%")
        print(f"   - Original features: {X.shape[1]}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocessing"""
        print("Preprocessing dati...")
        
        # 1. Pulizia
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # 2. Split train/validation  
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.VALIDATION_SPLIT, 
            random_state=self.config.RANDOM_SEED,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 3. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 4. PCA
        pca = PCA(n_components=self.config.PCA_COMPONENTS, random_state=self.config.RANDOM_SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"   ✅ PCA: {X.shape[1]} → {X_train_pca.shape[1]} features")
        print(f"   ✅ Varianza spiegata: {variance_explained*100:.1f}%")
        
        return X_train_pca, X_val_pca, y_train, y_val

# MODEL BUILDER
class OptimizedModelBuilder:
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def build_model(self, trial, input_shape):
        """
        Costruisce modello con iperparametri suggeriti da Optuna.
        Args:
            trial: Trial Optuna per suggerimenti parametri
            input_shape: Numero di feature input
        Returns:
            tuple: (model, params_dict)
        """
        
        # IPERPARAMETRI DA OTTIMIZZARE CON OPTUNA
        params = {
            # Learning rate - range logaritmico per migliore esplorazione
            'learning_rate': trial.suggest_float(
                'learning_rate', 
                self.config.LR_MIN, 
                self.config.LR_MAX, 
                log=True
            ),
            
            # L2 regularization - range logaritmico
            'l2_reg': trial.suggest_float(
                'l2_reg', 
                self.config.L2_MIN, 
                self.config.L2_MAX, 
                log=True
            ),
            
            # Architettura 4-layer
            'layer1_neurons': trial.suggest_int(
                'layer1_neurons', 
                64, 256, step=16
            ),
            'layer2_neurons': trial.suggest_int(
                'layer2_neurons', 
                32, 128, step=8
            ),
            'layer3_neurons': trial.suggest_int(
                'layer3_neurons', 
                16, 64, step=4
            ),
            'layer4_neurons': trial.suggest_int(
                'layer4_neurons', 
                8, 32, step=2
            ),
            
            # Dropout rates per layer
            'dropout1': trial.suggest_float(
                'dropout1', 
                self.config.DROPOUT_MIN, 
                self.config.DROPOUT_MAX, 
                step=0.05
            ),
            'dropout2': trial.suggest_float(
                'dropout2', 
                self.config.DROPOUT_MIN, 
                self.config.DROPOUT_MAX, 
                step=0.05
            ),
            'dropout3': trial.suggest_float(
                'dropout3', 
                self.config.DROPOUT_MIN, 
                self.config.DROPOUT_MAX, 
                step=0.05
            ),
            'dropout4': trial.suggest_float(
                'dropout4', 
                self.config.DROPOUT_MIN, 
                self.config.DROPOUT_MAX, 
                step=0.05
            ),
            
            # Funzione di attivazione
            'activation': trial.suggest_categorical(
                'activation', 
                ['relu', 'leaky_relu', 'selu', 'elu']
            ),
            
            # Tipo ottimizzatore
            'optimizer_type': trial.suggest_categorical(
                'optimizer_type', 
                ['adam', 'adamw', 'nadam']
            ),
            
            # Configurazioni aggiuntive
            'use_batch_norm': trial.suggest_categorical(
                'use_batch_norm', 
                [True, False]
            ),
            
            # Beta parameters per Adam
            'beta_1': trial.suggest_float(
                'beta_1', 
                0.85, 0.95, step=0.01
            ),
            'beta_2': trial.suggest_float(
                'beta_2', 
                0.990, 0.999, step=0.001
            ),
            
            # Gradient clipping
            'clipnorm': trial.suggest_float(
                'clipnorm', 
                0.5, 2.0, step=0.1
            )
        }
        
        # CONFIGURAZIONE FUNZIONE DI ATTIVAZIONE
        if params['activation'] == 'leaky_relu':
            activation_fn = lambda: layers.LeakyReLU(alpha=0.1)
            initializer = 'he_normal'
        elif params['activation'] == 'selu':
            activation_fn = lambda: layers.Activation('selu')
            initializer = 'lecun_normal'
        elif params['activation'] == 'elu':
            activation_fn = lambda: layers.ELU(alpha=1.0)
            initializer = 'he_normal'
        else:  # relu default
            activation_fn = lambda: layers.Activation('relu')
            initializer = 'he_normal'

        # COSTRUZIONE ARCHITETTURA OTTIMIZZATA
        model_layers = [
            # Input layer esplicito
            layers.Input(shape=(input_shape,), name='input_features'),
            
            # === LAYER 1 ===
            layers.Dense(
                params['layer1_neurons'],
                kernel_regularizer=regularizers.L2(params['l2_reg']),
                kernel_initializer=initializer,
                name='dense_1'
            ),
            activation_fn(),
        ]
        
        # Batch Normalization condizionale
        if params['use_batch_norm']:
            model_layers.append(layers.BatchNormalization(name='batch_norm_1'))
        
        model_layers.extend([
            layers.Dropout(params['dropout1'], name='dropout_1'),
            
            # === LAYER 2 ===
            layers.Dense(
                params['layer2_neurons'],
                kernel_regularizer=regularizers.L2(params['l2_reg']),
                kernel_initializer=initializer,
                name='dense_2'
            ),
            activation_fn(),
        ])
        
        if params['use_batch_norm']:
            model_layers.append(layers.BatchNormalization(name='batch_norm_2'))
        
        model_layers.extend([
            layers.Dropout(params['dropout2'], name='dropout_2'),
            
            # === LAYER 3 ===
            layers.Dense(
                params['layer3_neurons'],
                kernel_regularizer=regularizers.L2(params['l2_reg']),
                kernel_initializer=initializer,
                name='dense_3'
            ),
            activation_fn(),
        ])
        
        if params['use_batch_norm']:
            model_layers.append(layers.BatchNormalization(name='batch_norm_3'))
        
        model_layers.extend([
            layers.Dropout(params['dropout3'], name='dropout_3'),
            
            # === LAYER 4 ===
            layers.Dense(
                params['layer4_neurons'],
                kernel_regularizer=regularizers.L2(params['l2_reg']),
                kernel_initializer=initializer,
                name='dense_4'
            ),
            activation_fn(),
        ])
        
        if params['use_batch_norm']:
            model_layers.append(layers.BatchNormalization(name='batch_norm_4'))
        
        model_layers.extend([
            layers.Dropout(params['dropout4'], name='dropout_4'),
            
            # === OUTPUT LAYER ===
            layers.Dense(
                1, 
                activation='sigmoid',
                kernel_initializer='glorot_uniform',
                name='output'
            )
        ])
        
        # Crea modello sequenziale
        model = keras.Sequential(model_layers, name=f'SmartGrid_Optimized_{trial.number}')

        # CONFIGURAZIONE OTTIMIZZATORE
        if params['optimizer_type'] == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=params['learning_rate'],
                weight_decay=params['l2_reg'] * 0.1,  # Weight decay proporzionale a L2
                beta_1=params['beta_1'],
                beta_2=params['beta_2'],
                epsilon=1e-7,
                clipnorm=params['clipnorm']
            )
        elif params['optimizer_type'] == 'nadam':
            optimizer = keras.optimizers.Nadam(
                learning_rate=params['learning_rate'],
                beta_1=params['beta_1'],
                beta_2=params['beta_2'],
                epsilon=1e-7,
                clipnorm=params['clipnorm']
            )
        else:  # adam default
            optimizer = keras.optimizers.Adam(
                learning_rate=params['learning_rate'],
                beta_1=params['beta_1'],
                beta_2=params['beta_2'],
                epsilon=1e-7,
                clipnorm=params['clipnorm']
            )
        
        # COMPILAZIONE CON METRICHE COMPLETE
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.F1Score(name='f1_score'),
                keras.metrics.AUC(name='auc_roc'),
                keras.metrics.AUC(name='auc_pr', curve='PR')  # Precision-Recall AUC
            ]
        )

        # AGGIUNGI METADATA AI PARAMETRI
        params.update({
            'model_name': model.name,
            'total_parameters': model.count_params(),
            'architecture_summary': f"{params['layer1_neurons']}→{params['layer2_neurons']}→{params['layer3_neurons']}→{params['layer4_neurons']}→1",
            'input_shape': input_shape,
            'activation_function': params['activation'],
            'optimizer_name': params['optimizer_type'],
            'regularization_type': 'L2',
            'batch_normalization': params['use_batch_norm'],
            'total_layers': 6,  # 4 hidden + input + output
            'timestamp': '2025-08-20 13:23:48',
            'optimized_by': 'francescaapellegrino',
            'optuna_trial': trial.number
        })

        # LOG ARCHITETTURA (per debug)
        if trial.number % 10 == 0:  # Log ogni 10 trial
            print(f"\nTrial {trial.number} - Architettura costruita:")
            print(f"   - {params['architecture_summary']}")
            print(f"   - LR: {params['learning_rate']:.6f}")
            print(f"   - L2: {params['l2_reg']:.6f}")
            print(f"   - Optimizer: {params['optimizer_type']}")
            print(f"   - Activation: {params['activation']}")
            print(f"   - BatchNorm: {params['use_batch_norm']}")
            print(f"   - Parametri totali: {params['total_parameters']:,}")
        
        return model, params
    
    def validate_model_architecture(self, model, params):
        """Valida che il modello sia costruito correttamente.
        Args:
            model: Modello Keras
            params: Dizionario parametri
        Returns:
            bool: True se valido
        """
        try:
            # Verifica input shape
            expected_input = (None, self.config.PCA_COMPONENTS)
            actual_input = model.input_shape
            
            if actual_input != expected_input:
                print(f"⚠️ Warning: Input shape mismatch. Expected: {expected_input}, Got: {actual_input}")
                return False
            
            # Verifica output shape
            expected_output = (None, 1)
            actual_output = model.output_shape
            
            if actual_output != expected_output:
                print(f"⚠️ Warning: Output shape mismatch. Expected: {expected_output}, Got: {actual_output}")
                return False
            
            # Verifica numero layer
            expected_layers = 6  # Input + 4 hidden + output
            actual_layers = len([l for l in model.layers if isinstance(l, layers.Dense)])
            
            if actual_layers != expected_layers:
                print(f"⚠️ Warning: Layer count mismatch. Expected: {expected_layers}, Got: {actual_layers}")
                return False
            
            # Verifica compilazione
            if not hasattr(model, 'optimizer'):
                print(f"⚠️ Warning: Model not compiled")
                return False
            
            print(f"✅ Model validation passed for trial {params.get('optuna_trial', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False

# FUNZIONE OBIETTIVO PER OPTUNA 
class SmartGridObjective:
    
    def __init__(self, X_train, X_val, y_train, y_val, config: OptimizationConfig):
        self.X_train = X_train
        self.X_val = X_val  
        self.y_train = y_train
        self.y_val = y_val
        self.config = config
        self.model_builder = OptimizedModelBuilder(config)
        self.trial_count = 0
        
    def __call__(self, trial):
        """Funzione obiettivo chiamata da Optuna"""
        self.trial_count += 1
        
        try:
            # Costruisci modello
            model, params = self.model_builder.build_model(trial, self.X_train.shape[1])
            
            # Class weights per dataset sbilanciato
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=2,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            # Training
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            # Valutazione
            y_pred_prob = model.predict(self.X_val, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Metriche
            f1_weighted = f1_score(self.y_val, y_pred, average='weighted')
            f1_macro = f1_score(self.y_val, y_pred, average='macro')
            balanced_acc = balanced_accuracy_score(self.y_val, y_pred)
            
            try:
                auc_score = roc_auc_score(self.y_val, y_pred_prob)
            except:
                auc_score = 0.5
            
            # Score combinato
            val_loss = history.history['val_loss'][-1]
            if self.config.OBJECTIVE_METRIC == 'val_loss':
                return val_loss
            
            """
            if self.config.OBJECTIVE_METRIC == 'f1_weighted':
                objective_score = f1_weighted
            elif self.config.OBJECTIVE_METRIC == 'balanced_accuracy':
                objective_score = balanced_acc
            else:  # composite
                objective_score = (f1_weighted + balanced_acc + auc_score) / 3
            """

            # Log progresso ogni 10 trial
            if self.trial_count % 10 == 0:
                print(f"Trial {self.trial_count}: Score={objective_score:.4f} "
                      f"(F1={f1_weighted:.4f}, BAcc={balanced_acc:.4f}, AUC={auc_score:.4f})")
                print(f"   LR={params['learning_rate']:.6f}, L2={params['l2_reg']:.6f}")
                print(f"   Arch: {params['layer1_neurons']}→{params['layer2_neurons']}→{params['layer3_neurons']}→{params['layer4_neurons']}→1")
            
            return objective_score
            
        except Exception as e:
            print(f"Trial {self.trial_count} fallito: {e}")
            return 0.0

# FUNZIONE PRINCIPALE PER L'OTTIMIZZAZIONE
def optimize_smartgrid_hyperparameters():    
    print("OTTIMIZZAZIONE SCIENTIFICA SMARTGRID CON OPTUNA")
    print("=" * 70)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Utente: Francesca Pellegrino")
    print("=" * 70)
    
    config = OptimizationConfig()
    
    print(f"Configurazione:")
    print(f"   - Trial: {config.N_TRIALS}")
    print(f"   - Timeout: {config.TIMEOUT_HOURS}h")
    print(f"   - Metrica obiettivo: {config.OBJECTIVE_METRIC}")
    print(f"   - Client utilizzati: {config.CLIENT_IDS}")
    print(f"   - PCA components: {config.PCA_COMPONENTS}")
    
    try:
        # Carica e preprocessa dati
        data_loader = SmartGridDataLoader(config)
        X, y = data_loader.load_combined_data()
        X_train, X_val, y_train, y_val = data_loader.preprocess_data(X, y)
        
        # Crea studio Optuna
        study_name = f'smartgrid_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        study = optuna.create_study(
            direction='minimize', # o maximize a seconda della metrica
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                seed=config.RANDOM_SEED,
                n_startup_trials=max(10, config.N_TRIALS // 10),
                multivariate=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3
            )
        )
        
        # Crea funzione obiettivo
        objective = SmartGridObjective(X_train, X_val, y_train, y_val, config)

        print(f"\nAvvio ottimizzazione...")
        print(f"Obiettivo: Minimizzare {config.OBJECTIVE_METRIC}")
        print(f"Timeout: {config.TIMEOUT_HOURS} ore")

        # Esegui ottimizzazione
        study.optimize(
            objective,
            n_trials=config.N_TRIALS,
            timeout=config.TIMEOUT_HOURS * 3600,
            show_progress_bar=True
        )
        
        # Risultati
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\nOTTIMIZZAZIONE COMPLETATA!")
        print(f"Trial totali: {len(study.trials)}")
        print(f"Miglior score: {best_score:.6f}")
        print(f"Migliori parametri:")

        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"   {param}: {value:.6f}")
            else:
                print(f"   {param}: {value}")
        
        # Salva risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'best_params': best_params,
            'best_score': float(best_score),
            'timestamp': timestamp,
            'user': 'francescaapellegrino',
            'n_trials_completed': len(study.trials),
            'n_trials_requested': config.N_TRIALS,
            'objective_metric': config.OBJECTIVE_METRIC,
            'config': {
                'pca_components': config.PCA_COMPONENTS,
                'client_ids': config.CLIENT_IDS,
                'epochs': config.EPOCHS,
                'batch_size': config.BATCH_SIZE
            }
        }
        
        results_file = os.path.join(results_dir, f"smartgrid_optimization_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Risultati salvati: {results_file}")
        
        # Genera codice per implementazione
        generate_implementation_code(best_params, timestamp, best_score)
        
        return best_params, best_score
        
    except Exception as e:
        print(f"\n❌ Errore durante ottimizzazione: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# GENERATORE CODICE PER IMPLEMENTARE I PARAMETRI OTTIMIZZATI
def generate_implementation_code(best_params, timestamp, best_score=None):
    """
    Args:
        best_params: Dizionario parametri ottimali
        timestamp: Timestamp dell'ottimizzazione  
        best_score: Score migliore ottenuto (opzionale)
    """
    print(f"\nGenerazione codice implementazione...")
    
    # Genera config ottimizzata
    config_code = f'''# CONFIGURAZIONE OTTIMIZZATA CON OPTUNA
# Generata automaticamente il {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Utente: Francesca Pellegrino
# Timestamp: {timestamp}

class OptimizedConfig:
    """Configurazione ottimizzata scientificamente con Optuna"""
    
    # PARAMETRI OTTIMIZZATI SCIENTIFICAMENTE
    LEARNING_RATE = {best_params['learning_rate']:.10f}
    L2_REG = {best_params['l2_reg']:.10f}
    
    # ARCHITETTURA OTTIMIZZATA (4-layer)
    HIDDEN_LAYERS = [{best_params['layer1_neurons']}, {best_params['layer2_neurons']}, {best_params['layer3_neurons']}, {best_params['layer4_neurons']}]
    DROPOUT_RATES = [{best_params['dropout1']:.3f}, {best_params['dropout2']:.3f}, {best_params['dropout3']:.3f}, {best_params['dropout4']:.3f}]
    
    # CONFIGURAZIONE OTTIMIZZATA
    ACTIVATION_FUNCTION = "{best_params['activation']}"
    OPTIMIZER_TYPE = "{best_params['optimizer_type']}"
    USE_BATCH_NORM = {best_params['use_batch_norm']}
    
    # PARAMETRI OTTIMIZZATORE OTTIMIZZATI
    BETA_1 = {best_params['beta_1']:.3f}
    BETA_2 = {best_params['beta_2']:.3f}
    CLIPNORM = {best_params['clipnorm']:.1f}

    # PARAMETRI FISSI (COMPROVATI)
    PCA_COMPONENTS = 20
    EPOCHS_PER_ROUND = 15
    BATCH_SIZE = 32
    RANDOM_SEED = 42
    
    # METADATA OTTIMIZZAZIONE
    OPTIMIZATION_TIMESTAMP = "{timestamp}"
    OPTIMIZATION_SCORE = {best_score if best_score is not None else 0.0:.6f}
    OPTIMIZATION_METHOD = "optuna"
    VERSION = "scientifically_optimized"
    OPTIMIZED_BY = "francescaapellegrino"
    
    # ARCHITETTURA RIASSUNTO
    ARCHITECTURE_SUMMARY = "{best_params['layer1_neurons']}→{best_params['layer2_neurons']}→{best_params['layer3_neurons']}→{best_params['layer4_neurons']}→1"
    TOTAL_FEATURES = 20  # PCA fisso
    '''
    
    # Salva file 
    os.makedirs('optimization_results', exist_ok=True)
    config_file = f"optimization_results/optimized_config_{timestamp}.py"
    with open(config_file, 'w') as f:
        f.write(config_code)
    
    print(f"✅ Config ottimizzata salvata: {config_file}")
    
    architecture_summary = f"{best_params['layer1_neurons']}→{best_params['layer2_neurons']}→{best_params['layer3_neurons']}→{best_params['layer4_neurons']}→1"
    

if __name__ == "__main__":
    optimize_smartgrid_hyperparameters()