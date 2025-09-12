"""
Modello SmartGrid
Francesca Pellegrino
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Creazione modello migliorato
def create_improved_model(input_shape: int, config=None):
    
    # VERIFICA COMPATIBILITÀ INPUT
    if input_shape != 30:
        print(f"⚠️ Warning: input_shape {input_shape} != 30, forzo a 30 per compatibilità")
        input_shape = 30
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    print(f"🔧 MODELLO:")
    print(f"   📐 Input garantito: {input_shape} features")
    
    # CONFIGURAZIONE
    class CompatibleImprovedConfig:
        # Architettura
        HIDDEN_LAYERS = [256, 128, 64, 32]  # Più ampia del sistema attuale
        DROPOUT_RATES = [0.3, 0.4, 0.3, 0.2]  # Dropout progressivo

        # Parametri per convergenza
        LEARNING_RATE = 0.0015  # Più conservativo per stabilità
        L2_REG = 0.0001  # Regularization moderata
        
        # Ottimizzazioni avanzate
        USE_BATCH_NORM = True
        ACTIVATION = 'relu'
        
        # Optimizer
        OPTIMIZER = 'adam_compatible'
        BETA_1 = 0.9
        BETA_2 = 0.999
        CLIPNORM = 1.0
    
    improved_config = CompatibleImprovedConfig()
    
    # Costruzione modello 
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(input_shape,), name="input_features"),
        
        # Layer 1 (più ampio)
        keras.layers.Dense(
            improved_config.HIDDEN_LAYERS[0],  # 256 neuroni
            kernel_regularizer=keras.regularizers.L2(improved_config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_1"
        ),
        keras.layers.Activation('relu', name="activation_1"),
        keras.layers.BatchNormalization(name="batch_norm_1") if improved_config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(improved_config.DROPOUT_RATES[0], name="dropout_1"),
        
        # Layer 2 (migliorato)
        keras.layers.Dense(
            improved_config.HIDDEN_LAYERS[1],  # 128 neuroni
            kernel_regularizer=keras.regularizers.L2(improved_config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_2"
        ),
        keras.layers.Activation('relu', name="activation_2"),
        keras.layers.BatchNormalization(name="batch_norm_2") if improved_config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(improved_config.DROPOUT_RATES[1], name="dropout_2"),
        
        # Layer 3 (migliorato)
        keras.layers.Dense(
            improved_config.HIDDEN_LAYERS[2],  # 64 neuroni
            kernel_regularizer=keras.regularizers.L2(improved_config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_3"
        ),
        keras.layers.Activation('relu', name="activation_3"),
        keras.layers.BatchNormalization(name="batch_norm_3") if improved_config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(improved_config.DROPOUT_RATES[2], name="dropout_3"),
        
        # Layer 4 (nuovo, migliorato)
        keras.layers.Dense(
            improved_config.HIDDEN_LAYERS[3],  # 32 neuroni
            kernel_regularizer=keras.regularizers.L2(improved_config.L2_REG),
            kernel_initializer='he_normal',
            name="dense_4"
        ),
        keras.layers.Activation('relu', name="activation_4"),
        keras.layers.BatchNormalization(name="batch_norm_4") if improved_config.USE_BATCH_NORM else keras.layers.Identity(),
        keras.layers.Dropout(improved_config.DROPOUT_RATES[3], name="dropout_4"),
        
        # Output layer (identico)
        keras.layers.Dense(
            1, 
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name="output"
        )
    ], name="SmartGrid_Improved_Compatible")
    
    # OPTIMIZER
    optimizer = keras.optimizers.Adam(
        learning_rate=improved_config.LEARNING_RATE,
        beta_1=improved_config.BETA_1,
        beta_2=improved_config.BETA_2,
        clipnorm=improved_config.CLIPNORM
    )
    
    def weighted_binary_crossentropy(pos_weight=2.0):
        """Loss pesata per migliorare recall mantenendo precision"""
        def loss_fn(y_true, y_pred):
            # Converti a float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Clip per stabilità numerica
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            
            # Binary crossentropy pesata
            loss_pos = -y_true * tf.math.log(y_pred) * pos_weight
            loss_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
            
            return tf.reduce_mean(loss_pos + loss_neg)
        
        return loss_fn
    
    # Compilazione con loss migliorata
    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(pos_weight=2.5),  # Peso maggiore per attacchi
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
            keras.metrics.AUC(name="auc", curve='ROC'),
            keras.metrics.AUC(name="auc_pr", curve='PR')
        ]
    )
    
    print(f"🎯 MODELLO CREATO:")
    print(f"   📐 Architettura: {' → '.join(map(str, improved_config.HIDDEN_LAYERS))} → 1")
    print(f"   🎛️ Parametri totali: {model.count_params():,}")
    print(f"   🧠 Activation: {improved_config.ACTIVATION}")
    
    return model

def create_advanced_callbacks(config=None):
    """Callbacks avanzati per training ottimizzato"""
    
    callbacks = [
        # Early Stopping più paziente
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.0001
        ),

        # Learning Rate Scheduler
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Riduzione moderata
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),

        # Learning Rate Decay
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.0015 * (0.95 ** epoch),  # Decay graduale
            verbose=0
        )
    ]
    
    print(f"📊 CALLBACKS:")
    print(f"EarlyStopping: patience=12, min_delta=0.0001")
    print(f"ReduceLROnPlateau: factor=0.7, patience=5")
    print(f"LearningRateScheduler: exponential decay")

    return callbacks