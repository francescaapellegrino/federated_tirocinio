import flwr as fl
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import KerasClassifier

def load_client_data(client_id):
    """Carica i dati per il client malevolo."""
    print(f"=== CARICAMENTO DATI CLIENT MALEVOLO {client_id} ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "..", "data", "SmartGrid", f"data{client_id}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
        
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"Dataset preparato per attacco MIA:")
    print(f"  - Training: {len(X_train)} campioni")
    print(f"  - Test: {len(X_test)} campioni")
    
    return X_train, y_train, X_test, y_test

def create_model(input_shape):
    """Crea il modello."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model

class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)
        self.model = create_model(self.X_train.shape[1])
        print("[MALICIOUS] Client inizializzato per attacco MIA")
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def perform_mia_attack(self):
        """Attacco MIA DEVASTANTE con gestione robusta dei NaN."""
        print("\n[MALICIOUS] ===== ATTACCO MIA DEVASTANTE OTTIMIZZATO =====")
        
        try:
            # === CONFIGURAZIONE ATTACCO ===
            n_samples = min(len(self.X_train), len(self.X_test))
            print(f"[MALICIOUS] Target: {n_samples * 2} campioni totali")
            
            X_train_attack = self.X_train[:n_samples]
            y_train_attack = self.y_train.iloc[:n_samples].to_numpy()
            X_test_attack = self.X_test[:n_samples]
            y_test_attack = self.y_test.iloc[:n_samples].to_numpy()
            
            # === FASE 1: ANALISI PROFONDA PREDIZIONI ===
            print("[MALICIOUS] FASE 1: Analisi profonda predizioni...")
            
            # Predizioni multiple con dropout (se presente)
            train_predictions = []
            test_predictions = []
            
            for run in range(50):  # 50 run per massima informazione
                train_pred = self.model(X_train_attack, training=True).numpy().flatten()
                test_pred = self.model(X_test_attack, training=True).numpy().flatten()
                train_predictions.append(train_pred)
                test_predictions.append(test_pred)
            
            train_predictions = np.array(train_predictions)
            test_predictions = np.array(test_predictions)
            
            # Statistiche ensemble
            train_mean = np.mean(train_predictions, axis=0)
            test_mean = np.mean(test_predictions, axis=0)
            train_var = np.var(train_predictions, axis=0)
            test_var = np.var(test_predictions, axis=0)
            train_std = np.std(train_predictions, axis=0)
            test_std = np.std(test_predictions, axis=0)
            train_min = np.min(train_predictions, axis=0)
            test_min = np.min(test_predictions, axis=0)
            train_max = np.max(train_predictions, axis=0)
            test_max = np.max(test_predictions, axis=0)
            
            # Predizioni deterministiche
            train_det = self.model.predict(X_train_attack, verbose=0).flatten()
            test_det = self.model.predict(X_test_attack, verbose=0).flatten()
            
            print(f"  - Varianza media train: {np.mean(train_var):.6f}")
            print(f"  - Varianza media test: {np.mean(test_var):.6f}")
            print(f"  - Ratio varianza: {np.mean(train_var) / (np.mean(test_var) + 1e-10):.4f}")
            
            # === FASE 2: ANALISI LOSS INDIVIDUALIZZATA ===
            print("[MALICIOUS] FASE 2: Analisi loss devastante...")
            
            # Loss per ogni run
            train_losses_all = []
            test_losses_all = []
            
            for run in range(train_predictions.shape[0]):
                train_loss_run = -(y_train_attack * np.log(np.clip(train_predictions[run], 1e-12, 1-1e-12)) + 
                                (1-y_train_attack) * np.log(np.clip(1-train_predictions[run], 1e-12, 1-1e-12)))
                test_loss_run = -(y_test_attack * np.log(np.clip(test_predictions[run], 1e-12, 1-1e-12)) + 
                                (1-y_test_attack) * np.log(np.clip(1-test_predictions[run], 1e-12, 1-1e-12)))
                train_losses_all.append(train_loss_run)
                test_losses_all.append(test_loss_run)
            
            train_losses_all = np.array(train_losses_all)
            test_losses_all = np.array(test_losses_all)
            
            # Statistiche loss
            train_loss_mean = np.mean(train_losses_all, axis=0)
            test_loss_mean = np.mean(test_losses_all, axis=0)
            train_loss_var = np.var(train_losses_all, axis=0)
            test_loss_var = np.var(test_losses_all, axis=0)
            train_loss_min = np.min(train_losses_all, axis=0)
            test_loss_min = np.min(test_losses_all, axis=0)
            
            print(f"  - Loss medio train: {np.mean(train_loss_mean):.6f}")
            print(f"  - Loss medio test: {np.mean(test_loss_mean):.6f}")
            print(f"  - Differenza loss: {np.mean(train_loss_mean) - np.mean(test_loss_mean):.6f}")
            
            # === FASE 3: ANALISI GRADIENTI MASSIVA ===
            print("[MALICIOUS] FASE 3: Analisi gradienti massiva...")
            
            # Gradienti per batch di campioni
            train_grad_norms = []
            test_grad_norms = []
            
            batch_size = 16
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                
                try:
                    # Converte in tensori con forma corretta
                    X_train_batch = tf.convert_to_tensor(X_train_attack[i:end_idx], dtype=tf.float32)
                    y_train_batch = tf.convert_to_tensor(y_train_attack[i:end_idx], dtype=tf.float32)
                    X_test_batch = tf.convert_to_tensor(X_test_attack[i:end_idx], dtype=tf.float32)
                    y_test_batch = tf.convert_to_tensor(y_test_attack[i:end_idx], dtype=tf.float32)
                    
                    # Gradiente train
                    with tf.GradientTape() as tape:
                        preds = self.model(X_train_batch, training=True)
                        preds_clipped = tf.clip_by_value(preds, 1e-7, 1-1e-7)
                        preds_flat = tf.reshape(preds_clipped, [-1])
                        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                            y_train_batch, preds_flat, from_logits=False))
                    
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    grad_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grads if g is not None)).numpy()
                    
                    # Gradiente test
                    with tf.GradientTape() as tape:
                        preds = self.model(X_test_batch, training=True)
                        preds_clipped = tf.clip_by_value(preds, 1e-7, 1-1e-7)
                        preds_flat = tf.reshape(preds_clipped, [-1])
                        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                            y_test_batch, preds_flat, from_logits=False))
                    
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    grad_norm_test = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grads if g is not None)).numpy()
                    
                except Exception as e:
                    print(f"    Errore calcolo gradienti batch {i}: {e}")
                    grad_norm = 1.0  # Valore fallback
                    grad_norm_test = 1.0
                
                # Gestione valori NaN/Inf
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    grad_norm = 1.0
                if np.isnan(grad_norm_test) or np.isinf(grad_norm_test):
                    grad_norm_test = 1.0
                
                # Estendi a tutti i campioni nel batch
                for _ in range(end_idx - i):
                    train_grad_norms.append(grad_norm)
                    test_grad_norms.append(grad_norm_test)
            
            train_grad_norms = np.array(train_grad_norms[:n_samples])
            test_grad_norms = np.array(test_grad_norms[:n_samples])
            
            print(f"  - Grad norm medio train: {np.mean(train_grad_norms):.6f}")
            print(f"  - Grad norm medio test: {np.mean(test_grad_norms):.6f}")
            
            # === FASE 4: FEATURE ENGINEERING DEVASTANTE ===
            print("[MALICIOUS] FASE 4: Feature engineering devastante...")
            
            # Feature base con clipping per evitare NaN
            train_det_clipped = np.clip(train_det, 1e-12, 1-1e-12)
            test_det_clipped = np.clip(test_det, 1e-12, 1-1e-12)
            
            train_conf = np.abs(train_det_clipped - 0.5)
            test_conf = np.abs(test_det_clipped - 0.5)
            
            # Feature entropiche
            train_entropy = -(train_det_clipped * np.log(train_det_clipped) + 
                            (1-train_det_clipped) * np.log(1-train_det_clipped))
            test_entropy = -(test_det_clipped * np.log(test_det_clipped) + 
                            (1-test_det_clipped) * np.log(1-test_det_clipped))
            
            # Logit space
            train_logits = np.log(train_det_clipped) - np.log(1-train_det_clipped)
            test_logits = np.log(test_det_clipped) - np.log(1-test_det_clipped)
            
            # Clip logits per evitare infiniti
            train_logits = np.clip(train_logits, -10, 10)
            test_logits = np.clip(test_logits, -10, 10)
            
            # Feature statistiche avanzate con gestione NaN
            train_skew = []
            test_skew = []
            train_kurt = []
            test_kurt = []
            
            from scipy.stats import skew, kurtosis
            
            for i in range(n_samples):
                try:
                    train_sk = skew(train_predictions[:, i])
                    test_sk = skew(test_predictions[:, i])
                    train_kr = kurtosis(train_predictions[:, i])
                    test_kr = kurtosis(test_predictions[:, i])
                    
                    # Gestione NaN
                    train_sk = 0.0 if np.isnan(train_sk) or np.isinf(train_sk) else train_sk
                    test_sk = 0.0 if np.isnan(test_sk) or np.isinf(test_sk) else test_sk
                    train_kr = 0.0 if np.isnan(train_kr) or np.isinf(train_kr) else train_kr
                    test_kr = 0.0 if np.isnan(test_kr) or np.isinf(test_kr) else test_kr
                    
                    train_skew.append(train_sk)
                    test_skew.append(test_sk)
                    train_kurt.append(train_kr)
                    test_kurt.append(test_kr)
                    
                except Exception:
                    train_skew.append(0.0)
                    test_skew.append(0.0)
                    train_kurt.append(0.0)
                    test_kurt.append(0.0)
            
            train_skew = np.array(train_skew)
            test_skew = np.array(test_skew)
            train_kurt = np.array(train_kurt)
            test_kurt = np.array(test_kurt)
            
            # Rankings e percentili
            train_loss_rank = np.argsort(np.argsort(train_loss_mean)) / n_samples
            test_loss_rank = np.argsort(np.argsort(test_loss_mean)) / n_samples
            train_conf_rank = np.argsort(np.argsort(train_conf)) / n_samples
            test_conf_rank = np.argsort(np.argsort(test_conf)) / n_samples
            
            # === FASE 5: ANALISI ROBUSTEZZA SEMPLIFICATA ===
            print("[MALICIOUS] FASE 5: Analisi robustezza...")
            
            # Analisi robustezza semplificata per evitare complessità eccessiva
            try:
                epsilon = 0.01
                train_noise = np.random.normal(0, epsilon, X_train_attack.shape)
                test_noise = np.random.normal(0, epsilon, X_test_attack.shape)
                
                X_train_pert = X_train_attack + train_noise
                X_test_pert = X_test_attack + test_noise
                
                train_pred_pert = self.model.predict(X_train_pert, verbose=0).flatten()
                test_pred_pert = self.model.predict(X_test_pert, verbose=0).flatten()
                
                train_robustness = np.abs(train_pred_pert - train_det)
                test_robustness = np.abs(test_pred_pert - test_det)
                
            except Exception as e:
                print(f"    Errore analisi robustezza: {e}")
                train_robustness = np.ones(n_samples) * 0.01
                test_robustness = np.ones(n_samples) * 0.01
            
            # MEGA FEATURE MATRIX con controllo NaN
            def safe_feature(arr, default_value=0.0):
                """Sostituisce NaN e Inf con valore di default."""
                arr = np.array(arr)
                arr[np.isnan(arr) | np.isinf(arr)] = default_value
                return arr
            
            train_features = np.column_stack([
                # Base predictions (4 features)
                safe_feature(train_det),
                safe_feature(train_mean),
                safe_feature(train_conf),
                safe_feature(train_entropy),
                
                # Prediction statistics (8 features)
                safe_feature(train_var),
                safe_feature(train_std),
                safe_feature(train_min),
                safe_feature(train_max),
                safe_feature(train_skew),
                safe_feature(train_kurt),
                
                # Loss features (4 features)
                safe_feature(train_loss_mean),
                safe_feature(train_loss_var),
                safe_feature(train_loss_min),
                safe_feature(train_logits),
                
                # Gradient features (1 feature)
                safe_feature(train_grad_norms),
                
                # Ranking features (2 features)
                safe_feature(train_loss_rank),
                safe_feature(train_conf_rank),
                
                # Robustness features (1 feature)
                safe_feature(train_robustness),
                
                # Polynomial features (6 features)
                safe_feature(train_det ** 2),
                safe_feature(train_det ** 3),
                safe_feature(train_conf ** 2),
                safe_feature(np.sqrt(train_conf + 1e-10)),
                safe_feature(train_mean ** 2),
                safe_feature(np.sqrt(train_var + 1e-10)),
                
                # Interaction features (6 features)
                safe_feature(train_conf * train_loss_mean),
                safe_feature(train_det * train_loss_mean),
                safe_feature(train_var * train_conf),
                safe_feature(train_entropy * train_conf),
                safe_feature(train_std * train_loss_var),
                safe_feature(train_skew * train_kurt),
                
                # Statistical indicators (4 features)
                np.where(train_loss_mean > np.percentile(train_loss_mean, 90), 1, 0),
                np.where(train_var > np.percentile(train_var, 75), 1, 0),
                np.where(train_conf > np.percentile(train_conf, 85), 1, 0),
                np.where(np.abs(train_skew) > np.percentile(np.abs(train_skew), 80), 1, 0),
            ])
            
            test_features = np.column_stack([
                safe_feature(test_det), safe_feature(test_mean), safe_feature(test_conf), safe_feature(test_entropy),
                safe_feature(test_var), safe_feature(test_std), safe_feature(test_min), safe_feature(test_max), 
                safe_feature(test_skew), safe_feature(test_kurt),
                safe_feature(test_loss_mean), safe_feature(test_loss_var), safe_feature(test_loss_min), safe_feature(test_logits),
                safe_feature(test_grad_norms),
                safe_feature(test_loss_rank), safe_feature(test_conf_rank),
                safe_feature(test_robustness),
                safe_feature(test_det ** 2), safe_feature(test_det ** 3), safe_feature(test_conf ** 2), 
                safe_feature(np.sqrt(test_conf + 1e-10)), safe_feature(test_mean ** 2), safe_feature(np.sqrt(test_var + 1e-10)),
                safe_feature(test_conf * test_loss_mean), safe_feature(test_det * test_loss_mean),
                safe_feature(test_var * test_conf), safe_feature(test_entropy * test_conf),
                safe_feature(test_std * test_loss_var), safe_feature(test_skew * test_kurt),
                np.where(test_loss_mean > np.percentile(test_loss_mean, 90), 1, 0),
                np.where(test_var > np.percentile(test_var, 75), 1, 0),
                np.where(test_conf > np.percentile(test_conf, 85), 1, 0),
                np.where(np.abs(test_skew) > np.percentile(np.abs(test_skew), 80), 1, 0),
            ])
            
            print(f"  - Feature totali: {train_features.shape[1]}")
            
            # Controllo finale per NaN
            print(f"  - NaN in train_features: {np.isnan(train_features).sum()}")
            print(f"  - NaN in test_features: {np.isnan(test_features).sum()}")
            
            # === FASE 6: SUPER ENSEMBLE ROBUSTO ===
            print("[MALICIOUS] FASE 6: Super ensemble robusto...")
            
            X_attack = np.concatenate([train_features, test_features])
            y_attack = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
            
            from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                                        VotingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
            from sklearn.linear_model import LogisticRegression, RidgeClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import roc_auc_score
            from sklearn.naive_bayes import GaussianNB
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            
            # Pipeline con imputer per gestire eventuali NaN residui
            def create_robust_pipeline(classifier):
                return Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', classifier)
                ])
            
            # Classificatori robusti ai NaN
            classifiers = {
                'RF_Deep': create_robust_pipeline(RandomForestClassifier(
                    n_estimators=500, max_depth=20, min_samples_split=5, random_state=42)),
                'ExtraTrees': create_robust_pipeline(ExtraTreesClassifier(
                    n_estimators=400, max_depth=15, min_samples_split=5, random_state=42)),
                'HistGBM': create_robust_pipeline(HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.1, max_depth=8, random_state=42)),
                'AdaBoost': create_robust_pipeline(AdaBoostClassifier(
                    n_estimators=100, learning_rate=0.8, random_state=42)),
                'LogReg': create_robust_pipeline(LogisticRegression(
                    random_state=42, max_iter=2000, C=1.0)),
                'Ridge': create_robust_pipeline(RidgeClassifier(random_state=42)),
                'SVM': create_robust_pipeline(SVC(
                    probability=True, C=10, kernel='rbf', random_state=42)),
                'MLP': create_robust_pipeline(MLPClassifier(
                    hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42, early_stopping=True)),
                'DecisionTree': create_robust_pipeline(DecisionTreeClassifier(
                    max_depth=15, min_samples_split=10, random_state=42)),
                'GaussianNB': create_robust_pipeline(GaussianNB())
            }
            
            # Test e selezione modelli
            best_models = []
            
            print("    Valutazione modelli:")
            for name, pipeline in classifiers.items():
                try:
                    scores = cross_val_score(pipeline, X_attack, y_attack, cv=3, scoring='roc_auc')
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    print(f"      {name}: {mean_score:.4f} ± {std_score:.4f}")
                    
                    if mean_score > 0.55:  # Soglia permissiva
                        pipeline.fit(X_attack, y_attack)
                        best_models.append((name, pipeline, mean_score))
                        
                except Exception as e:
                    print(f"      {name}: Error - {str(e)[:50]}...")
            
            # Sorting modelli per performance
            best_models.sort(key=lambda x: x[2], reverse=True)
            
            # Voting ensemble finale
            if len(best_models) >= 3:
                voting_estimators = [(name, pipeline) for name, pipeline, score in best_models[:5]]
                
                final_ensemble = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft'
                )
                
                final_ensemble.fit(X_attack, y_attack)
                final_predictions = final_ensemble.predict(X_attack)
                final_probabilities = final_ensemble.predict_proba(X_attack)[:, 1]
            else:
                # Fallback al miglior modello singolo
                if best_models:
                    _, best_pipeline, _ = best_models[0]
                    final_predictions = best_pipeline.predict(X_attack)
                    final_probabilities = best_pipeline.predict_proba(X_attack)[:, 1]
                else:
                    print("    Nessun modello efficace trovato!")
                    return 0.0
            
            # === RISULTATI DEVASTANTI ===
            print(f"\n[MALICIOUS] ===== RISULTATI ATTACCO DEVASTANTE =====")
            
            final_accuracy = np.mean(final_predictions == y_attack)
            auc_score = roc_auc_score(y_attack, final_probabilities)
            
            # Metriche dettagliate
            tp = np.sum((final_predictions == 1) & (y_attack == 1))
            fp = np.sum((final_predictions == 1) & (y_attack == 0))
            tn = np.sum((final_predictions == 0) & (y_attack == 0))
            fn = np.sum((final_predictions == 0) & (y_attack == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"[MALICIOUS] 💀 ATTACCO FINALE 💀")
            print(f"[MALICIOUS] Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")
            print(f"[MALICIOUS] AUC Score: {auc_score:.4f} ({auc_score*100:.1f}%)")
            print(f"[MALICIOUS] Precision: {precision:.4f}")
            print(f"[MALICIOUS] Recall: {recall:.4f}")
            print(f"[MALICIOUS] F1-Score: {f1:.4f}")
            print(f"[MALICIOUS]")
            print(f"[MALICIOUS] COMPROMISSIONE SISTEMA:")
            print(f"[MALICIOUS]   🎯 Membri identificati: {tp}/{n_samples} ({tp/n_samples*100:.1f}%)")
            print(f"[MALICIOUS]   ✅ Non-membri identificati: {tn}/{n_samples} ({tn/n_samples*100:.1f}%)")
            print(f"[MALICIOUS]   ❌ Errori totali: {fp+fn}/{n_samples*2} ({(fp+fn)/(n_samples*2)*100:.1f}%)")
            
            # Analisi separazione
            member_probs = final_probabilities[y_attack == 1]
            non_member_probs = final_probabilities[y_attack == 0]
            separation = np.mean(member_probs) - np.mean(non_member_probs)
            
            print(f"[MALICIOUS]")
            print(f"[MALICIOUS] ANALISI SEPARAZIONE:")
            print(f"[MALICIOUS]   Probabilità membri: {np.mean(member_probs):.4f} ± {np.std(member_probs):.4f}")
            print(f"[MALICIOUS]   Probabilità non-membri: {np.mean(non_member_probs):.4f} ± {np.std(non_member_probs):.4f}")
            print(f"[MALICIOUS]   Separazione: {separation:.4f}")
            
            # Interpretazione DEVASTANTE
            composite_score = (final_accuracy + auc_score) / 2
            
            if composite_score > 0.9:
                print(f"\n[MALICIOUS] 💀💀💀 SISTEMA COMPLETAMENTE DISTRUTTO! 💀💀💀")
                print(f"[MALICIOUS]   🔥 PRIVACY TOTALMENTE ANNIENTATA!")
                print(f"[MALICIOUS]   🔥 FEDERATED LEARNING COMPROMESSO!")
            elif composite_score > 0.8:
                print(f"\n[MALICIOUS] 🔴🔴🔴 SISTEMA GRAVEMENTE COMPROMESSO! 🔴🔴🔴")
                print(f"[MALICIOUS]   💥 PRIVACY SERIAMENTE VIOLATA!")
                print(f"[MALICIOUS]   💥 ATTACCO DEVASTANTE RIUSCITO!")
            elif composite_score > 0.7:
                print(f"\n[MALICIOUS] 🔴🔴 COMPROMISSIONE SIGNIFICATIVA! 🔴🔴")
                print(f"[MALICIOUS]   ⚡ ATTACCO MOLTO EFFICACE!")
            elif composite_score > 0.6:
                print(f"\n[MALICIOUS] 🟡 VULNERABILITÀ SOSTANZIALE")
                print(f"[MALICIOUS]   🎯 ATTACCO PARZIALMENTE RIUSCITO")
            elif composite_score > 0.55:
                print(f"\n[MALICIOUS] ⚠️ SEGNALI DI VULNERABILITÀ")
                print(f"[MALICIOUS]   📊 Possibile compromissione")
            else:
                print(f"\n[MALICIOUS] ✅ SISTEMA RESISTENTE")
                print(f"[MALICIOUS]   🛡️ Privacy preservata")
            
            print(f"[MALICIOUS]   📊 Score composito: {composite_score:.4f}")
            print(f"[MALICIOUS] ===============================================")
            
            return composite_score
            
        except Exception as e:
            print(f"[MALICIOUS] Errore durante l'attacco: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def fit(self, parameters, config):
        print(f"\n[MALICIOUS] Client {self.client_id} - Avvio training locale...")
    
        self.model.set_weights(parameters)
    
        history = self.model.fit(
        self.X_train, self.y_train,
        epochs=1,
        batch_size=32,
        verbose=0
        )
    
          # Esegui l'attacco MIA
        mia_accuracy = self.perform_mia_attack()
    
         # Assicurati che tutte le metriche siano valori scalari
        results = {
        'loss': float(history.history['loss'][-1]),
        'accuracy': float(history.history['accuracy'][-1]),
        'precision': float(history.history['precision'][-1]),
        'recall': float(history.history['recall'][-1]),
        'mia_accuracy': float(mia_accuracy)
        }
    
        print(f"[MALICIOUS] Training completato:")
        print(f"  - Loss: {results['loss']:.4f}")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
        print(f"  - MIA Accuracy: {results['mia_accuracy']:.4f}")
    
        return self.model.get_weights(), len(self.X_train), results
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        print(f"\n[MALICIOUS] Valutazione locale:")
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def main():
    if len(sys.argv) != 2:
        print("Uso: python client_mal.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 1 or client_id > 15:
            raise ValueError("Client ID deve essere tra 1 e 15")
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)
    
    print(f"\n=== AVVIO CLIENT MALEVOLO {client_id} ===")
    print("Questo client eseguirà attacchi Membership Inference")
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=MaliciousClient(client_id)
    )

if __name__ == "__main__":
    main()