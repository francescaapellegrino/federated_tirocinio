"""
Random Forest centralizzato per dataset Smart Grid
Francesca Pellegrino
"""


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

from preprocessing import load_improved_client_data

# CONFIGURAZIONE
N_TREES = 100         # Numero alberi Random Forest
CRITERION = "gini"    # Splitting rule: "gini" oppure "entropy"
RANDOM_STATE = 42     # Per riproducibilità
CLIENT_IDS = list(range(1, 16))  # Unisci tutti i client (1-15)
METRICS_FILENAME = "centralized_RF_summary_metrics.txt"

def load_centralized_data(client_ids):
    """
    Unisce i dati preprocessati di tutti i client in un unico dataset.
    Restituisce: X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    for cid in client_ids:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_improved_client_data(cid, None)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
    # Unisci tutti i dati
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    print(f"Centralized data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    Calcola le metriche principali per classificazione binaria.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    specificity = (cm[0,0] / (cm[0,0] + cm[0,1])) if cm.shape == (2,2) else 0.0
    loss = log_loss(y_true, y_pred_proba)
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc_roc = 0.5
    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "specificity": specificity,
        "confusion": cm
    }

def print_metrics_table(metrics, title="METRICHE"):
    """
    Stampa una tabella riassuntiva delle metriche principali.
    """
    print(f"\n{title}:")
    print("=" * 80)
    print(f"{'Set':<10} {'Loss':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC_ROC':<10} {'Specificity':<10}")
    print("-" * 80)
    for row in metrics:
        print(
            f"{row['set']:<10} "
            f"{row['loss']:<10.4f} "
            f"{row['accuracy']:<10.4f} "
            f"{row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} "
            f"{row['f1']:<10.4f} "
            f"{row['auc_roc']:<10.4f} "
            f"{row['specificity']:<10.4f}"
        )
    print("=" * 80)

def save_metrics_to_file(metrics, filename):
    """
    Salva le metriche finali in un file di testo tabellare.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("TABELLA METRICHE RANDOM FOREST CENTRALIZZATO:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Set':<10} {'Loss':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC_ROC':<10} {'Specificity':<10}\n")
        f.write("-" * 80 + "\n")
        for row in metrics:
            f.write(
                f"{row['set']:<10} "
                f"{row['loss']:<10.4f} "
                f"{row['accuracy']:<10.4f} "
                f"{row['precision']:<10.4f} "
                f"{row['recall']:<10.4f} "
                f"{row['f1']:<10.4f} "
                f"{row['auc_roc']:<10.4f} "
                f"{row['specificity']:<10.4f}\n"
            )
        f.write("=" * 80 + "\n\n")
        f.write("Confusion Matrix Validation:\n")
        f.write(str(metrics[0]['confusion']) + "\n")
        f.write("\nConfusion Matrix Test:\n")
        f.write(str(metrics[1]['confusion']) + "\n")
        f.write("=" * 80 + "\n")
        f.write("Addestramento centralizzato completato.\n")

def main():
    print("\n=== RANDOM FOREST CENTRALIZZATO ===")
    print("=" * 60)
    print(f"Numero alberi: {N_TREES}")
    print(f"Splitting rule: {CRITERION}")
    print(f"Client IDs: {CLIENT_IDS}")
    print("=" * 60)

    # Carica e unisci tutti i dati
    X_train, y_train, X_val, y_val, X_test, y_test = load_centralized_data(CLIENT_IDS)

    # Addestra Random Forest centralizzato
    model = RandomForestClassifier(
        n_estimators=N_TREES,
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    print("Addestramento Random Forest centralizzato...")
    model.fit(X_train, y_train)

    # Valutazione su validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_pred_proba)
    val_metrics['set'] = "Validation"

    # Valutazione su test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_pred_proba)
    test_metrics['set'] = "Test"

    # Stampa tabella riassuntiva delle metriche
    all_metrics = [val_metrics, test_metrics]
    print_metrics_table(all_metrics, title="TABELLONE METRICHE RANDOM FOREST CENTRALIZZATO")

    # Salva le metriche finali in file di testo
    save_metrics_to_file(all_metrics, METRICS_FILENAME)
    print(f"\nMetriche finali salvate in {METRICS_FILENAME}")

    print("=" * 60)
    print("Addestramento centralizzato completato.")

if __name__ == "__main__":
    main()