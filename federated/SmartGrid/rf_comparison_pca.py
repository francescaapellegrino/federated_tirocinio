import pandas as pd
import matplotlib.pyplot as plt

# ================================
# CARICAMENTO DATI METRICHE
# ================================
# Carica le metriche dal file SENZA PCA
df_nopca = pd.read_csv("metriche_rf_federato_nopre.txt", sep="\t")

# Carica le metriche dal file CON PCA
df_pca = pd.read_csv("metriche_rf_federato_pre.txt", sep="\t")

# ================================
# ALLINEA I ROUND
# ================================
n_rounds = min(len(df_nopca), len(df_pca))
rounds = df_nopca["round"][:n_rounds]

# ================================
# LISTA DELLE METRICHE DA CONFRONTARE
# ================================
metrics = ["accuracy", "precision", "recall", "f1", "specificity", "loss"]
titles = {
    "accuracy": "Accuratezza",
    "precision": "Precisione",
    "recall": "Recall",
    "f1": "F1 Score",
    "specificity": "Specificità",
    "loss": "Log Loss"
}

# ================================
# GRAFICO COMPARATIVO CON RANGE [0, 1] E SALVATAGGIO IMMAGINE
# ================================
plt.figure(figsize=(18, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i + 1)
    plt.plot(rounds, df_pca[metric][:n_rounds], label="CON PCA", color="blue")
    plt.plot(rounds, df_nopca[metric][:n_rounds], label="SENZA PCA", color="orange")
    plt.title(titles.get(metric, metric))
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # Fissa il range dell'asse y tra 0 e 1 per tutte le metriche

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle("Confronto Metriche Federated Random Forest: CON PCA vs SENZA PCA", fontsize=16)

# Salva il grafico come immagine PNG
plt.savefig("confronto_rf_pca_vs_nopca.png", dpi=150)
print("Grafico salvato come 'confronto_rf_pca_vs_nopca.png'.")

plt.show()