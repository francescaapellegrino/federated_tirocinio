#!/usr/bin/env python3
"""
SmartGrid Client Distribution Analysis Script

Analizza la distribuzione dei dati tra i client per il federated learning SmartGrid.
Fornisce metriche quantitative, analisi del bilanciamento classi, similarità tra client,
identificazione outlier e visualizzazioni per la tesi.

Author: francescaapellegrino
Date: 2025-08-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurazione per grafici di alta qualità
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15
})

class SmartGridClientAnalyzer:
    """
    Analizzatore completo della distribuzione dei client SmartGrid per federated learning.
    """
    
    def __init__(self, client_ids=None, n_components=20, output_dir=None):
        """
        Inizializza l'analizzatore.
        
        Args:
            client_ids: Lista degli ID dei client da analizzare (default: 1-13 per training)
            n_components: Numero di componenti PCA (default: 20)
            output_dir: Directory per salvare report e grafici (default: current dir)
        """
        self.client_ids = client_ids if client_ids else list(range(1, 14))  # 1-13 per training
        self.n_components = n_components
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.client_data = {}
        self.client_stats = {}
        self.similarity_matrix = None
        self.clustering_results = {}
        
        # Crea directory di output se non esiste
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"=== SMARTGRID CLIENT DISTRIBUTION ANALYZER ===")
        print(f"Client da analizzare: {self.client_ids}")
        print(f"Componenti PCA: {self.n_components}")
        print(f"Directory output: {self.output_dir}")
        print("=" * 60)
    
    def load_client_data(self):
        """
        Carica i dati di tutti i client specificati.
        """
        print("\n📂 CARICAMENTO DATI CLIENT...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
        
        for client_id in self.client_ids:
            file_path = os.path.join(data_dir, f"data{client_id}.csv")
            
            try:
                df = pd.read_csv(file_path)
                
                # Separazione feature e target
                X = df.drop(columns=["marker"])
                y = (df["marker"] != "Natural").astype(int)  # 1 = attacco, 0 = naturale
                
                # Pulizia dati (gestione NaN e valori infiniti)
                nan_count = X.isnull().sum().sum()
                if nan_count > 0:
                    X.fillna(X.median(), inplace=True)
                
                # Gestione valori infiniti
                inf_count = np.isinf(X).sum().sum()
                if inf_count > 0:
                    print(f"    ⚠️  Trovati {inf_count} valori infiniti, sostituiti con valori estremi")
                    X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()], inplace=True)
                
                # Gestione valori troppo grandi (oltre float64)
                max_val = np.finfo(np.float64).max / 100  # Margine di sicurezza
                large_val_mask = np.abs(X) > max_val
                large_count = large_val_mask.sum().sum()
                if large_count > 0:
                    print(f"    ⚠️  Trovati {large_count} valori troppo grandi, limitati")
                    X = X.clip(-max_val, max_val)
                
                self.client_data[client_id] = {
                    'X': X,
                    'y': y,
                    'raw_samples': len(df),
                    'features': X.shape[1],
                    'nan_count': nan_count
                }
                
                print(f"  ✅ Client {client_id}: {len(df)} campioni, {X.shape[1]} feature")
                
            except FileNotFoundError:
                print(f"  ❌ File data{client_id}.csv non trovato")
                continue
        
        print(f"\n📊 Caricati {len(self.client_data)} client su {len(self.client_ids)} richiesti")
    
    def compute_client_statistics(self):
        """
        Calcola statistiche dettagliate per ogni client.
        """
        print("\n📊 CALCOLO STATISTICHE CLIENT...")
        
        for client_id, data in self.client_data.items():
            X, y = data['X'], data['y']
            
            # Statistiche di base
            total_samples = len(y)
            attack_samples = y.sum()
            normal_samples = (y == 0).sum()
            attack_ratio = y.mean()
            
            # Statistiche feature
            feature_means = X.mean()
            feature_stds = X.std()
            feature_medians = X.median()
            
            # Outlier detection (usando Z-score)
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            outlier_count = (z_scores > 3).sum().sum()
            outlier_ratio = outlier_count / (X.shape[0] * X.shape[1])
            
            self.client_stats[client_id] = {
                # Campioni
                'total_samples': total_samples,
                'attack_samples': attack_samples,
                'normal_samples': normal_samples,
                'attack_ratio': attack_ratio,
                'normal_ratio': 1 - attack_ratio,
                
                # Feature statistiche  
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'feature_medians': feature_medians,
                'feature_ranges': X.max() - X.min(),
                
                # Outlier
                'outlier_count': outlier_count,
                'outlier_ratio': outlier_ratio,
                
                # Diversità interna
                'feature_variance_mean': feature_stds.mean(),
                'feature_variance_std': feature_stds.std(),
                
                # Bilanciamento classi (quanto è sbilanciato)
                'class_imbalance_ratio': min(attack_ratio, 1-attack_ratio) / max(attack_ratio, 1-attack_ratio)
            }
            
            print(f"  Client {client_id}: {total_samples} campioni, {attack_ratio*100:.1f}% attacchi, {outlier_count} outlier")
    
    def compute_client_similarity(self):
        """
        Calcola la similarità tra client usando diverse metriche.
        """
        print("\n🔍 CALCOLO SIMILARITÀ TRA CLIENT...")
        
        # Prepara dati per confronto (medie delle feature per ogni client)
        client_profiles = []
        client_labels = []
        
        for client_id in sorted(self.client_data.keys()):
            profile = self.client_stats[client_id]['feature_means'].values
            client_profiles.append(profile)
            client_labels.append(f"Client {client_id}")
        
        client_profiles = np.array(client_profiles)
        
        # Controllo e pulizia valori problematici
        if np.any(np.isinf(client_profiles)) or np.any(np.isnan(client_profiles)):
            print("  ⚠️  Valori infiniti/NaN nei profili client, pulizia in corso...")
            client_profiles = np.nan_to_num(client_profiles, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalizza i profili per il confronto
        scaler = StandardScaler()
        client_profiles_normalized = scaler.fit_transform(client_profiles)
        
        # Calcola matrice di similarità coseno
        self.similarity_matrix = cosine_similarity(client_profiles_normalized)
        
        # Converti in DataFrame per facilità d'uso
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=client_labels,
            columns=client_labels
        )
        
        print(f"  ✅ Matrice similarità {len(client_labels)}x{len(client_labels)} calcolata")
        
        # Trova client più simili e più diversi
        # Esclude la diagonale (similarità con se stesso = 1.0)
        similarity_no_diag = self.similarity_matrix.copy()
        np.fill_diagonal(similarity_no_diag, np.nan)  # Usa NaN invece di -1 per evitare confusione
        
        # Crea una maschera per i valori validi (non NaN)
        valid_mask = ~np.isnan(similarity_no_diag)
        valid_similarities = similarity_no_diag[valid_mask]
        
        if len(valid_similarities) > 0:
            # Client più simili
            max_sim_idx = np.nanargmax(similarity_no_diag)
            max_sim_idx = np.unravel_index(max_sim_idx, similarity_no_diag.shape)
            max_similarity = similarity_no_diag[max_sim_idx]
            most_similar_pair = (client_labels[max_sim_idx[0]], client_labels[max_sim_idx[1]])
            
            # Client più diversi
            min_sim_idx = np.nanargmin(similarity_no_diag)
            min_sim_idx = np.unravel_index(min_sim_idx, similarity_no_diag.shape)
            min_similarity = similarity_no_diag[min_sim_idx]
            most_different_pair = (client_labels[min_sim_idx[0]], client_labels[min_sim_idx[1]])
            
            mean_similarity = np.nanmean(valid_similarities)
            std_similarity = np.nanstd(valid_similarities)
        else:
            # Fallback se non ci sono valori validi
            most_similar_pair = ("N/A", "N/A")
            max_similarity = 0.0
            most_different_pair = ("N/A", "N/A")
            min_similarity = 0.0
            mean_similarity = 0.0
            std_similarity = 0.0
        
        print(f"  📈 Client più simili: {most_similar_pair[0]} - {most_similar_pair[1]} (sim: {max_similarity:.3f})")
        print(f"  📉 Client più diversi: {most_different_pair[0]} - {most_different_pair[1]} (sim: {min_similarity:.3f})")
        
        return {
            'most_similar_pair': most_similar_pair,
            'max_similarity': max_similarity,
            'most_different_pair': most_different_pair,
            'min_similarity': min_similarity,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity
        }
    
    def perform_client_clustering(self, n_clusters=3):
        """
        Raggruppa i client in cluster basati sulla similarità.
        """
        client_ids_ordered = sorted(self.client_data.keys())
        num_clients = len(client_ids_ordered)
        
        # Adatta numero di cluster al numero di client disponibili
        max_clusters = min(n_clusters, num_clients - 1) if num_clients > 1 else 1
        if max_clusters < 2:
            print(f"\n🎯 CLUSTERING CLIENT: troppi pochi client ({num_clients}) per clustering, saltato")
            self.clustering_results = {
                'n_clusters': 1,
                'cluster_labels': [0] * num_clients,
                'clusters': {0: client_ids_ordered},
                'silhouette_score': 0.0,
                'centroids': None
            }
            return self.clustering_results
        
        print(f"\n🎯 CLUSTERING CLIENT (k={max_clusters})...")
        
        # Usa le feature medie normalizzate
        client_profiles = []
        
        for client_id in client_ids_ordered:
            profile = self.client_stats[client_id]['feature_means'].values
            client_profiles.append(profile)
        
        client_profiles = np.array(client_profiles)
        
        # Controllo e pulizia valori problematici
        if np.any(np.isinf(client_profiles)) or np.any(np.isnan(client_profiles)):
            print("  ⚠️  Valori infiniti/NaN nei profili client, pulizia in corso...")
            client_profiles = np.nan_to_num(client_profiles, nan=0.0, posinf=1e10, neginf=-1e10)
        
        scaler = StandardScaler()
        client_profiles_normalized = scaler.fit_transform(client_profiles)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(client_profiles_normalized)
        
        # Calcola silhouette score solo se abbiamo abbastanza campioni e cluster diversi
        if num_clients >= 2 and len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(client_profiles_normalized, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        # Organizza risultati per cluster
        clusters = {}
        for i, client_id in enumerate(client_ids_ordered):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
        
        self.clustering_results = {
            'n_clusters': max_clusters,
            'cluster_labels': cluster_labels,
            'clusters': clusters,
            'silhouette_score': silhouette_avg,
            'centroids': kmeans.cluster_centers_
        }
        
        print(f"  ✅ Clustering completato, Silhouette Score: {silhouette_avg:.3f}")
        for cluster_id, members in clusters.items():
            print(f"    Cluster {cluster_id}: {members}")
        
        return self.clustering_results
    
    def analyze_non_iid_metrics(self):
        """
        Calcola metriche di eterogeneità (Non-IID) dei dati.
        """
        print("\n🔬 ANALISI ETEROGENEITÀ (NON-IID)...")
        
        # Raccoglie statistiche per analisi Non-IID
        attack_ratios = []
        sample_counts = []
        feature_variances = []
        
        for client_id in sorted(self.client_data.keys()):
            stats = self.client_stats[client_id]
            attack_ratios.append(stats['attack_ratio'])
            sample_counts.append(stats['total_samples'])
            feature_variances.append(stats['feature_variance_mean'])
        
        # Metriche di eterogeneità
        metrics = {
            # Variabilità distribuzione classi
            'class_distribution_variance': np.var(attack_ratios),
            'class_distribution_std': np.std(attack_ratios),
            'class_distribution_cv': np.std(attack_ratios) / np.mean(attack_ratios) if np.mean(attack_ratios) > 0 else 0,
            
            # Variabilità numero campioni
            'sample_count_variance': np.var(sample_counts),
            'sample_count_std': np.std(sample_counts),
            'sample_count_cv': np.std(sample_counts) / np.mean(sample_counts),
            
            # Variabilità feature
            'feature_variance_mean': np.mean(feature_variances),
            'feature_variance_std': np.std(feature_variances),
            
            # Indice di eterogeneità complessivo (combinazione pesata)
            'heterogeneity_index': None  # Calcolato sotto
        }
        
        # Calcola indice di eterogeneità complessivo (0 = omogeneo, 1 = molto eterogeneo)
        # Normalizza ogni componente tra 0 e 1
        class_het = min(metrics['class_distribution_cv'] / 0.5, 1.0)  # CV > 0.5 considera molto eterogeneo
        sample_het = min(metrics['sample_count_cv'] / 0.2, 1.0)       # CV > 0.2 considera eterogeneo
        
        # Media pesata (classe più importante per federated learning)
        metrics['heterogeneity_index'] = 0.7 * class_het + 0.3 * sample_het
        
        print(f"  📊 Varianza distribuzione classi: {metrics['class_distribution_variance']:.4f}")
        print(f"  📊 CV distribuzione classi: {metrics['class_distribution_cv']:.4f}")
        print(f"  📊 CV numero campioni: {metrics['sample_count_cv']:.4f}")
        print(f"  🎯 Indice eterogeneità: {metrics['heterogeneity_index']:.3f} (0=omogeneo, 1=eterogeneo)")
        
        return metrics
    
    def identify_outlier_clients(self):
        """
        Identifica client outlier usando metodi statistici.
        """
        print("\n🚨 IDENTIFICAZIONE CLIENT OUTLIER...")
        
        # Raccoglie metriche per ogni client
        metrics_matrix = []
        client_ids_ordered = sorted(self.client_data.keys())
        
        for client_id in client_ids_ordered:
            stats = self.client_stats[client_id]
            metrics = [
                stats['attack_ratio'],
                stats['total_samples'],
                stats['feature_variance_mean'],
                stats['outlier_ratio'],
                stats['class_imbalance_ratio']
            ]
            metrics_matrix.append(metrics)
        
        metrics_matrix = np.array(metrics_matrix)
        
        # Controllo e pulizia valori problematici
        if np.any(np.isinf(metrics_matrix)) or np.any(np.isnan(metrics_matrix)):
            print("  ⚠️  Valori infiniti/NaN nelle metriche, pulizia in corso...")
            metrics_matrix = np.nan_to_num(metrics_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Standardizza le metriche
        scaler = StandardScaler()
        metrics_normalized = scaler.fit_transform(metrics_matrix)
        
        # Calcola Z-score per ogni client (distanza dalla media)
        client_z_scores = np.linalg.norm(metrics_normalized, axis=1)
        
        # Identifica outlier (Z-score > 2 deviazioni standard)
        z_threshold = 2.0
        outlier_mask = client_z_scores > z_threshold
        
        outlier_clients = []
        normal_clients = []
        
        for i, client_id in enumerate(client_ids_ordered):
            if outlier_mask[i]:
                outlier_clients.append({
                    'client_id': client_id,
                    'z_score': client_z_scores[i],
                    'reasons': self._analyze_outlier_reasons(client_id, metrics_normalized[i])
                })
            else:
                normal_clients.append(client_id)
        
        print(f"  ✅ Analisi completata:")
        print(f"    📍 Client outlier: {len(outlier_clients)}")
        print(f"    📍 Client normali: {len(normal_clients)}")
        
        for outlier in outlier_clients:
            print(f"    🚨 Client {outlier['client_id']}: Z-score={outlier['z_score']:.2f}, Motivi: {', '.join(outlier['reasons'])}")
        
        return {
            'outlier_clients': outlier_clients,
            'normal_clients': normal_clients,
            'z_scores': dict(zip(client_ids_ordered, client_z_scores)),
            'threshold': z_threshold
        }
    
    def _analyze_outlier_reasons(self, client_id, normalized_metrics):
        """
        Analizza i motivi per cui un client è considerato outlier.
        """
        reasons = []
        
        # Indici delle metriche: [attack_ratio, total_samples, feature_variance_mean, outlier_ratio, class_imbalance_ratio]
        thresholds = [1.5, 1.5, 1.5, 1.5, 1.5]  # Soglie per considerare un valore estremo
        reason_names = [
            'Attack ratio estremo',
            'Numero campioni anomalo', 
            'Varianza feature anomala',
            'Troppi outlier interni',
            'Sbilanciamento classi estremo'
        ]
        
        for i, (value, threshold, name) in enumerate(zip(normalized_metrics, thresholds, reason_names)):
            if abs(value) > threshold:
                reasons.append(name)
        
        return reasons if reasons else ['Combinazione di fattori']
    
    def generate_visualizations(self):
        """
        Genera tutte le visualizzazioni per l'analisi.
        """
        print("\n📊 GENERAZIONE VISUALIZZAZIONI...")
        
        # Configurazione per subplot
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Distribuzione campioni per client
        plt.subplot(4, 3, 1)
        self._plot_sample_distribution()
        
        # 2. Distribuzione classi per client  
        plt.subplot(4, 3, 2)
        self._plot_class_distribution()
        
        # 3. Heatmap similarità client
        plt.subplot(4, 3, 3)
        self._plot_similarity_heatmap()
        
        # 4. Box plot statistiche campioni
        plt.subplot(4, 3, 4)
        self._plot_sample_statistics_boxplot()
        
        # 5. Box plot ratio attacchi
        plt.subplot(4, 3, 5)
        self._plot_attack_ratio_boxplot()
        
        # 6. Clustering client
        plt.subplot(4, 3, 6)
        self._plot_client_clustering()
        
        # 7. Varianza feature per client
        plt.subplot(4, 3, 7)
        self._plot_feature_variance()
        
        # 8. Outlier detection
        plt.subplot(4, 3, 8)
        self._plot_outlier_detection()
        
        # 9. Matrice correlazione metriche
        plt.subplot(4, 3, 9)
        self._plot_metrics_correlation()
        
        # 10. Distribuzione eterogeneità 
        plt.subplot(4, 3, 10)
        self._plot_heterogeneity_distribution()
        
        # 11. Impatto su FedAvg (simulazione pesi)
        plt.subplot(4, 3, 11)
        self._plot_fedavg_impact()
        
        # 12. Riepilogo metriche chiave
        plt.subplot(4, 3, 12)
        self._plot_key_metrics_summary()
        
        plt.tight_layout()
        
        # Salva figura principale
        output_path = os.path.join(self.output_dir, 'smartgrid_client_distribution_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Visualizzazioni salvate: {output_path}")
        
        plt.show()
        
        # Genera anche visualizzazioni separate per dettagli
        self._generate_detailed_plots()
    
    def _plot_sample_distribution(self):
        """Grafico distribuzione campioni per client."""
        client_ids = sorted(self.client_data.keys())
        sample_counts = [self.client_stats[cid]['total_samples'] for cid in client_ids]
        
        bars = plt.bar(client_ids, sample_counts, color='skyblue', alpha=0.7)
        plt.xlabel('Client ID')
        plt.ylabel('Numero Campioni')
        plt.title('Distribuzione Campioni per Client')
        plt.grid(True, alpha=0.3)
        
        # Aggiungi valori sopra le barre
        for bar, count in zip(bars, sample_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=9)
    
    def _plot_class_distribution(self):
        """Grafico distribuzione classi per client."""
        client_ids = sorted(self.client_data.keys())
        attack_ratios = [self.client_stats[cid]['attack_ratio'] * 100 for cid in client_ids]
        normal_ratios = [100 - ratio for ratio in attack_ratios]
        
        x = np.arange(len(client_ids))
        width = 0.8
        
        plt.bar(x, attack_ratios, width, label='Attacchi (%)', color='red', alpha=0.7)
        plt.bar(x, normal_ratios, width, bottom=attack_ratios, label='Normali (%)', color='green', alpha=0.7)
        
        plt.xlabel('Client ID')
        plt.ylabel('Percentuale')
        plt.title('Distribuzione Classi per Client')
        plt.xticks(x, client_ids)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_similarity_heatmap(self):
        """Heatmap similarità tra client."""
        sns.heatmap(self.similarity_df, annot=True, cmap='coolwarm', center=0.5,
                   square=True, cbar_kws={'label': 'Similarità Coseno'})
        plt.title('Similarità tra Client')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    
    def _plot_sample_statistics_boxplot(self):
        """Box plot statistiche campioni."""
        client_ids = sorted(self.client_data.keys())
        sample_counts = [self.client_stats[cid]['total_samples'] for cid in client_ids]
        
        plt.boxplot(sample_counts)
        plt.ylabel('Numero Campioni')
        plt.title('Statistiche Distribuzione Campioni')
        plt.grid(True, alpha=0.3)
        
        # Aggiungi statistiche
        mean_samples = np.mean(sample_counts)
        std_samples = np.std(sample_counts)
        plt.text(0.7, max(sample_counts), f'μ={mean_samples:.0f}\nσ={std_samples:.0f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def _plot_attack_ratio_boxplot(self):
        """Box plot ratio attacchi."""
        client_ids = sorted(self.client_data.keys())
        attack_ratios = [self.client_stats[cid]['attack_ratio'] * 100 for cid in client_ids]
        
        plt.boxplot(attack_ratios)
        plt.ylabel('Percentuale Attacchi (%)')
        plt.title('Distribuzione Ratio Attacchi')
        plt.grid(True, alpha=0.3)
        
        # Aggiungi statistiche
        mean_ratio = np.mean(attack_ratios)
        std_ratio = np.std(attack_ratios)
        plt.text(0.7, max(attack_ratios), f'μ={mean_ratio:.1f}%\nσ={std_ratio:.1f}%', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    def _plot_client_clustering(self):
        """Visualizzazione clustering client."""
        if not hasattr(self, 'clustering_results') or not self.clustering_results:
            plt.text(0.5, 0.5, 'Clustering non eseguito', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Clustering Client')
            return
        
        # Usa prime due componenti principali per visualizzazione 2D
        client_profiles = []
        client_ids_ordered = sorted(self.client_data.keys())
        
        for client_id in client_ids_ordered:
            profile = self.client_stats[client_id]['feature_means'].values
            client_profiles.append(profile)
        
        client_profiles = np.array(client_profiles)
        
        # Controllo e pulizia valori problematici
        if np.any(np.isinf(client_profiles)) or np.any(np.isnan(client_profiles)):
            client_profiles = np.nan_to_num(client_profiles, nan=0.0, posinf=1e10, neginf=-1e10)
        
        scaler = StandardScaler()
        client_profiles_normalized = scaler.fit_transform(client_profiles)
        
        # PCA per visualizzazione 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(client_profiles_normalized)
        
        # Plot con colori per cluster
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        cluster_labels = self.clustering_results['cluster_labels']
        
        for cluster_id in range(self.clustering_results['n_clusters']):
            mask = cluster_labels == cluster_id
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=colors[cluster_id % len(colors)], 
                       label=f'Cluster {cluster_id}', alpha=0.7, s=100)
        
        # Annotazioni client
        for i, client_id in enumerate(client_ids_ordered):
            plt.annotate(f'C{client_id}', (coords_2d[i, 0], coords_2d[i, 1]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'Clustering Client (Silhouette: {self.clustering_results["silhouette_score"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_feature_variance(self):
        """Grafico varianza feature per client."""
        client_ids = sorted(self.client_data.keys())
        variances = [self.client_stats[cid]['feature_variance_mean'] for cid in client_ids]
        
        plt.bar(client_ids, variances, color='orange', alpha=0.7)
        plt.xlabel('Client ID')
        plt.ylabel('Varianza Media Feature')
        plt.title('Varianza Feature per Client')
        plt.grid(True, alpha=0.3)
        plt.xticks(client_ids)
    
    def _plot_outlier_detection(self):
        """Grafico detection outlier."""
        if not hasattr(self, 'outlier_results'):
            self.outlier_results = self.identify_outlier_clients()
        
        client_ids = sorted(self.client_data.keys())
        z_scores = [self.outlier_results['z_scores'][cid] for cid in client_ids]
        
        colors = ['red' if cid in [o['client_id'] for o in self.outlier_results['outlier_clients']] 
                 else 'blue' for cid in client_ids]
        
        bars = plt.bar(client_ids, z_scores, color=colors, alpha=0.7)
        plt.axhline(y=self.outlier_results['threshold'], color='red', linestyle='--', 
                   label=f'Soglia outlier ({self.outlier_results["threshold"]})')
        plt.xlabel('Client ID')
        plt.ylabel('Z-Score')
        plt.title('Outlier Detection per Client')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(client_ids)
    
    def _plot_metrics_correlation(self):
        """Matrice correlazione tra metriche."""
        # Raccoglie tutte le metriche numeriche
        metrics_data = {}
        for client_id in sorted(self.client_data.keys()):
            stats = self.client_stats[client_id]
            metrics_data[client_id] = {
                'Campioni': stats['total_samples'],
                'Ratio_Attacchi': stats['attack_ratio'],
                'Var_Feature': stats['feature_variance_mean'],
                'Outlier_Ratio': stats['outlier_ratio'],
                'Imbalance': stats['class_imbalance_ratio']
            }
        
        df_metrics = pd.DataFrame(metrics_data).T
        corr_matrix = df_metrics.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Correlazione'})
        plt.title('Correlazione tra Metriche Client')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    
    def _plot_heterogeneity_distribution(self):
        """Distribuzione eterogeneità."""
        if not hasattr(self, 'noniid_metrics'):
            self.noniid_metrics = self.analyze_non_iid_metrics()
        
        # Crea grafico a radar per le metriche di eterogeneità
        metrics = ['CV Classi', 'CV Campioni', 'Var Feature', 'Outlier Ratio']
        values = [
            self.noniid_metrics['class_distribution_cv'],
            self.noniid_metrics['sample_count_cv'],
            self.noniid_metrics['feature_variance_std'],
            np.mean([self.client_stats[cid]['outlier_ratio'] for cid in self.client_data.keys()])
        ]
        
        # Normalizza valori per visualizzazione
        values_norm = [min(v/0.5, 1.0) for v in values]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values_norm += values_norm[:1]  # Chiudi il poligono
        angles += angles[:1]
        
        plt.subplot(polar=True)
        plt.plot(angles, values_norm, 'o-', linewidth=2, color='red', alpha=0.7)
        plt.fill(angles, values_norm, alpha=0.25, color='red')
        plt.xticks(angles[:-1], metrics)
        plt.ylim(0, 1)
        plt.title(f'Eterogeneità Complessiva\n(Indice: {self.noniid_metrics["heterogeneity_index"]:.3f})')
        
        # Ripristina subplot normale
        plt.subplot(4, 3, 10, polar=False)
        
        # Grafico a barre semplice come fallback
        plt.bar(range(len(metrics)), values_norm[:-1], color='red', alpha=0.7)
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.ylabel('Eterogeneità Normalizzata')
        plt.title(f'Metriche Eterogeneità\n(Indice: {self.noniid_metrics["heterogeneity_index"]:.3f})')
        plt.grid(True, alpha=0.3)
    
    def _plot_fedavg_impact(self):
        """Impatto simulato su FedAvg."""
        client_ids = sorted(self.client_data.keys())
        
        # Calcola pesi FedAvg standard (proporzionale al numero di campioni)
        sample_counts = [self.client_stats[cid]['total_samples'] for cid in client_ids]
        total_samples = sum(sample_counts)
        standard_weights = [count/total_samples for count in sample_counts]
        
        # Simula impatto delle distribuzioni sbilanciate
        attack_ratios = [self.client_stats[cid]['attack_ratio'] for cid in client_ids]
        
        # Weighted accuracy simulata (client con più attacchi hanno impatto diverso)
        simulated_accuracy = []
        for weight, ratio in zip(standard_weights, attack_ratios):
            # Simula che client molto sbilanciati abbiano accuracy locale più bassa
            local_acc = 0.85 - 0.1 * abs(ratio - 0.5)  # Penalty per sbilanciamento
            simulated_accuracy.append(weight * local_acc)
        
        x = np.arange(len(client_ids))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, standard_weights, width, label='Peso FedAvg', alpha=0.7)
        bars2 = plt.bar(x + width/2, simulated_accuracy, width, label='Accuracy pesata', alpha=0.7)
        
        plt.xlabel('Client ID')
        plt.ylabel('Valore')
        plt.title('Impatto su FedAvg')
        plt.xticks(x, client_ids)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_key_metrics_summary(self):
        """Riepilogo metriche chiave."""
        # Statistiche globali
        total_clients = len(self.client_data)
        total_samples = sum(self.client_stats[cid]['total_samples'] for cid in self.client_data.keys())
        mean_attack_ratio = np.mean([self.client_stats[cid]['attack_ratio'] for cid in self.client_data.keys()])
        
        if not hasattr(self, 'noniid_metrics'):
            self.noniid_metrics = self.analyze_non_iid_metrics()
        
        if not hasattr(self, 'outlier_results'):
            self.outlier_results = self.identify_outlier_clients()
        
        # Testo riassuntivo
        summary_text = f"""
RIEPILOGO ANALISI SMARTGRID
{'='*35}

📊 DATI GENERALI:
   • Client analizzati: {total_clients}
   • Campioni totali: {total_samples:,}
   • Ratio attacchi medio: {mean_attack_ratio*100:.1f}%

🔍 ETEROGENEITÀ:
   • Indice Non-IID: {self.noniid_metrics['heterogeneity_index']:.3f}
   • CV distribuzione classi: {self.noniid_metrics['class_distribution_cv']:.3f}
   • CV numero campioni: {self.noniid_metrics['sample_count_cv']:.3f}

🚨 OUTLIER:
   • Client outlier: {len(self.outlier_results['outlier_clients'])}
   • Client normali: {len(self.outlier_results['normal_clients'])}

🎯 RACCOMANDAZIONI:
   • Strategia bilanciata necessaria: {'Sì' if self.noniid_metrics['heterogeneity_index'] > 0.3 else 'No'}
   • Client selection adattiva: {'Sì' if len(self.outlier_results['outlier_clients']) > 0 else 'No'}
   • Weighted aggregation: {'Consigliata' if self.noniid_metrics['class_distribution_cv'] > 0.1 else 'Opzionale'}
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        plt.axis('off')
        plt.title('Riepilogo Analisi')
    
    def _generate_detailed_plots(self):
        """Genera grafici dettagliati separati."""
        # TODO: Implementa grafici dettagliati aggiuntivi se necessario
        pass
    
    def generate_comprehensive_report(self):
        """
        Genera un report completo dell'analisi.
        """
        print("\n📝 GENERAZIONE REPORT COMPLETO...")
        
        # Assicurati che tutte le analisi siano state eseguite
        if not hasattr(self, 'similarity_matrix'):
            self.compute_client_similarity()
        if not hasattr(self, 'clustering_results'):
            self.perform_client_clustering()
        if not hasattr(self, 'noniid_metrics'):
            self.noniid_metrics = self.analyze_non_iid_metrics()
        if not hasattr(self, 'outlier_results'):
            self.outlier_results = self.identify_outlier_clients()
        
        # Genera report testuale
        report_path = os.path.join(self.output_dir, 'smartgrid_distribution_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SMARTGRID FEDERATED LEARNING - ANALISI DISTRIBUZIONE CLIENT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data analisi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Client analizzati: {self.client_ids}\n")
            f.write(f"Componenti PCA: {self.n_components}\n\n")
            
            # SEZIONE 1: Statistiche generali
            f.write("1. STATISTICHE GENERALI\n")
            f.write("-" * 30 + "\n")
            total_samples = sum(self.client_stats[cid]['total_samples'] for cid in self.client_data.keys())
            f.write(f"Numero client: {len(self.client_data)}\n")
            f.write(f"Campioni totali: {total_samples:,}\n")
            f.write(f"Campioni per client: {total_samples // len(self.client_data):,} (media)\n\n")
            
            # Dettagli per client
            f.write("Dettagli per client:\n")
            for client_id in sorted(self.client_data.keys()):
                stats = self.client_stats[client_id]
                f.write(f"  Client {client_id:2d}: {stats['total_samples']:5d} campioni, "
                       f"{stats['attack_ratio']*100:5.1f}% attacchi, "
                       f"{stats['outlier_count']:4d} outlier\n")
            f.write("\n")
            
            # SEZIONE 2: Analisi eterogeneità
            f.write("2. ANALISI ETEROGENEITÀ (NON-IID)\n")
            f.write("-" * 35 + "\n")
            f.write(f"Indice eterogeneità complessivo: {self.noniid_metrics['heterogeneity_index']:.3f}\n")
            f.write(f"  (0.0 = omogeneo, 1.0 = molto eterogeneo)\n\n")
            
            f.write("Metriche dettagliate:\n")
            f.write(f"  • CV distribuzione classi: {self.noniid_metrics['class_distribution_cv']:.4f}\n")
            f.write(f"  • CV numero campioni: {self.noniid_metrics['sample_count_cv']:.4f}\n")
            f.write(f"  • Varianza distribuzione classi: {self.noniid_metrics['class_distribution_variance']:.4f}\n")
            f.write(f"  • Std distribuzione classi: {self.noniid_metrics['class_distribution_std']:.4f}\n\n")
            
            # SEZIONE 3: Clustering e similarità
            f.write("3. CLUSTERING E SIMILARITÀ\n")
            f.write("-" * 30 + "\n")
            f.write(f"Numero cluster ottimale: {self.clustering_results['n_clusters']}\n")
            f.write(f"Silhouette score: {self.clustering_results['silhouette_score']:.3f}\n\n")
            
            f.write("Composizione cluster:\n")
            for cluster_id, members in self.clustering_results['clusters'].items():
                f.write(f"  Cluster {cluster_id}: {members}\n")
            f.write("\n")
            
            # Similarità
            similarity_results = self.compute_client_similarity()
            f.write("Similarità tra client:\n")
            f.write(f"  • Più simili: {similarity_results['most_similar_pair'][0]} - {similarity_results['most_similar_pair'][1]} "
                   f"(similarità: {similarity_results['max_similarity']:.3f})\n")
            f.write(f"  • Più diversi: {similarity_results['most_different_pair'][0]} - {similarity_results['most_different_pair'][1]} "
                   f"(similarità: {similarity_results['min_similarity']:.3f})\n")
            f.write(f"  • Similarità media: {similarity_results['mean_similarity']:.3f}\n")
            f.write(f"  • Deviazione standard: {similarity_results['std_similarity']:.3f}\n\n")
            
            # SEZIONE 4: Client outlier
            f.write("4. CLIENT OUTLIER\n")
            f.write("-" * 20 + "\n")
            f.write(f"Client outlier identificati: {len(self.outlier_results['outlier_clients'])}\n")
            f.write(f"Soglia Z-score: {self.outlier_results['threshold']}\n\n")
            
            if self.outlier_results['outlier_clients']:
                f.write("Dettagli outlier:\n")
                for outlier in self.outlier_results['outlier_clients']:
                    f.write(f"  Client {outlier['client_id']}: Z-score = {outlier['z_score']:.3f}\n")
                    f.write(f"    Motivi: {', '.join(outlier['reasons'])}\n")
            else:
                f.write("Nessun client outlier identificato.\n")
            f.write("\n")
            
            # SEZIONE 5: Raccomandazioni per federated learning
            f.write("5. RACCOMANDAZIONI PER FEDERATED LEARNING\n")
            f.write("-" * 45 + "\n")
            
            het_index = self.noniid_metrics['heterogeneity_index']
            cv_classes = self.noniid_metrics['class_distribution_cv']
            n_outliers = len(self.outlier_results['outlier_clients'])
            
            f.write("Analisi raccomandazioni:\n\n")
            
            # Strategia di aggregazione
            if het_index > 0.4:
                f.write("📊 STRATEGIA AGGREGAZIONE: BILANCIA PESATA NECESSARIA\n")
                f.write("   • L'alto indice di eterogeneità richiede una strategia di aggregazione bilanciata\n")
                f.write("   • Implementare weighted FedAvg basato su distribuzione classi\n")
                f.write("   • Ridurre peso di client con distribuzioni estreme\n\n")
            elif het_index > 0.2:
                f.write("📊 STRATEGIA AGGREGAZIONE: BILANCIA PESATA CONSIGLIATA\n")
                f.write("   • Moderata eterogeneità, strategia bilanciata migliorerebbe le performance\n")
                f.write("   • Considerare weighted FedAvg con pesi moderati\n\n")
            else:
                f.write("📊 STRATEGIA AGGREGAZIONE: FEDAVG STANDARD SUFFICIENTE\n")
                f.write("   • Bassa eterogeneità, FedAvg standard dovrebbe funzionare bene\n\n")
            
            # Client selection
            if n_outliers > 2:
                f.write("🎯 CLIENT SELECTION: SELEZIONE ADATTIVA NECESSARIA\n")
                f.write("   • Diversi client outlier identificati\n")
                f.write("   • Implementare client selection intelligente per round\n")
                f.write("   • Evitare di selezionare troppi outlier nello stesso round\n\n")
            elif n_outliers > 0:
                f.write("🎯 CLIENT SELECTION: MONITORAGGIO OUTLIER CONSIGLIATO\n")
                f.write("   • Alcuni client outlier identificati\n")
                f.write("   • Monitorare impatto outlier sulle performance globali\n\n")
            else:
                f.write("🎯 CLIENT SELECTION: SELEZIONE RANDOM SUFFICIENTE\n")
                f.write("   • Nessun outlier significativo identificato\n\n")
            
            # Learning rates adattivi
            if cv_classes > 0.15:
                f.write("🔧 LEARNING RATES: ADATTIVI NECESSARI\n")
                f.write("   • Alta variabilità distribuzione classi\n")
                f.write("   • Implementare learning rates personalizzati per client\n")
                f.write("   • Client con distribuzioni estreme dovrebbero avere LR più bassi\n\n")
            elif cv_classes > 0.08:
                f.write("🔧 LEARNING RATES: ADATTIVI CONSIGLIATI\n")
                f.write("   • Moderata variabilità, learning rates adattivi potrebbero aiutare\n\n")
            else:
                f.write("🔧 LEARNING RATES: UNIFORMI SUFFICIENTI\n")
                f.write("   • Bassa variabilità, learning rate uniforme appropriato\n\n")
            
            # Configurazioni specifiche raccomandate
            f.write("CONFIGURAZIONI RACCOMANDATE:\n")
            f.write("-" * 30 + "\n")
            
            if het_index > 0.3:
                f.write("• Fraction_fit: 0.6-0.8 (evita troppi client per round)\n")
                f.write("• Min_fit_clients: 3-5 (garantisce diversità)\n")
                f.write("• Weighted aggregation: Sì\n")
                f.write("• Adaptive LR: Sì\n")
                f.write("• Client selection: Intelligente\n")
            else:
                f.write("• Fraction_fit: 0.8-1.0 (può usare più client)\n")
                f.write("• Min_fit_clients: 2-3\n")
                f.write("• Weighted aggregation: Opzionale\n")
                f.write("• Adaptive LR: Opzionale\n")
                f.write("• Client selection: Random\n")
            
            f.write(f"\n• PCA components: {self.n_components} (appropriato per comunicazione FL)\n")
            f.write("• Rounds: 50-100 (per convergenza con questa eterogeneità)\n")
            f.write("• Evaluation frequency: Ogni 5-10 round\n\n")
            
            # SEZIONE 6: Metriche per benchmark
            f.write("6. METRICHE PER BENCHMARK\n")
            f.write("-" * 25 + "\n")
            f.write("Metriche da monitorare nel confronto baseline vs ottimizzato:\n\n")
            f.write("Performance:\n")
            f.write("  • Global accuracy su test set\n")
            f.write("  • F1-score (importante per classi sbilanciate)\n")
            f.write("  • AUC-ROC\n")
            f.write("  • Precision e Recall per classe\n\n")
            
            f.write("Convergenza:\n")
            f.write("  • Numero round per convergenza\n")
            f.write("  • Stabilità loss nelle ultime epoch\n")
            f.write("  • Tempo di convergenza\n\n")
            
            f.write("Fairness:\n")
            f.write("  • Varianza accuracy tra client\n")
            f.write("  • Performance per cluster di client\n")
            f.write("  • Impatto client outlier\n\n")
            
            f.write("Efficienza:\n")
            f.write("  • Tempo per round\n")
            f.write("  • Comunicazione overhead\n")
            f.write("  • Numero client partecipanti per round\n\n")
        
        print(f"  ✅ Report completo salvato: {report_path}")
        
        # Genera anche CSV con statistiche raw per ulteriori analisi
        self._export_raw_statistics()
        
        return report_path
    
    def _export_raw_statistics(self):
        """Esporta statistiche grezze in formato CSV per analisi ulteriori."""
        
        # DataFrame con statistiche per client
        stats_data = []
        for client_id in sorted(self.client_data.keys()):
            stats = self.client_stats[client_id]
            stats_data.append({
                'client_id': client_id,
                'total_samples': stats['total_samples'],
                'attack_samples': stats['attack_samples'],
                'normal_samples': stats['normal_samples'],
                'attack_ratio': stats['attack_ratio'],
                'normal_ratio': stats['normal_ratio'],
                'feature_variance_mean': stats['feature_variance_mean'],
                'feature_variance_std': stats['feature_variance_std'],
                'outlier_count': stats['outlier_count'],
                'outlier_ratio': stats['outlier_ratio'],
                'class_imbalance_ratio': stats['class_imbalance_ratio']
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_path = os.path.join(self.output_dir, 'client_statistics.csv')
        df_stats.to_csv(stats_path, index=False)
        
        # Matrice similarità
        similarity_path = os.path.join(self.output_dir, 'client_similarity_matrix.csv')
        self.similarity_df.to_csv(similarity_path)
        
        print(f"  ✅ Statistiche CSV salvate: {stats_path}")
        print(f"  ✅ Matrice similarità salvata: {similarity_path}")
    
    def run_complete_analysis(self):
        """
        Esegue l'analisi completa della distribuzione client.
        """
        print(f"\n🚀 AVVIO ANALISI COMPLETA DISTRIBUZIONE CLIENT SMARTGRID")
        print("=" * 80)
        
        try:
            # 1. Caricamento dati
            self.load_client_data()
            
            # 2. Calcolo statistiche
            self.compute_client_statistics()
            
            # 3. Analisi similarità
            similarity_results = self.compute_client_similarity()
            
            # 4. Clustering
            clustering_results = self.perform_client_clustering()
            
            # 5. Analisi Non-IID
            noniid_metrics = self.analyze_non_iid_metrics()
            
            # 6. Identificazione outlier
            outlier_results = self.identify_outlier_clients()
            
            # 7. Generazione visualizzazioni
            self.generate_visualizations()
            
            # 8. Report completo
            report_path = self.generate_comprehensive_report()
            
            print(f"\n✅ ANALISI COMPLETATA CON SUCCESSO!")
            print("=" * 50)
            print(f"📁 File generati in: {self.output_dir}")
            print(f"📊 Visualizzazioni: smartgrid_client_distribution_analysis.png")
            print(f"📝 Report completo: smartgrid_distribution_report.txt")
            print(f"📈 Statistiche CSV: client_statistics.csv")
            print(f"🔍 Matrice similarità: client_similarity_matrix.csv")
            
            # Riepilogo chiave
            print(f"\n📋 RIEPILOGO CHIAVE:")
            print(f"   • Client analizzati: {len(self.client_data)}")
            print(f"   • Indice eterogeneità: {noniid_metrics['heterogeneity_index']:.3f}")
            print(f"   • Client outlier: {len(outlier_results['outlier_clients'])}")
            print(f"   • Silhouette clustering: {clustering_results['silhouette_score']:.3f}")
            
            # Raccomandazioni immediate
            het_index = noniid_metrics['heterogeneity_index']
            if het_index > 0.4:
                print(f"   🎯 RACCOMANDAZIONE: Strategia bilanciata NECESSARIA")
            elif het_index > 0.2:
                print(f"   🎯 RACCOMANDAZIONE: Strategia bilanciata CONSIGLIATA")
            else:
                print(f"   🎯 RACCOMANDAZIONE: FedAvg standard SUFFICIENTE")
            
            return {
                'similarity_results': similarity_results,
                'clustering_results': clustering_results,
                'noniid_metrics': noniid_metrics,
                'outlier_results': outlier_results,
                'report_path': report_path,
                'client_stats': self.client_stats
            }
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE L'ANALISI: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Funzione principale per eseguire l'analisi.
    """
    print("🔍 SMARTGRID FEDERATED LEARNING - ANALISI DISTRIBUZIONE CLIENT")
    print("=" * 80)
    print("📅 Data:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("👤 Autore: francescaapellegrino")
    print("🎯 Obiettivo: Analisi completa distribuzione client per ottimizzazioni FL")
    print("=" * 80)
    
    # Configurazione analisi
    client_ids = list(range(1, 14))  # Client 1-13 per training (14-15 riservati per validazione)
    n_components = 20  # Componenti PCA standard dal sistema esistente
    
    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'analysis_output')
    
    # Esegui analisi
    analyzer = SmartGridClientAnalyzer(
        client_ids=client_ids,
        n_components=n_components,
        output_dir=output_dir
    )
    
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\n🎉 ANALISI COMPLETATA!")
        print(f"✨ Usa i risultati per implementare le ottimizzazioni consigliate")
        return analyzer
    else:
        print(f"\n💥 ANALISI FALLITA!")
        return None


if __name__ == "__main__":
    analyzer = main()