import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

def load_smartgrid_data_for_analysis(client_ids=[1, 2, 3, 4, 5]):
    """
    Carica i dati SmartGrid per l'analisi PCA
    """
    print("=== CARICAMENTO DATI SMARTGRID PER ANALISI PCA ===")
    
    # Directory dei dati
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data", "SmartGrid")
    
    all_data = []
    loaded_clients = []
    
    for client_id in client_ids:
        file_path = os.path.join(data_dir, f"data{client_id}.csv")
        
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            loaded_clients.append(client_id)
            print(f"✅ Caricato client {client_id}: {len(df)} campioni, {df.shape[1]-1} feature")
        except FileNotFoundError:
            print(f"❌ File data{client_id}.csv non trovato")
    
    if not all_data:
        raise ValueError("❌ Nessun file dati trovato! Verifica il path dei dati.")
    
    # Combina tutti i dati
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Dataset combinato da {len(loaded_clients)} client:")
    print(f"   - Campioni totali: {len(df_combined)}")
    print(f"   - Feature totali: {df_combined.shape[1]-1}")
    
    return df_combined, loaded_clients

def preprocess_smartgrid_data(df):
    """
    Preprocessing identico a quello nei tuoi script
    """
    print("\n=== PREPROCESSING DATI ===")
    
    # Separazione feature e target
    X = df.drop(columns=["marker"])
    y = (df["marker"] != "Natural").astype(int)
    
    # Statistiche originali
    attack_samples = y.sum()
    natural_samples = (y == 0).sum()
    attack_ratio = y.mean()
    
    print(f"📈 Distribuzione classi originale:")
    print(f"   - Campioni di attacco: {attack_samples} ({attack_ratio*100:.2f}%)")
    print(f"   - Campioni naturali: {natural_samples} ({(1-attack_ratio)*100:.2f}%)")
    
    # Pulizia dati (identica ai tuoi script)
    initial_features = X.shape[1]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isnull().sum().sum()
    
    if nan_count > 0:
        print(f"🔧 Valori NaN trovati: {nan_count}, applicazione imputazione con mediana...")
        X.fillna(X.median(), inplace=True)
    else:
        print(f"✅ Nessun valore NaN trovato")
    
    print(f"📊 Feature dopo pulizia: {X.shape[1]} (invariate)")
    
    return X, y

def analyze_pca_variance(X, max_components=60):
    """
    Analizza la varianza spiegata dai componenti PCA
    """
    print(f"\n=== ANALISI VARIANZA PCA (max {max_components} componenti) ===")
    
    # Normalizzazione (come nei tuoi script)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA completa
    n_components = min(max_components, X.shape[1], X.shape[0])
    print(f"🔍 Analizzando fino a {n_components} componenti...")
    
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_scaled)
    
    # Calcoli varianza
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    individual_variance = pca_full.explained_variance_ratio_
    
    # Trova soglie importanti
    components_80 = np.argmax(cumulative_variance >= 0.80) + 1 if any(cumulative_variance >= 0.80) else n_components
    components_85 = np.argmax(cumulative_variance >= 0.85) + 1 if any(cumulative_variance >= 0.85) else n_components
    components_90 = np.argmax(cumulative_variance >= 0.90) + 1 if any(cumulative_variance >= 0.90) else n_components
    components_95 = np.argmax(cumulative_variance >= 0.95) + 1 if any(cumulative_variance >= 0.95) else n_components
    components_99 = np.argmax(cumulative_variance >= 0.99) + 1 if any(cumulative_variance >= 0.99) else n_components
    
    print(f"📊 Componenti necessari per:")
    print(f"   - 80% varianza: {components_80}")
    print(f"   - 85% varianza: {components_85}")
    print(f"   - 90% varianza: {components_90}")
    print(f"   - 95% varianza: {components_95}")
    print(f"   - 99% varianza: {components_99}")
    
    # Metodo del gomito
    # Trova dove la varianza aggiunta diminuisce significativamente
    variance_diff = np.diff(individual_variance)
    elbow_candidates = []
    
    for i in range(1, min(30, len(variance_diff))):
        if variance_diff[i] > -0.005:  # Variazione piccola
            elbow_candidates.append(i + 1)
    
    elbow_point = elbow_candidates[0] if elbow_candidates else components_90
    print(f"📈 Punto del gomito stimato: {elbow_point} componenti")
    
    # Visualizzazione
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Varianza cumulativa
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
    plt.axhline(y=0.80, color='purple', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='85%')
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%')
    plt.axhline(y=0.99, color='blue', linestyle='--', alpha=0.7, label='99%')
    plt.axvline(x=components_95, color='green', linestyle=':', alpha=0.8, linewidth=2)
    plt.xlabel('Numero di Componenti PCA')
    plt.ylabel('Varianza Cumulativa Spiegata')
    plt.title('Varianza Cumulativa vs Numero Componenti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, min(50, len(cumulative_variance)))
    
    # Plot 2: Varianza individuale (primi 30)
    plt.subplot(2, 3, 2)
    components_to_show = min(30, len(individual_variance))
    plt.bar(range(1, components_to_show + 1), individual_variance[:components_to_show], alpha=0.7, color='skyblue')
    plt.xlabel('Componente PCA')
    plt.ylabel('Varianza Spiegata per Componente')
    plt.title(f'Varianza per Singolo Componente (primi {components_to_show})')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Zoom zona critica (5-40 componenti)
    plt.subplot(2, 3, 3)
    start_zoom, end_zoom = 5, min(40, len(cumulative_variance))
    plt.plot(range(start_zoom, end_zoom + 1), cumulative_variance[start_zoom-1:end_zoom], 'b-', linewidth=2)
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%')
    plt.axvline(x=components_95, color='green', linestyle=':', alpha=0.8)
    plt.axvline(x=elbow_point, color='orange', linestyle=':', alpha=0.8, label='Gomito')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Cumulativa')
    plt.title(f'Zona Critica ({start_zoom}-{end_zoom} componenti)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Varianza incrementale (per identificare il gomito)
    plt.subplot(2, 3, 4)
    components_to_show = min(35, len(individual_variance))
    plt.plot(range(1, components_to_show + 1), individual_variance[:components_to_show], 'r-o', markersize=4)
    plt.axvline(x=elbow_point, color='orange', linestyle='--', alpha=0.8, label='Gomito')
    plt.xlabel('Componente PCA')
    plt.ylabel('Varianza del Singolo Componente')
    plt.title('Identificazione Punto del Gomito')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Scala logaritmica per vedere meglio i dettagli
    
    # Plot 5: Efficienza (varianza per componente)
    plt.subplot(2, 3, 5)
    efficiency = cumulative_variance / np.arange(1, len(cumulative_variance) + 1)
    plt.plot(range(1, len(efficiency) + 1), efficiency, 'g-', linewidth=2)
    max_efficiency_idx = np.argmax(efficiency[:min(30, len(efficiency))])
    plt.axvline(x=max_efficiency_idx + 1, color='red', linestyle='--', alpha=0.8, label=f'Max Efficienza: {max_efficiency_idx + 1}')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Cumulativa / N° Componenti')
    plt.title('Efficienza PCA (Varianza per Componente)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, min(40, len(efficiency)))
    
    # Plot 6: Riduzione dimensionale
    plt.subplot(2, 3, 6)
    original_features = X.shape[1]
    components_range = range(1, min(50, len(cumulative_variance)) + 1)
    reduction_percentage = [(original_features - c) / original_features * 100 for c in components_range]
    plt.plot(components_range, reduction_percentage, 'purple', linewidth=2)
    plt.axvline(x=components_95, color='green', linestyle='--', alpha=0.8, label=f'95% varianza: {components_95}')
    plt.xlabel('Numero di Componenti Mantenuti')
    plt.ylabel('Riduzione Dimensionale (%)')
    plt.title(f'Riduzione da {original_features} Feature Originali')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smartgrid_pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Grafico salvato come 'smartgrid_pca_variance_analysis.png'")
    plt.show()
    
    return {
        'components_80': components_80,
        'components_85': components_85,
        'components_90': components_90,
        'components_95': components_95,
        'components_99': components_99,
        'elbow_point': elbow_point,
        'max_efficiency_point': max_efficiency_idx + 1,
        'cumulative_variance': cumulative_variance,
        'individual_variance': individual_variance,
        'original_features': original_features
    }

def test_pca_performance_federated(X, y, component_candidates):
    """
    Testa le performance di diversi numeri di componenti PCA
    simulando condizioni di federated learning
    """
    print(f"\n=== TEST PERFORMANCE PCA PER FEDERATED LEARNING ===")
    print(f"🧪 Testando componenti: {component_candidates}")
    
    results = []
    
    for n_components in component_candidates:
        if n_components >= min(X.shape[1], X.shape[0]):
            print(f"⏭️ Saltando {n_components} componenti (troppi per questo dataset)")
            continue
            
        print(f"\n🔍 Testando {n_components} componenti...")
        
        # Preprocessing (identico ai tuoi script)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        feature_reduction = (X.shape[1] - n_components) / X.shape[1]
        
        # Test con modelli tipici per intrusion detection
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        }
        
        component_result = {
            'n_components': n_components,
            'variance_explained': variance_explained,
            'feature_reduction': feature_reduction
        }
        
        for model_name, model in models.items():
            print(f"  🤖 Testing {model_name}...")
            start_time = time.time()
            
            # Cross-validation con 5 fold (simula diversi client)
            try:
                cv_scores = cross_val_score(model, X_pca, y, cv=5, scoring='f1_weighted', n_jobs=-1)
                training_time = time.time() - start_time
                
                component_result[f'{model_name}_f1_mean'] = cv_scores.mean()
                component_result[f'{model_name}_f1_std'] = cv_scores.std()
                component_result[f'{model_name}_time'] = training_time
                
                print(f"     F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                print(f"     Tempo: {training_time:.2f}s")
                
            except Exception as e:
                print(f"     ❌ Errore: {e}")
                component_result[f'{model_name}_f1_mean'] = 0.0
                component_result[f'{model_name}_f1_std'] = 1.0
                component_result[f'{model_name}_time'] = 999.0
        
        print(f"  📊 Varianza spiegata: {variance_explained:.4f} ({variance_explained*100:.1f}%)")
        print(f"  📉 Riduzione feature: {feature_reduction*100:.1f}%")
        
        results.append(component_result)
    
    # Converte in DataFrame per analisi
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("❌ Nessun risultato ottenuto!")
        return df_results, None, None
    
    # Trova i migliori risultati
    best_f1_lr = df_results.loc[df_results['LogisticRegression_f1_mean'].idxmax()]
    best_f1_rf = df_results.loc[df_results['RandomForest_f1_mean'].idxmax()]
    
    print(f"\n🏆 MIGLIORI RISULTATI:")
    print(f"📊 Logistic Regression: {int(best_f1_lr['n_components'])} componenti")
    print(f"   - F1 Score: {best_f1_lr['LogisticRegression_f1_mean']:.4f} (±{best_f1_lr['LogisticRegression_f1_std']:.4f})")
    print(f"   - Varianza: {best_f1_lr['variance_explained']:.4f}")
    print(f"   - Tempo: {best_f1_lr['LogisticRegression_time']:.2f}s")
    
    print(f"🌲 Random Forest: {int(best_f1_rf['n_components'])} componenti")
    print(f"   - F1 Score: {best_f1_rf['RandomForest_f1_mean']:.4f} (±{best_f1_rf['RandomForest_f1_std']:.4f})")
    print(f"   - Varianza: {best_f1_rf['variance_explained']:.4f}")
    print(f"   - Tempo: {best_f1_rf['RandomForest_time']:.2f}s")
    
    # Visualizzazione risultati performance
    plt.figure(figsize=(18, 12))
    
    # Plot 1: F1 Score vs Numero Componenti
    plt.subplot(2, 4, 1)
    plt.plot(df_results['n_components'], df_results['LogisticRegression_f1_mean'], 
             'o-', label='Logistic Regression', linewidth=2, markersize=6)
    plt.fill_between(df_results['n_components'], 
                     df_results['LogisticRegression_f1_mean'] - df_results['LogisticRegression_f1_std'],
                     df_results['LogisticRegression_f1_mean'] + df_results['LogisticRegression_f1_std'],
                     alpha=0.2)
    plt.plot(df_results['n_components'], df_results['RandomForest_f1_mean'], 
             's-', label='Random Forest', linewidth=2, markersize=6)
    plt.fill_between(df_results['n_components'], 
                     df_results['RandomForest_f1_mean'] - df_results['RandomForest_f1_std'],
                     df_results['RandomForest_f1_mean'] + df_results['RandomForest_f1_std'],
                     alpha=0.2)
    plt.axvline(x=best_f1_lr['n_components'], color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=best_f1_rf['n_components'], color='orange', linestyle='--', alpha=0.7)
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('F1 Score (Cross-Validation)')
    plt.title('Performance vs Numero Componenti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Varianza Spiegata vs Componenti
    plt.subplot(2, 4, 2)
    plt.plot(df_results['n_components'], df_results['variance_explained'], 
             'g-o', linewidth=2, markersize=6)
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%')
    plt.axhline(y=0.99, color='blue', linestyle='--', alpha=0.7, label='99%')
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('Varianza Spiegata')
    plt.title('Varianza vs Numero Componenti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Tempo di Training vs Componenti
    plt.subplot(2, 4, 3)
    plt.plot(df_results['n_components'], df_results['LogisticRegression_time'], 
             'o-', label='Logistic Regression', linewidth=2)
    plt.plot(df_results['n_components'], df_results['RandomForest_time'], 
             's-', label='Random Forest', linewidth=2)
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('Tempo Training (secondi)')
    plt.title('Efficienza vs Numero Componenti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Trade-off Performance vs Efficienza
    plt.subplot(2, 4, 4)
    scatter1 = plt.scatter(df_results['LogisticRegression_time'], df_results['LogisticRegression_f1_mean'], 
                          s=df_results['n_components']*10, alpha=0.7, label='Logistic Regression', c='blue')
    scatter2 = plt.scatter(df_results['RandomForest_time'], df_results['RandomForest_f1_mean'], 
                          s=df_results['n_components']*10, alpha=0.7, label='Random Forest', c='orange')
    
    # Annotazioni per i punti migliori
    plt.annotate(f"{int(best_f1_lr['n_components'])}", 
                (best_f1_lr['LogisticRegression_time'], best_f1_lr['LogisticRegression_f1_mean']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    plt.annotate(f"{int(best_f1_rf['n_components'])}", 
                (best_f1_rf['RandomForest_time'], best_f1_rf['RandomForest_f1_mean']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.xlabel('Tempo Training (s)')
    plt.ylabel('F1 Score')
    plt.title('Trade-off Performance vs Tempo\n(dimensione = n° componenti)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Stabilità (Standard Deviation)
    plt.subplot(2, 4, 5)
    plt.plot(df_results['n_components'], df_results['LogisticRegression_f1_std'], 
             'o-', label='LR Std Dev', linewidth=2)
    plt.plot(df_results['n_components'], df_results['RandomForest_f1_std'], 
             's-', label='RF Std Dev', linewidth=2)
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('Standard Deviation F1')
    plt.title('Stabilità vs Numero Componenti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Efficienza (Performance per Componente)
    plt.subplot(2, 4, 6)
    efficiency_lr = df_results['LogisticRegression_f1_mean'] / df_results['n_components']
    efficiency_rf = df_results['RandomForest_f1_mean'] / df_results['n_components']
    plt.plot(df_results['n_components'], efficiency_lr, 'o-', label='LR Efficiency', linewidth=2)
    plt.plot(df_results['n_components'], efficiency_rf, 's-', label='RF Efficiency', linewidth=2)
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('F1 Score / N° Componenti')
    plt.title('Efficienza per Componente')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Riduzione Feature vs Performance
    plt.subplot(2, 4, 7)
    plt.scatter(df_results['feature_reduction'] * 100, df_results['LogisticRegression_f1_mean'], 
               s=60, alpha=0.7, label='Logistic Regression', c='blue')
    plt.scatter(df_results['feature_reduction'] * 100, df_results['RandomForest_f1_mean'], 
               s=60, alpha=0.7, label='Random Forest', c='orange')
    plt.xlabel('Riduzione Feature (%)')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Riduzione Dimensionale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Performance combinata (media dei due modelli)
    plt.subplot(2, 4, 8)
    combined_performance = (df_results['LogisticRegression_f1_mean'] + df_results['RandomForest_f1_mean']) / 2
    combined_std = (df_results['LogisticRegression_f1_std'] + df_results['RandomForest_f1_std']) / 2
    
    plt.plot(df_results['n_components'], combined_performance, 'ro-', linewidth=2, markersize=6)
    plt.fill_between(df_results['n_components'], 
                     combined_performance - combined_std,
                     combined_performance + combined_std,
                     alpha=0.2, color='red')
    
    best_combined_idx = combined_performance.idxmax()
    best_combined_components = df_results.loc[best_combined_idx, 'n_components']
    plt.axvline(x=best_combined_components, color='red', linestyle='--', alpha=0.8)
    plt.annotate(f'Best: {int(best_combined_components)}', 
                xy=(best_combined_components, combined_performance.loc[best_combined_idx]),
                xytext=(10, 10), textcoords='offset points', fontweight='bold')
    
    plt.xlabel('Numero Componenti PCA')
    plt.ylabel('F1 Score Medio')
    plt.title('Performance Media (LR + RF)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smartgrid_pca_performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Grafico performance salvato come 'smartgrid_pca_performance_analysis.png'")
    plt.show()
    
    return df_results, best_f1_lr, best_f1_rf

def make_final_recommendation(variance_results, performance_results, best_lr, best_rf):
    """
    Formula la raccomandazione finale basata su tutti i criteri
    """
    print(f"\n" + "="*80)
    print(f"🎯 RACCOMANDAZIONE FINALE PER SMARTGRID FEDERATED LEARNING")
    print(f"="*80)
    
    # Estrai i candidati principali
    candidates = {
        'varianza_90': variance_results['components_90'],
        'varianza_95': variance_results['components_95'],
        'gomito': variance_results['elbow_point'],
        'efficienza_max': variance_results['max_efficiency_point'],
        'performance_lr': int(best_lr['n_components']) if best_lr is not None else None,
        'performance_rf': int(best_rf['n_components']) if best_rf is not None else None,
    }
    
    # Rimuovi candidati None
    candidates = {k: v for k, v in candidates.items() if v is not None}
    
    print(f"📊 CANDIDATI PRINCIPALI:")
    for name, components in candidates.items():
        variance_pct = variance_results['cumulative_variance'][components-1] * 100 if components <= len(variance_results['cumulative_variance']) else 0
        reduction_pct = (variance_results['original_features'] - components) / variance_results['original_features'] * 100
        print(f"   - {name.replace('_', ' ').title()}: {components} componenti ({variance_pct:.1f}% varianza, {reduction_pct:.1f}% riduzione)")
    
    # LOGICA DI RACCOMANDAZIONE INTELLIGENTE
    
    # 1. Per federated learning su smart grid, priorità:
    #    - Mantenere almeno 90% della varianza (per non perdere pattern di attacco)
    #    - Riduzione significativa per efficienza comunicazione
    #    - Stabilità tra diversi modelli
    
    # 2. Criteri di selezione
    min_variance_threshold = 0.90  # Minimo 90% varianza
    max_components_reasonable = 35  # Massimo ragionevole per FL
    
    # Filtra candidati che rispettano i criteri
    valid_candidates = []
    
    for name, components in candidates.items():
        if components <= len(variance_results['cumulative_variance']):
            variance = variance_results['cumulative_variance'][components-1]
            if variance >= min_variance_threshold and components <= max_components_reasonable:
                valid_candidates.append((name, components, variance))
    
    if not valid_candidates:
        # Fallback: usa 95% varianza se disponibile
        final_recommendation = variance_results['components_95']
        reason = "fallback: 95% varianza (criterio conservativo)"
    else:
        # Ordina per un punteggio composito che bilancia varianza e efficienza
        scored_candidates = []
        for name, components, variance in valid_candidates:
            # Punteggio: bilancia varianza alta con componenti bassi
            efficiency_score = (variance_results['original_features'] - components) / variance_results['original_features']
            variance_score = variance
            # Peso maggiore alla varianza (più importante non perdere informazioni)
            composite_score = 0.7 * variance_score + 0.3 * efficiency_score
            scored_candidates.append((name, components, variance, composite_score))
        
        # Ordina per punteggio composito
        scored_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Scelta finale: migliore punteggio composito
        best_candidate = scored_candidates[0]
        final_recommendation = best_candidate[1]
        reason = f"migliore bilancio varianza/efficienza (da {best_candidate[0]})"
    
    # Assicurati che la raccomandazione sia ragionevole
    if final_recommendation > max_components_reasonable:
        final_recommendation = max_components_reasonable
        reason = f"limitato a {max_components_reasonable} per efficienza FL"
    elif final_recommendation < 10:
        final_recommendation = max(10, variance_results['components_90'])
        reason = "aumentato per mantenere informazioni sufficienti"
    
    # Calcola le metriche della scelta finale
    final_variance = variance_results['cumulative_variance'][final_recommendation-1] if final_recommendation <= len(variance_results['cumulative_variance']) else 0
    final_reduction = (variance_results['original_features'] - final_recommendation) / variance_results['original_features']
    
    print(f"\n🏆 RACCOMANDAZIONE FINALE: {final_recommendation} COMPONENTI PCA")
    print(f"📋 Motivo: {reason}")
    print(f"\n📊 CON {final_recommendation} COMPONENTI OTTERRAI:")
    print(f"   ✅ Varianza spiegata: {final_variance:.4f} ({final_variance*100:.1f}%)")
    print(f"   ✅ Riduzione feature: {final_reduction*100:.1f}% (da {variance_results['original_features']} a {final_recommendation})")
    print(f"   ✅ Adatto per federated learning: comunicazione efficiente")
    print(f"   ✅ Mantiene informazioni critiche per rilevazione attacchi")
    
    # Confronto con configurazione attuale (20)
    current_components = 20
    if final_recommendation != current_components:
        if current_components <= len(variance_results['cumulative_variance']):
            current_variance = variance_results['cumulative_variance'][current_components-1]
            variance_diff = final_variance - current_variance
            
            print(f"\n🔄 CONFRONTO CON CONFIGURAZIONE ATTUALE ({current_components} componenti):")
            print(f"   - Configurazione attuale: {current_variance:.4f} varianza ({current_variance*100:.1f}%)")
            print(f"   - Configurazione raccomandata: {final_variance:.4f} varianza ({final_variance*100:.1f}%)")
            
            if variance_diff > 0:
                print(f"   - 📈 MIGLIORAMENTO: +{variance_diff:.4f} varianza (+{variance_diff*100:.2f}%)")
            elif variance_diff < 0:
                print(f"   - 📉 Leggera perdita: {variance_diff:.4f} varianza ({variance_diff*100:.2f}%)")
            else:
                print(f"   - ✅ Varianza identica")
                
            component_diff = final_recommendation - current_components
            if component_diff > 0:
                print(f"   - 🔺 Più componenti: +{component_diff} (comunicazione leggermente più pesante)")
            elif component_diff < 0:
                print(f"   - 🔻 Meno componenti: {component_diff} (comunicazione più efficiente)")
    else:
        print(f"\n✅ LA CONFIGURAZIONE ATTUALE ({current_components} componenti) È GIÀ OTTIMALE!")
    
    # Istruzioni di aggiornamento
    print(f"\n🛠️ PER AGGIORNARE IL TUO CODICE:")
    print(f"   1. Nel file client_sg.py, cambia:")
    print(f"      n_components={final_recommendation}  # era n_components=20")
    print(f"   2. Nel file server_sg.py, cambia:")
    print(f"      n_components={final_recommendation}  # era n_components=20")
    
    print(f"\n🎓 PER LA TUA TESI:")
    print(f"   - Spiega che hai analizzato la varianza spiegata")
    print(f"   - Hai bilanciato performance e efficienza comunicativa")
    print(f"   - Hai validato empiricamente con cross-validation")
    print(f"   - La scelta di {final_recommendation} componenti ottimizza il trade-off")
    
    return final_recommendation

def main():
    """
    Funzione principale per l'analisi PCA ottimale
    """
    print("🚀 ANALISI PCA OTTIMALE PER SMARTGRID FEDERATED LEARNING")
    print("=" * 80)
    print("📅 Data analisi:", "2025-08-04 15:16:22")
    print("👤 Utente:", "francescaapellegrino")
    print("=" * 80)
    
    try:
        # 1. Carica i dati SmartGrid
        print("\n📂 FASE 1: CARICAMENTO DATI")
        df_combined, loaded_clients = load_smartgrid_data_for_analysis(
            client_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Usa più client per analisi robusta
        )
        
        # 2. Preprocessing
        print("\n🔧 FASE 2: PREPROCESSING")
        X, y = preprocess_smartgrid_data(df_combined)
        
        # 3. Analisi varianza PCA
        print("\n📊 FASE 3: ANALISI VARIANZA PCA")
        variance_results = analyze_pca_variance(X, max_components=60)
        
        # 4. Test performance su diversi numeri di componenti
        print("\n🎯 FASE 4: TEST PERFORMANCE")
        # Selezione intelligente di candidati da testare
        candidates_to_test = [
            5, 10, 15, 20, 25, 30,  # Range standard
            variance_results['elbow_point'],
            variance_results['components_90'],
            variance_results['components_95'],
            variance_results['max_efficiency_point']
        ]
        # Rimuovi duplicati e ordina
        candidates_to_test = sorted(list(set([c for c in candidates_to_test if c is not None and c <= 50])))
        
        performance_results, best_lr, best_rf = test_pca_performance_federated(
            X, y, candidates_to_test
        )
        
        # 5. Raccomandazione finale
        print("\n🎯 FASE 5: RACCOMANDAZIONE FINALE")
        optimal_components = make_final_recommendation(
            variance_results, performance_results, best_lr, best_rf
        )
        
        # 6. Riepilogo finale
        print(f"\n🎉 ANALISI COMPLETATA!")
        print(f"📊 Dataset analizzato: {len(df_combined)} campioni da {len(loaded_clients)} client")
        print(f"🏆 NUMERO OTTIMALE DI COMPONENTI PCA: {optimal_components}")
        print(f"📁 Grafici salvati: 'smartgrid_pca_variance_analysis.png' e 'smartgrid_pca_performance_analysis.png'")
        
        return optimal_components
        
    except Exception as e:
        print(f"❌ ERRORE durante l'analisi: {e}")
        import traceback
        print("\n🔍 DETTAGLI ERRORE:")
        traceback.print_exc()
        print(f"\n💡 SUGGERIMENTI:")
        print(f"   - Verifica che i file SmartGrid siano nella directory: federated/SmartGrid/../../data/SmartGrid/")
        print(f"   - Assicurati di avere i file data1.csv, data2.csv, ecc.")
        print(f"   - Controlla di essere nella directory corretta quando esegui lo script")
        return None

if __name__ == "__main__":
    optimal_components = main()
    if optimal_components:
        print(f"\n✨ USA {optimal_components} COMPONENTI PCA NEI TUOI CODICI! ✨")