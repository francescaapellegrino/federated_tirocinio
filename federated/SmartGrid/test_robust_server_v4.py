#!/usr/bin/env python3
"""
Test script per verificare la compatibilità del robust_server_final.py
con l'architettura v4 richiesta.

Questo script verifica:
1. Architettura del modello (22 tensori di peso)
2. Pipeline di preprocessing (128→40 features)
3. Compatibilità con client v4 simulato
4. Valutazione globale
"""

import numpy as np
import sys
import os
sys.path.append('/home/runner/work/federated_tirocinio/federated_tirocinio/federated/SmartGrid')

from robust_server_final import (
    create_v4_model_with_attention,
    load_and_preprocess_global_validation_data,
    create_v4_feature_engineering,
    get_v4_global_validation_fn
)

def test_model_architecture():
    """Test dell'architettura del modello v4."""
    print("🧪 TEST 1: Architettura Modello v4")
    print("-" * 50)
    
    model = create_v4_model_with_attention(40)
    weights = model.get_weights()
    
    print(f"✅ Modello creato con successo")
    print(f"📊 Parametri totali: {model.count_params():,}")
    print(f"🎯 Tensori di peso: {len(weights)}")
    
    # Verifica architettura
    expected_layers = [
        "input_layer", "dense_256", "batch_norm_256", "dropout_256",
        "attention_dense", "attention_multiply",
        "dense_128", "batch_norm_128", "dropout_128",
        "dense_64", "batch_norm_64", "dropout_64",
        "output_layer"
    ]
    
    actual_layers = [layer.name for layer in model.layers]
    print(f"📐 Layer presenti: {len(actual_layers)}")
    
    # Verifica presenza attention mechanism
    attention_layers = [l for l in actual_layers if 'attention' in l]
    print(f"🔍 Layer attention: {attention_layers}")
    
    # Verifica tensori di peso (deve essere 22)
    assert len(weights) == 22, f"Attesi 22 tensori, trovati {len(weights)}"
    print("✅ ARCHITETTURA VERIFICATA: 22 tensori di peso")
    
    return True

def test_preprocessing_pipeline():
    """Test della pipeline di preprocessing."""
    print("\n🧪 TEST 2: Pipeline Preprocessing v4")
    print("-" * 50)
    
    try:
        X, y, info = load_and_preprocess_global_validation_data()
        
        print(f"✅ Preprocessing completato")
        print(f"📊 Shape finale: {X.shape}")
        print(f"🎯 Features finali: {X.shape[1]}")
        print(f"📈 Campioni: {X.shape[0]}")
        print(f"⚖️  Ratio attacchi: {info['attack_ratio']:.2%}")
        print(f"📊 Varianza PCA spiegata: {info['variance_explained']:.2%}")
        
        # Verifica shape
        assert X.shape[1] == 40, f"Attese 40 features, trovate {X.shape[1]}"
        print("✅ PREPROCESSING VERIFICATO: 40 features finali")
        
        return X, y, info
        
    except Exception as e:
        print(f"❌ Errore preprocessing: {e}")
        return None, None, None

def test_feature_engineering():
    """Test del feature engineering 30→40."""
    print("\n🧪 TEST 3: Feature Engineering")
    print("-" * 50)
    
    # Simula 30 componenti PCA
    np.random.seed(42)
    X_pca = np.random.randn(100, 30)
    
    print(f"📊 Input PCA: {X_pca.shape}")
    
    X_engineered = create_v4_feature_engineering(X_pca)
    
    print(f"🎯 Output engineered: {X_engineered.shape}")
    print(f"📈 Features aggiunte: {X_engineered.shape[1] - X_pca.shape[1]}")
    
    # Verifica che si abbiano esattamente 40 features
    assert X_engineered.shape[1] == 40, f"Attese 40 features, trovate {X_engineered.shape[1]}"
    print("✅ FEATURE ENGINEERING VERIFICATO: 30→40 features")
    
    return True

def test_global_evaluation():
    """Test della funzione di valutazione globale."""
    print("\n🧪 TEST 4: Valutazione Globale")
    print("-" * 50)
    
    # Crea funzione di valutazione
    eval_fn = get_v4_global_validation_fn()
    
    if eval_fn is None:
        print("❌ Impossibile creare funzione di valutazione")
        return False
    
    print("✅ Funzione di valutazione creata")
    
    # Crea pesi fittizi per test
    model = create_v4_model_with_attention(40)
    test_weights = model.get_weights()
    
    print(f"🎯 Test con {len(test_weights)} tensori di peso")
    
    try:
        # Esegui valutazione di test
        loss, metrics = eval_fn(
            server_round=1,
            parameters=test_weights,
            config={}
        )
        
        print(f"✅ Valutazione completata")
        print(f"💰 Loss: {loss:.4f}")
        print(f"📊 Metriche: {len(metrics)} parametri")
        print(f"🎯 Accuracy: {metrics.get('global_accuracy', 'N/A')}")
        
        # Verifica metriche essenziali
        required_metrics = ['global_accuracy', 'global_precision', 'global_recall', 'global_f1_score']
        for metric in required_metrics:
            assert metric in metrics, f"Metrica mancante: {metric}"
        
        print("✅ VALUTAZIONE GLOBALE VERIFICATA")
        return True
        
    except Exception as e:
        print(f"❌ Errore valutazione: {e}")
        return False

def test_compatibility_v4():
    """Test di compatibilità complessiva v4."""
    print("\n🧪 TEST 5: Compatibilità v4 Completa")
    print("-" * 50)
    
    try:
        # 1. Test architettura
        model = create_v4_model_with_attention(40)
        weights = model.get_weights()
        
        # 2. Test preprocessing
        X, y, info = load_and_preprocess_global_validation_data()
        
        # 3. Test predizione
        predictions = model.predict(X[:100], verbose=0)  # Test su 100 campioni
        
        print(f"✅ Predizioni generate: {predictions.shape}")
        print(f"📊 Predizioni range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # 4. Verifica compatibilità dimensioni
        assert predictions.shape[1] == 1, "Output deve essere singolo"
        assert X.shape[1] == 40, "Input deve essere 40 features"
        assert len(weights) == 22, "Deve avere 22 tensori di peso"
        
        print("✅ COMPATIBILITÀ v4 VERIFICATA COMPLETAMENTE")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore compatibilità: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Esegue tutti i test di verifica."""
    print("🚀 TEST SUITE ROBUST SERVER v4")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Architettura
        if test_model_architecture():
            tests_passed += 1
        
        # Test 2: Preprocessing
        if test_preprocessing_pipeline()[0] is not None:
            tests_passed += 1
        
        # Test 3: Feature Engineering
        if test_feature_engineering():
            tests_passed += 1
        
        # Test 4: Valutazione globale
        if test_global_evaluation():
            tests_passed += 1
        
        # Test 5: Compatibilità complessiva
        if test_compatibility_v4():
            tests_passed += 1
    
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO NEI TEST: {e}")
        import traceback
        traceback.print_exc()
    
    # Risultati finali
    print("\n" + "=" * 60)
    print("📊 RISULTATI TEST SUITE")
    print("=" * 60)
    print(f"✅ Test superati: {tests_passed}/{total_tests}")
    print(f"📈 Percentuale successo: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 TUTTI I TEST SUPERATI - ROBUST SERVER v4 PRONTO!")
        print("\n💡 Il server è completamente compatibile con client v4")
        print("🚀 Utilizza: python robust_server_final.py")
    else:
        print(f"⚠️  {total_tests-tests_passed} test falliti")
        print("🔧 Ricontrolla l'implementazione")
    
    print("=" * 60)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)