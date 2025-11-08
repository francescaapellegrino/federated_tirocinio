"""
Script per eseguire attacchi di inferenza su un modello Random Forest federato già salvato (.pkl).

UTILIZZO:
    python run_attacks_on_saved_model.py <path_al_modello.pkl>
    
ESEMPIO:
    python run_attacks_on_saved_model.py models/federated_rf_final.pkl

DESCRIZIONE:
    Questo script carica un modello Random Forest federato già addestrato e salvato,
    poi esegue i 4 tipi di attacchi di inferenza per valutarne le vulnerabilità:
    1. Membership Inference Attack
    2. Attribute Inference Attack
    3. Model Inversion Attack
    4. Reconstruction Attack
"""

import sys
import os
import pickle
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importa il framework degli attacchi esistente
from inference_attacks_integration import FederatedInferenceAttacker
from clientRFtmp_privacy import load_client_smartgrid_data

def load_saved_model(model_path):
    """
    Carica un modello Random Forest salvato da file .pkl
    
    Args:
        model_path: Path al file .pkl contenente il modello
        
    Returns:
        model: Modello Random Forest caricato
    """
    print(f"\n{'='*80}")
    print(f"📦 CARICAMENTO MODELLO SALVATO")
    print(f"{'='*80}")
    print(f"Path modello: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ File modello non trovato: {model_path}")
    
    try:
        # Carica il modello dal file pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Modello caricato con successo!")
        
        # Mostra informazioni sul modello
        print(f"\n📊 INFORMAZIONI MODELLO:")
        print(f"   Tipo: {type(model).__name__}")
        
        if hasattr(model, 'n_estimators'):
            print(f"   N. Estimatori (alberi): {model.n_estimators}")
        
        if hasattr(model, 'estimators_'):
            print(f"   N. Alberi addestrati: {len(model.estimators_)}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"   N. Features: {model.n_features_in_}")
        
        if hasattr(model, 'classes_'):
            print(f"   Classi: {model.classes_}")
        
        if hasattr(model, 'max_depth'):
            print(f"   Max Depth: {model.max_depth}")
        
        if hasattr(model, 'criterion'):
            print(f"   Criterio: {model.criterion}")
        
        # Verifica che il modello sia addestrato
        if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
            raise ValueError("❌ Il modello non sembra essere stato addestrato (nessun albero trovato)")
        
        return model
        
    except Exception as e:
        print(f"❌ Errore durante caricamento modello: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_model_on_data(model, X_test, y_test, dataset_name="test"):
    """
    Testa il modello caricato su dati di test per verificare che funzioni.
    
    Args:
        model: Modello da testare
        X_test: Dati di test
        y_test: Etichette di test
        dataset_name: Nome del dataset per il report
    """
    print(f"\n{'='*80}")
    print(f"🔍 VERIFICA FUNZIONAMENTO MODELLO SU DATI {dataset_name.upper()}")
    print(f"{'='*80}")
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Predizioni
        y_pred = model.predict(X_test)
        
        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"✅ Modello funzionante!")
        print(f"\n📊 METRICHE SU DATI {dataset_name.upper()}:")
        print(f"   Campioni testati: {len(X_test)}")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(X_test)
        }
        
    except Exception as e:
        print(f"❌ Errore durante test modello: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_attacks_on_saved_model(model_path, 
                                member_clients=[1, 2, 3], 
                                nonmember_clients=[14, 15],
                                output_dir="attack_results"):
    """
    Esegue attacchi di inferenza su un modello salvato.
    
    Args:
        model_path: Path al file .pkl del modello
        member_clients: Client considerati "membri" (dati nel training)
        nonmember_clients: Client considerati "non-membri" (dati NON nel training)
        output_dir: Directory dove salvare i risultati
        
    Returns:
        dict: Risultati completi degli attacchi
    """
    print(f"\n{'='*80}")
    print(f"🛡️  ATTACCHI DI INFERENZA SU MODELLO SALVATO")
    print(f"{'='*80}")
    print(f"Data/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modello: {model_path}")
    print(f"Output: {output_dir}")
    
    try:
        # STEP 1: Carica il modello salvato
        print(f"\n[STEP 1/4] Caricamento modello...")
        model = load_saved_model(model_path)
        
        # STEP 2: Carica dati per gli attacchi (usando il TUO preprocessing)
        print(f"\n[STEP 2/4] Caricamento dati con preprocessing del tuo sistema...")
        print(f"   Client membri: {member_clients}")
        print(f"   Client non-membri: {nonmember_clients}")
        
        # Carica dati membri (erano nel training)
        X_members_list, y_members_list = [], []
        for client_id in member_clients:
            try:
                X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                X_members_list.append(X_combined)
                y_members_list.append(y_combined)
                print(f"   ✅ Client {client_id} (membro): {len(X_combined)} campioni, {X_combined.shape[1]} features")
            except Exception as e:
                print(f"   ⚠️  Client {client_id}: {e}")
        
        # Carica dati non-membri (NON erano nel training)
        X_nonmembers_list, y_nonmembers_list = [], []
        for client_id in nonmember_clients:
            try:
                X_train, y_train, X_val, y_val, dataset_info = load_client_smartgrid_data(client_id)
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                X_nonmembers_list.append(X_combined)
                y_nonmembers_list.append(y_combined)
                print(f"   ✅ Client {client_id} (non-membro): {len(X_combined)} campioni, {X_combined.shape[1]} features")
            except Exception as e:
                print(f"   ⚠️  Client {client_id}: {e}")
        
        if not X_members_list or not X_nonmembers_list:
            raise ValueError("❌ Impossibile caricare dati membri/non-membri")
        
        # Combina i dati
        X_members = np.vstack(X_members_list)
        y_members = np.hstack(y_members_list)
        X_nonmembers = np.vstack(X_nonmembers_list)
        y_nonmembers = np.hstack(y_nonmembers_list)
        
        print(f"\n   📊 DATI CARICATI:")
        print(f"   Membri: {len(X_members)} campioni")
        print(f"   Non-membri: {len(X_nonmembers)} campioni")
        print(f"   Features: {X_members.shape[1]}")
        
        # STEP 3: Verifica che il modello funzioni sui dati
        print(f"\n[STEP 3/4] Verifica funzionamento modello...")
        
        # Test su dati membri
        test_members_metrics = test_model_on_data(
            model, 
            X_members[:min(500, len(X_members))], 
            y_members[:min(500, len(y_members))],
            "membri"
        )
        
        # Test su dati non-membri
        test_nonmembers_metrics = test_model_on_data(
            model,
            X_nonmembers[:min(500, len(X_nonmembers))],
            y_nonmembers[:min(500, len(y_nonmembers))],
            "non-membri"
        )
        
        # STEP 4: Esegui attacchi di inferenza
        print(f"\n[STEP 4/4] Esecuzione attacchi di inferenza...")
        
        # Inizializza l'attaccante
        attacker = FederatedInferenceAttacker(output_dir=output_dir)
        
        # Assegna il modello caricato come target
        attacker.target_model = model
        
        # Prepara info sul training (simulata per il modello caricato)
        attacker.training_info = {
            'training_clients': member_clients,
            'n_samples': len(X_members),
            'n_features': X_members.shape[1],
            'accuracy': test_members_metrics['accuracy'] if test_members_metrics else 0.0,
            'attack_ratio': np.mean(y_members),
            'model_source': 'loaded_from_pkl',
            'model_path': model_path
        }
        
        # Prepara dati per gli attacchi
        n_test_members = min(500, len(X_members) // 2)
        n_test_nonmembers = min(500, len(X_nonmembers) // 2)
        
        X_test = np.vstack([
            X_members[:n_test_members],
            X_nonmembers[:n_test_nonmembers]
        ])
        y_test = np.hstack([
            y_members[:n_test_members],
            y_nonmembers[:n_test_nonmembers]
        ])
        
        membership_ground_truth = np.hstack([
            np.ones(n_test_members),
            np.zeros(n_test_nonmembers)
        ])
        
        attack_data = {
            'X_members': X_members,
            'y_members': y_members,
            'X_nonmembers': X_nonmembers,
            'y_nonmembers': y_nonmembers,
            'X_test': X_test,
            'y_test': y_test,
            'membership_ground_truth': membership_ground_truth,
            'member_clients': member_clients,
            'nonmember_clients': nonmember_clients,
            'n_features': X_members.shape[1]
        }
        
        # Esegui i 4 attacchi
        all_results = {
            'evaluation_id': f"saved_model_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'target_system': 'loaded_federated_random_forest',
            'training_info': attacker.training_info,
            'model_performance': {
                'members': test_members_metrics,
                'nonmembers': test_nonmembers_metrics
            },
            'attack_data_info': {
                'member_clients': member_clients,
                'nonmember_clients': nonmember_clients,
                'n_features': attack_data['n_features']
            },
            'attacks': {}
        }
        
        print(f"\n🎯 Esecuzione 4 attacchi di inferenza...")
        
        # 1. Membership Inference Attack
        print(f"\n{'='*80}")
        print(f"🎯 [1/4] MEMBERSHIP INFERENCE ATTACK")
        print(f"{'='*80}")
        all_results['attacks']['membership_inference'] = attacker.membership_inference_attack(attack_data)
        
        # 2. Attribute Inference Attack
        print(f"\n{'='*80}")
        print(f"🎯 [2/4] ATTRIBUTE INFERENCE ATTACK")
        print(f"{'='*80}")
        all_results['attacks']['attribute_inference'] = attacker.attribute_inference_attack(attack_data)
        
        # 3. Model Inversion Attack
        print(f"\n{'='*80}")
        print(f"🎯 [3/4] MODEL INVERSION ATTACK")
        print(f"{'='*80}")
        all_results['attacks']['model_inversion'] = attacker.model_inversion_attack(attack_data)
        
        # 4. Reconstruction Attack
        print(f"\n{'='*80}")
        print(f"🎯 [4/4] RECONSTRUCTION ATTACK")
        print(f"{'='*80}")
        all_results['attacks']['reconstruction'] = attacker.reconstruction_attack(attack_data)
        
        # Calcola summary
        successful_attacks = [name for name, result in all_results['attacks'].items() 
                            if result.get('success', False)]
        
        # Calcola vulnerability score
        vulnerability_scores = []
        for attack_name, result in all_results['attacks'].items():
            if not result.get('success', False):
                continue
            
            if attack_name == 'membership_inference':
                score = result.get('attack_advantage', 0) * 100
            elif attack_name == 'attribute_inference':
                score = result.get('average_advantage', 0) * 100
            elif attack_name == 'model_inversion':
                score = result.get('max_confidence', 0) * 100
            elif attack_name == 'reconstruction':
                score = result.get('success_rate', 0) * 100
            else:
                score = 0
            
            vulnerability_scores.append(max(0, score))
        
        overall_vulnerability = np.mean(vulnerability_scores) if vulnerability_scores else 0.0
        
        # Risk assessment
        def assess_risk(vuln_score, n_successful):
            if vuln_score > 50 or n_successful >= 3:
                return "🔴 ALTO RISCHIO"
            elif vuln_score > 25 or n_successful >= 2:
                return "🟡 MEDIO RISCHIO"
            else:
                return "🟢 BASSO RISCHIO"
        
        all_results['summary'] = {
            'total_attacks': 4,
            'successful_attacks': len(successful_attacks),
            'success_rate': len(successful_attacks) / 4,
            'successful_attack_types': successful_attacks,
            'overall_vulnerability_score': float(overall_vulnerability),
            'risk_assessment': assess_risk(overall_vulnerability, len(successful_attacks))
        }
        
        # Stampa riassunto finale
        print(f"\n{'='*80}")
        print(f"📊 RIASSUNTO ATTACCHI SUL MODELLO SALVATO")
        print(f"{'='*80}")
        print(f"Modello: {model_path}")
        print(f"Attacchi totali: 4")
        print(f"Attacchi riusciti: {len(successful_attacks)}/4")
        print(f"Success rate: {all_results['summary']['success_rate']*100:.1f}%")
        print(f"Vulnerability score: {overall_vulnerability:.1f}%")
        print(f"Risk assessment: {all_results['summary']['risk_assessment']}")
        
        if successful_attacks:
            print(f"\n✅ Attacchi riusciti:")
            for attack_name in successful_attacks:
                print(f"   - {attack_name.replace('_', ' ').title()}")
        else:
            print(f"\n🛡️  Nessun attacco riuscito - Modello resistente!")
        
        print(f"\n📋 INTERPRETAZIONI PER ATTACCO:")
        for attack_name, result in all_results['attacks'].items():
            if result.get('success', False):
                print(f"\n{attack_name.replace('_', ' ').title()}:")
                print(f"   {result.get('interpretation', 'N/A')}")
        
        # Salva risultati
        attacker._save_results(all_results)
        
        print(f"\n{'='*80}")
        print(f"✅ VALUTAZIONE COMPLETATA!")
        print(f"📁 Risultati salvati in: {output_dir}")
        print(f"{'='*80}")
        
        return all_results
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERRORE DURANTE VALUTAZIONE")
        print(f"{'='*80}")
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Funzione principale dello script.
    """
    print(f"\n{'='*80}")
    print(f"🛡️  SCRIPT ATTACCHI DI INFERENZA SU MODELLO SALVATO")
    print(f"{'='*80}")
    
    # Verifica argomenti
    if len(sys.argv) < 2:
        print(f"\n❌ ERRORE: Devi fornire il path al file .pkl del modello")
        print(f"\nUTILIZZO:")
        print(f"   python run_attacks_on_saved_model.py <path_al_modello.pkl>")
        print(f"\nESEMPIO:")
        print(f"   python run_attacks_on_saved_model.py models/federated_rf_final.pkl")
        print(f"\nDESCRIZIONE:")
        print(f"   Carica un modello Random Forest federato già addestrato e")
        print(f"   esegue i 4 attacchi di inferenza per valutarne le vulnerabilità:")
        print(f"   1. Membership Inference Attack")
        print(f"   2. Attribute Inference Attack")
        print(f"   3. Model Inversion Attack")
        print(f"   4. Reconstruction Attack")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Parametri opzionali
    member_clients = [1, 2, 3]      # Client considerati nel training
    nonmember_clients = [14, 15]    # Client NON nel training
    output_dir = "attack_results"   # Directory output
    
    # Se forniti parametri aggiuntivi
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    try:
        # Esegui attacchi
        results = run_attacks_on_saved_model(
            model_path=model_path,
            member_clients=member_clients,
            nonmember_clients=nonmember_clients,
            output_dir=output_dir
        )
        
        if results:
            print(f"\n✅ Tutti gli attacchi sono stati eseguiti con successo!")
            print(f"📊 Controlla '{output_dir}' per i risultati dettagliati")
            
            # Suggerimenti per la tesi
            print(f"\n{'='*80}")
            print(f"💡 SUGGERIMENTI PER LA TESI")
            print(f"{'='*80}")
            print(f"Usa questi risultati nella tua tesi per:")
            print(f"1. Valutare la privacy del modello federato")
            print(f"2. Confrontare vulnerabilità con/senza tecniche di protezione")
            print(f"3. Discutere implicazioni degli attacchi riusciti")
            print(f"4. Proporre contromisure (es. differential privacy)")
            
            vuln_score = results['summary']['overall_vulnerability_score']
            if vuln_score > 50:
                print(f"\n⚠️  Il modello presenta vulnerabilità significative.")
                print(f"   Considera l'implementazione di tecniche di privacy-preserving.")
            elif vuln_score > 25:
                print(f"\n⚠️  Il modello presenta vulnerabilità moderate.")
                print(f"   Potrebbero essere necessarie contromisure aggiuntive.")
            else:
                print(f"\n✅ Il modello è relativamente resistente agli attacchi.")
                print(f"   L'approccio federato offre buona protezione della privacy.")
            
        else:
            print(f"\n❌ Errore durante esecuzione attacchi")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Esecuzione interrotta dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()