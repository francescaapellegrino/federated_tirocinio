import os
import json
from glob import glob
import numpy as np

def read_jsons(directory, pattern="attack_results_client_*.json"):
    """Carica tutti i file JSON di attacco nella directory specificata."""
    return [json.load(open(f)) for f in glob(os.path.join(directory, pattern))]

def safe_mean(values):
    """Restituisce la media ignorando None e NaN."""
    arr = [v for v in values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
    return float(np.mean(arr)) if arr else None

def aggregate_means(json_list):
    metrics = {
        "mia_confidence_based_accuracy": [],
        "mia_combined_accuracy": [],
        "mia_privacy_breach_score": [],
        "property_success_rate": [],
        "property_estimated_attack_ratio": [],
        "property_estimation_accuracy": [],
        "property_properties_detected": [],
        "model_inv_information_leakage_score": [],
        "model_inv_avg_confidence": [],
        "attack_success_rate": [],
        "privacy_risk_score": [],
    }
    n = len(json_list)
    for data in json_list:
        mia = data.get("membership_inference", {})
        prop = data.get("property_inference", {})
        minv = data.get("model_inversion", {})
        recap = data.get("attack_summary", {})
        # Membership Inference
        metrics["mia_confidence_based_accuracy"].append(mia.get("confidence_based_accuracy"))
        metrics["mia_combined_accuracy"].append(mia.get("combined_accuracy"))
        metrics["mia_privacy_breach_score"].append(mia.get("privacy_breach_score"))
        # Property Inference
        metrics["property_success_rate"].append(prop.get("success_rate"))
        metrics["property_estimated_attack_ratio"].append(prop.get("estimated_attack_ratio"))
        metrics["property_estimation_accuracy"].append(prop.get("estimation_accuracy"))
        metrics["property_properties_detected"].append(prop.get("properties_detected"))
        # Model Inversion
        metrics["model_inv_information_leakage_score"].append(minv.get("information_leakage_score"))
        metrics["model_inv_avg_confidence"].append(minv.get("avg_confidence"))
        # Attack summary
        metrics["attack_success_rate"].append(recap.get("attack_success_rate"))
        metrics["privacy_risk_score"].append(recap.get("privacy_risk_score"))
    means = {k: safe_mean(v) for k, v in metrics.items()}
    means["num_clients"] = n

    # Spiegazioni dettagliate per ogni metrica
    explanations = {
        "num_clients": "Numero totale di client/fogli di risultato aggregati.",
        "mia_confidence_based_accuracy": "Percentuale media di dati per cui l'attacco di Membership Inference riesce a indovinare correttamente se un dato era nel training set del modello. Valori >0.5 indicano rischio privacy reale per il dataset federato.",
        "mia_combined_accuracy": "Accuratezza media ottenuta combinando le tecniche di attacco MIA. Valori >0.5 indicano che l'attacco è superiore al caso casuale.",
        "mia_privacy_breach_score": "Score medio che quantifica il rischio di violazione della privacy dovuto a MIA. 0 significa nessun rischio, 1 rischio massimo.",
        "property_success_rate": "Percentuale media di proprietà sensibili effettivamente individuate tramite property inference. Più alto è il valore, maggiore è il rischio di leakage di proprietà collettive.",
        "property_properties_detected": "Numero medio di proprietà sensibili (su tutte quelle possibili/testate) che l'attacco è riuscito a inferire correttamente.",
        "property_estimated_attack_ratio": "Proporzione media stimata di dati/clienti colpiti dall'attacco di property inference, secondo le stime dell'attaccante.",
        "property_estimation_accuracy": "Accuratezza media delle stime fatte dall'attaccante sulla frazione reale di dati/proprietà colpite. 1 = stima perfetta.",
        "model_inv_information_leakage_score": "Score medio che esprime quanto l'attacco di inversione è riuscito a ricostruire dati sensibili a partire dal modello. Più alto = maggiore leakage.",
        "model_inv_avg_confidence": "Confidenza media dei prototipi/gli input ricostruiti tramite inversione. Più alto = maggiore efficacia dell’attacco.",
        "attack_success_rate": "Percentuale media di attacchi (tra MIA, property, inversion) che hanno raggiunto il criterio di successo nei client analizzati.",
        "privacy_risk_score": "Score medio aggregato di rischio privacy finale, tenendo conto di tutte le tecniche di attacco usate."
    }

    # Costruisci dizionario finale con spiegazioni
    results_with_expl = {}
    for k in means:
        results_with_expl[k] = {
            "media": means[k],
            "spiegazione": explanations.get(k, "")
        }
    return results_with_expl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate means of malicious attack metrics with explanations.")
    parser.add_argument("--dir", type=str, default=".", help="Directory con i file JSON attacco.")
    parser.add_argument("--out", type=str, default="aggregated_attack_means.json", help="Nome file output JSON.")
    args = parser.parse_args()

    jsons = read_jsons(args.dir)
    if not jsons:
        print("Nessun file attack_results_client_*.json trovato!")
        exit(1)

    means_expl = aggregate_means(jsons)
    with open(args.out, "w") as f:
        json.dump(means_expl, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Medie delle metriche con spiegazioni salvate in {args.out}")