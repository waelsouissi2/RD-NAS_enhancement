# Fichier src/run_experiments.py (Version corrigée)

from rd_nas_core import evaluate_ranking_consistency
from rl_agent import RLPPOAgent
import numpy as np

def run_comparison(num_runs=3):
    """
    Exécute et compare la Baseline RD-NAS (aléatoire) et l'approche Améliorée (RL+Transfert).
    Utilise les vrais calculs issus de rd_nas_core.py.
    """
    baseline_tau_results = []
    rl_enhanced_tau_results = []
    
    print("--- Démarrage de l'Évaluation Réelle ---")
    
    for i in range(num_runs):
        print(f"\n--- RUN {i+1}/{num_runs} ---")
        
        # 1. Exécution de la Baseline (Aléatoire)
        # On passe agent=None pour utiliser la recherche aléatoire
        np.random.seed(42 + i)
        print("Calcul de la Baseline...")
        baseline_tau = evaluate_ranking_consistency(agent=None, num_samples=100)
        baseline_tau_results.append(baseline_tau)
        
        # 2. Exécution de l'Amélioration (RL + Transfert)
        # On crée un vrai agent PPO qui va apprendre durant le run
        np.random.seed(42 + i)
        print("Calcul de l'Agent RL (PPO)...")
        agent = RLPPOAgent(learning_rate=3e-5) 
        
        # Ici, l'agent va vraiment modifier ses paramètres via agent.update_policy
        rl_tau = evaluate_ranking_consistency(agent=agent, num_samples=100)
        rl_enhanced_tau_results.append(rl_tau)

    # --- AFFICHAGE DES VRAIS RÉSULTATS ---
    
    avg_baseline = np.mean(baseline_tau_results)
    avg_rl = np.mean(rl_enhanced_tau_results)
    
    print("\n" + "="*30)
    print("      RÉSULTATS FINAUX       ")
    print("="*30)
    print(f"Moyenne Baseline (Aléatoire) : {avg_baseline:.4f}")
    print(f"Moyenne RL + Transfert       : {avg_rl:.4f}")
    print("-" * 30)
    
    improvement = ((avg_rl - avg_baseline) / avg_baseline) * 100
    print(f"Amélioration relative : {improvement:.2f}%")
    
    if avg_rl > avg_baseline:
        print("\nConclusion : L'agent RL a appris à maximiser le Proxy plus efficacement que le hasard.")
    else:
        print("\nConclusion : L'agent nécessite plus d'échantillons ou de pré-entraînement.")

if __name__ == '__main__':
    run_comparison(num_runs=3)