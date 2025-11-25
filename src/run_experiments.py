# Fichier src/run_experiments.py

from rd_nas_core import evaluate_ranking_consistency
from rl_agent import RLPPOAgent
import numpy as np

def run_comparison(num_runs=3):
    """
    Exécute et compare la Baseline RD-NAS (aléatoire) et l'approche Améliorée (RL+Transfert).
    """
    baseline_tau_results = []
    rl_enhanced_tau_results = []
    
    print("--- Démarrage de l'Évaluation Comparée (Semaine 8) ---")
    
    for i in range(num_runs):
        print(f"\n--- RUN {i+1}/{num_runs} ---")
        
        # 1. Run Baseline (Aléatoire)
        np.random.seed(42 + i)
        baseline_tau = evaluate_ranking_consistency(agent=None, num_samples=200)
        baseline_tau_results.append(baseline_tau)
        
        # 2. Run Amélioration (RL + Transfert)
        np.random.seed(42 + i)
        agent = RLPPOAgent(learning_rate=3e-5) # Utilisation d'un learning rate plus bas
        rl_tau = evaluate_ranking_consistency(agent=agent, num_samples=200)
        rl_enhanced_tau_results.append(rl_tau)

    # Affichage des résultats finaux (Simulés)
    avg_baseline = np.mean(baseline_tau_results)
    avg_rl = np.mean(rl_enhanced_tau_results)
    
    print("\n==============================================")
    print(f"Moyenne Cohérence Baseline (Tau): {avg_baseline:.4f}")
    print(f"Moyenne Cohérence Améliorée (Tau): {avg_rl:.4f} (Gain: {avg_rl - avg_baseline:.4f})")
    print("==============================================")


if __name__ == '__main__':
    # Simuler le débogage PPO en modifiant les paramètres si le run échoue
    run_comparison()
