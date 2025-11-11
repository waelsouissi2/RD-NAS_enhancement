# Fichier src/rl_agent.py (Design Final de l'Agent PPO)

import torch
import torch.nn as nn
import torch.nn.functional as F

# NAS-Bench-201 Architecture (simplifiée pour l'agent) :
# 5 bords (edges) sélectionnables entre 4 nœuds. Chaque bord a 5 opérations possibles.
# Espace d'Action : 5 (choix de l'opération)
# Espace d'État : Représentation séquentielle de l'architecture.

class PolicyNetwork(nn.Module):
    """Réseau Acteur (Policy) pour l'Agent PPO."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Un réseau simple pour simuler la politique
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Logits pour la distribution catégorielle des actions
        return self.fc2(x)

class RLPPOAgent:
    """
    Agent PPO pour la recherche d'architecture dans le Supernet RD-NAS.
    """
    # Dimensions pour NAS-Bench-201 (6 opérations possibles sur chaque bord, 6 bords dans un graph DAG)
    STATE_DIM = 6 * 6  # Représentation One-Hot ou embedding de l'architecture
    ACTION_DIM = 6     # 6 opérations possibles à choisir (ou 6 bords à modifier)

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        print(f"Agent RL finalisé : État dim={state_dim}, Action dim={action_dim}")

    def select_action(self, state):
        """Sélectionne la prochaine architecture à évaluer via le Policy Network."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits = self.policy_network(state_tensor)
        
        # Simulation d'échantillonnage de la distribution
        probs = F.softmax(logits, dim=-1)
        action_index = torch.multinomial(probs, 1).item()
        
        return action_index 

    def calculate_reward(self, current_proxy_rank, previous_proxy_rank):
        """
        Fonction de Récompense basée sur le classement du Proxy (cœur de l'intégration RD-NAS).
        
        Récompense : 
        - Positive si le nouveau classement (rank) est meilleur (plus bas).
        - Négative si le nouveau classement est moins bon (plus haut).
        """
        if current_proxy_rank < previous_proxy_rank:
            # Meilleur classement (plus proche du top)
            reward = 1.0 
        elif current_proxy_rank > previous_proxy_rank:
            # Moins bon classement
            reward = -0.5 # Pénalité moins forte que la récompense pour encourager l'exploration
        else:
            # Même classement
            reward = 0.0
            
        return reward

    def update_policy(self, experience_buffer):
        """Mise à jour des réseaux Acteur et Critique (à implémenter en S6/S7)."""
        pass
    
    def load_pretrained(self, path):
        """Charge les poids d'un pré-entraînement (Transfer Learning)."""
        pass
