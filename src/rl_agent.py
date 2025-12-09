# Fichier src/rl_agent.py (Design Final de l'Agent PPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# NAS-Bench-201 Architecture (simplifiée pour l'agent) :
# 5 bords (edges) sélectionnables entre 4 nœuds. Chaque bord a 5 opérations possibles.
# Espace d'Action : 5 (choix de l'opération)
# Espace d'État : Représentation séquentielle de l'architecture.

class ActorCritic(nn.Module):
    """Réseaux Acteur (Policy) et Critique (Value) pour l'Agent PPO."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Réseau Acteur (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Output: Logits pour les actions
        )

        # Réseau Critique (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Valeur V(s) de l'état
        )

    def forward(self, x):
        # Pour une implémentation complète, on ne passe pas le forward ainsi, 
        # mais ici nous définissons les réseaux.
        pass

    def get_action_and_value(self, x):
        """Calcule l'action, son log-probabilité et la valeur V(s)."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), self.critic(x).squeeze(-1)

class RLPPOAgent:
    """
    Agent PPO pour la recherche d'architecture dans le Supernet RD-NAS.
    """
    STATE_DIM = 6 * 6 
    ACTION_DIM = 6     
    GAMMA = 0.99  # Facteur d'actualisation
    CLIP_EPS = 0.2 # PPO clipping parameter
    PPO_EPOCHS = 4 # Nombre d'époques d'optimisation sur le même buffer

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Utilisation du réseau ActorCritic
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Buffer pour stocker les expériences (états, actions, récompenses)
        self.buffer = [] 
        
        print(f"Agent PPO (Acteur-Critique) implémenté.")

    def select_action(self, state):
        """Sélectionne l'action via le réseau Acteur."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # La fonction get_action_and_value retourne tout
        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(state_tensor)
        
        # Stocke l'expérience pour l'apprentissage futur
        self.buffer.append((state, action.item(), log_prob.item(), value.item()))
        
        return action.item() 

    def calculate_reward(self, current_proxy_rank, previous_proxy_rank):
        """Fonction de Récompense (inchangée)."""
        if current_proxy_rank < previous_proxy_rank:
            reward = 1.0 
        elif current_proxy_rank > previous_proxy_rank:
            reward = -0.5 
        else:
            reward = 0.0
        return reward

    def update_policy(self, experience_buffer):
        """
        Mise à jour des réseaux Acteur et Critique via l'algorithme PPO.
        C'est ici que l'apprentissage par renforcement a lieu.
        """
        if not experience_buffer:
            return

        # 1. Préparation des Tenseurs
        # Conversion du buffer en tenseurs pour l'optimisation
        states, actions, log_probs_old, values, rewards, dones = zip(*experience_buffer)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 2. Calcul de la Value Target et des Avantages (A_t)
        # Simple GAE (Generalized Advantage Estimation) ou simple R_t
        # Pour une implémentation simple, on utilise le Monte Carlo Return
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.GAMMA * R 
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Avantages : La différence entre la récompense réelle (Return) et la prédiction du Critique (Value)
        advantages = returns - values 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Boucle d'Optimisation PPO
        for _ in range(self.PPO_EPOCHS):
            # Calcul des nouvelles log_probs et valeurs V
            logits = self.network.actor(states)
            new_probs = Categorical(logits=logits)
            log_probs_new = new_probs.log_prob(actions)
            values_new = self.network.critic(states).squeeze(-1)

            # Ratio de Probabilité
            ratio = torch.exp(log_probs_new - log_probs_old)

            # Perte de l'Acteur (Policy Loss) - PPO Clipping
            pg_loss1 = advantages * ratio
            pg_loss2 = advantages * torch.clamp(ratio, 1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
            actor_loss = -torch.min(pg_loss1, pg_loss2).mean()

            # Perte du Critique (Value Loss)
            critic_loss = F.mse_loss(values_new, returns)

            # Perte Totale (avec terme d'entropie pour encourager l'exploration)
            entropy_loss = new_probs.entropy().mean()
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss 

            # Optimisation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Le buffer est vidé après l'optimisation
        self.buffer = []

    def load_pretrained(self, path):
        """Charge les poids d'un pré-entraînement (Transfer Learning)."""
        pass
