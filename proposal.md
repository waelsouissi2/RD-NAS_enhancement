# Proposition d'Amélioration : RD-NAS + RL (Transfer Learning)

## Problématique

La méthode RD-NAS améliore la cohérence de classement du Supernet grâce à la distillation des proxys zéro-coût. Cependant, la phase de recherche d'architecture elle-même reste une simple exploration aléatoire ou une approche gourmande.

## Amélioration Proposée

Nous proposons d'intégrer un **Agent de Recherche basé sur l'Apprentissage par Renforcement (RL)**, avec une phase de **Transfer Learning** (inspiré par *Task Adaptation of Reinforcement Learning-Based NAS Agents...*).

1.  **Agent RL (PPO) :** L'agent remplacera la méthode d'échantillonnage pour sélectionner les architectures dans le Supernet.
2.  **Fonction de Récompense :** La récompense sera directement tirée de la **cohérence de classement distillée par RD-NAS**. Plus le proxy prédit un bon classement pour l'architecture choisie, plus la récompense est positive.
3.  **Transfer Learning :** L'agent RL sera **pré-entraîné** sur une tâche de NAS simple (e.g., recherche sur un sous-ensemble de données ou un autre petit benchmark) avant d'être transféré à la tâche principale de RD-NAS.

## Objectifs de l'Amélioration

- **Améliorer la Vitesse de Recherche :** L'agent RL pré-entraîné devrait trouver les meilleures architectures beaucoup plus rapidement que l'échantillonnage aléatoire.
- **Maintenir (ou améliorer) la Cohérence de Classement :** L'exploration intelligente par RL devrait se concentrer sur les régions du Supernet où la cohérence de classement est la plus forte.
