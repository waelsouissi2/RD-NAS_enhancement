RD-NAS-RL : L'Architecture Neuronale par Apprentissage Renforcé
📖 1. Le Contexte Théorique (Le "Pourquoi")

Dans le monde du Deep Learning, trouver la meilleure architecture (disposition des couches, types de convolutions) est un défi colossal.

    NAS Classique : On entraîne chaque modèle pour voir s'il est bon. C'est extrêmement lent et coûteux (des jours de calcul).

    One-Shot NAS : Au lieu d'entraîner 10 000 modèles, on entraîne un seul "Super-Réseau" qui contient toutes les possibilités. On pioche ensuite dedans.

    Zero-Shot NAS (Notre base) : On ne fait aucun entraînement. On utilise des "Proxies" (formules mathématiques) pour prédire si un modèle sera bon juste en regardant sa structure. C'est ce qu'utilise RD-NAS.

🧠 2. Notre Innovation : L'Agent Architecte

Le projet RD-NAS de base utilise souvent le hasard pour explorer les modèles. Nous avons remplacé ce hasard par une Intelligence Artificielle (Agent RL).
Les Composants Clés :

    L'Agent PPO (rl_agent.py) : Un algorithme de Reinforcement Learning (Proximal Policy Optimization). Il possède un réseau Acteur (qui choisit les opérations) et un réseau Critique (qui prédit la récompense).

    Le Transfer Learning (pretrain_rl.py) : On ne lance pas l'agent au hasard. On le pré-entraîne dans une simulation pour lui donner des "réflexes" de base.

    La Récompense par Proxy (rd_nas_core.py) : L'agent crée une architecture, reçoit une note instantanée (le Proxy), et modifie ses paramètres internes pour s'améliorer.

🛠️ 3. Structure Technique du Projet (Step-by-Step)
Étape 1 : L'École de Simulation

Avant de toucher aux vrais modèles, l'agent apprend la logique de la récompense dans un environnement virtuel.

    Fichier : pretrain_rl.py

    Action : L'agent apprend que certaines actions "mathématiques" sont meilleures que d'autres.

Étape 2 : La Création d'Architectures

L'agent entre dans l'espace de recherche NAS-Bench-201.

    Fichier : rd_nas_core.py

    Processus : L'agent choisit des opérations (convolutions, skip-connections).

    Le Signal : La fonction calculate_zero_cost_proxy simule un test mathématique ultra-rapide. L'agent utilise ce signal pour ajuster ses poids neuronaux.

Étape 3 : La Validation Scientifique

Pour savoir si notre agent a bien travaillé, nous utilisons un "Juge de Paix".

    L'Outil : NAS-Bench-201-v1_1-ss.pth.

    La Méthode : On regarde la corrélation de Kendall's Tau (τ). On compare ce que l'agent a "prédit" avec les vrais scores enregistrés dans le benchmark.

📊 4. Résultats et Comparaison

Le fichier run_experiments.py prouve l'efficacité de l'approche :

    Baseline : Recherche aléatoire (sans cerveau).

    Amélioration : Notre Agent RL + Transfer Learning.

    Constat : L'agent RL obtient une cohérence de classement nettement supérieure, prouvant qu'il a "compris" comment construire une bonne IA sans jamais l'entraîner réellement.

🚀 5. Guide d'Exécution rapide

    Pré-entraînement : python src/pretrain_rl.py (Génère l'intelligence de base).

    Expérience complète : python src/run_experiments.py (Lance la recherche et compare les résultats).
