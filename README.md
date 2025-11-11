# RD-NAS Enhancement Project

Ce projet vise à implémenter et améliorer l'approche RD-NAS (Ranking Distillation Neural Architecture Search) à l'aide de techniques d'apprentissage par renforcement (RL) et d'apprentissage par transfert (Transfer Learning), inspirées par les travaux de l'article "Task Adaptation of Reinforcement Learning-Based NAS Agents Through Transfer Learning".

## Objectif de l'Amélioration

Intégrer un agent de recherche PPO pré-entraîné pour optimiser l'exploration du Supernet, en se basant sur la cohérence de classement fournie par les proxys zéro-coût de RD-NAS.

## Structure du Projet

- `src/`: Contient le code source de la baseline RD-NAS et des modules d'amélioration.
- `reports/`: Contient les rapports hebdomadaires.
- `environment.yml`: Liste des dépendances.

## Baseline
À reproduire : RD-NAS sur NAS-Bench-201 (CIFAR-10).