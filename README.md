# 🤖 RL Project : The Triple Acrobot Challenge
**Cours :** LINFO2275 - Data mining and decision making (Partie II)  
**Auteurs :** [Vos Noms et N° de groupe]  

---

## 📑 Table des Matières
1. [Contexte et Objectifs](#1-contexte-et-objectifs)
2. [L'Environnement : Le Triple Acrobot](#2-lenvironnement--le-triple-acrobot)
3. [Démarche Scientifique et Algorithmes](#3-démarche-scientifique-et-algorithmes)
4. [🧪 Nos Découvertes Empiriques (Résultats)](#4--nos-découvertes-empiriques-résultats)
5. [Architecture du Projet](#5-architecture-du-projet)
6. [Installation et Prérequis](#6-installation-et-prérequis)
7. [Guide d'Utilisation](#7-guide-dutilisation)

---

## 1. Contexte et Objectifs
Dans le cadre de la seconde partie du cours LINFO2275, nous avons exploré les limites des algorithmes classiques d'Apprentissage par Renforcement (RL). Au lieu d'utiliser l'environnement standard `Acrobot-v1` (un pendule inversé à 2 segments, facilement résoluble), nous avons choisi de repousser les limites de la complexité en modélisant un **Acrobot à 3 segments**.

L'objectif de ce projet est d'étudier comment la **malédiction de la dimensionnalité**, couplée à un système physique chaotique et des récompenses "creuses" (*sparse rewards*), fait s'effondrer les algorithmes de base (comme le Deep Q-Learning), et comment l'ingénierie des récompenses et des algorithmes plus avancés (PPO) permettent de résoudre ce problème.

---

## 2. L'Environnement : Le Triple Acrobot
Le Triple Acrobot est un Processus de Décision Markovien (MDP) modélisant un pendule à trois liens suspendu dans le vide.

* **Sous-actionnement extrême :** Sur les 3 articulations, seul le moteur situé entre le 1er et le 2ème segment peut être contrôlé. Le 3ème segment est totalement libre et soumis aux forces d'inertie et de gravité.
* **Espace d'Actions (Discret) :** $\mathcal{A} \in \{-1, 0, 1\}$ (Couple négatif, nul, ou positif appliqué à la 2ème articulation).
* **Espace d'États (Continu - 9D) :** L'observation $\mathcal{S}$ contient 9 dimensions : $[\cos(\theta_1), \sin(\theta_1), \cos(\theta_2), \sin(\theta_2), \cos(\theta_3), \sin(\theta_3), \dot{\theta_1}, \dot{\theta_2}, \dot{\theta_3}]$.
* **Récompense standard (Sparse) :** $-1$ à chaque *timestep*. L'épisode se termine (victoire) lorsque l'extrémité du 3ème segment franchit une certaine hauteur. La limite de temps est fixée à 1000 étapes.

---

## 3. Démarche Scientifique et Algorithmes
Pour résoudre cet environnement, nous avons mené deux expérimentations majeures en utilisant la librairie `stable-baselines3`. Les entraînements ont été réalisés en vectorisant 4 environnements en parallèle (`SubprocVecEnv`) pour maximiser l'efficacité d'échantillonnage.

1. **La Baseline (DQN) :** Utilisation d'un Deep Q-Network classique avec une politique d'exploration $\epsilon$-greedy.
2. **L'Approche Avancée (PPO + Reward Shaping) :** Utilisation de l'algorithme on-policy *Proximal Policy Optimization* combiné à un `Wrapper` d'environnement modifiant la fonction de récompense.

---

## 4. 🧪 Nos Découvertes Empiriques (Résultats)

L'un des apports majeurs de ce projet est l'analyse de l'échec du DQN face à la complexité de l'environnement, comparé au succès du PPO.

### A. L'échec du DQN : L'Aiguille dans la Botte de Foin
Notre première expérimentation avec le DQN a abouti à un échec d'apprentissage total. Après 500 000 *timesteps*, la durée moyenne des épisodes (`ep_len_mean`) est restée bloquée au maximum (1000 étapes), et la récompense moyenne à -1000.

**Explication théorique :**
L'échec ne provient pas de l'architecture du réseau, mais de la nature du signal d'apprentissage. Avec un espace continu à 9 dimensions, la probabilité que l'agent parvienne à balancer les 3 segments au-dessus de la ligne d'objectif *par pur hasard* lors de sa phase d'exploration est infinitésimale.
Puisque l'agent ne rencontre jamais l'état de victoire, il ne reçoit qu'un flux infini de récompenses de `-1`. Le réseau de neurones déduit simplement que toutes les actions mènent au même résultat négatif. Le gradient s'aplatit, et l'apprentissage stagne.

### B. La Solution : Le *Reward Shaping*
Pour résoudre ce problème de *Sparse Reward* (récompense creuse), nous avons implémenté la technique du façonnage de récompense (*Reward Shaping*). 
Nous avons créé un `Gym Wrapper` qui modifie la récompense à chaque étape :
$$R'(s) = -1.0 + \text{Hauteur\_du\_segment\_3}(s)$$
Au lieu de chercher la victoire finale au hasard, l'agent reçoit un **signal dense**. S'il fait monter le pendule, il est moins pénalisé. S'il atteint la cible, il gagne un bonus massif (+100).

### C. Le Succès du PPO
Couplé à ce *Reward Shaping*, nous avons entraîné un agent PPO (Proximal Policy Optimization). PPO, en tant que méthode d'optimisation de politique (Actor-Critic), gère beaucoup mieux les dynamiques physiques continues que les méthodes basées sur la valeur (Q-Learning).

**Résultats obtenus :**
* L'agent PPO a convergé avec succès.
* En 480 000 étapes, la durée moyenne pour atteindre l'objectif (`ep_len_mean`) est passée de **1000 étapes à seulement ~330 étapes**.
* Visuellement, l'agent a développé une technique de "pompage" sophistiquée, accumulant de l'énergie cinétique avant de "fouetter" le troisième segment avec un timing parfait. La variance expliquée du Critique a atteint **96%**, prouvant que le réseau modélise parfaitement la gravité et l'élan du système.

---

## 5. Architecture du Projet

```text
projet2/
│
├── triple_acrobot.py       # Le cœur physique (Résolution Lagrangienne, Wrapper de récompense)
├── train_agent.py          # Script d'entraînement optimisé pour Kaggle/GPU (j'ai pas utilisé au final pcq trop long et donc j'ai mis sur kaggle)
├── enjoy_agent.py          # Script interactif pour visionner les parties
│
├── models/                 # Modèles sérialisés
│   ├── best_modelDQN.zip   # "Cerveau" ayant échoué (Preuve empirique n°1)
│   └── best_modelPPO.zip   # "Cerveau" victorieux (Preuve empirique n°2)
│
└── tensorboard_logs/       # Historique des métriques (Loss, Rewards, Episodes length) pas mis a jour
├── kaggle/                 # Script lancé sur kaggle pour trouver les modèles
│   ├── dqn_kaggle.py   # Script ayant échoué (Preuve empirique n°1)
│   └── ppo_kaggle.py   # Script victorieux (Preuve empirique n°2)

```
---

## 6. Architecture du Projet

```text
pip install gymnasium pygame
pip install "stable-baselines3[extra]" tensorboard# acrobot
```