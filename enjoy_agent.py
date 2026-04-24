"""
Script de test universel pour le Triple Acrobot.
Permet de visualiser et comparer facilement les modèles entraînés (DQN et PPO).
"""

import time
import os
# On importe les deux algorithmes
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

# Importation de l'environnement depuis votre fichier local
from triple_acrobot import TripleAcrobotEnv 

# ==========================================
# 1. PRÉPARATION DE L'ENVIRONNEMENT
# ==========================================
def make_env():
    env = TripleAcrobotEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=1000)
    return env

vec_env = DummyVecEnv([make_env])

# ==========================================
# 2. MENU INTERACTIF (Choix du modèle)
# ==========================================
print("\n" + "="*50)
print("🕹️  SIMULATEUR TRIPLE ACROBOT  🕹️")
print("="*50)
print("Quel modèle souhaitez-vous tester ?")
print("1. DQN (Deep Q-Network - Modèle de base / Échec)")
print("2. PPO (Proximal Policy Optimization - Succès avec Reward Shaping)")
choix = input("Entrez 1 ou 2 : ")

# ==========================================
# 3. CHARGEMENT DYNAMIQUE
# ==========================================
if choix == "1":
    chemin_fichier = "models/best_modelDQN"
    print(f"\n🧠 Chargement du cerveau DQN depuis '{chemin_fichier}'...")
    try:
        model = DQN.load(chemin_fichier)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {chemin_fichier}.zip est introuvable.")
        exit()

elif choix == "2":
    chemin_fichier = "models/best_modelPPO"
    print(f"\n🧠 Chargement du cerveau PPO depuis '{chemin_fichier}'...")
    try:
        model = PPO.load(chemin_fichier)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {chemin_fichier}.zip est introuvable.")
        exit()
else:
    print("❌ Choix invalide. Veuillez relancer le script.")
    exit()

# ==========================================
# 4. BOUCLE DE SIMULATION
# ==========================================
obs = vec_env.reset()
print("\n▶️  Démarrage de l'animation ! Observez la technique de l'agent...\n")

# On laisse tourner la simulation pour 3000 étapes maximum
for i in range(3000):
    # L'agent prédit l'action optimale de manière déterministe
    action, _states = model.predict(obs, deterministic=True)
    
    # On applique l'action
    obs, reward, done, info = vec_env.step(action)
    
    # Pause pour la fluidité visuelle
    time.sleep(0.02) 
    
    # Si le bout du 3ème segment dépasse la ligne ou si le temps est écoulé
    if done:
        if choix == "2":
            print("🏆 Épisode terminé (L'agent PPO a probablement réussi !) Réinitialisation...")
        else:
            print("⏱️ Épisode terminé (L'agent DQN a probablement atteint la limite de temps). Réinitialisation...")
            
        time.sleep(1.5) # Pause pour observer la position finale

vec_env.close()
print("Fin de la simulation.")