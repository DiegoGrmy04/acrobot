"""
Script de test universel pour le Triple Acrobot.
Permet de visualiser et comparer facilement les modèles entraînés (DQN, PPO, REINFORCE).
"""

import time

import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from reinforce_policy import PolicyNetwork
from triple_acrobot import TripleAcrobotEnv


def make_env():
    env = TripleAcrobotEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=1000)
    return env


print("\n" + "=" * 50)
print("🕹️  SIMULATEUR TRIPLE ACROBOT  🕹️")
print("=" * 50)
print("Quel modèle souhaitez-vous tester ?")
print("1. DQN       (Deep Q-Network — baseline / échec)")
print("2. PPO       (Proximal Policy Optimization — succès avec reward shaping)")
print("3. REINFORCE (Policy gradient Monte-Carlo — Williams 1992)")
choix = input("Entrez 1, 2 ou 3 : ")

vec_env = DummyVecEnv([make_env])

if choix == "1":
    chemin = "models/best_modelDQN"
    print(f"\n🧠 Chargement DQN depuis '{chemin}'...")
    model = DQN.load(chemin)
    obs = vec_env.reset()
    predict = lambda obs: model.predict(obs, deterministic=True)[0]

elif choix == "2":
    chemin = "models/best_modelPPO"
    print(f"\n🧠 Chargement PPO depuis '{chemin}'...")
    model = PPO.load(chemin)
    obs = vec_env.reset()
    predict = lambda obs: model.predict(obs, deterministic=True)[0]

elif choix == "3":
    chemin = "models/best_modelREINFORCE.pt"
    print(f"\n🧠 Chargement REINFORCE depuis '{chemin}'...")
    ckpt = torch.load(chemin, map_location="cpu")
    hidden = ckpt.get("hyper", {}).get("hidden", 64)
    policy = PolicyNetwork(obs_dim=9, n_actions=3, hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    obs = vec_env.reset()

    def predict(obs):
        obs_t = torch.as_tensor(obs[0], dtype=torch.float32)
        return [policy.predict(obs_t, deterministic=True)]

else:
    print("❌ Choix invalide. Veuillez relancer le script.")
    raise SystemExit(1)

print("\n▶️  Démarrage de l'animation...\n")
for i in range(3000):
    action = predict(obs)
    obs, reward, done, info = vec_env.step(action)
    time.sleep(0.02)
    if done[0]:
        if choix == "1":
            print("⏱️ Épisode terminé (DQN — probablement timeout).")
        else:
            print("🏆 Épisode terminé (objectif probablement atteint).")
        time.sleep(1.5)

vec_env.close()
print("Fin de la simulation.")
