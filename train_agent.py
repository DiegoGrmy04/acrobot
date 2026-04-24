import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from triple_acrobot import TripleAcrobotEnv

# Création des dossiers pour sauvegarder les modèles et les graphiques
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Initialisation de l'environnement d'entraînement
# On n'affiche pas la fenêtre (render_mode=None) pour que l'entraînement soit ultra-rapide
env = TripleAcrobotEnv(render_mode=None)
# Le wrapper Monitor enregistre les récompenses et la durée des épisodes
env = Monitor(env, "logs/train_log") 

# 2. Initialisation de l'environnement d'évaluation
# Permet de tester l'agent périodiquement pendant son entraînement
eval_env = Monitor(TripleAcrobotEnv(render_mode=None), "logs/eval_log")
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./models/',
    log_path='./logs/', 
    eval_freq=5000, # Évalue l'agent toutes les 5000 étapes
    deterministic=True, 
    render=False
)

# 3. Création du modèle IA (Deep Q-Network avec Decay)
# MlpPolicy : Utilise un réseau de neurones classique (Multi-Layer Perceptron)
# exploration_fraction : C'est le fameux "decay" ! L'agent va explorer au hasard 
# pendant les premiers 50% de l'entraînement, puis se fier de plus en plus à ce qu'il a appris.
model = DQN(
    "MlpPolicy", 
    env, 
    learning_rate=1e-3,
    buffer_size=50000,
    exploration_fraction=0.5, # L'exploration diminue sur 50% du temps total
    exploration_initial_eps=1.0, # Commence avec 100% d'exploration (au hasard)
    exploration_final_eps=0.05,  # Finit avec 5% d'exploration (très sûr de lui)
    verbose=1,
    tensorboard_log="./tensorboard_logs/" # Pour tracer les courbes d'apprentissage
)

# 4. Lancement de l'entraînement
print("Début de l'apprentissage (DQN sur Triple Acrobot)...")
# Note : 200 000 timesteps prennent quelques minutes. Pour de vrais résultats finaux, 
# vous devrez probablement monter à 500 000 ou 1 000 000.
model.learn(total_timesteps=200000, callback=eval_callback, tb_log_name="DQN_TripleAcrobot")
print("Apprentissage terminé !")

# 5. Sauvegarde du modèle final
model.save("models/dqn_triple_acrobot_final")
env.close()