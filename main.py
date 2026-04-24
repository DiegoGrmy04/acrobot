import time
from triple_acrobot import TripleAcrobotEnv

# 1. On instancie notre environnement personnalisé directement
# render_mode="human" permet d'afficher la fenêtre PyGame
env = TripleAcrobotEnv(render_mode="human")

# 2. On réinitialise l'environnement pour obtenir l'état de départ
observation, info = env.reset()

print("Début de la simulation. L'agent joue au hasard !")

# 3. Boucle de test (ex: 500 étapes)
for step in range(500):
    # L'IA choisit une action au hasard (0, 1 ou 2)
    action = env.action_space.sample()
    
    # On applique l'action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Petite pause pour que l'affichage ne soit pas trop rapide à l'œil nu
    time.sleep(0.02)
    
    # Si le 3ème segment dépasse la ligne (victoire), on recommence
    if terminated:
        print(f"Objectif atteint en {step} étapes ! Réinitialisation...")
        observation, info = env.reset()
        time.sleep(1) # Pause d'une seconde pour observer la victoire

# 4. On ferme proprement la fenêtre
env.close()
print("Fin du test.")