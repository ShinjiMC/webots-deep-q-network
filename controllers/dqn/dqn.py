# evaluation.py
from controller import Supervisor
from environment import EPuckEnv
from agent import DQNAgent # Necesita agent.py y environment.py
import sys
import os
import numpy as np

# --- CONFIGURACIÓN ---
TIME_STEP = 64
MAX_STEPS_PER_EPISODE = 1000
STATE_DIM = 8
ACTION_DIM = 3

BEST_MODEL_FILE = 'dqn_best_model.h5' 

supervisor = Supervisor()
env = EPuckEnv(supervisor, TIME_STEP)
agent = DQNAgent(STATE_DIM, ACTION_DIM)

if os.path.exists(BEST_MODEL_FILE):
    print("--- MODO EVALUACIÓN ---")
    print(f"Cargando el MEJOR modelo desde: {BEST_MODEL_FILE}")
    agent.model.load_weights(BEST_MODEL_FILE) 
    agent.epsilon = 0.0 
    print("Epsilon fijado en 0.0 (Modo 'Greedy'). No hay aleatoriedad.")
else:
    print(f"Error: No se encontró el archivo del mejor modelo: {BEST_MODEL_FILE}")
    print("Por favor, ejecuta el entrenamiento (dqn.py) primero.")
    sys.exit()

supervisor.step(TIME_STEP) 
state = env.get_state()
total_reward = 0
step_counter = 0

print("Iniciando bucle de evaluación. El robot repetirá el camino óptimo...")

while supervisor.step(TIME_STEP) != -1:
    action = agent.choose_action(state)
    env.apply_action(action)
    next_state = env.get_state()
    reward, done = env.get_reward_and_done(action, state)
    total_reward += reward
    state = next_state
    step_counter += 1
    if done or step_counter > MAX_STEPS_PER_EPISODE:
        if done:
            print(f"¡META ALCANZADA! Pasos: {step_counter}, Recompensa Final: {total_reward:.2f}")
        else:
             print(f"¡TIEMPO LÍMITE! Pasos: {step_counter}, Recompensa Final: {total_reward:.2f}")
        print("--- Reiniciando mundo para repetir la demostración ---")
        supervisor.worldReload()
        supervisor.step(TIME_STEP)
        state = env.get_state()
        total_reward = 0
        step_counter = 0