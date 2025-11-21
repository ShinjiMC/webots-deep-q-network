# my_controller.py
from controller import Supervisor
from environment import EPuckEnv
from agent import DQNAgent
import sys
import os
import numpy as np
import time

LOG_FILE = "training_log.txt"
METRICS_LOG_FILE = "training_metrics.txt"

supervisor = Supervisor()
TIME_STEP = 64
MAX_STEPS_PER_EPISODE = 1000
UPDATE_TARGET_EVERY = 100
STATE_DIM = 8
ACTION_DIM = 3

env = EPuckEnv(supervisor, TIME_STEP)
agent = DQNAgent(STATE_DIM, ACTION_DIM)

BEST_MODEL_FILE = 'dqn_best_model.h5'
BEST_METRICS_FILE = 'dqn_best_metrics.npy'

if os.path.exists(BEST_METRICS_FILE):
    print("Cargando métricas del mejor modelo...")
    metrics = np.load(BEST_METRICS_FILE)
    if len(metrics) == 3:
        BEST_TIME = metrics[0]
        BEST_REWARD = metrics[1]
        BEST_AVG_LOSS = metrics[2]
    elif len(metrics) == 2:
        print("Detectado archivo de métricas antiguo (2 valores). Actualizando a 3.")
        BEST_TIME = np.inf
        BEST_REWARD = metrics[0]
        BEST_AVG_LOSS = metrics[1]
    else:
        print("Error: Archivo de métricas corrupto. Empezando de cero.")
        BEST_TIME = np.inf
        BEST_REWARD = -np.inf
        BEST_AVG_LOSS = np.inf
    print(f"Mejor Tiempo cargado: {BEST_TIME:.2f} segundos")
    print(f"Mejor Recompensa cargada: {BEST_REWARD:.2f}")
else:
    print("No se encontraron métricas. Empezando 'Mejor Modelo' de cero.")
    BEST_TIME = np.inf
    BEST_REWARD = -np.inf
    BEST_AVG_LOSS = np.inf

if not os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, "w") as f:
            f.write("Epsilon | Pasos | Resultado | Recompensa_Total\n")
            f.write("------------------------------------------\n")
    except Exception as e:
        print(f"Error al crear el encabezado del log: {e}")

if not os.path.exists(METRICS_LOG_FILE):
    try:
        with open(METRICS_LOG_FILE, "w") as f:
            f.write("Avg_Loss | Total_Reward | Duration\n")
            f.write("----------------------------------------\n")
    except Exception as e:
        print(f"Error al crear el log de métricas: {e}")

agent.load_state()
print(f"--- Iniciando Episodio (Epsilon actual: {agent.epsilon:.3f}) ---")
print(f"--- Tamaño del Buffer: {len(agent.buffer)} ---")
start_time = time.time()
supervisor.step(TIME_STEP)
state = env.get_state()
total_reward = 0
step_counter = 0
episode_losses = []

while supervisor.step(TIME_STEP) != -1:
    action = agent.choose_action(state)
    env.apply_action(action)
    next_state = env.get_state()
    reward, done = env.get_reward_and_done(action, state)
    total_reward += reward
    agent.store_transition(state, action, reward, next_state, done)
    loss = agent.learn()
    if loss is not None:
        episode_losses.append(loss)
        if step_counter % 20 == 0:
            print(f"Paso {step_counter}, Recompensa: {total_reward:.2f}, Loss actual: {loss:.4f}")
    state = next_state
    step_counter += 1
    if step_counter % UPDATE_TARGET_EVERY == 0:
        agent.update_target_model()
    if done or step_counter > MAX_STEPS_PER_EPISODE:
        end_time = time.time()
        duration_sec = end_time - start_time
        minutes, seconds = divmod(int(duration_sec), 60)
        duration_formatted = f"{minutes:02d}:{seconds:02d}"
        reason = "Meta" if done else "Fracaso"
        try:
            current_epsilon = agent.epsilon
            log_line = f"{current_epsilon:.4f} | {step_counter} | {reason} | {total_reward:.2f}\n"
            with open(LOG_FILE, "a") as f:
                f.write(log_line)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            if np.isnan(avg_loss): avg_loss = np.inf
            avg_loss_str = "nan" if avg_loss == np.inf else f"{avg_loss:.6f}"
            
            metrics_line = f"{avg_loss_str} | {total_reward:.2f} | {duration_formatted}\n"
            with open(METRICS_LOG_FILE, "a") as f:
                f.write(metrics_line)
        except Exception as e:
            print(f"Error escribiendo logs: {e}")
            
        if done:
            print(f"¡META ALCANZADA! Pasos: {step_counter}, Recompensa Final: {total_reward:.2f}")
        else:
            print(f"¡TIEMPO LÍMITE! Pasos: {step_counter}, Recompensa Final: {total_reward:.2f}")
        agent.decay_epsilon()
        agent.save_state()
        if reason == "Meta" and duration_sec < BEST_TIME:
            print("\n--- ¡NUEVO RÉCORD DE VELOCIDAD! ---")
            print(f"Tiempo: {duration_sec:.2f}s (Récord anterior: {BEST_TIME:.2f}s)")
            
            BEST_TIME = duration_sec
            BEST_REWARD = total_reward  
            BEST_AVG_LOSS = avg_loss    
            
            agent.model.save_weights(BEST_MODEL_FILE)
            print(f"Mejor modelo guardado en: {BEST_MODEL_FILE}")
            metrics_to_save = np.array([BEST_TIME, BEST_REWARD, BEST_AVG_LOSS])
            np.save(BEST_METRICS_FILE, metrics_to_save)
            print("Métricas actualizadas.\n")
        else:
            print("Info: No es un nuevo récord.")
            if reason != "Meta":
                print("  (Razón: No llegó a la meta)")
            elif duration_sec >= BEST_TIME:
                print(f"  (Razón: Tiempo {duration_sec:.2f}s no supera el récord de {BEST_TIME:.2f}s)")
        
        if agent.epsilon <= agent.epsilon_min:
            print("--- ENTRENAMIENTO COMPLETADO (Epsilon Mínimo alcanzado) ---")
            print("Cerrando controlador.")
            break
        else:
            print("--- Reiniciando para el próximo episodio... ---")
            supervisor.worldReload()
            supervisor.step(TIME_STEP)
            state = env.get_state()
            total_reward = 0
            step_counter = 0
            episode_losses = []
            start_time = time.time()
            print(f"--- Iniciando Episodio (Epsilon actual: {agent.epsilon:.3f}) ---")
            print(f"--- Tamaño del Buffer: {len(agent.buffer)} ---")