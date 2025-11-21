# agent.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import pickle
import random
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Detectada y configurada: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("!!!! No se detectó GPU. Se usará la CPU (esto es normal si no tienes CUDA instalado).")

def build_model(input_dim, output_dim):
    """
    Crea la red neuronal que predice los valores Q.
    Entrada: 8 (sensores)
    Salida: 3 (Q-valor para cada acción: Avanzar, Izquierda, Derecha)
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0), loss='huber')
    return model

class ReplayBuffer:
    def __init__(self, max_size=20000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 8
        self.buffer = ReplayBuffer()
        self.model = build_model(state_dim, action_dim)
        self.target_model = build_model(state_dim, action_dim)
        self.update_target_model()
        self.model_weights_file = 'dqn_model_weights.h5'
        self.epsilon_file = 'dqn_epsilon.npy'
        self.buffer_file = 'dqn_buffer.pkl'

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = state[np.newaxis, :]
        q_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        q_next = self.target_model.predict(next_states, verbose=0)
        q_next_max = np.max(q_next, axis=1)
        
        targets = rewards + self.gamma * q_next_max * (1 - dones)
        q_current = self.model.predict(states, verbose=0)
        
        for i in range(self.batch_size):
            q_current[i, actions[i]] = targets[i]
        
        history = self.model.fit(states, q_current, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        return loss

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_state(self):
        print(f"Guardando estado... Epsilon actual: {self.epsilon}")
        self.model.save_weights(self.model_weights_file)
        np.save(self.epsilon_file, [self.epsilon])
        try:
            with open(self.buffer_file, 'wb') as f:
                pickle.dump(self.buffer, f)
            print(f"Buffer guardado (Tamaño: {len(self.buffer)})")
        except Exception as e:
            print(f"Error al guardar el buffer: {e}")

    def load_state(self):
        if os.path.exists(self.model_weights_file):
            print("Cargando estado guardado...")
            self.model.load_weights(self.model_weights_file)
            self.epsilon = np.load(self.epsilon_file)[0]
            print(f"Estado cargado. Epsilon reanudado en: {self.epsilon}")
            self.update_target_model()
        else:
            print("No se encontró estado guardado. Empezando de cero.")
        if os.path.exists(self.buffer_file):
            try:
                with open(self.buffer_file, 'rb') as f:
                    self.buffer = pickle.load(f)
                print(f"Buffer cargado (Tamaño: {len(self.buffer)})")
            except Exception as e:
                print(f"Error al cargar el buffer: {e}")