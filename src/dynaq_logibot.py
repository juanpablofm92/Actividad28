"""
Actividad 28: Algoritmo Híbrido Dyna-Q para el Robot Móvil LogiBot-RL
Maestría en Inteligencia Artificial - TecNM / ITSU
Materia: Aprendizaje Automático
Autor: Juan Pablo Figueroa Moran (M26040059)
Repositorio: https://github.com/juanpablofm92/Actividad28.git
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class LogiBotWarehouseEnv:
    """
    Entorno de Almacén Logístico LogiBot-RL (Sutton & Barto GridWorld).
    Dimensiones: 6 filas x 9 columnas.
    Inicio: (2, 0)
    Bahía de Despacho (Meta): (0, 8) -> Recompensa +100.0
    Estanterías de Almacén (Obstáculos):
      - Columna 2: Filas 1, 2, 3
      - Columna 5: Fila 4
      - Columna 7: Filas 0, 1, 2
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Arriba, Abajo, Izquierda, Derecha
    ACTION_NAMES = ["Arriba", "Abajo", "Izquierda", "Derecha"]

    def __init__(self, rows: int = 6, cols: int = 9):
        self.rows = rows
        self.cols = cols
        self.start_pos = (2, 0)
        self.goal_pos = (0, 8)
        self.obstacles = {
            (1, 2), (2, 2), (3, 2),
            (4, 5),
            (0, 7), (1, 7), (2, 7)
        }
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action_idx: int):
        move = self.ACTIONS[action_idx]
        next_r = self.current_pos[0] + move[0]
        next_c = self.current_pos[1] + move[1]

        # Validar límites de la cuadrícula
        if 0 <= next_r < self.rows and 0 <= next_c < self.cols:
            # Validar que no choque con estanterías
            if (next_r, next_c) not in self.obstacles:
                self.current_pos = (next_r, next_c)

        if self.current_pos == self.goal_pos:
            reward = 100.0
            done = True
        else:
            reward = -1.0  # Costo por paso para fomentar la ruta más corta
            done = False

        return self.current_pos, reward, done


class DynaQAgent:
    """
    Agente Dyna-Q que integra Aprendizaje Directo por Refuerzo con Planificación Simulada.
    Basado en Sutton & Barto (Capítulo 8).
    """
    def __init__(self, actions: list, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, planning_steps_k: int = 10):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.k = planning_steps_k

        # Tabla Q: diccionario de diccionarios
        self.q_table = {}
        # Modelo del entorno aprendido: (s, a) -> (next_state, reward)
        self.model = {}
        # Registro de pares (estado, acción) ya experimentados
        self.observed_states = set()
        self.observed_actions_per_state = {}

    def get_q(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def set_q(self, state, action, value):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        self.q_table[state][action] = value

    def max_q(self, state):
        if state not in self.q_table:
            return 0.0
        return max(self.q_table[state].values())

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        if state not in self.q_table:
            return random.choice(self.actions)
        q_vals = self.q_table[state]
        max_v = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_v]
        return random.choice(best_actions)

    def train_step(self, s, a, r, s_prime, done):
        # 1. APRENDIZAJE DIRECTO DE Q (Direct RL)
        q_curr = self.get_q(s, a)
        max_q_next = 0.0 if done else self.max_q(s_prime)
        td_target = r + self.gamma * max_q_next
        new_q = q_curr + self.alpha * (td_target - q_curr)
        self.set_q(s, a, new_q)

        # 2. APRENDIZAJE DEL MODELO INTERNO: M(s, a) -> (s', r)
        self.model[(s, a)] = (s_prime, r, done)
        self.observed_states.add(s)
        if s not in self.observed_actions_per_state:
            self.observed_actions_per_state[s] = set()
        self.observed_actions_per_state[s].add(a)

        # 3. PLANIFICACIÓN SIMULADA (k pasos de 'ensayo mental')
        for _ in range(self.k):
            # Muestrear estado previamente experimentado
            sim_s = random.choice(list(self.observed_states))
            # Muestrear acción previamente ejecutada en sim_s
            sim_a = random.choice(list(self.observed_actions_per_state[sim_s]))

            sim_next, sim_r, sim_done = self.model[(sim_s, sim_a)]

            sim_q = self.get_q(sim_s, sim_a)
            sim_max_q = 0.0 if sim_done else self.max_q(sim_next)
            sim_td_target = sim_r + self.gamma * sim_max_q
            sim_new_q = sim_q + self.alpha * (sim_td_target - sim_q)
            self.set_q(sim_s, sim_a, sim_new_q)


def run_experiment(k_planning: int, num_episodes: int = 50, num_runs: int = 10, max_steps_per_ep: int = 2000):
    all_runs_steps = np.zeros((num_runs, num_episodes))
    actions = [0, 1, 2, 3]

    for run in range(num_runs):
        agent = DynaQAgent(actions=actions, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps_k=k_planning)
        env = LogiBotWarehouseEnv()

        for ep in range(num_episodes):
            s = env.reset()
            steps = 0
            done = False

            while not done and steps < max_steps_per_ep:
                steps += 1
                a = agent.choose_action(s)
                s_prime, r, done = env.step(a)
                agent.train_step(s, a, r, s_prime, done)
                s = s_prime

            all_runs_steps[run, ep] = steps

    return np.mean(all_runs_steps, axis=0)


def main():
    print("*" * 80)
    print(" TECNM / INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN - MAESTRÍA EN IA")
    print(" ACTIVIDAD 28: AGENTE DYNA-Q PARA LOGIBOT-RL (LOGÍSTICA)")
    print(" Alumno: Juan Pablo Figueroa Moran | Matrícula: M26040059")
    print("*" * 80)

    num_episodes = 50
    num_runs = 15

    print(f"\n[+] Ejecutando evaluación comparativa ({num_runs} réplicas, {num_episodes} episodios cada una)...")
    print("  -> Configuración A: Q-Learning Puro (k = 0 pasos de planificación)")
    qlearning_steps = run_experiment(k_planning=0, num_episodes=num_episodes, num_runs=num_runs)

    print("  -> Configuración B: Dyna-Q Híbrido (k = 10 pasos de planificación simulada)")
    dynaq_steps = run_experiment(k_planning=10, num_episodes=num_episodes, num_runs=num_runs)

    print("\n" + "=" * 80)
    print(" RESUMEN CUANTITATIVO DE RENDIMIENTO (PASOS PROMEDIO AL OBJETIVO)")
    print("=" * 80)
    print(f"{'Episodio':<12} | {'Q-Learning (k=0)':<20} | {'Dyna-Q (k=10)':<20} | {'Aceleración':<15}")
    print("-" * 75)
    sample_checkpoints = [1, 2, 5, 10, 20, 30, 40, 50]
    for ep in sample_checkpoints:
        idx = ep - 1
        q_val = qlearning_steps[idx]
        d_val = dynaq_steps[idx]
        speedup = f"{q_val / d_val:.1f}x" if d_val > 0 else "N/A"
        print(f"Episodio {ep:<4} | {q_val:18.1f} | {d_val:18.1f} | {speedup:<15}")
    print("-" * 75)

    # Gráfico comparativo de convergencia
    os.makedirs("docs", exist_ok=True)
    plt.figure(figsize=(11, 6))

    episodes = range(1, num_episodes + 1)
    plt.plot(episodes, qlearning_steps, label='Q-Learning Puro (k=0 planificación)', color='#DC2626', linewidth=2.5, marker='o', markersize=4)
    plt.plot(episodes, dynaq_steps, label='Dyna-Q Híbrido (k=10 planificación)', color='#2563EB', linewidth=2.5, marker='s', markersize=4)

    # Ruta óptima teórica (distancia Manhattan mínima sorteando obstáculos = 14 pasos)
    plt.axhline(14, color='#16A34A', linestyle='--', linewidth=2, label='Cota Óptima Teórica (~14 pasos)')

    plt.yscale('log')
    plt.title('Comparativa de Eficiencia Muestral: Dyna-Q ($k=10$) vs. Q-Learning Puro ($k=0$)\nEntorno de Almacén Automatizado LogiBot-RL', fontsize=12, fontweight='bold')
    plt.xlabel('Episodio de Aprendizaje', fontsize=11)
    plt.ylabel('Pasos para Alcanzar la Bahía de Despacho (Escala Log)', fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    out_file = os.path.join("docs", "comparativa_dynaq_vs_qlearning.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"\n[+] Gráfico comparativo generado exitosamente en: {out_file}")


if __name__ == "__main__":
    main()
