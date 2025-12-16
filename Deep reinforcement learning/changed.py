
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# -----------------------------
# Graph Definition
# -----------------------------
edges = [
    (0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
    (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
    (8, 9), (7, 8), (1, 7), (3, 9)
]

GOAL = 10
N_STATES = 11
GAMMA = 0.75
EPISODES = 1000

# -----------------------------
# Visualization
# -----------------------------
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, node_size=500)
plt.title("State Graph")
plt.show()

# -----------------------------
# Reward Matrix Initialization
# -----------------------------
R = np.full((N_STATES, N_STATES), -1)

for s, a in edges:
    R[s, a] = 100 if a == GOAL else 0
    R[a, s] = 100 if s == GOAL else 0

R[GOAL, GOAL] = 100

# -----------------------------
# Q Matrix Initialization
# -----------------------------
Q = np.zeros((N_STATES, N_STATES))

# -----------------------------
# Helper Functions
# -----------------------------
def available_actions(state):
    return np.where(R[state] >= 0)[0]

def choose_action(actions):
    return random.choice(actions)

def update_q(state, action):
    best_next = np.max(Q[action])
    Q[state, action] = R[state, action] + GAMMA * best_next

# -----------------------------
# Training Phase
# -----------------------------
scores = []

for _ in range(EPISODES):
    state = random.randint(0, N_STATES - 1)
    actions = available_actions(state)
    action = choose_action(actions)
    update_q(state, action)
    scores.append(np.sum(Q))

# -----------------------------
# Policy Testing
# -----------------------------
current_state = 0
path = [current_state]

while current_state != GOAL:
    next_state = np.argmax(Q[current_state])
    path.append(next_state)
    current_state = next_state

print("Most Efficient Path:")
print(path)

# -----------------------------
# Training Progress Plot
# -----------------------------
plt.plot(scores)
plt.xlabel("Episodes")
plt.ylabel("Total Q Value")
plt.title("Learning Progress")
plt.show()

# -----------------------------
# Environment Awareness Extension
# -----------------------------
police = [2, 4, 5]
drug_traces = [3, 8, 9]

env_police = np.zeros_like(Q)
env_drugs = np.zeros_like(Q)

def collect_environment(action):
    if action in police:
        return 'p'
    if action in drug_traces:
        return 'd'
    return None

def update_q_with_env(state, action):
    update_q(state, action)
    env = collect_environment(action)
    if env == 'p':
        env_police[state, action] += 1
    if env == 'd':
        env_drugs[state, action] += 1

# -----------------------------
# Training with Environment Help
# -----------------------------
scores_env = []

for _ in range(EPISODES):
    state = random.randint(0, N_STATES - 1)
    actions = available_actions(state)
    action = choose_action(actions)
    update_q_with_env(state, action)
    scores_env.append(np.sum(Q))

plt.plot(scores_env)
plt.xlabel("Episodes")
plt.ylabel("Total Q Value")
plt.title("Learning with Environment Awareness")
plt.show()

print("Police Matrix:")
print(env_police)
print("\nDrug Trace Matrix:")
print(env_drugs)
