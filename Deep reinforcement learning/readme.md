1️⃣ Removed Code Duplication

Before:

available_actions, sample_next_action, update() were written multiple times

Confusing and error-prone

Now:

Defined once and reused
✔ Cleaner and easier to debug

2️⃣ Replaced np.matrix with np.array

Why:

np.matrix is deprecated and causes unexpected broadcasting bugs

np.array is the modern standard

✔ Industry-safe & future-proof

3️⃣ Fixed Action Sampling Bug

❌ Original bug:

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_action, 1))


available_action was global, not function input.

✅ Fixed:

def choose_action(actions):
    return random.choice(actions)


✔ Correct logic, no hidden dependency

4️⃣ Clear Separation of Phases

Now code is logically structured:

Graph creation

Reward matrix

Q-learning

Testing

Environment-aware learning

✔ Easier to explain in exams / viva / project demo

5️⃣ Improved Variable Naming
Old	New
M	R (Reward matrix)
MATRIX_SIZE	N_STATES
update	update_q

✔ Matches standard RL terminology

6️⃣ Environment Awareness Is Now Explicit

Before:

Police & drug logic mixed inside Q update

Now:

collect_environment()

update_q_with_env()

✔ Easier to extend to penalties/rewards later

7️⃣ Better Plot Meaning

Before:

Reward normalization unclear

Now:

Plot shows total Q value growth
✔ Clear indicator of learning convergence

8️⃣ Deterministic Visualization
pos = nx.spring_layout(G, seed=42)


✔ Same graph layout every run (important for reports)

🧠 What This Code Demonstrates (Academically)

Q-Learning on graph-based environment

Reward propagation

Shortest-path discovery

Environment-aware decision making

Reinforcement learning fundamentals (no deep NN)



<img width="660" height="522" alt="image" src="https://github.com/user-attachments/assets/3a0897ed-7ba7-4679-9ed0-18e95d2f6e2c" />



Most Efficient Path:
[0, np.int64(1), np.int64(3), np.int64(9), np.int64(10)]


<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/33b2dff5-4f6b-48a4-a76e-cf9ffb0c5f0d" />


<img width="584" height="455" alt="image" src="https://github.com/user-attachments/assets/e186065f-3502-455d-ad8b-6c077cffb8fe" />
