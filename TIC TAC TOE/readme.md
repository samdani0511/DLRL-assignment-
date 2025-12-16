Here’s a **professional README.md** for your improved Tic-Tac-Toe Reinforcement Learning project:

````markdown
# Tic-Tac-Toe Reinforcement Learning (Python)

## Overview
This project implements a **Tic-Tac-Toe game** where an AI agent learns to play optimally using **reinforcement learning (RL)**.  
The AI learns through self-play and can play against a human using a trained policy.

---

## Features

- **Reinforcement Learning AI**: Learns optimal moves using state-value updates.
- **Human vs Computer Mode**: Play against the trained AI.
- **Self-Play Training**: AI trains by playing against itself.
- **Policy Saving and Loading**: Save trained policies to reuse in future sessions.
- **Board Hashing**: Unique board states stored for efficient learning.
- **Reward Backpropagation**: AI updates the value of all states visited in a game based on game outcome.

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd tic-tac-toe-rl
````

2. Install dependencies:

```bash
pip install numpy
```

---

## How to Use

### Training AI

```python
p1 = Player("p1")
p2 = Player("p2")
game = State(p1, p2)
game.play(rounds=50000)  # Train AI with self-play
p1.savePolicy()           # Save learned policy
```

### Play Against AI

```python
ai_player = Player("computer", exp_rate=0)
ai_player.loadPolicy("policy_p1")
human = HumanPlayer("human")
game = State(ai_player, human)

game.playHuman()  # Start the game with human
```

* `exp_rate=0` ensures AI uses only the learned policy without random moves.
* Human inputs row and column (0-indexed) to make a move.

---

## How It Works

1. **State Representation**: Board is stored as a 3x3 numpy array; empty cells = 0, X = 1, O = -1.
2. **State Hashing**: Each board state is converted to a unique string for AI evaluation.
3. **AI Decision Making**:

   * **Epsilon-Greedy Policy**: Chooses random moves with probability `exp_rate` to explore.
   * Otherwise, selects the move with the highest state value.
4. **Reward System**:

   * Win: 1 point
   * Loss: 0 points
   * Tie: 0.5 points
5. **Learning**: Uses **temporal-difference (TD) learning** to update state values after each game.

---

## Project Structure

```
tic-tac-toe-rl/
│
├── tictactoe.py          # Main game logic and RL AI
├── policy_p1             # Saved AI policy after training
└── README.md
```

---

## Improvements in This Version

* Cleaned and modularized code for readability and maintainability.
* Fixed initialization and player switching issues.
* Corrected diagonal and tie-checking logic.
* Added clear constants for player symbols.
* Enhanced reward propagation logic.
* Better handling of human inputs.
* Progress display during training every 5000 rounds.

---


<img width="220" height="268" alt="image" src="https://github.com/user-attachments/assets/412374f3-2442-4e2b-b9e3-61b77366b26f" />
<img width="202" height="339" alt="image" src="https://github.com/user-attachments/assets/3d927d95-144b-4957-afcd-33f08f61de89" />

