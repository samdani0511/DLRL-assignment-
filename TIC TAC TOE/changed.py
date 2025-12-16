import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
PLAYER_X = 1
PLAYER_O = -1


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = PLAYER_X  # p1 starts

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_ROWS * BOARD_COLS))
        return self.boardHash

    def winner(self):
        # check rows and columns
        for i in range(BOARD_ROWS):
            if abs(sum(self.board[i, :])) == 3:
                self.isEnd = True
                return PLAYER_X if sum(self.board[i, :]) == 3 else PLAYER_O
            if abs(sum(self.board[:, i])) == 3:
                self.isEnd = True
                return PLAYER_X if sum(self.board[:, i]) == 3 else PLAYER_O

        # check diagonals
        diag1 = sum([self.board[i, i] for i in range(BOARD_ROWS)])
        diag2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_ROWS)])
        if abs(diag1) == 3 or abs(diag2) == 3:
            self.isEnd = True
            if diag1 == 3 or diag2 == 3:
                return PLAYER_X
            else:
                return PLAYER_O

        # check tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0  # tie

        return None  # game not finished

    def availablePositions(self):
        return [(i, j) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) if self.board[i, j] == 0]

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch player
        self.playerSymbol = PLAYER_O if self.playerSymbol == PLAYER_X else PLAYER_X

    def giveReward(self):
        result = self.winner()
        if result == PLAYER_X:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == PLAYER_O:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:  # tie
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = PLAYER_X

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 5000 == 0:
                print(f"Training round: {i}")
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p1.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                # Player 2
                positions = self.availablePositions()
                action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p2.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

    def playHuman(self):
        while not self.isEnd:
            positions = self.availablePositions()
            action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(action)
            self.showBoard()
            if self.winner() is not None:
                self.displayWinner()
                self.reset()
                break

            positions = self.availablePositions()
            action = self.p2.chooseAction(positions)
            self.updateState(action)
            self.showBoard()
            if self.winner() is not None:
                self.displayWinner()
                self.reset()
                break

    def showBoard(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                token = 'x' if self.board[i, j] == PLAYER_X else 'o' if self.board[i, j] == PLAYER_O else ' '
                out += token + ' | '
            print(out)
        print('-------------')

    def displayWinner(self):
        win = self.winner()
        if win == PLAYER_X:
            print(self.p1.name, "wins!")
        elif win == PLAYER_O:
            print(self.p2.name, "wins!")
        else:
            print("Tie!")

# --- Player Classes ---

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(BOARD_ROWS * BOARD_COLS))

    def chooseAction(self, positions, board, symbol):
        if np.random.uniform() <= self.exp_rate:
            return positions[np.random.choice(len(positions))]
        else:
            value_max = -999
            for p in positions:
                next_board = board.copy()
                next_board[p] = symbol
                hash_val = self.getHash(next_board)
                value = self.states_value.get(hash_val, 0)
                if value >= value_max:
                    value_max = value
                    action = p
            return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            self.states_value[st] = self.states_value.get(st, 0) + self.lr * (self.decay_gamma * reward - self.states_value.get(st, 0))
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open(f'policy_{self.name}', 'wb') as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, file):
        with open(file, 'rb') as f:
            self.states_value = pickle.load(f)

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            try:
                row = int(input("Row (0-2): "))
                col = int(input("Col (0-2): "))
                action = (row, col)
                if action in positions:
                    return action
            except:
                print("Invalid input. Try again.")

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass

# --- Main ---

if __name__ == "__main__":
    # Training AI vs AI
    p1 = Player("p1")
    p2 = Player("p2")
    game = State(p1, p2)
    print("Training AI...")
    game.play(50000)

    # Save policy
    p1.savePolicy()

    # Human vs AI
    ai_player = Player("computer", exp_rate=0)
    ai_player.loadPolicy("policy_p1")
    human = HumanPlayer("human")
    game = State(ai_player, human)

    cont = 'y'
    while cont.lower() == 'y':
        game.playHuman()
        cont = input("Play again? (y/n): ")
