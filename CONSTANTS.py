from enum import Enum, auto

class NetworkArchitecture(Enum):
    FF = auto()
    CONV = auto()

class SelectionPolicy(Enum):
    SAMPLE = auto()
    MAX = auto()

class TrainingGame(Enum):
    HEX = auto()
    C2 = auto()
    TTT = auto()


# 0: no debugging
# 1: light debugging
# 2: heavy debugging
DEBUG_LEVEL = 0


# CONSTANTS IN NEURAL NET TRAINER
NUM_EPOCHS = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 4


# NET TOPOLOGY
NETWORK_ARCHITECTURE = NetworkArchitecture.FF

# FF
LAYERS = [100, 100, 100]


# CONSTANTS IN MCTS
MCTS_EPSILON = 0.1
GAME_MOVE_EPSILON = 0.1
SQRT_2 = 1.41
ROLLOUTS = 500


# Multi threading
M_THREAD = True
CORES = 4


# Agent behaviour
AGENT_SELECTION_POLICY = SelectionPolicy.MAX

# Agent training 
GAMES_PER_SAVE = 500
NUM_SAVES = 4

GAME = TrainingGame.TTT

# Hex
HEX_SIZE = 3

# TOURNEY 
NEURAL_AGENT_TIMESTAMP = "2023-03-03_9:51"
TOURNEY_NUM_GAMES = 50