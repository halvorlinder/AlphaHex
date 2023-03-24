import torch
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
NUM_EPOCHS = 2
LEARNING_RATE = 0.005
BATCH_SIZE = 4
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# NET TOPOLOGY
NETWORK_ARCHITECTURE = NetworkArchitecture.FF

# FF
LAYERS = [512, 512, 512]


# CONSTANTS IN MCTS
MCTS_EPSILON = 0.2
GAME_MOVE_EPSILON = 0.1
SQRT_2 = 1.41
ROLLOUTS = 1000


# Multi threading
M_THREAD = True
CORES = 4


# Agent behaviour
AGENT_SELECTION_POLICY = SelectionPolicy.MAX

# Agent training 
GAMES_PER_SAVE = 100
NUM_SAVES = 3

GAME = TrainingGame.HEX

# Hex
HEX_SIZE = 3

# TOURNEY 
NEURAL_AGENT_TIMESTAMP = "2023-03-24_13:13"
TOURNEY_NUM_GAMES = 10
RANDOM_IN_TOURNEY = True
