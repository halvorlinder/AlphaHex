from enum import Enum, auto

class NetworkArchitecture(Enum):
    FF = auto()
    CONV = auto()
    RESNET = auto()

class SelectionPolicy(Enum):
    SAMPLE = auto()
    MAX = auto()

class TrainingGame(Enum):
    HEX = auto()
    C2 = auto()
    TTT = auto()

class Optimizer(Enum):
    ADAGRAD = auto()
    SGD = auto()
    RMSPROP = auto()
    ADAM = auto()
    ADAMW = auto()

class HiddenNodeActivation():
    LINEAR = auto()
    TANH = auto()
    RELU = auto()
    SIGMOID = auto()


# 0: no debugging
# 1: light debugging
# 2: heavy debugging
DEBUG_LEVEL = 0


# CONSTANTS IN NEURAL NET TRAINER
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 16
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# NET TOPOLOGY
NETWORK_ARCHITECTURE = NetworkArchitecture.CONV
OPTIMIZER = Optimizer.ADAM
HIDDEN_NODE_ACTIVATION = HiddenNodeActivation.RELU

# FF
LAYERS = [100, 100]
DROPOUT_RATE = 0.2

# CONSTANTS IN MCTS
MCTS_EPSILON = 0.2
GAME_MOVE_EPSILON = 0.1
SQRT_2 = 1.41
ROLLOUTS = 1000


# Multi threading
M_THREAD = True
CORES = 10

# Agent behaviour
AGENT_SELECTION_POLICY = SelectionPolicy.MAX

# Agent training 
GAMES_PER_SAVE = 100
NUM_SAVES = 5

GAME = TrainingGame.HEX

# Replay buffer
REPLAY_BUFFER_MAX_SIZE = 1000
REPLAY_BUFFER_MOVES_CHOSEN = 50

# Hex
HEX_SIZE = 4

# TOURNEY 
NEURAL_AGENT_TIMESTAMP = "2023-04-13_12:35"
TOURNEY_NUM_GAMES = 50
RANDOM_IN_TOURNEY = True

# Wandb
ENABLE_WANDB = False
