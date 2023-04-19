from enum import Enum, auto

class NetworkArchitecture(Enum):
    FF = auto()
    CONV = auto()
    RESNET = auto()

class SelectionPolicy(Enum):
    SAMPLE = auto()
    MAX = auto()
    SMART_SAMPLE = auto()

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
NUM_EPOCHS = 1
LEARNING_RATE = 0.005
BATCH_SIZE = 32
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# NET TOPOLOGY
NETWORK_ARCHITECTURE = NetworkArchitecture.FF
OPTIMIZER = Optimizer.ADAMW
HIDDEN_NODE_ACTIVATION = HiddenNodeActivation.RELU

# FF
LAYERS = [512, 512, 512]
DROPOUT_RATE = 0.15

# RESNET
NUMBER_RES_BLOCKS = 3
CHANNELS_RES_BLOCK = 64

# CONSTANTS IN MCTS
MCTS_EPSILON = 0.15
GAME_MOVE_EPSILON = 0.1
SQRT_2 = 1.41
ROLLOUTS = 2500
MAX_ROLLOUT_TIME_SECONDS = 30
MIN_NUMBER_ROLLOUTS = 200 # must be > 1, has precedence over MAX_ROLLOUT_TIME_SECONDS


# Multi threading
M_THREAD = True
CORES = 10

# Agent behaviour
AGENT_SELECTION_POLICY = SelectionPolicy.SMART_SAMPLE

# Agent training 
GAMES_PER_SAVE = 50
NUM_SAVES = 10

GAME = TrainingGame.HEX

# Replay buffer
REPLAY_BUFFER_MAX_SIZE = 2048
REPLAY_BUFFER_MOVES_CHOSEN = 0
MCTS_MOVES_CHOSEN = 256

# Hex
HEX_SIZE = 7

# TOURNEY 
NEURAL_AGENT_TIMESTAMP = "2023-04-16_17:50"
TOURNEY_NUM_GAMES = 50
RANDOM_IN_TOURNEY = True

# Wandb
ENABLE_WANDB = False
