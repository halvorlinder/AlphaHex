# 0: no debugging
# 1: light debugging
# 2: heavy debugging
from enum import Enum, auto


DEBUG_LEVEL = 0

# CONSTANTS IN NEURAL NET TRAINER
NUM_EPOCHS = 3
LEARNING_RATE = 0.05
BATCH_SIZE = 4


# CONSTANTS IN MCTS
MCTS_EPSILON = 0.1
GAME_MOVE_EPSILON = 0.1
SQRT_2 = 1.41
ROLLOUTS = 100

# Multi threading
M_THREAD = True
CORES = 4

# Agent behaviour
class SelectionPolicy(Enum):
    SAMPLE = auto()
    MAX = auto()

AGENT_SELECTION_POLICY = SelectionPolicy.MAX