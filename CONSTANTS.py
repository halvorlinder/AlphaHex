import torch

# 0: no debugging
# 1: light debugging
# 2: heavy debugging
DEBUG_LEVEL = 0

# CONSTANTS IN NEURAL NET TRAINER
NUM_EPOCHS = 1
LEARNING_RATE = 0.05
BATCH_SIZE = 32
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# CONSTANTS IN MCTS
MCTS_EPSILON = 0.2
GAME_MOVE_EPSILON = 0.2
SQRT_2 = 1.41
ROLLOUTS = 200

# Multi threading
M_THREAD = True
CORES = 10