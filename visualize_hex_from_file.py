from hex import HexState, HexMove
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


with open("examples.txt", "r") as f:
    data = f.readlines()

for i, d in enumerate(data):
    data[i] = d.strip().split(";")

for i in range(0, len(data), 2):
    data[i] = [int(x) for x in data[i]]
    data[i+1] = [float(x) for x in data[i+1]]

def visualize_game_board(gameboard):
    board_size = math.isqrt(len(gameboard))
    arr = list(np.array(gameboard).reshape(board_size, board_size))
    for i, sub_arr in enumerate(arr):
        arr[i] = list(sub_arr)
    gs = HexState.from_list(arr)
    gs.plot()

def create_heat_map_from_prob_dist(prob_dist):
    pass

visualize_game_board(data[6])


