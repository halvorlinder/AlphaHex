import math
import numpy as np
from API_token import API_TOKEN
from RL import get_neural_agents, NeuralAgent
from gen_agents import RandomAgent
from hex import Hex, HexState, HexMove
import CONSTANTS
from ANET import ConvResNet, PytorchNN

SIZE = 7

def get_MCTS_agent():
    net = ConvResNet(
                board_dimension_depth=game.conv_net_layers, 
                channels=CONSTANTS.CHANNELS_RES_BLOCK, 
                num_res_blocks=CONSTANTS.NUMBER_RES_BLOCKS, 
                board_state_length=game.state_representation_length, 
                move_cardinality=game.move_cardinality, 
            )
    pynet = PytorchNN()
    pynet.load(net, f'agents/hex7/MCTS/49')
    actor = NeuralAgent(pynet, f'50')
    return actor

game = Hex(SIZE)
# actor = RandomAgent(Hex(SIZE), "RandomMan")
# actor = get_neural_agents(game, CONSTANTS.NEURAL_AGENT_TIMESTAMP)[-1]
actor = get_MCTS_agent()

def rotate_matrix(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1,-1,-1)]

from ActorClient import ActorClient

class MyClient(ActorClient):

    def handle_game_start(self, start_player):
        self.start_player = start_player
        self.logger.info('Game start: start_player=%s', start_player)

    def handle_get_action(self, state):
        board = state[1:]
        if self.start_player == 2:
            board = [0 if p == 0 else 1 if p == 2 else 2 for p in board]
        board = [[board[j] for j in range(i, len(board), SIZE)][::-1]for i in range(SIZE)]
        if self.start_player == 1:
            board = rotate_matrix(board)
        gamestate = HexState.from_list(board)
        row, col = HexMove.from_int_representation(actor.get_next_move(gamestate), SIZE).as_tuple()
        if self.start_player == 2:
            return int(SIZE-1-col), int(row)
        if self.start_player == 1:
            return int(row), int(col)

if __name__ == '__main__':
    client = MyClient(auth=API_TOKEN, qualify=True)
    client.run()