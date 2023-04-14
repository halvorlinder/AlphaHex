import math
import numpy as np
from API_token import API_TOKEN
from RL import get_neural_agents
from gen_agents import RandomAgent
from hex import Hex, HexState, HexMove
import CONSTANTS

SIZE = 7

game = Hex(SIZE)
# actor = RandomAgent(Hex(SIZE), "RandomMan")
actor = get_neural_agents(game, CONSTANTS.NEURAL_AGENT_TIMESTAMP)[0]

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
        gamestate = HexState.from_list(board)
        row, col = HexMove.from_int_representation(actor.get_next_move(gamestate), SIZE).as_tuple()
        print(row, col)
        return SIZE-1-col, row

if __name__ == '__main__':
    client = MyClient(auth=API_TOKEN, qualify=False)
    client.run()