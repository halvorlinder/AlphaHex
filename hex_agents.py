from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING
from agent import SpecializedAgent
from hex import HexState, HexMove, Piece, Hex
from connect2 import Connect2Gamestate, Connect2Move
from MCTS import MCTS, Node
if TYPE_CHECKING:
    from agent import Agent

from random import randrange
import numpy as np

# class HexAgentType(Enum):
#     RANDOM = auto()
#     HUMAN = auto()

#     def construct_agent(self, params : dict['str', 'str']) -> Agent:
#         match self:
#             case HexAgentType.RANDOM:
#                 return RandomHexAgent(params)
#             case HexAgentType.HUMAN:
#                 return HumanHexAgent(params)
        
# @dataclass
# class HexAgentConfig:
#     agent_type : HexAgentType
#     params : dict['str', 'str']

# @dataclass
# class HexAgentsConfig:
#     agent_1: HexAgentConfig
#     agent_2: HexAgentConfig

class RandomHexAgent(SpecializedAgent):

    def __init__(self, name) -> None:
        self.name = name

    def get_next_move(self, gamestate: HexState) -> int:
        while True:
            n = randrange(0, gamestate.board_size*gamestate.board_size)
            if gamestate.is_open(gamestate.create_move(n)):
                return n

class MCTSHexAgent(SpecializedAgent):

    def __init__(self, name : str, n_rollouts : int, board_size : int) -> None:
        self.name = name
        self.n = n_rollouts
        self.game = Hex(board_size)

    def get_next_move(self, gamestate: HexState) -> int:
        mcts = MCTS(self.game, root = Node(gamestate))
        probs = mcts.run_simulations(self.n) 
        move = np.argmax(probs)
        return move
            
class RandomConnect2Agent(SpecializedAgent):

    def __init__(self, name) -> None:
        self.name = name

    def get_next_move(self, gamestate: Connect2Gamestate) -> int:
        while True:
            n = randrange(0, 4)
            if gamestate.board_state[n] == 0:
                return n

class HumanHexAgent(SpecializedAgent):

    def __init__(self) -> None:
        self.name = input("Please enter your name: ")

    def get_next_move(self, gamestate: HexState) -> int:
        while True:
            try: 
                coords = input("Human player to move: (SE, NE): \n").split(" ")
                return HexMove(int(coords[0]), int(coords[1]), gamestate.board_size).get_int_representation()
            except:
                print("Illegal move, try again")
