from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING
from agent import SpecializedAgent
from hex import HexState, HexMove, Piece
if TYPE_CHECKING:
    from agent import Agent

from random import randrange

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

    def __init__(self) -> None:
        pass

    def get_next_move(self, gamestate: HexState) -> int:
        while True:
            n = randrange(0, gamestate.board_size*gamestate.board_size)
            if gamestate.is_open(gamestate.create_move(n)):
                return n

class HumanHexAgent(SpecializedAgent):

    def __init__(self) -> None:
        pass

    def get_next_move(self, gamestate: HexState) -> int:
        while True:
            try: 
                coords = input("Human player to move: (SE, NE): \n").split(" ")
                return HexMove(int(coords[0]), int(coords[1]), gamestate.board_size).get_int_representation()
            except:
                print("Illegal move, try again")
