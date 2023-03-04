from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from representations import StateRepresentation

# if TYPE_CHECKING:
#     from agent import Agent

class Game(ABC):

    state_representation_length : int
    move_cardinality : int
    num_agents : int = 2
    conv_net_layers: int

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_initial_position(self) -> Gamestate:
        pass

    @abstractmethod
    def from_int_list_representation(self, list_rep : list[int]) -> Gamestate:
        pass

class Gamestate(ABC):

    @abstractmethod
    def play(self, move : Move) -> Gamestate:
        pass

    @abstractmethod
    def play_move_int(self, move_idx : int) -> Gamestate:
        pass

    @abstractmethod
    def is_legal_move(self, move : Move) -> bool:
        pass

    @abstractmethod
    def get_legal_moves(self) -> list[bool]:
        pass

    @abstractmethod
    def create_move(self, int_representation : int) -> Move:
        pass

    @abstractmethod
    def reward(self) -> list[int]:
        pass

    @abstractmethod
    def get_agent_index(self) -> int:
        pass

    @abstractmethod
    def get_representation(self, representation : StateRepresentation) -> np.ndarray:
        pass

    # @abstractmethod
    # def get_int_list_representation(self) -> list[int]:
    #     pass

    # @abstractmethod
    # def get_layered_representation(self) -> list[list[list[int]]]:
    #     pass

    

class Move(ABC):

    @abstractmethod
    def get_int_representation(self) -> int:
        pass

    @abstractmethod
    def from_int_representation(self, int_representation : int) -> Move:
        pass