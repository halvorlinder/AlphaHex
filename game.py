from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from agent import Agent

class Game(ABC):

    move_cardinality : int

    @abstractmethod
    def get_initial_position(self) -> Gamestate:
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
    def reward(self) -> int:
        pass

    def get_agent_index(self) -> int:
        pass

class Move(ABC):

    @abstractmethod
    def get_int_representation(self):
        pass

    @abstractmethod
    def from_int_representation(self, int_representation : int) -> Move:
        pass