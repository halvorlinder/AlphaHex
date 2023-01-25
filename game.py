from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod

class Game(ABC):

    move_cardinality : int

    @abstractmethod
    def play(self, gamestate : Gamestate, move : Move) -> Gamestate:
        pass

    @abstractmethod
    def is_legal_move(self, gamestate : Gamestate, move : Move) -> bool:
        pass

    @abstractmethod
    def get_legal_moves(self, gamestate: Gamestate) -> list[bool]:
        pass

    @abstractmethod
    def create_move(self, int_representation : int) -> Move :
        pass

class Gamestate(ABC):

    @abstractmethod
    def reward(self) -> int:
        pass

class Move(ABC):

    @abstractmethod
    def get_int_representation(self):
        pass
