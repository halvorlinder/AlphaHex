from abc import ABC, abstractmethod

from game import Gamestate

class Agent(ABC):

    name : str = "NO_NAME"

    @abstractmethod
    def get_next_move(self, gamestate : Gamestate) -> int:
        pass

class GeneralAgent(Agent):
    pass

class SpecializedAgent(Agent):
    pass