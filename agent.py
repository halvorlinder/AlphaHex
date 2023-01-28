from abc import ABC, abstractmethod

from game import Gamestate

class Agent(ABC):
    @abstractmethod
    def get_next_move(self, gamestate : Gamestate):
        pass

class GeneralAgent(Agent):
    pass

class SpecializedAgent(Agent):
    pass