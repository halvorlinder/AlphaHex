from __future__ import annotations
from agent import Agent
from game import Game, Gamestate


class RL:

    def __init__(self, game : Game) -> None:
        self.game = game
        self.agent : NeuralAgent = None
    
    def get_agent(self) -> NeuralAgent:
        return self.agent

    def get_agent_from_file(self, filename) -> NeuralAgent:
        self.load_agent(filename)
        return self.get_agent()

    def save_agent(self, filename : str) -> None:
        pass

    def new_agent(self, something_that_describes_the_NN) -> None:
        pass

    def load_agent(self, filename : str) -> None:
        pass

    def train_agent(self, num_games : int) -> None:
        # Might need some more args to configure the MCTS
        for _ in num_games:
            pass

class NeuralAgent(Agent):

    def __init__(self, neural_network) -> None:
        self.neural_network = neural_network

    def get_next_move(self, gamestate: Gamestate, epsilon = 0):
        # Generate random number, compare with epsilon, do either random move or max over probs
        return 

    def get_action_probabilities(self, gamestate: Gamestate) -> list[float]:
        # Just predict with the net 
        return None
