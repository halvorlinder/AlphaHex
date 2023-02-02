from __future__ import annotations
from agent import Agent
from game import Game, Gamestate
from ANET import FFNet, Trainer
import random
import numpy as np
from MCTS import MCTS, UCB


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
        self.agent.neural_network.save(filename)

    def new_agent(self, something_that_describes_the_NN) -> None:
        self.agent = FFNet(self.game.state_representation_lenght, self.game.move_cardinality)

    def load_agent(self, filename : str) -> None:
        self.agent = FFNet(self.game.state_representation_lenght, self.game.move_cardinality)
        self.agent.load(filename)

    def train_agent(self, num_games : int) -> None:
        # Might need some more args to configure the MCTS
        trainer = Trainer(self.agent.neural_network, 1, 0.05, 8)
        for _ in num_games:
            mcts = MCTS(self.game, self.game.get_initial_position, UCB, self.agent.neural_network)
            action_probs, next_root = mcts.run_simulations(50)
            training_examples = self.play_game()
            trainer.train(training_examples)

    def play_game(self) -> np.ndarray:


            

class NeuralAgent(Agent):

    def __init__(self, neural_network : FFNet) -> None:
        self.neural_network = neural_network

    def get_next_move(self, gamestate: Gamestate, epsilon = 0) -> int:
        choose_random = random.random()
        if choose_random < epsilon:
            true_idx = np.argwhere(np.array(gamestate.get_legal_moves()))
            random_idx = random.randint(0, len(true_idx) - 1)
            return true_idx[random_idx][0]
        else:
            return np.argmax(self.neural_network.forward(gamestate))

    # def get_action_probabilities(self, gamestate: Gamestate) -> list[float]:
    #     return self.neural_network.forward(gamestate)
