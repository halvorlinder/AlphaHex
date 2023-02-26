from __future__ import annotations
from agent import Agent
from game import Game, Gamestate
from ANET import ConvNet, FFNet, PytorchNN, Trainer
import random
import numpy as np
from MCTS import MCTS, UCB
from hex_agents import RandomHexAgent, RandomConnect2Agent, MCTSHexAgent
from neural_net import NeuralNet
from tournament import TournamentPlayer
from utils import epsilon_greedy_choise, filter_and_normalize

import CONSTANTS


class RL:

    def __init__(self, game: Game, model: NeuralNet, epsilon: int = 0) -> None:
        self.game = game
        self.model: NeuralNet = model
        self.epsilon = epsilon

    # def get_agent(self) -> NeuralAgent:
    #     return self.agent

    # def get_agent_from_file(self, filename) -> NeuralAgent:
    #     self.load_agent(filename)
    #     return self.get_agent()

    def save_model(self, filename: str) -> None:
        self.model.save(filename)

    # def new_agent(self, something_that_describes_the_NN) -> None:
    #     self.agent = NeuralAgent(FFNet(self.game.state_representation_length, self.game.move_cardinality))

    def load_agent(self, filename: str) -> None:
        # self.agent = NeuralAgent(FFNet(self.game.state_representation_length, self.game.move_cardinality))
        self.model.load(filename)

    def train_agent(self, num_games: int) -> None:
        # Might need some more args to configure the MCTS
        # trainer = Trainer(self.agent.neural_network, 1, 0.05, 8)
        for n in range(num_games):
            print(n)
            inputs, labels = self.play_game()
            self.model.train(inputs, labels)
        # self.model.save('heisann')

    def play_game(self) -> np.ndarray:
        # TODO not quite sure of the numpy code here
        gamestate = self.game.get_initial_position()
        training_states = []
        training_probs = []
        training_visits = []
        training_states_full = []
        next_root = None
        while gamestate.reward()==None:
            mcts = MCTS(self.game, root=next_root,
                        predict_func=self.model.predict, representation=self.model.model.state_representation)
            action_probs = mcts.run_simulations(CONSTANTS.ROLLOUTS)

            selected_move = epsilon_greedy_choise(
                action_probs, gamestate.get_legal_moves(), epsilon=self.epsilon)
            training_states.append(gamestate.get_representation(self.model.model.state_representation))
            training_states_full.append(gamestate)
            training_probs.append(action_probs)
            training_visits.append(mcts.get_visits())
            gamestate = gamestate.play_move_int(selected_move)
            next_root = mcts.get_next_root(selected_move)
        print(training_states)
        print(training_probs)
        # for full, state, prob, visit in zip(training_states_full, training_states, training_probs, training_visits):
        #     print(full)
        #     print(state)
        #     print(list(map(lambda p: str(p)[:4], prob)))
        #     print(visit)
        #     print(sum(visit))
        #     print()
        return [np.array(training_states), np.array(training_probs)]


class NeuralAgent(Agent):

    def __init__(self, neural_net: NeuralNet, name: str = None) -> None:
        self.neural_net = neural_net
        if name:
            self.name = name

    def get_next_move(self, gamestate: Gamestate) -> int:
        prediction = self.neural_net.predict(gamestate.get_representation(self.neural_net.model.state_representation))
        # print(prediction)
        # print(filter_and_normalize(prediction, gamestate.get_legal_moves()))
        prediction = np.exp(prediction)/sum(np.exp(prediction)) # apply softmax to avoid negative probabilities
        probs = filter_and_normalize(prediction, gamestate.get_legal_moves())
        move = np.random.choice([n for n in range(len(probs))], p=probs)
        # move = epsilon_greedy_choise(filter_and_normalize(prediction, gamestate.get_legal_moves()), gamestate.get_legal_moves(), epsilon=0)
        # print(move)
        return move

if __name__ == "__main__":
    from hex import Hex
    from connect2 import Connect2
    hex = Hex(3)
    connect2 = Connect2()
    game = connect2
    # net_1 = FFNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_1 = PytorchNN()
    # pynet_1.load(net_1, 'agent_49')
    # rl = RL(hex, pynet_1)

    rl = RL(
        game, 
        PytorchNN(
            model=FFNet(
            game.state_representation_length, 
            game.move_cardinality
            )
        ),
        epsilon=CONSTANTS.GAME_MOVE_EPSILON
    )
    
    # rl.train_agent(1000)
    # rl.model.save('agent_50')
    # rl.train_agent(50)
    # rl.model.save('agent_100')
    # rl.train_agent(50)
    # rl.model.save('agent_150')
    # rl.train_agent(50)
    # rl.model.save('agent_200')

    net_50 = FFNet(game.state_representation_length, game.move_cardinality)
    pynet_50 = PytorchNN()
    pynet_50.load(net_50, 'agent_50')

    # net_100 = ConvNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_100 = PytorchNN()
    # pynet_100.load(net_100, 'agent_100')

    # net_150 = ConvNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_150 = PytorchNN()
    # pynet_150.load(net_150, 'agent_150')

    # net_200 = ConvNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_200 = PytorchNN()
    # pynet_200.load(net_200, 'agent_200')

    # net_1 = FFNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_100 = PytorchNN()
    # pynet_100.load(net_1, 'agent_100')

    # tourney = TournamentPlayer(Hex(3), [RandomHexAgent('random'), NeuralAgent(pynet_50, '50'), NeuralAgent(pynet_100, '100'), NeuralAgent(pynet_150, '150'), NeuralAgent(pynet_200, '200') ][::-1], 30, True)
    # scores, wins = tourney.play_tournament()
    # print(wins)

    gs = game.get_initial_position()
    print(pynet_50.predict(gs.get_int_list_representation()))
    # print(pynet_100.predict(gs.get_int_list_representation()))
    # print(pynet_150.predict(gs.get_int_list_representation()))
    # print(pynet_200.predict(gs.get_int_list_representation()))

    # tourney = TournamentPlayer(Hex(3), [NeuralAgent(pynet_1, '1'), RandomHexAgent('random'),], 100, True)
    # scores, wins = tourney.play_tournament()
    # print(wins)


    # tourney = TournamentPlayer(Hex(4), [MCTSHexAgent("MCTS", 1000, 4), RandomHexAgent('random')][::-1], 100, True)
    # scores, wins = tourney.play_tournament()
    # print(wins)