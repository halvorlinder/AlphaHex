from __future__ import annotations
from agent import Agent
from game import Game, Gamestate
from ANET import ConvNet, FFNet, PytorchNN, Trainer
import numpy as np
from MCTS import MCTS, UCB
from game_player import GameInstance
from neural_net import NeuralNet
from tournament import TournamentPlayer
from utils import epsilon_greedy_choise, filter_and_normalize
import multiprocessing as mp
import wandb

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
        if CONSTANTS.M_THREAD:
            for n in range(num_games//CONSTANTS.CORES):
                print(n*CONSTANTS.CORES)
                examples = list(mp.Pool(CONSTANTS.CORES).map(RL.play_game, [self]*CONSTANTS.CORES))
                inputs = np.concatenate([inputs for (inputs, _) in examples])
                labels = np.concatenate([labels for (_, labels) in examples])
                print(inputs)
                print(labels)
                for inp in inputs:
                    print(self.model.predict(inp))
                self.model.train(inputs, labels)
                avg_epoch_loss = self.model.train(inputs, labels)
                wandb.log({'loss': avg_epoch_loss})

        else:
            for n in range(num_games):
                print(n)
                inputs, labels = self.play_game()
                print(inputs)
                print(labels)
                avg_epoch_loss = self.model.train(inputs, labels)
                wandb.log({'loss': avg_epoch_loss})

    def play_game(self) -> np.ndarray:
        # TODO not quite sure of the numpy code here
        gamestate = self.game.get_initial_position()
        training_states = []
        training_probs = []
        training_visits = []
        training_states_full = []
        next_root = None
        while gamestate.reward()==None:
            print(gamestate)
            mcts = MCTS(self.game, root=next_root,
                        predict_func=self.model.predict, representation=self.model.model.state_representation)
            action_probs = mcts.run_simulations(CONSTANTS.ROLLOUTS)
            # print(gamestate.board)
            # print(action_probs)
            # print(gamestate.get_legal_moves())

            selected_move = epsilon_greedy_choise(
                action_probs, gamestate.get_legal_moves(), epsilon=CONSTANTS.GAME_MOVE_EPSILON)
            training_states.append(gamestate.get_representation(self.model.model.state_representation))
            training_states_full.append(gamestate)
            training_probs.append(action_probs)
            training_visits.append(mcts.get_visits())
            gamestate = gamestate.play_move_int(selected_move)
            next_root = mcts.get_next_root(selected_move)
        # print(training_states)
        # print(training_probs)
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
        print(prediction)
        probs = filter_and_normalize(prediction, gamestate.get_legal_moves())
        match CONSTANTS.AGENT_SELECTION_POLICY:
            case CONSTANTS.SelectionPolicy.MAX:
                move = np.random.choice([n for n in range(len(probs))], p=probs)
            case CONSTANTS.SelectionPolicy.SAMPLE:
                move = epsilon_greedy_choise(filter_and_normalize(prediction, gamestate.get_legal_moves()), gamestate.get_legal_moves(), epsilon=0)
        return move

if __name__ == "__main__":
    from gen_agents import HumanAgent, RandomAgent, MCTSAgent
    from tic_tac_toe import TicTacToeGame
    from hex import Hex
    from connect2 import Connect2

    wandb.init(project="RL-hex")
    hex = Hex(3)
    connect2 = Connect2()
    game = hex
    # net_1 = FFNet(hex.state_representation_length, hex.move_cardinality)
    # pynet_1 = PytorchNN()
    # pynet_1.load(net_1, 'agent_49')
    # rl = RL(hex, pynet_1)

    hex = Hex(3)
    connect2 = Connect2()
    tic = TicTacToeGame()

    game = hex

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
    
    # rl.train_agent(500)
    # rl.model.save('agent_50_3')
    # rl.train_agent(500)
    # rl.model.save('agent_100_3')
    # rl.train_agent(500)
    # rl.model.save('agent_150_3')
    # rl.train_agent(500)
    # rl.model.save('agent_200_3')
    # rl.train_agent(100)
    # rl.model.save('agent_250_4')
    # rl.train_agent(100)
    # rl.model.save('agent_300_4')
    # rl.train_agent(100)
    # rl.model.save('agent_350_4')
    # rl.train_agent(100)
    # rl.model.save('agent_400_4')
    # rl.train_agent(100)
    # rl.model.save('agent_450_4')
    # rl.train_agent(100)
    # rl.model.save('agent_500_4')
    # rl.train_agent(100)
    # rl.model.save('agent_550_4')
    # rl.train_agent(100)
    # rl.model.save('agent_600_4')

    net_50 = FFNet(game.state_representation_length, game.move_cardinality)
    pynet_50 = PytorchNN()
    pynet_50.load(net_50, 'agent_50_4')

    net_100 = FFNet(game.state_representation_length, game.move_cardinality)
    pynet_100 = PytorchNN()
    pynet_100.load(net_100, 'agent_100_4')

    net_150 = FFNet(game.state_representation_length, game.move_cardinality)
    pynet_150 = PytorchNN()
    pynet_150.load(net_150, 'agent_150_4')

    net_200 = FFNet(game.state_representation_length, game.move_cardinality)
    pynet_200 = PytorchNN()
    pynet_200.load(net_200, 'agent_200_4')

    # game_inst = GameInstance(game, [HumanAgent(), NeuralAgent(pynet_50, '50')][::-1], True)
    # game_inst.start()

    tourney = TournamentPlayer(game, [RandomAgent(game, 'random'), NeuralAgent(pynet_50, '50'), NeuralAgent(pynet_200, '200') ][::-1], 100, True)

    scores, wins = tourney.play_tournament()
    print(wins)