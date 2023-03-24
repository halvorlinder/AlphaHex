from __future__ import annotations
from datetime import datetime
import json
import os
from connect2 import Connect2
from hex import Hex
from tic_tac_toe import TicTacToeGame
from agent import Agent
from game import Game, Gamestate
from ANET import ConvNet, FFNet, PytorchNN, Trainer
import numpy as np
from MCTS import MCTS, UCB
from neural_net import NeuralNet
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
                examples = list(mp.Pool(CONSTANTS.CORES).map(
                    RL.play_game, [self]*CONSTANTS.CORES))
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
        while gamestate.reward() == None:
            print(gamestate)
            mcts = MCTS(self.game, root=next_root,
                        predict_func=self.model.predict, representation=self.model.model.state_representation)
            action_probs = mcts.run_simulations(CONSTANTS.ROLLOUTS)
            # print(gamestate.board)
            # print(action_probs)
            # print(gamestate.get_legal_moves())

            selected_move = epsilon_greedy_choise(
                action_probs, gamestate.get_legal_moves(), epsilon=CONSTANTS.GAME_MOVE_EPSILON)
            training_states.append(gamestate.get_representation(
                self.model.model.state_representation))
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
        prediction = self.neural_net.predict(gamestate.get_representation(
            self.neural_net.model.state_representation))
        print(prediction)
        probs = filter_and_normalize(prediction, gamestate.get_legal_moves())
        match CONSTANTS.AGENT_SELECTION_POLICY:
            case CONSTANTS.SelectionPolicy.MAX:
                move = np.random.choice(
                    [n for n in range(len(probs))], p=probs)
            case CONSTANTS.SelectionPolicy.SAMPLE:
                move = epsilon_greedy_choise(filter_and_normalize(
                    prediction, gamestate.get_legal_moves()), gamestate.get_legal_moves(), epsilon=0)
        return move

def get_agent_folder() -> str:
    match CONSTANTS.GAME:
        case CONSTANTS.TrainingGame.HEX:
            return f'agents/hex{CONSTANTS.HEX_SIZE}/'
        case CONSTANTS.TrainingGame.C2:
            return f'agents/c2/'
        case CONSTANTS.TrainingGame.TTT:
            return f'agents/ttt/'

def train_from_conf() -> None:

    match CONSTANTS.GAME:
        case CONSTANTS.TrainingGame.HEX:
            game = Hex(CONSTANTS.HEX_SIZE)
        case CONSTANTS.TrainingGame.TTT:
            game = TicTacToeGame()
        case CONSTANTS.TrainingGame.C2:
            game = Connect2()

    match CONSTANTS.NETWORK_ARCHITECTURE:
        case CONSTANTS.NetworkArchitecture.FF:
            net = FFNet(
                game.state_representation_length,
                game.move_cardinality
            )
        case CONSTANTS.NetworkArchitecture.CONV:
            raise NotImplementedError()

    rl = RL(
        game,
        PytorchNN(model=net),
        epsilon=CONSTANTS.GAME_MOVE_EPSILON
    )
    t = datetime.now()
    time_stamp = f'{t.date()}_{t.hour}:{t.minute}'.replace(" " , "_")
    
    os.makedirs(f'{get_agent_folder()}/{time_stamp}', exist_ok=True)
    constants = dict(list(map(lambda kv: (str(kv[0]), str(kv[1])), filter(lambda kv: kv[0].isupper(), CONSTANTS.__dict__.items()))))
    with open(f'{get_agent_folder()}/{time_stamp}/METADATA.json', "w") as metadata:
        json.dump(constants, metadata)
    for i in range(CONSTANTS.NUM_SAVES):
        rl.train_agent(CONSTANTS.GAMES_PER_SAVE)
        rl.model.save(f'{get_agent_folder()}/{time_stamp}/{i}')

def get_neural_agents(game : Game, time_stamp : str, indicies : list[int] = None):
    with open(f'agents/{game.get_name()}/{time_stamp}/METADATA.json') as json_file:
        data = json.load(json_file)
        match data['NETWORK_ARCHITECTURE']:
            case 'NetworkArchitecture.FF':
                net_gen = lambda:FFNet(game.state_representation_length, game.move_cardinality)
            case 'NetworkArchitecture.CONV':
                raise(NotImplementedError())
        all_indicies = list(range(int(data['NUM_SAVES'])))
        games_per_save = int(data['GAMES_PER_SAVE'])

    agents = []
    for i in all_indicies if not indicies else indicies:
        net = net_gen()
        pynet = PytorchNN()
        pynet.load(net, f'agents/{game.get_name()}/{time_stamp}/{i}')
        agents.append(NeuralAgent(pynet, f'{games_per_save*(i+1)}'))
    print(agents)
    return agents


if __name__ == "__main__":
    wandb.init(project="RL-hex")
    train_from_conf()
    # game = Hex(3)
    # get_neural_agents(game, '2023-03-03_9:51',)
