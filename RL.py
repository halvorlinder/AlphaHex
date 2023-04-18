from __future__ import annotations
from datetime import datetime
import json
import os
from connect2 import Connect2
from hex import Hex
from tic_tac_toe import TicTacToeGame
from agent import Agent
from game import Game, Gamestate
from ANET import ConvNet, FFNet, PytorchNN, Trainer, ConvResNet
import numpy as np
from MCTS import MCTS, UCB
from neural_net import NeuralNet
from utils import epsilon_greedy_choise, filter_and_normalize
import multiprocessing as mp
import wandb
import random
import torch.nn.functional as F

import CONSTANTS


class RL:

    def __init__(self, game: Game, model: NeuralNet, epsilon: int = 0) -> None:
        self.game = game
        self.model: NeuralNet = model
        self.epsilon = epsilon
        self.move_buffer = {"inputs" : np.array([]), "labels" : np.array([])}
        self.mcts_examples = {"inputs" : np.array([]), "labels" : np.array([])}
        self.load_MCTS_data()

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

    def load_MCTS_data(self) -> None:
        states = []
        probs = []
        for i in range(8):
            with open(f'MCTS_data/data_{i}.txt', 'r') as f:
                for line in f:
                    state, prob = line.split(';')
                    state = self.game.from_int_list_representation(list(map(int, state.split(',')))).get_representation(self.model.model.state_representation)
                    prob = list(map(float, prob.split(',')))
                    states.append(state)
                    probs.append(prob)
        
        self.mcts_examples['inputs'] = np.array(states)
        self.mcts_examples['labels'] = np.array(probs)

    def get_training_examples(self):
        print(self.move_buffer["inputs"])

        buffer_length = self.move_buffer["inputs"].shape[0]
        chosen_indexes = random.sample(range(0, buffer_length), min(CONSTANTS.REPLAY_BUFFER_MOVES_CHOSEN, buffer_length))
        chosen_inputs = self.move_buffer["inputs"][chosen_indexes]
        chosen_labels = self.move_buffer["labels"][chosen_indexes]

        buffer_length = self.mcts_examples["inputs"].shape[0]
        chosen_indexes = random.sample(range(0, buffer_length), min(CONSTANTS.MCTS_MOVES_CHOSEN, buffer_length))
        chosen_inputs_2 = self.mcts_examples["inputs"][chosen_indexes]
        chosen_labels_2 = self.mcts_examples["labels"][chosen_indexes]

        return np.concatenate((chosen_inputs, chosen_inputs_2)), np.concatenate((chosen_labels, chosen_labels_2))
    
    def add_training_examples(self, inputs, labels):
        self.move_buffer["inputs"] = np.concatenate((self.move_buffer["inputs"], inputs)) if self.move_buffer["inputs"].size > 0 else inputs
        self.move_buffer["labels"] = np.concatenate((self.move_buffer["labels"], labels)) if self.move_buffer["labels"].size > 0 else labels
        buffer_length = self.move_buffer["inputs"].shape[0]
        if buffer_length > CONSTANTS.REPLAY_BUFFER_MAX_SIZE:
            print("BUFFER TOO LARGE, DELETING OLD EXAMPLES")
            self.move_buffer["inputs"] = self.move_buffer["inputs"][:CONSTANTS.REPLAY_BUFFER_MAX_SIZE]
            self.move_buffer["labels"] = self.move_buffer["labels"][:CONSTANTS.REPLAY_BUFFER_MAX_SIZE]
                

    def train_agent(self, num_games: int) -> None:
        if CONSTANTS.M_THREAD:
            for n in range(num_games//CONSTANTS.CORES):
                print(n*CONSTANTS.CORES)
                examples = list(mp.Pool(CONSTANTS.CORES).map(
                    RL.play_game, [self]*CONSTANTS.CORES))
                inputs = np.concatenate([inputs for (inputs, _) in examples])
                labels = np.concatenate([labels for (_, labels) in examples])
                self.add_training_examples(inputs=inputs, labels=labels)
                chosen_inputs, chosen_labels = self.get_training_examples()
                avg_epoch_loss = self.model.train(chosen_inputs, chosen_labels)
                if CONSTANTS.ENABLE_WANDB:
                    wandb.log({'loss': avg_epoch_loss, "buffer_capacity": self.move_buffer["inputs"].shape[0] / CONSTANTS.REPLAY_BUFFER_MAX_SIZE})

        else:
            for n in range(num_games):
                inputs, labels = self.play_game()
                self.add_training_examples(inputs=inputs, labels=labels)
                chosen_inputs, chosen_labels = self.get_training_examples()
                avg_epoch_loss = self.model.train(inputs, labels)
                if CONSTANTS.ENABLE_WANDB:
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
        # print(prediction)
        probs = filter_and_normalize(prediction, gamestate.get_legal_moves())
        match CONSTANTS.AGENT_SELECTION_POLICY:
            case CONSTANTS.SelectionPolicy.SAMPLE:
                move = np.random.choice(
                    [n for n in range(len(probs))], p=(probs**2)/sum(probs**2))
            case CONSTANTS.SelectionPolicy.MAX:
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

    match CONSTANTS.HIDDEN_NODE_ACTIVATION:
        case CONSTANTS.HiddenNodeActivation.LINEAR:
            hidden_node_activation = F.linear
        case CONSTANTS.HiddenNodeActivation.TANH:
            hidden_node_activation = F.tanh
        case CONSTANTS.HiddenNodeActivation.SIGMOID:
            hidden_node_activation = F.sigmoid
        case CONSTANTS.HiddenNodeActivation.RELU:
            hidden_node_activation = F.relu

    match CONSTANTS.NETWORK_ARCHITECTURE:
        case CONSTANTS.NetworkArchitecture.FF:
            net = FFNet(
                game.state_representation_length,
                game.move_cardinality, 
                hidden_node_activation=hidden_node_activation
            )
        case CONSTANTS.NetworkArchitecture.CONV:
            net = ConvNet(
                board_state_length=game.state_representation_length, 
                move_cardinality=game.move_cardinality, 
                board_dimension_depth=game.conv_net_layers, 
                hidden_node_activation=hidden_node_activation
            )
        case CONSTANTS.NetworkArchitecture.RESNET:
            net = ConvResNet(
                board_dimension_depth=game.conv_net_layers, 
                channels=CONSTANTS.CHANNELS_RES_BLOCK, 
                num_res_blocks=CONSTANTS.NUMBER_RES_BLOCKS, 
                board_state_length=game.state_representation_length, 
                move_cardinality=game.move_cardinality, 
            )

    pytorch_nn = PytorchNN(model=net)
    # pytorch_nn.load(model=net, filename="agents/hex7/2023-04-15_18:40/2")
    rl = RL(
        game,
        pytorch_nn,
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
                net_gen = lambda:ConvNet(
                    board_state_length=game.state_representation_length, 
                    move_cardinality=game.move_cardinality, 
                    board_dimension_depth=game.conv_net_layers
                    )
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
    if CONSTANTS.ENABLE_WANDB:
        wandb.init(project="RL-hex")
    train_from_conf()
    # game = Hex(3)
    # get_neural_agents(game, '2023-03-03_9:51',)
