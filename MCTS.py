from functools import partial
import itertools
import numpy as np
import random
from game import Game, Gamestate, Move
from connect2 import Connect2, Connect2Gamestate
from hex import Hex
from torch import float32, float16, float64
import time

from representations import StateRepresentation
import CONSTANTS
from tic_tac_toe import TicTacToeGame

DEBUG = False

class Node():

    def __init__(self, gamestate: Gamestate):
        self.gamestate = gamestate
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if (prob > 0):
                #TODO: Should check legal instead 
                new_move = self.gamestate.create_move(action)
                new_gamestate = self.gamestate.play(move=new_move)
                self.children[action] = Node(gamestate=new_gamestate)

    def select_child(self, score_func):
        # max_score = -float("inf")
        # selected_child = None
        scores = np.array([score_func(parent=self, child=child) for (action, child) in self.children.items()])
        index = np.random.choice(np.flatnonzero(scores == scores.max()))
        # for action,child in self.children.items():
        #     score = score_func(parent=self, child=child)
        #     if DEBUG:
        #         print(f'\t\tAction: {action}\tUCB: {score}')
        #     if score > max_score:
        #         selected_child = child
        #         selected_action = action
        #         max_score = score
        # if DEBUG:
        #     print(f'\tSelecting {selected_action}')
        # print(max_score==scores[index])
        return list(self.children.items())[index][1]
        # return selected_child


# def UCB(parent, child):
#     prior_score = child.prior * np.sqrt(parent.visits) / (child.visits + 1)
#     value_score = child.value / child.visits if child.visits > 0 else 0
#     return prior_score + value_score


def UCB(parent, child):
    explore = CONSTANTS.SQRT_2 * np.sqrt(np.log(parent.visits) / (child.visits + 1))
    exploit = child.value / (child.visits + 1)
    return explore + exploit


def dummy_predict(n: int, _: Gamestate):
    policy = [1/n] * n
    return policy


class MCTS():

    def __init__(self, game: Game, score_func=UCB, root: Node = None, predict_func=None, representation : StateRepresentation = StateRepresentation.FLAT ) -> None:
        if DEBUG:
            print(f'Starting MCTS from')
        self.game = game
        self.representation = representation
        if root:
            self.initial_gamestate = root.gamestate
            self.current_gamestate = root.gamestate
            self.root = root
        else:
            self.initial_gamestate = game.get_initial_position()
            self.current_gamestate = game.get_initial_position()
            self.root = Node(gamestate=self.initial_gamestate)
        if DEBUG:
            print(self.initial_gamestate)
            print(f'With {self.root.visits} previous visits')
            print()
        self.score_func = score_func
        self.predict_func, self.epsilon = (predict_func, CONSTANTS.MCTS_EPSILON) if predict_func else (partial(
            dummy_predict, game.move_cardinality), 1)
        

    def normalize_action_probs(self, action_probs, gamestate: Gamestate):
        norm_action_probs = np.array([prob * mask for prob,
                             mask in zip(action_probs, gamestate.get_legal_moves())])
        return norm_action_probs / np.sum(norm_action_probs)

    def run_simulation(self):

        self.current_gamestate = self.initial_gamestate
        node = self.root
        search_path = []

        # using tree policy to find leaf node
        if DEBUG:
            print(f'Root:')
            for action, nd in node.children.items():
                score = self.score_func(parent=node, child=nd)
                print(f'\t\tAction: {action}\tUCB: {score}, Visits: {nd.visits}, Value: {nd.value}')
            print(f'Starting tree search:')
        while (len(node.children) > 0):  # node has children, already expanded
            node = node.select_child(self.score_func)
            search_path.append(node)
            self.current_gamestate = node.gamestate

        if DEBUG: 
            print(f'The leaf node gamestate:')
            print(self.current_gamestate)
        # expanding leaf node if not in terminal state
        reward = node.gamestate.reward()
        if reward != None:
            if DEBUG:
                print(f'Returning early due to reward {reward}')
            self.increment_reward(search_path=search_path, reward=reward)
            return

        # print(node.gamestate)
        # print(np.array([node.gamestate.get_representation(self.representation)]))
        # print(len(np.array([node.gamestate.get_representation(self.representation)])[0]))
        action_probs = self.predict_func(
            node.gamestate.get_representation(self.representation))
        # print(action_probs)
        norm_action_probs = self.normalize_action_probs(
            action_probs=action_probs, gamestate=node.gamestate)
        if DEBUG:
            print(f'Action probabilities for expansion:\n\t{norm_action_probs}')
            print(f'Expanding...')
        node.expand(action_probs=norm_action_probs)

        leaf_node = node
        self.current_gamestate = leaf_node.gamestate

        if DEBUG:
            print(f'Starting rollout')
        while reward == None:  # i.e. we are not in a terminal state
            action_probs = self.predict_func(
                self.current_gamestate.get_representation(self.representation)
                )
            norm_action_probs = self.normalize_action_probs(
                action_probs=action_probs, 
                gamestate=self.current_gamestate
                )
            if DEBUG:
                print(f'\tMove probabilities: \n\t{norm_action_probs}')
            # select next move in rollout phase
            choose_random = random.random()
            move = None
            if choose_random < self.epsilon:
                if DEBUG: 
                    print(f'\tEpsilon choice')
                true_idx = np.argwhere(
                    np.array(self.current_gamestate.get_legal_moves()))
                random_idx = random.randint(0, len(true_idx) - 1)
                move = true_idx[random_idx][0]
            else:
                move = np.argmax(norm_action_probs)
            if DEBUG:
                print(f'\tSelected move: {move}')
            

            self.current_gamestate = self.current_gamestate.play_move_int(
                move_idx=move
                )
            if DEBUG: 
                print(f'\tNext gamestate:')
                print(self.current_gamestate)
            reward = self.current_gamestate.reward()
        if DEBUG:
            print(f'Incrementing values')
        self.increment_reward(search_path=search_path, reward=reward)

    def increment_reward(self, search_path : list[Node], reward: list[int]):
        player_in_root = self.initial_gamestate.get_agent_index()
        if DEBUG:
            print(f'Player: {player_in_root}')
            print(f'Reward: {reward}')

        for node in search_path:
            node.visits += 1
        self.root.visits+=1
        
        offset_reward = reward[player_in_root:] + reward[:player_in_root]
        for node, rew in zip(search_path, itertools.cycle(offset_reward)):
            node.value += rew

    def run_simulations(self, n: int) -> np.ndarray:
        start_time = time.time()
        i = 0
        while i < n:
            self.run_simulation()
            time_diff = time.time() - start_time
            if time_diff > CONSTANTS.MAX_ROLLOUT_TIME_SECONDS and i > 1:
                print(time_diff)
                print(i)
                break
            i += 1
        print(f"Number of rollouts: {i}")
        return self.get_norm_action_probs()

    def get_norm_action_probs(self) -> np.ndarray:
        action_probs = [0] * self.game.move_cardinality
        for action, child in self.root.children.items():
            action_probs[action] = child.visits
        norm_action_probs = self.normalize_action_probs(
            action_probs, self.initial_gamestate)
        return norm_action_probs

    def get_next_root(self, move_index: int) -> Node:
        return self.root.children[move_index]

    def get_visits(self) -> np.ndarray:
        action_probs = [0] * self.game.move_cardinality
        for action, child in self.root.children.items():
            action_probs[action] = child.visits
        return action_probs



if __name__ == "__main__":
    game = TicTacToeGame()
    gs = game.from_int_list_representation([0,2,1,1,2,2,0,1,1])
    mcts = MCTS(game, root=Node(gs))
    for _ in range(1):
        print(mcts.run_simulations(1000))
