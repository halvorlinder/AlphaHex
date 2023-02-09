from functools import partial
import numpy as np
import random
from game import Game, Gamestate, Move
from connect2 import Connect2, Connect2Gamestate
from hex import Hex
import copy
from torch import float32, float16, float64

DEBUG = False

class Node():

    def __init__(self, prior: float, gamestate: Gamestate):
        self.prior = prior
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
                self.children[action] = Node(
                    prior=prob, gamestate=new_gamestate)

    def select_child(self, score_func):
        # TODO Is this not just argmax, if so refactor to make it more readable (np.argmax)
        max_score = -float("inf")
        selected_child = None
        for action,child in self.children.items():
            score = score_func(parent=self, child=child)
            if DEBUG:
                print(f'\t\tAction: {action}\tUCB: {score}')
            if score > max_score:
                selected_child = child
                selected_action = action
                max_score = score
        if DEBUG:
            print(f'\tSelecting {selected_action}')
        return selected_child


# def UCB(parent, child):
#     prior_score = child.prior * np.sqrt(parent.visits) / (child.visits + 1)
#     value_score = child.value / child.visits if child.visits > 0 else 0
#     return prior_score + value_score
SQRT_2 = 1.41


def UCB(parent, child):
    explore = SQRT_2 * np.sqrt(np.log(parent.visits) / (child.visits + 1))
    exploit = child.value / (child.visits + 1)
    return explore + exploit


def dummy_predict(n: int, _: Gamestate):
    policy = [1/n] * n
    return policy


class MCTS():

    def __init__(self, game: Game, score_func=UCB, root: Node = None, predict_func=None) -> None:
        if DEBUG:
            print(f'Starting MCTS from')
        self.game = game
        if root:
            self.initial_gamestate = root.gamestate
            self.current_gamestate = root.gamestate
            # TODO is this okay? yes probably
            self.root = root
        else:
            self.initial_gamestate = game.get_initial_position()
            self.current_gamestate = game.get_initial_position()
            self.root = Node(prior=-1, gamestate=self.initial_gamestate)
        if DEBUG:
            print(self.initial_gamestate)
            print(f'With {self.root.visits} previous visits')
            print()
        self.score_func = score_func
        self.predict_func = predict_func if predict_func else partial(
            dummy_predict, game.move_cardinality)

    def normalize_action_probs(self, action_probs, gamestate: Gamestate):
        norm_action_probs = np.array([prob * mask for prob,
                             mask in zip(action_probs, gamestate.get_legal_moves())])
        return norm_action_probs / np.sum(norm_action_probs)

    def run_simulation(self):

        self.current_gamestate = self.initial_gamestate
        node = self.root
        search_path = [node]

        # using tree policy to find leaf node
        if DEBUG:
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
        # print(np.array([node.gamestate.get_int_list_representation()]))
        # print(len(np.array([node.gamestate.get_int_list_representation()])[0]))
        action_probs = self.predict_func(
            node.gamestate.get_int_list_representation())
        # print(action_probs)
        norm_action_probs = self.normalize_action_probs(
            action_probs=action_probs, gamestate=node.gamestate)
        if DEBUG:
            print(f'Action probabilities for expansion:\n\t{norm_action_probs}')
            print(f'Expanding...')
        node.expand(action_probs=norm_action_probs)

        leaf_node = node
        self.current_gamestate = leaf_node.gamestate
        # TODO constant
        epsilon = 0.1
        if DEBUG:
            print(f'Starting rollout')
        while reward == None:  # i.e. we are not in a terminal state
            action_probs = self.predict_func(
                self.current_gamestate.get_int_list_representation()
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
            if choose_random < epsilon:
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

    def increment_reward(self, search_path : list[Node], reward: int):
        player_in_root = self.initial_gamestate.get_agent_index()
        offset = 1 if ( player_in_root == 0 and reward == 1 ) or (player_in_root == 1 and reward == -1) else 0
        if DEBUG:
            print(f'Offset: {offset}')
            print(f'Player: {player_in_root}')
            print(f'Reward: {reward}')
        for node in search_path:
            node.visits += 1
        for node in search_path[offset::2]:
            node.value += 1

    def run_simulations(self, n: int) -> np.ndarray:
        for _ in range(n):
            self.run_simulation()
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
    game = Hex(4)
    gs = game.get_initial_position()
    mcts = MCTS(game, score_func=UCB)
    for i in range(1):
        mcts.run_simulation()

    print("Best place to start:")
    for _, child in mcts.root.children.items():
        print(child.value)
