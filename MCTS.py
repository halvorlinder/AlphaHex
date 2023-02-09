from functools import partial
import numpy as np
import random
from game import Game, Gamestate, Move
from connect2 import Connect2, Connect2Gamestate
from hex import Hex
import copy
from torch import float32, float16, float64


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
                new_move = self.gamestate.create_move(action)
                new_gamestate = self.gamestate.play(move=new_move)
                self.children[action] = Node(
                    prior=prob, gamestate=new_gamestate)

    def select_child(self, score_func):
        # TODO Is this not just argmax, if so refactor to make it more readable (np.argmax)
        max_score = -float("inf")
        selected_action = None
        selected_child = None
        for action, child in self.children.items():
            score = score_func(parent=self, child=child)
            if score > max_score:
                selected_action = action
                selected_child = child
                max_score = score
        return selected_action, selected_child


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
    value = 0.5
    policy = [1/n] * n
    return value, policy


class MCTS():

    def __init__(self, game: Game, score_func=UCB, root: Node = None, predict_func=None) -> None:
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
        self.score_func = score_func
        self.predict_func = predict_func if predict_func else partial(
            dummy_predict, game.move_cardinality)

    def normalize_action_probs(self, action_probs, gamestate: Gamestate):
        # TODO np.norm?
        norm_action_probs = [prob * mask for prob,
                             mask in zip(action_probs, gamestate.get_legal_moves())]
        prob_sum = sum(norm_action_probs)
        norm_action_probs = [prob / prob_sum for prob in norm_action_probs]
        return norm_action_probs

    def run_simulation(self):
        # TODO remove constant
        M = 1

        self.current_gamestate = self.initial_gamestate
        node = self.root
        search_path = [node]

        # using tree policy to find leaf node
        while (len(node.children) > 0):  # node has children, already expanded
            _, node = node.select_child(self.score_func)
            search_path.append(node)
            self.current_gamestate = node.gamestate

        # expanding leaf node if not in terminal state
        reward = node.gamestate.reward()
        if reward != None:
            for s_node in search_path:
                s_node.value += reward
                s_node.visits += 1
            return

        # print(node.gamestate)
        # print(np.array([node.gamestate.get_int_list_representation()]))
        # print(len(np.array([node.gamestate.get_int_list_representation()])[0]))
        action_probs = self.predict_func(
            node.gamestate.get_int_list_representation())
        # print(action_probs)
        norm_action_probs = self.normalize_action_probs(
            action_probs=action_probs, gamestate=node.gamestate)
        node.expand(action_probs=norm_action_probs)

        # perform M rollouts
        leaf_node = node
        for _ in range(M):
            self.current_gamestate = leaf_node.gamestate
            # TODO constant
            epsilon = 0.1
            while reward == None:  # i.e. we are not in a terminal state
                action_probs = self.predict_func(
                    self.current_gamestate.get_int_list_representation())
                norm_action_probs = self.normalize_action_probs(
                    action_probs=action_probs, gamestate=self.current_gamestate)
                # select next move in rollout phase
                choose_random = random.random()
                move = None
                if choose_random < epsilon:
                    true_idx = np.argwhere(
                        np.array(self.current_gamestate.get_legal_moves()))
                    random_idx = random.randint(0, len(true_idx) - 1)
                    move = true_idx[random_idx][0]
                else:
                    move = np.argmax(norm_action_probs)

                # TODO this can play illegal moves
                self.current_gamestate = self.current_gamestate.play_move_int(
                    move_idx=move)
                reward = self.current_gamestate.reward()

            for s_node in search_path:
                s_node.value += reward
                s_node.visits += 1

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
    for i in range(10000):
        mcts.run_simulation()

    print("Best place to start:")
    for _, child in mcts.root.children.items():
        print(child.value)
