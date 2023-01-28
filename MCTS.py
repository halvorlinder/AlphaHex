import numpy as np
import random
from game import Game, Gamestate, Move
from connect2 import Connect2, Connect2Gamestate
from hex import Hex, HexState, HexMove
import copy

class Node():

    def __init__(self, prior: float, game: Game, gamestate: Gamestate):
        self.prior = prior
        self.gamestate = gamestate
        self.game = game
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if(prob > 0):
                new_move = self.gamestate.create_move(action)
                new_gamestate = self.gamestate.play(move=new_move)
                self.children[action] = Node(prior=prob, gamestate=new_gamestate, game=self.game)

    def select_child(self, score_func):
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


def UCB(parent, child):
    prior_score = child.prior * np.sqrt(parent.visits) / (child.visits + 1)
    if child.visits > 0:
        value_score = child.value / child.visits
    else:
        value_score = 0
    
    return prior_score + value_score


def dummy_predict(gamestate: Gamestate, n: int):
    value = 0.5
    policy = [1/n] * n
    return value, policy

class MCTS():

    def __init__(self, game: Game, gamestate: Gamestate, score_func) -> None:
        self.game = game
        self.initial_gamestate = gamestate
        self.current_gamestate = gamestate
        self.score_func = score_func
        self.root = Node(prior=-1, gamestate=self.initial_gamestate, game=self.game)

    def normalize_action_probs(self, action_probs, gamestate: Gamestate):
        norm_action_probs = [prob * mask for prob, mask in zip(action_probs, gamestate.get_legal_moves())]
        prob_sum = sum(norm_action_probs)
        norm_action_probs = [prob / prob_sum for prob in norm_action_probs]
        return norm_action_probs

    def run_simulation(self):
        M = 100

        self.current_gamestate = self.initial_gamestate
        node = self.root
        search_path = [node]
        
        # using tree policy to find leaf node
        while(len(node.children) > 0): # node has children, already expanded
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
        
        value, action_probs = dummy_predict(node.gamestate, self.game.move_cardinality)
        norm_action_probs = self.normalize_action_probs(action_probs=action_probs, gamestate=node.gamestate)
        node.expand(action_probs=norm_action_probs)

        # perform M rollouts
        leaf_node = node
        for _ in range(M):
            self.current_gamestate = leaf_node.gamestate
            copy_game = copy.deepcopy(self.game)
            epsilon = 0.1
            while reward == None: # i.e. we are not in a terminal state
                value, action_probs = dummy_predict(self.current_gamestate, self.game.move_cardinality)
                # select next move in rollout phase
                choose_random = random.random()
                move = None
                if choose_random < epsilon:
                    true_idx = np.argwhere(np.array(self.current_gamestate.get_legal_moves()))
                    random_idx = random.randint(0, len(true_idx) - 1)
                    move = true_idx[random_idx][0]
                else:
                    move = np.argmax(action_probs)
                
                self.current_gamestate = self.current_gamestate.play_move_int(move_idx=move)
                reward = self.current_gamestate.reward()
                
            for s_node in search_path:
                s_node.value += reward
                s_node.visits += 1

if __name__ == "__main__":
    game = Hex()
    gs = HexState(4)
    mcts = MCTS(game=game, gamestate=gs, score_func=UCB)
    for i in range(100):
        mcts.run_simulation()
    
    print("Best place to start:")
    for _, child in mcts.root.children.items():
        print(child.value)

