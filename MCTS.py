import numpy as np
from game import Game, Gamestate, Move
from connect2 import Connect2, Connect2Gamestate
from hex import Hex, HexState

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
                new_move = game.create_move(action)
                new_gamestate = self.game.play(gamestate=self.gamestate, move=new_move)
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
        self.gamestate = gamestate
        self.score_func = score_func
        self.root = Node(prior=-1, gamestate=self.gamestate, game=self.game)

    def normalize_action_probs(self, action_probs, gamestate: Gamestate):
        norm_action_probs = [prob * mask for prob, mask in zip(action_probs, self.game.get_legal_moves(gamestate))]
        prob_sum = sum(norm_action_probs)
        norm_action_probs = [prob / prob_sum for prob in norm_action_probs]
        return norm_action_probs

    def run_simulation(self):
        node = self.root
        search_path = [node]
        
        while(len(node.children) > 0): # node has children, already expanded
            _, node = node.select_child(self.score_func)
            search_path.append(node)

        reward = node.gamestate.reward()

        while reward == None: # i.e. we are not in a terminal state
            value, action_probs = dummy_predict(node.gamestate, self.game.move_cardinality)
            norm_action_probs = self.normalize_action_probs(action_probs=action_probs, gamestate=node.gamestate)
            node.expand(action_probs=norm_action_probs)
            action, node = node.select_child(self.score_func)
            search_path.append(node)
            reward = node.gamestate.reward()
        
                
        for s_node in search_path:
            s_node.value += reward
            s_node.visits += 1
        
        if(len(search_path) < 4):
            print(len(search_path))
        node = search_path[0]
        self.root = search_path[0]

if __name__ == "__main__":
    game = Hex()
    gs = HexState(4)
    mcts = MCTS(game=game, gamestate=gs, score_func=UCB)
    for i in range(1):
        mcts.run_simulation()
    
    print("Best place to start:")
    print(mcts.root.children[0].value)
    print(mcts.root.children[1].value)
    print(mcts.root.children[2].value)
    print(mcts.root.children[3].value)

