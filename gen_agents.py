from random import randrange

import numpy as np
from MCTS import MCTS, Node
from agent import Agent
from game import Game, Gamestate

class RandomAgent(Agent):

    def __init__(self, game : Game, name : str) -> None:
        self.game = game
        self.name = name

    def get_next_move(self, gamestate: Gamestate) -> int:
        while True:
            n = randrange(0, self.game.move_cardinality)
            if gamestate.is_legal_move(gamestate.create_move(n)):
                return n

class MCTSAgent(Agent):

    def __init__(self, game: Game, name : str, n_rollouts : int) -> None:
        self.name = name
        self.n = n_rollouts
        self.game = game

    def get_next_move(self, gamestate: Gamestate) -> int:
        mcts = MCTS(self.game, root = Node(gamestate))
        probs = mcts.run_simulations(self.n) 
        print(mcts.get_visits())
        move = np.argmax(probs)
        return move

class HumanAgent(Agent):

    def __init__(self) -> None:
        self.name = 'Human'

    def get_next_move(self, gamestate: Gamestate) -> int:
        while True:
            try: 
                return int(input("Provide int representation: \n"))
            except:
                print("Illegal move, try again")