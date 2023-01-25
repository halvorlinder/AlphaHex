import copy
import numpy as np
from game import Game, Gamestate, Move

class Connect2(Game):

    move_cardinality = 4

    def play(self, gamestate : Gamestate, move : Move) -> Gamestate:
        new_gamestate = copy.deepcopy(gamestate)
        new_gamestate.board_state[move.get_int_representation()] = new_gamestate.player_to_play
        new_gamestate.player_to_play *= -1
        return new_gamestate


    def is_legal_move(self, gamestate : Gamestate, move : Move) -> bool:
        return gamestate.board_state[move.get_int_representation] == 0

    def get_legal_moves(self, gamestate: Gamestate) -> list[bool]:
        return list(np.array(gamestate.board_state) == 0)

    def create_move(self, int_representation: int) -> Move:
        return Connect2Move(int_representation)

    def visualize_gamestate(self, gamestate : Gamestate):
        print("---------")
        print("|" + str(gamestate.board_state[0]) + 
            "|" + str(gamestate.board_state[1]) + 
            "|" + str(gamestate.board_state[2]) + 
            "|" + str(gamestate.board_state[3]) + 
            "|")
        print("---------")

class Connect2Gamestate(Gamestate):

    def __init__(self) -> None:
        super().__init__()
        self.board_state = [0, 0, 0, 0]
        self.player_to_play = 1

    def reward(self) -> int:
        for i in range(3):
            if self.board_state[i] == self.board_state[i+1] != 0:
                return self.board_state[i]
                

        full = True
        for i in range(4):
            if self.board_state[i] == 0:
                full = False
        
        if full:
            return 0

        return None

class Connect2Move(Move):

    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def get_int_representation(self) -> int:
        return self.value

if __name__ == "__main__":
    gs = Connect2Gamestate()
    game = Connect2()
    game.visualize_gamestate(gs)
    mv0 = Connect2Move(0)
    mv1 = Connect2Move(1)
    mv2 = Connect2Move(2)
    mv3 = Connect2Move(3)

    gs = game.play(gamestate=gs, move=mv0)
    game.visualize_gamestate(gs)

    gs = game.play(gamestate=gs, move=mv3)
    game.visualize_gamestate(gs)

    print(game.get_legal_moves(gs))
