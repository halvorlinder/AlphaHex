import copy
import numpy as np
from game import Game, Gamestate, Move

class Connect2(Game):

    def __init__(self) -> None:
        self.move_cardinality = 4
        self.state_representation_length = 4

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
    
    def from_int_list_representation(self, list_rep: list[int]) -> Gamestate:
        return Connect2Gamestate.from_list(list_rep)
    
    def get_initial_position(self) -> Gamestate:
        return Connect2Gamestate()

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
        self.turn = 1

    def create_move(self, int_representation: int) -> Move:
        return Connect2Move(int_representation)
    
    def get_agent_index(self) -> int:
        return self.turn
    
    def get_int_list_representation(self) -> list[int]:
        return self.board_state
    
    def get_legal_moves(self) -> list[bool]:
        return list(np.array(self.board_state) == 0)
    
    def is_legal_move(self, move: Move) -> bool:
        return self.board_state[move.value] == 0
    
    def play(self, move: Move) -> Gamestate:
        assert(self.board_state[move.value] == 0)
        new_gs = copy.deepcopy(self)
        new_gs.board_state[move.value] = self.turn
        new_gs.turn *= -1
        return new_gs
    
    def play_move_int(self, move_idx: int) -> Gamestate:
        move = self.create_move(move_idx)
        return self.play(move=move)

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
    
    def from_list(l : list[int]):
        state = Connect2Gamestate()
        state.board_state = l
        placed = np.sum(np.abs(np.array(l)))
        state.turn = 1 if placed % 2 == 0 else -1
        return state
    
    def __str__(self) -> str:
        return "|" + str(self.board_state[0]) + "|" + str(self.board_state[1]) + "|" + str(self.board_state[2]) + "|" + str(self.board_state[3]) + "|"

class Connect2Move(Move):

    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def get_int_representation(self) -> int:
        return self.value
    
    def from_int_representation(self, int_representation: int) -> Move:
        return Connect2Move(value=int_representation)

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
