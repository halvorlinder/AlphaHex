from __future__ import annotations
from copy import deepcopy
import itertools

import numpy as np
from game import Game, Gamestate, Move

from representations import StateRepresentation


class TicTacToeGame(Game):

    wins = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]

    def __init__(self) -> None:
        self.state_representation_length: int = 9
        self.move_cardinality: int = 9
        self.num_agents: int = 2
        self.conv_net_layers: int = 3

    def get_initial_position(self) -> TicTacToeState:
        return TicTacToeState()

    def from_int_list_representation(self, list_rep: list[int]) -> TicTacToeState:
        raise NotImplementedError()


class TicTacToeState(Gamestate):

    def __init__(self) -> None:
        self.board = [[0]*3 for _ in range(3)]
        self.turn = 0

    def play(self, move: TicTacToeMove) -> TicTacToeState:
        if not self.is_legal_move(move):
            raise ValueError(f'Illegal move ({self.y}, {self.x})')
        new_gs = deepcopy(self)
        new_gs.board[move.y][move.x] = self.turn + 1
        new_gs.turn = 1 if self.turn == 0 else 0
        return new_gs

    def play_move_int(self, move_idx: int) -> TicTacToeState:
        move = TicTacToeMove.from_int_representation(move_idx)
        return self.play(move)

    def is_legal_move(self, move: TicTacToeMove) -> bool:
        return not (move.y < 0 or move.y > 2 or move.x < 0 or move.x > 2 or self.board[move.y][move.x] != 0)

    def get_legal_moves(self) -> list[bool]:
        return [self.board[i//3][i % 3] == 0 for i in range(9)]

    def create_move(self, int_representation: int) -> TicTacToeMove:
        return TicTacToeMove.from_int_representation(int_representation)

    def reward(self) -> list[int]:
        reward = [0,0]
        for p in (0,1):
            for win in TicTacToeGame.wins:
                if p+1 == self.get(win[0]) and p+1 == self.get(win[1]) and p+1 == self.get(win[2]):
                    reward[p] = 1
                    return reward
        if self.full():
            reward = [0.5,0,5]
            return reward 
        return None
        
    def full(self) -> bool:
        return all(map(lambda p: p!=0, itertools.chain(*self.board)))

    def get(self, pos : tuple[int, int]) -> int:
        return self.board[pos[0]][pos[1]]

    def get_agent_index(self) -> int:
        return self.turn

    def get_representation(self, representation: StateRepresentation) -> np.ndarray:
        match representation:
            case StateRepresentation.LAYERED:
                raise NotImplementedError()
            case StateRepresentation.FLAT:
                return np.array(list(map(lambda p: 1 if p==self.turn+1 else 0 if p==0 else -1, itertools.chain(*self.board))))

    def __str__(self) -> str:
        return '\n'.join([''.join(map(str, row)) for row in self.board])

class TicTacToeMove(Move):

    def __init__(self, y: int, x: int) -> None:
        self.x = x
        self.y = y

    def get_int_representation(self) -> int:
        return self.y*3 + self.x

    def from_int_representation(int_representation: int) -> TicTacToeMove:
        return TicTacToeMove(int_representation//3, int_representation % 3)
