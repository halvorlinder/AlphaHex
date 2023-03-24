from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import copy
import math
from typing import Callable, TYPE_CHECKING
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from game import Game, Gamestate, Move
from representations import StateRepresentation
# from agent import Agent
# from hex_agents import RandomHexAgent, HumanHexAgent

BOARD_SIZE = 4

# The six directions in the square array representation of the hex grid
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]


class bcolors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    PURPLE = '\033[35m'
    ENDC = '\033[0m'


class Player(Enum):
    P1 = auto()
    P2 = auto()

    def next_player(self) -> Player:
        match self:
            case Player.P1:
                return Player.P2
            case Player.P2:
                return Player.P1


class Piece(Enum):
    Played = Player
    Open = auto()

    def colored_string(piece) -> str:
        """Generates a colored string for a piece 

        Returns:
            str: A 0 character colored based on the type of piece
        """
        match piece:
            # Player one is blue
            case Player.P1:
                return f'{bcolors.BLUE}0{bcolors.ENDC}'
            case Player.P2:
                return f'{bcolors.RED}0{bcolors.ENDC}'
            case _:
                return '0'


class Hex(Game):

    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        self.move_cardinality = board_size * board_size
        self.board_size = board_size
        self.state_representation_length = self.move_cardinality
        self.conv_net_layers = 3

    def get_name(self) -> str:
        return f'hex{self.board_size}'

    def get_initial_position(self) -> HexState:
        return HexState(self.board_size)

    def from_int_list_representation(self, list_rep : list[int]) -> HexState:
        return HexState.from_list(np.split(np.array(list_rep), math.isqrt(len(list_rep)))) 

class HexState(Gamestate):

    def __init__(self, board_size: int) -> None:
        self.board_size = board_size
        self.board = [[Piece.Open for _ in range(
            board_size)] for _ in range(board_size)]
        self.turn = Player.P1
        self.win_path = []
        self.winner = None

    def play(self, move: HexMove) -> Gamestate:
        assert(self.is_open(move))
        new_gs = copy.deepcopy(self)
        new_gs.board[move.se_diag][move.ne_diag] = self.turn
        new_gs.turn = new_gs.turn.next_player()
        return new_gs
    
    def play_move_int(self, move_idx: int) -> Gamestate:
        move = self.create_move(move_idx)
        return self.play(move)

    def is_legal_move(self, move: HexMove) -> bool:
        match self.index(move):
            case Piece.Open:
                return True
            case _:
                return False

    def get_legal_moves(self) -> list[bool]:
        return [self.is_legal_move(HexMove.from_int_representation(m, self.board_size)) for m in range(self.board_size*self.board_size)]

    def create_move(self, int_representation: int) -> Move:
        return HexMove.from_int_representation(int_representation, self.board_size)

    def get_representation(self, representation: StateRepresentation) -> np.ndarray:
        match representation:
            case StateRepresentation.FLAT:
                return self.get_int_list_representation()
            case StateRepresentation.LAYERED:
                return self.get_layered_representation()
            case _ :
                raise('No representation format provided')

    def get_int_list_representation(self) -> list[int]:
        if(self.turn == Player.P1):
            return list(map(lambda piece: 0 if piece==Piece.Open else 1 if piece==Player.P1 else -1, np.array(self.board).flatten()))
        return list(map(lambda piece: 0 if piece==Piece.Open else -1 if piece==Player.P1 else 1, np.array(self.board).flatten()))

    def get_layered_representation(self) -> list[list[list[int]]]:
        return np.array([ list(map(lambda row: list(map(lambda p: 1 if p==piece else 0, row)), self.board)) for piece in [Piece.Open, Player.P1, Player.P2]])
        


    def from_list(l: list[list[int]]) -> HexState:
        """ Generates a gamestate from a grid of numbers (0,1,2)

        Args:
            l (list[list[int]]): 

        Returns:
            HexState: 
        """
        state = HexState(len(l))
        board = list(map(lambda diag: list(map(lambda n: Piece.Open if n ==
                     0 else Player.P1 if n == 1 else Player.P2, diag)), l))
        placed = sum(map(lambda diag: sum(
            map(lambda n: 0 if n == 0 else 1, diag)), l))
        state.board = board
        state.turn = Player.P1 if placed % 2 == 0 else Player.P2
        return state

    def index(self, move: HexMove) -> Piece:
        """
        Args:
            move (HexMove): A board position represented as a move

        Returns:
            Piece: The piece at the position 
        """
        return self.board[move.se_diag][move.ne_diag]

    def is_open(self, move : HexMove):
        return self.index(move) == Piece.Open


    def in_board(self, move: HexMove) -> bool:
        """
        Args:
            move (HexMove): The move to check 

        Returns:
            bool: The move in within the bounds of the board
        """
        return move.se_diag >= 0 and move.se_diag < self.board_size and move.ne_diag >= 0 and move.ne_diag < self.board_size

    def get_agent_index(self) -> int:
        match self.turn:
            case Player.P1:
                return 0
            case Player.P2:
                return 1

    def reward(self) -> int:
        start: list[HexMove] = [HexMove(0, ne, self.board_size) for ne in range(self.board_size)] if self.turn == Player.P2 else [
            HexMove(se, 0, self.board_size) for se in range(self.board_size)]
        end: Callable[[HexMove], bool] = lambda move: (
            move.se_diag if self.turn == Player.P2 else move.ne_diag) == self.board_size-1
        visited = [[False for _ in range(self.board_size)]
                   for _ in range(self.board_size)]
        for tile in start:
            if self.dfs(tile, visited, end):
                return [0,1] if self.turn == Player.P1 else [1,0]

    def dfs(self, tile: HexMove, visited: list[list[bool]], end: Callable[[HexMove], bool]) -> bool:
        if (not self.in_board(tile)) or self.index(tile) != self.turn.next_player() or visited[tile.se_diag][tile.ne_diag]:
            return False
        if end(tile):
            self.win_path = [tile.as_tuple()]
            self.winner = self.turn.next_player()
            return True
        visited[tile.se_diag][tile.ne_diag] = True
        on_path = any([self.dfs(HexMove(tile.se_diag+d[0], tile.ne_diag+d[1], self.board_size), visited, end) for d in DIRS])
        if on_path:
            self.win_path.append(tile.as_tuple())
        return on_path

    def get_diags(self) -> list[tuple[int, int]]:
        se_diag_starts = [(0, i) for i in range(
            self.board_size-1, -1, -1)] + [(i, 0) for i in range(1, self.board_size)]
        diag_lengths_top = [i for i in range(1, self.board_size)]
        diag_lengths = diag_lengths_top + \
            [self.board_size] + diag_lengths_top[::-1]
        return list(map(lambda t: [(t[0][0] + i, t[0][1] + i)
                     for i in range(t[1])], list(zip(se_diag_starts, diag_lengths))))

    def __str__(self) -> str:
        """Horrible function that creates a colored printable representation of the board

        Returns:
            str: The colored board
        """
        se_diag_starts = [(0, i) for i in range(
            self.board_size-1, -1, -1)] + [(i, 0) for i in range(1, self.board_size)]
        diag_lengths_top = [i for i in range(1, self.board_size)]
        diag_lengths = diag_lengths_top + \
            [self.board_size] + diag_lengths_top[::-1]
        diags = list(map(lambda t: [(t[0][0] + i, t[0][1] + i)
                     for i in range(t[1])], list(zip(se_diag_starts, diag_lengths))))
        res = " "*(self.board_size) + \
            f' {bcolors.PURPLE}#{bcolors.ENDC}' + '\n'
        for i, diag in enumerate(diags):
            diag_string = " ".join(map(lambda t: Piece.colored_string(
                self.index(HexMove(t[0], t[1], self.board_size))), diag))
            prefix = f' {bcolors.BLUE}#{bcolors.ENDC}' if i < self.board_size - \
                1 else f' {bcolors.PURPLE}#{bcolors.ENDC}' if i == self.board_size-1 else f' {bcolors.RED}#{bcolors.ENDC}'
            suffix = f'{bcolors.RED}#{bcolors.ENDC}' if i < self.board_size - \
                1 else f'{bcolors.PURPLE}#{bcolors.ENDC}' if i == self.board_size-1 else f'{bcolors.BLUE}#{bcolors.ENDC}'
            res += " "*((self.board_size*2-1-len(diag)*2+1)//2) + \
                prefix + diag_string + suffix + '\n'
        res += (" "*(self.board_size) +
                f' {bcolors.PURPLE}#{bcolors.ENDC}' + '\n')
        self.plot()
        return res

    def plot(self) -> None:
        plt.figure(2)
        plt.clf()
        ## Set axis limits
        plt.xlim(-1,2*self.board_size)
        plt.ylim(2*self.board_size + math.sqrt(3),-math.sqrt(3))

        G=nx.Graph()
        diags = self.get_diags()
        color_map = []
        size_map = [100]*(self.board_size*self.board_size)
        for i, diag in enumerate(diags):
            for j, pos in enumerate(diag):
                G.add_node(str(pos),pos=(((self.board_size*2-1-len(diag)*2+1)//2)/2+j, i*math.sqrt(3)/2))
                tile = HexMove(pos[0], pos[1], self.board_size)
                piece = self.index(tile)
                color_map.append('yellow' if piece == Piece.Open else 'blue' if piece==Player.P1 else 'red')
                # neighbors = map(lambda move: (move.as_tuple()), filter(self.in_board, [HexMove(tile.se_diag+d[0], tile.ne_diag+d[1], self.board_size) for d in DIRS]))
                # for neighbor in neighbors:
                #     if not (pos in self.win_path and neighbor in self.win_path):
                #         G.add_edge(str(pos), str(neighbor), color = 'black', weight = 1)
        for (node_1, node_2) in zip(self.win_path, self.win_path[1:]):
            size_map[list(G.nodes).index(str(node_1))] = 300
            color = ('blue' if self.winner == Player.P1 else 'red')
            G.add_edge(str(node_1), str(node_2), color=color, weight =7)
        # size_map[list(G.nodes).index(str(node_2))] = 300
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G,pos, node_color = color_map,  edge_color=colors, width=weights, node_size = size_map )

        ## Draw the border lines
        (x1, y1) = (((self.board_size*2-2)//2)/2, -(math.sqrt(3)))
        (x2, y2) = (self.board_size, (self.board_size-1)*math.sqrt(3)/2)
        (x3, y3) = (x1, 2*(self.board_size-1)*math.sqrt(3)/2+(math.sqrt(3)))
        (x4, y4) = (-1, y2)
        plt.plot([x1,x2], [y1,y2], color = 'red', linewidth = 10)
        plt.plot([x2,x3], [y2,y3], color = 'blue', linewidth = 10)
        plt.plot([x3,x4], [y3,y4], color = 'red', linewidth = 10)
        plt.plot([x4,x1], [y4,y1], color = 'blue', linewidth = 10)

class HexMove(Move):

    def __init__(self, se_diag, ne_diag, board_size) -> None:
        # assert(se_diag >= 0 and se_diag < board_size and ne_diag >= 0 and ne_diag < board_size)
        self.se_diag = se_diag
        self.ne_diag = ne_diag
        self.board_size = board_size

    def from_int_representation(int_representation: int, board_size: int) -> HexMove:
        """ Creates a move from an integer representation 

        Args:
            int_representation (int): The integer representation 
            board_size (int): The board size

        Returns:
            HexMove: The corresponding move
        """
        return HexMove(int_representation // board_size, int_representation % board_size, board_size)

    def get_int_representation(self):
        return self.se_diag * self.board_size + self.ne_diag

    def as_tuple(self) -> tuple[int, int]:
        return (self.se_diag, self.ne_diag)

    def __str__(self) -> str:
        return f'({self.se_diag}, {self.ne_diag}) -> {self.get_int_representation()}'


# for ne in range(4):
#     for se in range(4):
#         print(ne, se, HexMove(ne, se, 4).get_int_representation())
# p = Player.P1
# print(p.next_player())

if __name__ == "__main__":
    size = 4
    game = Hex(size)
    # gs = HexState(size)
    # print(gs)
    # gs2 = game.play(gs, HexMove(0, 0, size))
    # print(gs2)

    # gs3 = game.play(gs2, HexMove(3, 0, size))
    # print(gs3)
    # moves = game.get_legal_moves(gs3)
    # print(moves)
    # move = game.create_move(7)
    # print(move)
    # gs4 = game.play(gs3, move)
    # print(gs4)
    # print(gs4.reward())
    gs = HexState.from_list(
        [[1, 2, 0, 0], [0, 1, 2, 0], [0, 0, 1, 2], [0, 0, 0, 1],])
    print(gs)
    reward = gs.reward()
    assert(reward == 1)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[1, 2, 1, 0], [0, 1, 2, 0], [0, 0, 1, 2], [0, 0, 0, 2],])
    print(gs)
    reward = gs.reward()
    assert(reward == None)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[2, 1, 2, 0], [0, 1, 2, 0], [0, 1, 0, 2], [0, 1, 0, 2],])
    print(gs)
    reward = gs.reward()
    assert(reward == 1)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[2, 2, 2, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],])
    print(gs)
    reward = gs.reward()
    assert(reward == None)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[1, 2, 0, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 2, 0, 1],])
    print(gs)
    reward = gs.reward()
    assert(reward == None)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[1, 1, 1, 0], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 1, 0],])
    print(gs)
    reward = gs.reward()
    assert(reward == -1)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[1, 0, 0, 1], [0, 2, 2, 0], [0, 2, 2, 0], [1, 0, 0, 1],])
    print(gs)
    reward = gs.reward()
    assert(reward == None)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[0, 0, 0, 1], [0, 2, 1, 0], [0, 1, 2, 0], [1, 0, 0, 2],])
    print(gs)
    reward = gs.reward()
    assert(reward == None)
    print(f'Reward was the expected {reward}')

    size = 5
    game = Hex(size)

    gs = HexState.from_list(
        [[0, 0, 0, 1, 0],[0, 0, 2, 1, 0],[0, 0, 2, 1, 0],[0, 0, 2, 1, 0],[0, 0, 2, 1, 0]])
    print(gs)
    reward = gs.reward()
    assert(reward == 1)
    print(f'Reward was the expected {reward}')

    gs = HexState.from_list(
        [[1, 1, 2, 1, 2],[1, 1, 2, 1, 1],[2, 2, 2, 1, 2],[1, 1, 2, 1, 2],[1, 2, 2, 1, 2]])
    print(gs)
    reward = gs.reward()
    assert(reward == 1)
    print(f'Reward was the expected {reward}')