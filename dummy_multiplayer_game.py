from __future__ import annotations
from agent import Agent
from game import Game, Gamestate, Move
from copy import copy

from random import getrandbits


class DummyMultiAgentGame(Game):

    num_agents : int = 2

    def __init__(self, num_agents) -> None:
        self.state_representation_lenght = 1
        self.move_cardinality = 1
        self.num_agents = num_agents

    def get_initial_position(self) -> DummyMultiAgentGameState:
        return DummyMultiAgentGameState(self.num_agents)

class DummyMultiAgentGameState(Gamestate):

    def __init__(self, num_agents) -> None:
        self.agent_index = 0
        self.num_agents = num_agents

    def play(self, move : Move) -> DummyMultiAgentGameState:
        new_gs = copy(self)
        new_gs.agent_index = (new_gs.agent_index + 1) % self.num_agents
        return new_gs

    def play_move_int(self, move_idx : int) -> DummyMultiAgentGameState:
        return self.play(None)

    def is_legal_move(self, move : Move) -> bool:
        return True

    def get_legal_moves(self) -> list[bool]:
        return [True]

    def create_move(self, int_representation : int) -> Move:
        return DummyMultiAgentMove()

    def reward(self) -> int:
        win = bool(getrandbits(1))
        return 1 if win else None

    def get_agent_index(self) -> int:
        return self.agent_index

    def get_int_list_representation(self) -> list[int]:
        [0]

class DummyMultiAgentMove(Move):

    def get_int_representation(self):
        return 0

    def from_int_representation(self, int_representation : int) -> Move:
        return self

class DummyMultiAgent(Agent):
    def get_next_move(self, gamestate : Gamestate):
        return DummyMultiAgentMove()