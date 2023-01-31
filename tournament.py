from itertools import combinations
import numpy as np 

from agent import Agent
from game import Game
from game_player import GameInstance
from hex import Hex
from hex_agents import RandomHexAgent


class TournamentPlayer:
    
    def __init__(self, game : Game, agents : list[Agent], num_matches : int) -> None:
        self.agents = agents
        self.num_matches = num_matches
        self.scores = np.zeros((len(agents),)*game.num_agents, int)
        self.game = game

    def play_tournament(self) -> np.ndarray:
        for enum_agents in combinations(enumerate(self.agents), self.game.num_agents):
            for _ in range(self.num_matches):
                agents = list(map(lambda enum_agent: enum_agent[1], enum_agents))
                indices = list(map(lambda enum_agent: enum_agent[0], enum_agents))
                game_inst = GameInstance(self.game, agents)
                winner = indices[game_inst.start()]
                losers = set(indices)
                losers.remove(winner)
                losers = sorted(list(losers))
                score_index = tuple([ winner ] + losers)
                self.scores[score_index]+=1
        return self.scores

    
if __name__ == '__main__':
    tourney = TournamentPlayer(Hex(4), [RandomHexAgent(), RandomHexAgent(), RandomHexAgent(), RandomHexAgent()], 5)
    scores = tourney.play_tournament()
    print(scores)
