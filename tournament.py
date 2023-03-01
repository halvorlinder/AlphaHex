from itertools import combinations
import numpy as np 
import matplotlib.pyplot as plt
from math import comb
import os

from agent import Agent
from game import Game
from game_player import GameInstance
from gen_agents import RandomAgent


class TournamentPlayer:
    
    def __init__(self, game : Game, agents : list[Agent], num_matches : int, display = False) -> None:
        self.agents = agents
        self.num_matches = num_matches
        self.scores = np.zeros((len(agents),)*game.num_agents, int)
        self.game = game
        self.display = display
        self.agents_wins = [0]*len(agents)

    def play_tournament(self) -> tuple[np.ndarray[int], list[int]]:
        for enum_agents in combinations(enumerate(self.agents), self.game.num_agents):
            for _ in range(self.num_matches):
                agents = list(map(lambda enum_agent: enum_agent[1], enum_agents))
                indices = list(map(lambda enum_agent: enum_agent[0], enum_agents))
                game_inst = GameInstance(self.game, agents, self.display)
                scores = game_inst.start()
                for i, score in enumerate(scores):
                    current = indices[i]
                    other = set(indices) 
                    other.remove(current)
                    other = sorted(list(other))
                    score_index = tuple([ current ] + other)
                    self.scores[score_index]+=score
                    self.agents_wins[current]+=score
                if self.display:
                    self.plot_matchup(indices)
                    os.system('clear')
        if self.display:
            self.plot_wins()
                
        return self.scores, self.agents_wins

    def plot_matchup(self, players:list[int]) -> None:
        agents = list(map(lambda index: self.agents[index].name, players))
        indices = list(map(lambda player : tuple([player]+  list(set(players)-set([player]))), players))
        scores = list(map(lambda index: self.scores[index], indices))
        
        plt.figure(1)
        plt.bar(agents, scores)
        plt.ylim(0,self.num_matches) 
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    def plot_wins(self) -> None:
        plt.figure(1)
        plt.bar(list(map(lambda agent: agent.name ,self.agents)), self.agents_wins)
        plt.ylim(0,self.num_matches*comb(len(self.agents)-1, self.game.num_agents-1))
        plt.show()


    
if __name__ == '__main__':
    from hex import Hex
    from dummy_multiplayer_game import DummyMultiAgent, DummyMultiAgentGame
    game = hex(3)
    tourney = TournamentPlayer(game, [RandomAgent(game, '1'),RandomAgent(game, '2'),RandomAgent(game, '3'),RandomAgent(game, '4')], 10, True)
    scores, wins = tourney.play_tournament()
    print(wins)