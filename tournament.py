from itertools import combinations
import numpy as np 
import matplotlib.pyplot as plt
from math import comb
import os

from agent import Agent
from game import Game
from game_player import GameInstance
from hex import Hex
from hex_agents import RandomHexAgent
from dummy_multiplayer_game import DummyMultiAgent, DummyMultiAgentGame


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
                winner = indices[game_inst.start()]
                losers = set(indices)
                losers.remove(winner)
                losers = sorted(list(losers))
                score_index = tuple([ winner ] + losers)
                self.scores[score_index]+=1
                self.agents_wins[winner]+=1
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
        plt.pause(2)
        plt.clf()

    def plot_wins(self) -> None:
        plt.figure(1)
        plt.bar(list(map(lambda agent: agent.name ,self.agents)), self.agents_wins)
        plt.ylim(0,self.num_matches*comb(len(self.agents)-1, self.game.num_agents-1))
        plt.show()


    
if __name__ == '__main__':
    tourney = TournamentPlayer(Hex(6), [RandomHexAgent('1'), RandomHexAgent('2'), RandomHexAgent('3'), RandomHexAgent('4')], 10, True)
    scores, wins = tourney.play_tournament()
    print(wins)

    # tourney = TournamentPlayer(DummyMultiAgentGame(3), [DummyMultiAgent('1'), DummyMultiAgent('2'), DummyMultiAgent('3'), DummyMultiAgent('4'), DummyMultiAgent('5')], 10, True)
    # scores = tourney.play_tournament()
    # print(scores)
