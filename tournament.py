from itertools import combinations, permutations
import numpy as np 
import matplotlib.pyplot as plt
from math import comb
import os
from RL import get_neural_agents

from agent import Agent
from game import Game
from game_player import GameInstance
from gen_agents import MCTSAgent, RandomAgent
from tic_tac_toe import TicTacToeGame


class TournamentPlayer:
    
    def __init__(self, game : Game, agents : list[Agent], num_matches : int, display = False, random:bool = False) -> None:
        self.agents = agents if not random else [RandomAgent(game, 'Random')] + agents
        self.num_matches = num_matches
        self.scores = np.zeros((len(self.agents),)*game.num_agents, int)
        self.game = game
        self.display = display
        self.agents_wins = [0]*len(self.agents)

    def play_tournament(self) -> tuple[np.ndarray[int], list[int]]:
        for enum_agents in combinations(enumerate(self.agents), self.game.num_agents):
            perms = list(permutations(enum_agents))
            for perm in perms:
                for _ in range(self.num_matches//len(list(perms))):
                    agents = list(map(lambda enum_agent: enum_agent[1], perm))
                    indices = list(map(lambda enum_agent: enum_agent[0], perm))
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
                        # os.system('clear')
                    # print(self.agents_wins)
        if self.display:
            self.plot_wins()
                
        print(self.agents_wins)
        return self.scores, self.agents_wins

    def plot_matchup(self, players:list[int]) -> None:
        agents = list(map(lambda index: self.agents[index].name, players))
        indices = list(map(lambda player : tuple([player]+  list(set(players)-set([player]))), players))
        scores = list(map(lambda index: self.scores[index], indices))
        
        plt.figure(1)
        plt.bar(agents, scores)
        plt.ylim(0,self.num_matches) 
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    def plot_wins(self) -> None:
        plt.figure(1)
        plt.bar(list(map(lambda agent: agent.name ,self.agents)), self.agents_wins)
        plt.ylim(0,self.num_matches*comb(len(self.agents)-1, self.game.num_agents-1))
        plt.draw()
        plt.show()


    
if __name__ == '__main__':
    from gen_agents import RandomAgent
    from hex import Hex
    from tic_tac_toe import TicTacToeGame
    from connect2 import Connect2
    import CONSTANTS

    match CONSTANTS.GAME:
        case CONSTANTS.TrainingGame.HEX:
            game = Hex(CONSTANTS.HEX_SIZE)
        case CONSTANTS.TrainingGame.TTT:
            game = TicTacToeGame()
        case CONSTANTS.TrainingGame.C2:
            game = Connect2()

    tourney = TournamentPlayer(game, get_neural_agents(game, CONSTANTS.NEURAL_AGENT_TIMESTAMP), CONSTANTS.TOURNEY_NUM_GAMES, True, CONSTANTS.RANDOM_IN_TOURNEY)
    scores, wins = tourney.play_tournament()
    print(wins)