from agent import Agent
from connect2 import Connect2
from connect2agents import HumanConnect2Agent
from game import Game
from hex import Hex
from hex_agents import MCTSAgent, MCTSHexAgent
from tic_tac_toe import TicTacToeGame


class GameInstance:
    def __init__(self, game: Game, agents: list[Agent], display=False) -> None:
        self.game = game
        self.agents = agents
        self.display = display

    def start(self) -> list[int]:
        gamestate = self.game.get_initial_position()
        agent_to_play_index = gamestate.get_agent_index()
        while gamestate.reward() == None:
            print(gamestate)
            agent_to_play_index = gamestate.get_agent_index()
            agent_to_play = self.agents[agent_to_play_index]
            gamestate = gamestate.play(gamestate.create_move(
                agent_to_play.get_next_move(gamestate)))
        print(gamestate.reward())
        if self.display:
            print(gamestate)
        return gamestate.reward()
    
if __name__ == '__main__':
    game = Connect2()
    game = GameInstance(game, [MCTSAgent(game, 'Bot', 200), HumanConnect2Agent()][::-1], True)
    game.start()