from agent import Agent
from game import Game
class GameInstance:
    def __init__(self, game : Game, agents: list[Agent]) -> None:
        self.game = game
        self.agents = agents

    def start(self) -> int:
        gamestate = self.game.get_initial_position()
        while gamestate.reward() == None:
            agent_to_play_index = gamestate.get_agent_index()
            agent_to_play = self.agents[agent_to_play_index]
            print(gamestate)
            print(agent_to_play)
            gamestate = gamestate.play(gamestate.create_move(agent_to_play.get_next_move(gamestate)))
            print(gamestate)
        return agent_to_play_index
