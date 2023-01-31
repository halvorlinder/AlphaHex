from agent import Agent
from game import Game


class GameInstance:
    def __init__(self, game: Game, agents: list[Agent], display=False) -> None:
        self.game = game
        self.agents = agents
        self.display = display

    def start(self) -> int:
        gamestate = self.game.get_initial_position()
        agent_to_play_index = gamestate.get_agent_index()
        while gamestate.reward() == None:
            agent_to_play_index = gamestate.get_agent_index()
            agent_to_play = self.agents[agent_to_play_index]
            gamestate = gamestate.play(gamestate.create_move(
                agent_to_play.get_next_move(gamestate)))
        if self.display:
            print(gamestate)
        return agent_to_play_index
