from game_player import GameInstance
from hex import Hex
from hex_agents import HumanHexAgent, RandomHexAgent

game = Hex(4)
agents = [ HumanHexAgent(), RandomHexAgent() ]
game_instance = GameInstance(game, agents)
game_instance.start()
