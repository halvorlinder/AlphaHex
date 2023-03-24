from game_player import GameInstance
from gen_agents import HumanAgent, RandomAgent
from hex import Hex

game = Hex(4)
agents = [ HumanAgent(), RandomAgent() ]
game_instance = GameInstance(game, agents)
game_instance.start()
