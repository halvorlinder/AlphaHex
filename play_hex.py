from RL import get_neural_agents
from connect2 import Connect2
from game_player import GameInstance
from gen_agents import HumanAgent
from hex import Hex
import CONSTANTS
from tic_tac_toe import TicTacToeGame

if __name__ == '__main__':
    match CONSTANTS.GAME:
        case CONSTANTS.TrainingGame.HEX:
            game = Hex(CONSTANTS.HEX_SIZE)
        case CONSTANTS.TrainingGame.TTT:
            game = TicTacToeGame()
        case CONSTANTS.TrainingGame.C2:
            game = Connect2()
    agents = get_neural_agents(game, CONSTANTS.NEURAL_AGENT_TIMESTAMP)
    for agent in agents:
        game = GameInstance(game, [agent, HumanAgent()], True)
        game.start()
        game = GameInstance(game, [HumanAgent(), agent], True)
        game.start()