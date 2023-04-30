from RL import get_neural_agents
from connect2 import Connect2
from game_player import GameInstance
from gen_agents import HumanAgent
from hex import Hex
import CONSTANTS
from tic_tac_toe import TicTacToeGame

from ANET import ConvResNet, PytorchNN
from RL import NeuralAgent

if __name__ == '__main__':
    match CONSTANTS.GAME:
        case CONSTANTS.TrainingGame.HEX:
            game = Hex(CONSTANTS.HEX_SIZE)
        case CONSTANTS.TrainingGame.TTT:
            game = TicTacToeGame()
        case CONSTANTS.TrainingGame.C2:
            game = Connect2()
    # agents = get_neural_agents(game, CONSTANTS.NEURAL_AGENT_TIMESTAMP)
    game = Hex(7)
    net = ConvResNet(
                board_dimension_depth=game.conv_net_layers, 
                channels=CONSTANTS.CHANNELS_RES_BLOCK, 
                num_res_blocks=CONSTANTS.NUMBER_RES_BLOCKS, 
                board_state_length=game.state_representation_length, 
                move_cardinality=game.move_cardinality, 
            )
    pynet = PytorchNN()
    agents = []
    for i in range(49, 50, 10):
        pynet.load(net, f'agents/hex7/MCTS/{i}')
        agents.append(NeuralAgent(pynet, f'{i+1}'))
    for agent in agents:
        game = GameInstance(game, [agent, HumanAgent()], True)
        game.start()
        game = GameInstance(game, [HumanAgent(), agent], True)
        game.start()