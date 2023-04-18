from hex import Hex
import CONSTANTS
from RL import PytorchNN, NeuralAgent
from ANET import ConvResNet
from tournament import TournamentPlayer
from gen_agents import RandomAgent

if __name__ == "__main__":
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
    # for i in range(49, 50, 10):
    pynet.load(net, 'agents/hex7/MCTS/49')
    agents.append(NeuralAgent(pynet, f'MCTS'))

    pynet.load(net, 'agents/hex7/2023-04-18_13:48/0')
    pynet.load(net, 'agents/hex7/2023-04-18_13:48/1')
    pynet.load(net, 'agents/hex7/2023-04-18_13:48/2')
    agents.append(NeuralAgent(pynet, f'RL'))

    tourney = TournamentPlayer(game, agents, CONSTANTS.TOURNEY_NUM_GAMES, True, CONSTANTS.RANDOM_IN_TOURNEY)
    scores, wins = tourney.play_tournament()
    print(wins)