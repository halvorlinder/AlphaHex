from hex import Hex, HexState

game = Hex(4)
gs = game.get_initial_position()
print(gs.get_layered_representation())

gs2 = HexState.from_list([
    [0,0,1,2],
    [0,1,0,2],
    [1,0,1,2],
    [0,2,1,0],
    ])

print(gs2.get_layered_representation())