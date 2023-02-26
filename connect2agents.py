from agent import SpecializedAgent
from connect2 import Connect2Gamestate, Connect2Move


class HumanConnect2Agent(SpecializedAgent):

    def __init__(self) -> None:
        # self.name = input("Please enter your name: ")
        self.name = 'Human'

    def get_next_move(self, _: Connect2Gamestate) -> int:
        while True:
            try: 
                square = int(input("Select a square (0-3)"))
                return square
            except:
                print("Illegal move, try again")
