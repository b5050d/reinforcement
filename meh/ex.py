"""

"""

import numpy as np


BOARD_LENGTH = 64

def get_intensity(light_pos, player_pos, decay=.95):
    diff = abs(light_pos - player_pos)
    return decay**diff

class Environment():

    def __init__(self):
        """
        Start 
        """
        self.state = 0

    def reset(self):
        """
        Reset 
        """
        self.light_location = np.random.randint((0, BOARD_LENGTH))
        self.player_location = np.random.randint((0, BOARD_LENGTH))
        while self.player_location != self.light_location:
            self.player_location = np.random.randint((0, BOARD_LENGTH))

    def get_state(self):

    def perform_action(self, action):
        self.player_location = 
        


class Player:
    def __init__(self):
        pass

    def 

def player(self):
    pass