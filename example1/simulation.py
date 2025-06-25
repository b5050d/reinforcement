import numpy as np



def pick_light_location():
    ans = np.random.randint(0,2)
    if ans == 0:
        light_location = "left"
    else:
        light_location = "right"
    return light_location

def pick_spawn_point():
    ans = np.random.randint(3, 8)
    return ans


class Board():
    """
    Game board
    """
    def __init__(self): 
        self.light_location = pick_light_location()
        self.spawn_point = pick_spawn_point()

    def calculate_light_intensities(self):
        self.light_looking_left = []
        self.light_looking_right = []

        for i in range(0, 11):
            if self.light_location == "left":
                self.light_looking_right += 

class Player():
    def __init__(self):
        self.board = Board()

    
















