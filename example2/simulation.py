import numpy as np
from matplotlib import pyplot as plt

GRID_SIZE = 10
MAX_TURNS = 100

class Board:
    def __init__(self):
        self.define_game_board()

    def define_game_board(self):
        self.board = np.zeros((GRID_SIZE, GRID_SIZE))

        # Pick a random position to start and a random light position
        self.starting_pos = (
            np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
            np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
        )     
        self.finish_pos = (
            np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
            np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
        )     
        while self.starting_pos == self.finish_pos:
            self.finish_pos = (
                np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
                np.random.randint(int(.1 * GRID_SIZE),int(.9 * GRID_SIZE)),
            )            
    def clip_player_pos(self, position):
        x = position[0]
        y = position[1]

        new_pos = (0,0)

        if x >= GRID_SIZE:
            x = GRID_SIZE-1
        elif x<0:
            x=0
        if y >= GRID_SIZE:
            y = GRID_SIZE-1
        elif y<0:
            y =0

        return (x,y)
            

class Random_Player:
    def __init__(self):
        self.current_position = (-1, -1)

    def move(self):
        ans = np.random.randint(0,4)
        return ans


class Round:
    def __init__(self):
        self.player = Random_Player()
        self.board = Board()
        self.turn_count = 0
        self.score = 0
        self.trail = [self.board.starting_pos]
        self.player_position = self.board.starting_pos

    def turn(self):
        # Is the player on the goal
        if self.player_position == self.board.finish_pos:
            print("Won the game, +1 points")
            self.score+=1
            return 0

        # Are we out of turns?
        if self.turn_count == MAX_TURNS - 1:
            print("Out of Turns")
            return 0
        
        # Ok move
        move = self.player.move()
        if move == 0:
            self.player_position = (self.player_position[0], self.player_position[1]+1)
        elif move == 1:
            self.player_position = (self.player_position[0]+1, self.player_position[1])
        elif move == 2:
            self.player_position = (self.player_position[0], self.player_position[1]-1)
        elif move == 3:
            self.player_position = (self.player_position[0]-1, self.player_position[1])

        # Clip the player position to fit on the board
        # input(self.player_position)
        self.player_position = self.board.clip_player_pos(self.player_position)

        self.turn_count+=1
        self.trail.append(self.player_position)

        return 1
    
    def play(self):
        ans = 1
        while ans == 1:
            ans = self.turn()

if __name__ == "__main__":
    a = Round()
    a.play()

    # Plot the trail!
    mat = a.board.board
    mat[a.board.finish_pos]=1

    mats = []
    for t in a.trail:
        cp = mat.copy()
        cp[t]=.5
        mats.append(cp)

    plt.ion()
    img = plt.imshow(mat)
    for mat in mats:

        img.set_data(mat)
        plt.draw()
        plt.pause(.1)
    plt.ioff()


    # plt.show()


# plt.ion()  # turn on interactive mode
# fig, ax = plt.subplots()
# img = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=1)

# for frame in frames:
#     img.set_data(frame)
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     time.sleep(0.1)  # control frame rate

# plt.ioff()  # optional: turn off interactive mode
# plt.show()