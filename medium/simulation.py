# Simplify and Scale

# Handle Imports
import pygame
from collections import deque
from resources import sprite_path
import sys

# import random

# # Set the seed
# random.seed(42)

# # Generate 10 random numbers
# numbers = [random.random() for _ in range(10)]

# print(numbers)




# import numpy as np

# rng = np.random.default_rng(seed=42)
# numbers = rng.random(10)

# print(numbers)

# def get_random_foods(width, height, num = 10):
#     """
#     Get a random arrangement of foods about the arena
#     """
#     foods = {}
#     for i in range(num):
#         stop_condition = False
#         while not stop_condition:
#             x = np.random.randint(20, width- 20)
#             y = np.random.randint(20, height - 20)
#             rand_pos = (x, y)
#             if rand_pos not in foods:
#                 stop_condition = True
#                 foods[rand_pos] = 1
#     return foods

# def encode_foods(player_x, player_y, foods):
#     """
#     Encode the foods dictionary into an observation space
#     digestible by the model
#     """

#     diffs_active = []
#     keys_active = []

#     diffs_inactive = []
#     keys_inactive = []
#     for fpos in foods.keys():
#         if foods[fpos] ==  1:
#             diffs_active.append(find_euclidean_distance([player_x, player_y], fpos))
#             keys_active.append(fpos)
#         else:
#             diffs_inactive.append(find_euclidean_distance([player_x, player_y], fpos))
#             keys_inactive.append(fpos)
    
#     sorted_pairs_active = sorted(zip(diffs_active, keys_active))
#     sorted_pairs_inactive = sorted(zip(diffs_inactive, keys_inactive))

#     foods_array = []

#     # Add the players position here
#     foods_array.append(player_x/WIDTH)
#     foods_array.append(player_y/HEIGHT)

#     for euc, fpos in sorted_pairs_active:
#         dx, dy = find_x_y_delta([player_x, player_y], fpos)
#         foods_array.append(normalize_delta(WIDTH, dx))
#         foods_array.append(normalize_delta(HEIGHT, dy))
#         foods_array.append(foods[fpos])

#     for euc, fpos in sorted_pairs_inactive:
#         dx, dy = find_x_y_delta([player_x, player_y], fpos)
#         foods_array.append(normalize_delta(WIDTH, dx))
#         foods_array.append(normalize_delta(HEIGHT, dy))
#         foods_array.append(foods[fpos])
    
#     # print(foods_array)
#     assert max(foods_array) <= 1
#     assert min(foods_array) >= 0

#     return np.array(foods_array, dtype=np.float32)


class Environment():
    """
    The actual game environment, player and AI
    interact with this in order to play the game
    """

    def __init__(self):
        """
        Set up the environment
        """
        self.define_variables()
        self.reset()
    
    def define_variables(self):
        """
        Define needed variables for the simulation
        """
        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        # Super small
        self.N_FOODS = 1
        self.RANDOM_TYPE = "fixed"
        self.ARENA_SIZE = 30
        self.PLAYER_SPEED = 1
        self.FPS = 12
        self.RENDER_MULT = 10

        # # Original 300x300
        # self.N_FOODS = 10
        # self.RANDOM_TYPE = "fixed"
        # self.ARENA_SIZE = 300
        # self.PLAYER_SPEED = 1
        # self.FPS = 120
        # self.RENDER_MULT = 1
    

        self.player_position = [self.ARENA_SIZE//2, self.ARENA_SIZE//2]
        

    def set_up_game(self):
        """
        Set up the Pygame environment
        """
        pygame.init()

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 24)

        self.screen = pygame.display.set_mode((self.ARENA_SIZE * self.RENDER_MULT, self.ARENA_SIZE * self.RENDER_MULT))
        pygame.display.set_caption("Arena")

        # Load the Sprites
        self.player_image = pygame.image.load(sprite_path).convert_alpha()
        self.player_rect = self.player_image.get_rect(
            center=((self.ARENA_SIZE // 2) * self.RENDER_MULT, (self.ARENA_SIZE // 2) * self.RENDER_MULT)
        )

        self.game_timer = 0

    def update_game_rendering(self):
        """
        Update the rendering of the game
        """
        # Fill the screen
        self.screen.fill(self.GREEN)

        # Draw the player
        self.player_rect.center = (
            int(self.player_position[0] * self.RENDER_MULT),
            int(self.player_position[1] * self.RENDER_MULT),
        )

        self.screen.blit(self.player_image, self.player_rect)

        # Now draw the foods

        # Increment the clock
        self.clock.tick(self.FPS)

        self.game_timer+=1/self.FPS

        self.text_clock = self.font.render(f"{self.game_timer:.2f}", True, (255, 255, 255))
        self.text_rect = self.text_clock.get_rect(
            topright=((self.ARENA_SIZE * self.RENDER_MULT) - 10, 10)
        )

        self.screen.blit(self.text_clock, self.text_rect)

        # Update display
        pygame.display.flip()

    def reset(self):
        """
        Reset the game
        """
        # Reset the player position to the center

        # Reset the foods
        if self.RANDOM_TYPE == "fixed":
            # Load a fixed randomness
            pass
        elif self.RANDOM_TYPE == "semi-random":
            # Load a semi random
            pass
        elif self.RANDOM_TYPE == "random":
            # Load a fully random food list
            pass
        else:
            raise Exception("Error, unrecognized random type or not set")

    def perform_action(self, action):
        """
        Simply Take an Action
        """
        if action == 0:  #up
            self.player_position[1]-=self.PLAYER_SPEED
        elif action == 1: # Upper Left
            self.player_position[0]-=self.PLAYER_SPEED
            self.player_position[1]-=self.PLAYER_SPEED
        elif action == 2: # Left
            self.player_position[0]-=self.PLAYER_SPEED
        elif action == 3: # Down Left
            self.player_position[0]-=self.PLAYER_SPEED
            self.player_position[1]+=self.PLAYER_SPEED
        elif action == 4: # Down
            self.player_position[1]+=self.PLAYER_SPEED
        elif action == 5: # Down Right
            self.player_position[0]+=self.PLAYER_SPEED
            self.player_position[1]+=self.PLAYER_SPEED
        elif action == 6: # Right
            self.player_position[0]+=self.PLAYER_SPEED
        elif action == 7: # Up Right
            self.player_position[0]+=self.PLAYER_SPEED
            self.player_position[1]-=self.PLAYER_SPEED
        else:
            pass

        # Fix the player on the arena

        if self.player_position[0] >= self.ARENA_SIZE:
            self.player_position[0] = self.ARENA_SIZE-1
        elif self.player_position[0] <= 0:
            self.player_position[0]=0

        if self.player_position[1] >= self.ARENA_SIZE:
            self.player_position[1] = self.ARENA_SIZE-1
        elif self.player_position[1] <= 0:
            self.player_position[1]=0

    def step(self, action):
        """
        Perform an Action and Progress the game
        """

        # Perform action
        self.perform_action(action)

        # Check if we have reached a food

        # Update Observation Space

        return 0, 0, False

    def handle_keypresses(self):
        """
        Handle the Keypresses
        """
        keys = pygame.key.get_pressed()
        key_flags = [keys[pygame.K_w], keys[pygame.K_a], keys[pygame.K_s],  keys[pygame.K_d]]
        value = 0
        for i, bit in enumerate(reversed(key_flags)):
            value |= bit << i

        if value in [8, 13]:
            action = 0
        elif value == 12:
            action = 1
        elif value in [4, 14]:
            action = 2
        elif value == 6:
            action = 3
        elif value in [2, 7]:
            action = 4
        elif value == 3:
            action = 5
        elif value in [1, 11]:
            action = 6
        elif value == 9:
            action = 7
        else:
            action = -1

        return action

    def check_for_pygame_events(self):
        """
        
        """
        # Check if the usre exits
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def play(self, ai = False):
        """
        Play the Game
        """
        self.reset()

        self.set_up_game()

        stop_condition = False
        total_reward = 0
        while not stop_condition:
            # Check for pygame events
            if not ai:
                self.check_for_pygame_events()

            # Take an action
            if ai:
                pass
                raise NotImplementedError
            else:
                action = self.handle_keypresses()

            # Take the action
            next_state, reward, done = self.step(action)
            total_reward+=reward

            self.update_game_rendering()
        # Rendering Steps
        for f in self.foods:
            pass

        if done:
            pass
        
        # Print the end stats to let the player know how they did!


if __name__ == "__main__":
    env = Environment()
    env.play()