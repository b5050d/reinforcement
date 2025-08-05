"""
The Actual Simulation is made here

The user can run this script to play the game
"""

# Handle Imports
import pygame
from medium.resources import sprite_path
import sys
from medium.food_matrix import (
    get_random_food_positions,
    compute_directional_signals
)
from medium.spatial_methods import find_euclidean_distance


class Environment():
    """
    The actual game environment, player and AI
    interact with this in order to play the game
    """

    def __init__(self, config):
        """
        Set up the environment
        """
        self.config = config
        self.define_variables()
        self.reset()
    
    def define_variables(self):
        """
        Define needed variables for the simulation
        """

        self.N_FOODS = self.config['N_FOODS']
        self.RANDOM_TYPE = self.config['RANDOM_TYPE']
        self.ARENA_SIZE = self.config['ARENA_SIZE']
        self.PLAYER_SPEED = self.config['PLAYER_SPEED']
        self.FPS = self.config['FPS']
        self.RENDER_MULT = self.config['RENDER_MULT']
        self.PLAYER_RADIUS = self.config['PLAYER_RADIUS']

        # # Super small
        # self.N_FOODS = 1
        # self.RANDOM_TYPE = "fixed"
        # self.ARENA_SIZE = 20
        # self.PLAYER_SPEED = 1
        # self.FPS = 12
        # self.RENDER_MULT = 10
        # self.PLAYER_RADIUS = .5

        # # Original 300x300
        # self.N_FOODS = 10
        # self.RANDOM_TYPE = "fixed"
        # self.ARENA_SIZE = 300
        # self.PLAYER_SPEED = 1
        # self.FPS = 120
        # self.RENDER_MULT = 1

        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

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
        for f in self.foods:
            render_pos = (f[0]*self.RENDER_MULT, f[1]*self.RENDER_MULT)
            pygame.draw.circle(self.screen, self.YELLOW, render_pos, 5)

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
        self.player_position = [self.ARENA_SIZE//2, self.ARENA_SIZE//2]

        # Reset the foods
        if self.RANDOM_TYPE == 0:
            # Load a fixed randomness
            self.foods = get_random_food_positions(self.ARENA_SIZE, self.N_FOODS, 42)
        elif self.RANDOM_TYPE == 1:
            # Load a semi random
            pass
        elif self.RANDOM_TYPE == 2:
            # Load a fully random food list
            pass
        else:
            raise Exception("Error, unrecognized random type or not set")
        
        return compute_directional_signals(self.player_position, self.foods)

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
        reward = 0

        # Perform action
        self.perform_action(action)

        # Check if we have reached a food
        foods = []
        for f in self.foods:
            dist = find_euclidean_distance(f, self.player_position)

            if dist <= self.PLAYER_RADIUS:
                pass
                reward += 1
            else:
                foods.append(f)
        self.foods = foods

        if self.foods == []:
            done = True
        else:
            done = False

        # Update Observation Space
        next_state = compute_directional_signals(self.player_position, self.foods)

        return next_state, reward, done

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

    def play(self, ai=False):
        """
        Play the Game
        """
        print("Entered the Play Method")
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
                break
        
        # Print the end stats to let the player know how they did!
        print("Finished running the game")
        pygame.quit()
        sys.exit()
        return

if __name__ == "__main__":
    env = Environment()
    env.play()