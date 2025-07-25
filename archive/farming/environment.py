
import pygame
import numpy as np
import os
import sys
import random

# Set up the Resources needed
# sprite_path = os.path.join(currdir,"sprite.png")
currdir = os.path.dirname(__file__)
resource_folder = os.path.join(os.path.dirname(currdir), "resources")
lester_path = os.path.join(resource_folder, "farmer_sprite.png")

farm_path = os.path.join(resource_folder, "farm_unplanted.png")
farm0_path = os.path.join(resource_folder, "farm_0.png")
farm1_path = os.path.join(resource_folder, "farm_1.png")
farm2_path = os.path.join(resource_folder, "farm_2.png")
farm3_path = os.path.join(resource_folder, "farm_3.png")
farm4_path = os.path.join(resource_folder, "farm_4.png")
farm5_path = os.path.join(resource_folder, "farm_5.png")

grass_path = os.path.join(resource_folder, "grass_static.png")
stone_path = os.path.join(resource_folder, "stone.png")

SIZE_MULT = 100 # FOR DISPLAY
GREEN = (100, 200, 100)
# PLAYER_POS = [WIDTH // 2, HEIGHT // 2]
PLAYER_SPEED = 1
FPS = 10

class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player_pos = [1,1]
        self.seed_count = 0
        self.harvested_count = 0
        self.farm_tiles = [-1, -1, -1]
        self.farm_tiles_counters = [0, 0, 0]

    def step(self, action):
        """
        Take an action
        """
        if action == 0:  #up
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 1: # Upper Left
            self.player_pos[0]-=PLAYER_SPEED
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 2: # Left
            self.player_pos[0]-=PLAYER_SPEED
        elif action == 3: # Down Left
            self.player_pos[0]-=PLAYER_SPEED
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 4: # Down
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 5: # Down Right
            self.player_pos[0]+=PLAYER_SPEED
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 6: # Right
            self.player_pos[0]+=PLAYER_SPEED
        elif action == 7: # Up Right
            self.player_pos[0]+=PLAYER_SPEED
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 8: # Harvest
            # Check the state of the tile we are on
            print("Harvesting")
            # Check if we are on grass
            if self.player_pos[0] == 2:
                if random.random() <= .1:
                    print("Found a Seed!")
                    self.seed_count+=1
                # This is a grass tile
            elif self.player_pos[0] == 1:
                # This is a stone tile
                pass
            elif self.player_pos[0] == 0:
                # this is a farming tile
                # need to look up the state of the farm tile
                farm_idx = self.player_pos[1]
                if self.farm_tiles[farm_idx] == 5:
                    print("Yay Fully harvested!")
                    self.harvested_count+=1
                elif self.farm_tiles[farm_idx] > 0:
                    # Destroy the plant
                    print("Plant was too immature! you got nothing")
                self.farm_tiles[farm_idx] = -1
        
        elif action == 9: # Plant
            if self.player_pos[0] == 0:
                # this is a farming tile
                if self.seed_count == 0:
                    print("Cannot plant the seed, no seeds to plant")
                    pass
                else:
                    print("Planted Seed")
                    farm_idx = self.player_pos[1]
                    # Plant the seed
                    self.seed_count-=1
                    self.farm_tiles[farm_idx] = 0
                    self.farm_tiles_counters[farm_idx] = 10
        else:
            pass

        # clamp to grid bounds [0, 2]
        self.player_pos[0] = max(0, min(2, self.player_pos[0]))
        self.player_pos[1] = max(0, min(2, self.player_pos[1]))


        # Update all of the farm files
        for i in range(3):
            if self.farm_tiles[i] == -1 or self.farm_tiles[i] == 5:
                pass
            else:
                if self.farm_tiles_counters[i] == 0:
                    self.farm_tiles[i] += 1
                    self.farm_tiles_counters[i] = 10
                else:
                    self.farm_tiles_counters[i] -= 1



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
            # Not a movement
            if keys[pygame.K_o]:
                # Harvest
                action = 8
            elif keys[pygame.K_p]:
                # Plant (requires seed)
                action = 9
            else:
                action = -1

        return action

    def play(self):
        self.reset()

        clock = pygame.time.Clock()

        pygame.init()

        font = pygame.font.SysFont(None, 24)

        screen = pygame.display.set_mode((SIZE_MULT*3, SIZE_MULT*3))
        pygame.display.set_caption("Farming")

        # Load sprite
        player_image = pygame.image.load(lester_path).convert_alpha()
        player_rect = player_image.get_rect(center=(SIZE_MULT // 2, SIZE_MULT // 2))

        stone_image = pygame.image.load(stone_path).convert_alpha()
        stone_rect_0 = stone_image.get_rect(center = (SIZE_MULT, SIZE_MULT))

        grass_image = pygame.image.load(grass_path).convert_alpha()
        grass_rect_0 = grass_image.get_rect(center = (SIZE_MULT, SIZE_MULT))

        farm_image = pygame.image.load(farm_path).convert_alpha()
        farm_rect_0 = farm_image.get_rect(center = (SIZE_MULT, SIZE_MULT))

        farm_image_0 = pygame.image.load(farm0_path).convert_alpha()
        farm_image_1 = pygame.image.load(farm1_path).convert_alpha()
        farm_image_2 = pygame.image.load(farm2_path).convert_alpha()
        farm_image_3 = pygame.image.load(farm3_path).convert_alpha()
        farm_image_4 = pygame.image.load(farm4_path).convert_alpha()
        farm_image_5 = pygame.image.load(farm5_path).convert_alpha()

        farm_image_list = [
            farm_image_0,
            farm_image_1,
            farm_image_2,
            farm_image_3,
            farm_image_4,
            farm_image_5,
            farm_image
        ]

        # farm_0_image = pygame.image.load()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = self.handle_keypresses()

            # next_state, reward, done = self.step(action)
            self.step(action)

            # Draw the Blocks
            screen.fill(GREEN)


            # Draw the Stone
            stone_rect_0.center = (150, 50)
            screen.blit(stone_image, stone_rect_0)
            stone_rect_0.center = (150, 150)
            screen.blit(stone_image, stone_rect_0)
            stone_rect_0.center = (150, 250)
            screen.blit(stone_image, stone_rect_0)

            grass_rect_0.center = (250, 50)
            screen.blit(grass_image, grass_rect_0)
            grass_rect_0.center = (250, 150)
            screen.blit(grass_image, grass_rect_0)
            grass_rect_0.center = (250, 250)
            screen.blit(grass_image, grass_rect_0)

            farm_rect_0.center = (50, 50)
            screen.blit(farm_image_list[self.farm_tiles[0]], farm_rect_0)
            farm_rect_0.center = (50, 150)
            screen.blit(farm_image_list[self.farm_tiles[1]], farm_rect_0)
            farm_rect_0.center = (50, 250)
            screen.blit(farm_image_list[self.farm_tiles[2]], farm_rect_0)

            # Draw Player
            player_rect.center = ((int(self.player_pos[0])+.5)*SIZE_MULT, (int(self.player_pos[1])+.5)*SIZE_MULT)
            screen.blit(player_image, player_rect)

            clock.tick(FPS)

            # Update display
            pygame.display.flip()

if __name__ == "__main__":
    a = Environment()
    a.play()