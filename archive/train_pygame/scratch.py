"""
Written by b5050d
7.1.2025

Basic Pygame

"""

import pygame
import sys
import numpy as np
import time

import os


# Set up the Resources needed
currdir = os.path.dirname(__file__)
sprite_path = os.path.join(currdir, "sprite.png")


def find_euclidean_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


# Constants

arena_size = 100
WIDTH, HEIGHT = 300, 300
GREEN = (0, 200, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PLAYER_RADIUS = 5
PLAYER_POS = [WIDTH // 2, HEIGHT // 2]
PLAYER_SPEED = 1

FPS = 120
clock = pygame.time.Clock()

# Pick a random
foods = []
for i in range(10):
    stop_condition = False
    while not stop_condition:
        rand_pos_height = np.random.randint(20, HEIGHT - 20)
        rand_pos_width = np.random.randint(20, WIDTH - 20)
        rand_pos = np.array([rand_pos_width, rand_pos_height])
        if not any(np.array_equal(rand_pos, existing) for existing in foods):
            stop_condition = True
            foods.append(rand_pos)

# Initialize Pygame
pygame.init()

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Arena")

# Load sprite
player_image = pygame.image.load(sprite_path).convert_alpha()
player_rect = player_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))

# Game loop
cooldown = 0
start_time = time.time()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Handle keypresses
    keys = pygame.key.get_pressed()

    key_flags = [keys[pygame.K_w], keys[pygame.K_a], keys[pygame.K_s], keys[pygame.K_d]]
    value = 0
    for i, bit in enumerate(reversed(key_flags)):
        value |= bit << i

    print(value)
    if keys[pygame.K_w]:
        PLAYER_POS[1] -= PLAYER_SPEED
    if keys[pygame.K_s]:
        PLAYER_POS[1] += PLAYER_SPEED
    if keys[pygame.K_a]:
        PLAYER_POS[0] -= PLAYER_SPEED
    if keys[pygame.K_d]:
        PLAYER_POS[0] += PLAYER_SPEED

    if keys[pygame.K_x]:
        if cooldown == 0:
            print(PLAYER_POS)
            cooldown += 600

    # Fill background
    screen.fill(GREEN)

    # Draw player
    # pygame.draw.circle(screen, RED, PLAYER_POS, PLAYER_RADIUS)
    player_rect.center = (int(PLAYER_POS[0]), int(PLAYER_POS[1]))
    screen.blit(player_image, player_rect)

    # # Place some food there
    to_del = []
    for i, f in enumerate(foods):
        if find_euclidean_distance(f, np.array(PLAYER_POS)) < 5:
            print("Found the food")
            to_del.append(i)

    for i in to_del:
        foods.pop(i)

    for f in foods:
        pygame.draw.circle(screen, YELLOW, f, 5)

    if len(foods) == 0:
        print("You Win!")
        break

    # Update display
    pygame.display.flip()

    if cooldown > 0:
        cooldown = cooldown - 1

    clock.tick(FPS)

end_time = time.time()
elapsed = round(end_time - start_time, 3)

print(f"The time taken for you was: {elapsed}")
print("Game Over!")
