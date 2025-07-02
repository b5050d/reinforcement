

import pygame
import sys
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

def find_euclidean_distance(point_a, point_b):
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)



ARENA_SIZE = 300
WIDTH, HEIGHT = ARENA_SIZE, ARENA_SIZE
GREEN = (0, 200, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PLAYER_RADIUS = 5
PLAYER_POS = [WIDTH // 2, HEIGHT // 2]
PLAYER_SPEED = 1
FPS = 120
clock = pygame.time.Clock()

def find_x_y_delta(player_x, player_y, food_x, food_y):
    dx = food_x - player_x
    dy = food_y - player_y
    return dx, dy

def get_random_foods(width, height, num = 10):
    foods = {}
    for i in range(num):
        stop_condition = False
        while not stop_condition:
            x = np.random.randint(20, width- 20)
            y = np.random.randint(20, height - 20)
            rand_pos = (x, y)
            if rand_pos not in foods:
                stop_condition = True
                foods[rand_pos] = 1
    return foods

def encode_foods(player_x, player_y, foods):
    foods_array = []
    for fpos in foods.keys():
        dx, dy = find_x_y_delta(player_x, player_y, fpos[0], fpos[1])
        foods_array.append(fpos[0])
        foods_array.append(fpos[1])
        foods_array.append(foods[fpos])    
    return np.array(foods_array, dtype=np.uint16)

class Environment():
    def __init__(self):
        self.reset()

    def reset(self):
        # Get new foods
        self.foods = get_random_foods(WIDTH, HEIGHT, 10)
        self.player_pos = [150, 150]
        return encode_foods(self.player_pos[0], self.player_pos[1], self.foods)

    def step(self, action):
        if action == 0: # Down
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 1: # Left
            self.player_pos[0]-=PLAYER_SPEED
        elif action == 2: # Up
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 3: # Right
            self.player_pos[0]+=PLAYER_SPEED

        reward = 0
        # Check if we are in proximity of the food
        for f in self.foods:
            if self.foods[f] == 1: # active
                ans = find_euclidean_distance(self.player_pos, f)
                if ans <= 5:
                    self.foods[f] = 0
                    reward+=1
        if reward == 0:
            reward = -.01
            
        # Encode the Foods
        obs = encode_foods(self.player_pos[0], self.player_pos[1], self.foods)
        
        # Check how many foods remain
        done = True
        for f in self.foods:
            if self.foods[f] == 1:
                done = False
                break
        if done == True:
            reward = 10 # won the game! give a big reward
        return obs, reward, done

    def render(self):
        pass


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)
    






if __name__ == "__main__":
    env = Environment()
    ans = env.reset()

    nnet = DQN()
    LEARNING_RATE = 1e-3
    optimizer = optim.Adam(nnet.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = .99
    epsilon = 1.0
    epsilon_decay = .995
    epsilon_min = .1
    episodes = 500

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(2000):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_vals = nnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_vals).item()
            
            next_state, reward, done = env.step(action)
            replay_buffer.append(
                (state, action, reward, next_state, done)
            )
            state = next_state
            total_reward += reward

            # Train
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                s, a, r, s2, d = zip(*batch)

                #TODO - continue with the training!