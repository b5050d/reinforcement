import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


class Board:
    def __init__(self, size = 10):
        self.size = size
        self.reset()

    def reset(self):
        self.goal = np.random.randint(0, self.size, size=2)
        self.pos = np.random.randint(0, self.size, size=2)
        while np.array_equal(self.pos, self.goal):
            self.pos = np.random.randint(0, self.size, size=2)

        return self.__get_state()
    
    def __get_state(self):
        vec = self.goal - self.pos
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.zeros(2)
    
    def step(self, action):
        move = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }[action]

        self.pos = np.clip(
            self.pos + move,
            0,
            self.size-1
        )
        done = np.array_equal(self.pos, self.goal)
        reward = 1.0 if done else -0.01
        return self.__get_state(), reward, done
    

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)
    


# Alright lets do thr training loop now

game_board = Board()
nnet = DQN()
optimizer = optim.Adam(nnet.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=10000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
episodes = 500
max_steps = 100

counter = 0

from matplotlib import pyplot as plt

losses = []

for ep in range(episodes):
    state = game_board.reset()
    total_reward = 0

    for step in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = nnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(q_vals).item()

                # if counter%100:
                #     print(action)
                #     counter=0
                # counter+=1
        
        next_state, reward, done = game_board.step(action)
        
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # Training step
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            s, a, r, s2, d = zip(*batch)

            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(s2, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)

            q_vals = nnet(s)
            q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                max_q_s2 = nnet(s2).max(1)[0]
                target = r + gamma * max_q_s2 * (1-d)
            
            loss = loss_fn(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
