import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import numpy as np

SIZE = 4
EPISODES = 500
MAX_STEPS = 100
REWARD_PENALTY = -0.01


class Board:
    def __init__(self):
        self.size = SIZE
        self.reset()

    def reset(self):
        # Pick a random set
        self.vec = []
        for i in range(self.size):
            self.vec.append(np.random.randint(0, 2))

        return self.vec.copy()

    def flip_bit(self, action):
        return int(not self.vec[action])

    def step(self, action):
        self.vec[action] = self.flip_bit(action)

        # Check if done
        if 1 not in self.vec:
            done = True
        else:
            done = False

        reward = 1.0 if done else REWARD_PENALTY
        return self.vec.copy(), reward, done


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        node_num = 8
        self.model = nn.Sequential(
            nn.Linear(SIZE, node_num),
            nn.ReLU(),
            nn.Linear(node_num, node_num),
            nn.ReLU(),
            nn.Linear(node_num, SIZE),
        )

    def forward(self, x):
        return self.model(x)


game = Board()
nnet = DQN()
optimizer = optim.Adam(nnet.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
episodes = EPISODES
max_steps = MAX_STEPS

from matplotlib import pyplot as plt

rewards = []
steps_taken = []

for ep in range(episodes):
    state = game.reset()
    total_reward = 0

    for step in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, SIZE - 1)
        else:
            with torch.no_grad():
                q_vals = nnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(q_vals).item()

        next_state, reward, done = game.step(action)

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
                target = r + gamma * max_q_s2 * (1 - d)

            loss = loss_fn(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            steps_taken.append(step)
            break

    if not done:
        steps_taken.append(max_steps)
        total_reward = 0

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
    rewards.append(total_reward)

plt.plot(rewards)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward")
plt.show()

plt.plot(steps_taken)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Steps Taken")
plt.title("Steps Taken")
plt.show()
