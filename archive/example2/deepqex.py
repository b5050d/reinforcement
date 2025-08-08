import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)


env = GridEnv()
net = DQN()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
episodes = 500

for ep in range(episodes):  # Number of Episodes
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Number of steps in a game
        if (
            random.random() < epsilon
        ):  # Every so often, go ahead and perform a random action
            action = random.randint(0, 3)
        else:
            with torch.no_grad():  # no gradienet
                q_vals = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(q_vals).item()

        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Train
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            s, a, r, s2, d = zip(*batch)

            s = torch.tensor(s, dtype=torch.float32)
            a = torch.tensor(a)
            r = torch.tensor(r, dtype=torch.float32)
            s2 = torch.tensor(s2, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)

            q_vals = net(s)
            q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                max_q_s2 = net(s2).max(1)[0]
                target = r + gamma * max_q_s2 * (1 - d)

            loss = loss_fn(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
