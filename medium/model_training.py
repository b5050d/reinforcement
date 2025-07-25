"""
Model Training
"""

from environment import Environment
from model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import random
import numpy as np
import pygame
from model_management import save_ai_run


def training_loop():
    """
    Train the model on the environment
    """

    LEARNING_RATE = 1e-3

    total_steps = 0

    EVALUATION_INTERVAL = 10
    TARGET_UPDATE_N = 5 # Update the target network every N episodes

    # Establish the Environment to re-use
    env = Environment()

    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = .99
    epsilon = 1.0
    epsilon_decay = .995
    epsilon_min = .1
    episodes = 100
    max_episode_steps = 2000
    run_tag = "test"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # CPU is faster on this simple simulation
    print(f"Using device: {device}")

    # Set up the Training Network
    nnet = DQN()
    nnet.to(device)
    optimizer = optim.Adam(nnet.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Set up the Target Network
    nnet_target = DQN()
    nnet_target.load_state_dict(nnet.state_dict())
    nnet_target.eval()  # no gradients needed

    # Loop Through the specified amount of episodes
    for ep in range(episodes):
        start_time = time.time()
        print(f"Beginning Episode: {ep}")
        state = env.reset()
        total_reward = 0

        action_history = []
        for t in range(max_episode_steps):
            if random.random() < epsilon:
                action = random.randint(0, 7)
            else:
                with torch.no_grad():
                    q_vals = nnet(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device))
                    action = torch.argmax(q_vals).item()

            action_history.append(action)

            next_state, reward, done = env.step(action)
            replay_buffer.append(
                (state, action, reward, next_state, done)
            )
            state = next_state
            total_reward += reward

            # Train
            if (len(replay_buffer) >= batch_size) and (total_steps > 1000):
                batch = random.sample(replay_buffer, batch_size)
                s, a, r, s2, d = zip(*batch)
                s = np.array(s)
                a = np.array(a)
                r = np.array(r)
                s2 = np.array(s2)
                d = np.array(d)

                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)

                qvals = nnet(s)
                qval = qvals.gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_q_s2 = nnet_target(s2).max(1)[0]
                    target = r + gamma * max_q_s2 * (1 - d)

                loss = loss_fn(qval, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps+=1

            if done:
                break
                

        if ep%TARGET_UPDATE_N == 0:  # or every N episodes or steps
            nnet_target.load_state_dict(nnet.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        print(f"Episode Run time = {round(time.time() - start_time, 3)}")

        # Save the run every so often
        if ep%EVALUATION_INTERVAL == 0:
            evaluation_loop(run_tag, env, ep, nnet)
            pygame.quit()


def evaluation_loop(run_tag, env, ep, nnet):
    """
    Simple Evaluation Loop for the training of our model
    """
    print("Running Evaluation Loop")
    nnet.eval()

    episodes = 1
    total_reward = 0
    for epi in range(episodes):
        # Reset the env
        state = env.reset()
        foods = env.foods
        done = False

        action_history = []
        ep_reward = 0
        # while not done:
        for t in range(2000):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = nnet(state_tensor)
                action = torch.argmax(q_values).item()
            action_history.append(action)
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break

        total_reward += ep_reward

    avg_reward = total_reward / episodes
    print(f"[Eval] Avg reward over {episodes} eval runs: {avg_reward:.2f}")

    # Save a run
    save_ai_run(foods, action_history, ep, run_tag)

    nnet.train() # Set it back in training mode


if __name__ == "__main__":
    training_loop()

