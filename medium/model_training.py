"""
Model Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import random
import numpy as np
import pygame
from medium.model_management import save_ai_run
from medium.environment import Environment
from medium.model import DQN
from database_ops import *
from config import DATABASE_PATH, REPLAY_PATH


def training_loop(config, experiment_id):
    """
    Train the model on the environment
    """
    # Establish the Environment to re-use
    env = Environment(config)

    total_steps = 0
    replay_buffer = deque(maxlen=10000)

    epsilon = config["STARTING_EPSILON"]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # CPU is faster on this simple simulation
    print(f"Using device: {device}")

    # Set up the Training Network
    nnet = DQN()
    nnet.to(device)
    optimizer = optim.Adam(nnet.parameters(), lr = config['LEARNING_RATE'])
    loss_fn = nn.MSELoss()

    # Set up the Target Network
    nnet_target = DQN()
    nnet_target.load_state_dict(nnet.state_dict())
    nnet_target.eval()  # no gradients needed

    # Loop Through the specified amount of episodes
    for episode in range(config['EPISODES']):
        print(f"Beginning Episode: {episode}")
        start_time = time.time()

        state = env.reset()
        total_reward = 0

        action_history = []

        for step in range(config['MAX_EPISODE_STEPS']):
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
            if (len(replay_buffer) >= config['BATCH_SIZE']) and (total_steps > 1000):
                batch = random.sample(replay_buffer, config['BATCH_SIZE'])
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
                    target = r + config['GAMMA'] * max_q_s2 * (1 - d)

                loss = loss_fn(qval, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps+=1

            if done:
                break

        if episode%config['TARGET_UPDATE_N'] == 0:  # or every N episodes or steps
            nnet_target.load_state_dict(nnet.state_dict())

        epsilon = max(config['EPSILON_MIN'], epsilon * config['EPSILON_DECAY'])
        print(f"Episode {episode}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        print(f"Episode Run time = {round(time.time() - start_time, 3)}")


        # Save the run every so often
        if episode%config['SAVE_TRAINING_INTERVAL'] == 0:
            # Save the Training Run
            replay_config = config.copy()
            replay_config["ACTIONS"] = action_history

            replay_id = add_replay(DATABASE_PATH, experiment_id, replay_config, REPLAY_PATH)
        else:
            replay_id = None

        add_training_run(DATABASE_PATH, experiment_id, episode, round(epsilon,3), round(total_reward,3), replay_id, None)

        if episode%config['EVALUATION_INTERVAL'] == 0:
            # Run the Evaluation Loop
            eval_actions, eval_reward, = evaluation_loop(config, nnet)
            replay_config = config.copy()
            replay_config["ACTIONS"] = eval_actions

            replay_id = add_replay(DATABASE_PATH, experiment_id, replay_config, REPLAY_PATH)

            add_evaluation_run(DATABASE_PATH, experiment_id, episode, round(eval_reward,3), replay_id, None)

            pygame.quit()


def evaluation_loop(config, nnet):
    """
    Simple Evaluation Loop for the training of our model
    """
    print("Running Evaluation Loop")

    env = Environment(config)

    nnet.eval()

    total_reward = 0

    # Reset the env
    state = env.reset()

    done = False

    action_history = []
    # while not done:
    for t in range(2000):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = nnet(state_tensor)
            action = torch.argmax(q_values).item()
        action_history.append(action)
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"[Eval] Avg reward over {1} eval runs: {total_reward:.2f}")

    nnet.train() # Set it back in training mode

    return action_history, total_reward


if __name__ == "__main__":
    training_loop()

