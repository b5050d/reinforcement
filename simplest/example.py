"""
Reinforcement Learning most simple example

Lets use a Q learning table

The states can be 0 or 1
"""

import numpy as np
from matplotlib import pyplot as plt

ALPHA = .01 # Learning Rate
REWARD_CORRECT = 1
REWARD_FAILURE = -1

"""
Q Update Code
Normal Equation (factoring in future rewards)
Q(s, a) = Q(state, action) + alpha(reward + y(max(Q(next_state, :)) - Q(s, a))

Simplified Equation as future potential means nothing in this example
Q(s, a) = Q(s, a) + alpha(reward - Q(s, a))

"""


class Environment():
    def __init__(self):
        self.state = 0

    def get_new_state(self):
        """
        Get the State
        """
        self.state = np.random.randint(0,2)
        return self.state
    
    def perform_action(self, action):
        """
        Perform the Action and return a reward
        """
        if action == self.state:
            return REWARD_CORRECT
        else:
            return REWARD_FAILURE


class Player():
    def __init__(self):
        self.q_table = np.zeros((2, 2))

    def make_random_action(self):
        return  np.random.randint(0,2)

    def make_action(self, state):
        """
        Use the Q Table to make the best prediction
        """
        return np.argmax(self.q_table[state])
    
    def q_update(self, state, action, reward):
        """
        Q update
        """
        og = self.q_table[state][action]
        self.q_table[state][action] = og + (ALPHA*(reward - og))


class Training:
    """
    Class to handle the training of our unit
    """
    def __init__(self):
        self.player = Player()
        self.env = Environment()
        self.rewards = []

        self.num_models = 5
        self.num_steps = 20
    
    def train(self):
        """
        Train a model
        """
        for i in range(self.num_steps):
            state = self.env.get_new_state()
            action = self.player.make_action(state)
            reward = self.env.perform_action(action)
            self.player.q_update(state, action, reward)
            self.rewards.append(reward)


if __name__ == "__main__":
    """
    Run a few rounds of training to see how they converge
    """
    for i in range(5):
        train = Training()
        train.train()
        
        plt.plot(train.rewards)

    plt.grid()
    plt.title("Rewards per Round (Over 5 Models)")
    plt.xlabel("Round")
    plt.ylabel("Reward Earned in Round")
    # plt.savefig("simplest_peformance1.png")
    plt.show()