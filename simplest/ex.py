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
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")

        og = self.q_table[state][action]
        self.q_table[state][action] = og + (ALPHA*(reward - og))


if __name__ == "__main__":
    player = Player()
    env = Environment()

    rewards = []
    
    for i in range(20):
        # Alright lets train

        state = env.get_new_state()
        print(f"State: {state}")

        action = player.make_action(state)
        print(f"Action: {action}")

        reward = env.perform_action(action)
        print(f"Reward: {reward}")

        player.q_update(state, action, reward)

        rewards.append(reward)

    print(player.q_table)

    plt.plot(rewards)
    plt.show()