# Simplest Reinforcement Learning Example

## Overview

This is an extremely simple reinforcement learning exercise, done for learning purposes.

The Environment is a simple game. Every 'round' the environment will present a state (0, 1). It is up to the RL model to echo the environment with its actions, being (0, 1). 

The architecture for this is to use a simple Q-Learning Table.

| State | Action 0 | Action 1 |
|-------|----------|----------|
|   0   |    a     |    b     |
|   1   |    c     |    d     |

Where a, b, c, and d are the learned parameter values.
