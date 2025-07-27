# Reinforcement Learning Codebase

## Overview

I created this codebase to learn about Reinforcement Learning. I really am fascinated by the concept and would like to play around with different simulations/environments/architectures and see what kinds of cool things I can create. This codebase is intended to start very simply in order to learn the fundamentals.

## Elements
### Simplest
An extremely simple example of a Q Learning Table learning a binary choice 'game'. Its not really even a game - just choose 0 if the state is 0 and 1 if the state is 1. Nice to illustrate the simplest possible example.


## Medium
Work in progess. The idea here is to train a model to play a simple resource gathering game.

## Hard
This has not been started yet


## Database Schema
Using SQLite3 as a database for now.

### Experiment Table
| id (iter) | Description | Config  |
|----------|----------|----------|
| Value A  | Value B  | Value C  |

### Model Table

| id (uuid) | Experiment id |
|----------|----------|
| Value A  | Value B  |


### Replay Table
| id (uuid) | Experiment id |
|----------|----------|
| Value A  | Value B  |

### Training Table
| id (uuid) | Episode id | episode | epsilon | reward | Replay id | Model id |
|----------|----------|----------|----------|----------|----------|----------|
| Value A  | Value B  | Value C  | Value A | Value B  | Value C  | Value A |


### Evaluation Table
| id (uuid) | Episode id | episode | reward | Replay id | Model id |
|----------|----------|----------|----------|----------|----------|
| Value A  | Value B  | Value C  | Value A | Value B  | Value C 


And one other thing to note is that I will be adjusting the way that the rewards are done and stuff, so that means that the environment will change over time, meaning it might be confusing when observing past runs. To counter this it would be a good idea for some way to track the version of the game. An initial though is that each training loop could pick a new instnace iterator, but if we run multiple trainings of the same version, then that could get confusing. or vice versa. I cant think


