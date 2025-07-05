# Training on Pygame Simulation

Alright so the thought here is to create a basic simulation that both the human and the AI can participate in.

Pygame can be used to allow the user to play but the game should be accessible to the AI and run headless


Thoughts on the encoding of environment observation
dist_x, dist_y, active and repeating as such
This keeps the observation space consistent

## Notes:

The original reward model was not quick enough, it was finding a food like 1/10000 steps, so it would never get enough signal to effectively learn. Maybe after a really long time it would figure it out, but yeah.

Alright to get the ai to work correctly, I will need to change the way the action space is organized. Right now I had the user be able to do multiple directions at once. So if you pressed up and right you would move diagonally. 

The way the game was set up the AI could only move 1 of the 4 cardinal directions. This should be changed to 1 of the 8 directoinns including the diagonals

Direction Setup:
Do nothing = 0, 5, 10, 15
Go up = 8, 13
Go Left = 4, 14
Go down = 2, 7
Go right = 1 ,11
go up left = 12
Go down left = 6
go down right = 3
go up right = 9


0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

# translation into move indexes
up = 0
up left = 1
left =2
down left = 3
down = 4
down right = 5
right = 6
up right = 7
