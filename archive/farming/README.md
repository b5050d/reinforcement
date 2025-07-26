# Farming Sim

This is a step above the other simulations, and its trying to get the model to learn a somewhat complex task.

The objective of the simulation is to farm 21 pieces of wheat

The model can take the following actions:
- Plant (consumes seeds if it has any)
- Till (destroys any growing seeds)
- Harvest (destroys any growing crops that have not finished yet, but when done on grass, it gives a chance of getting seeds)

The arena is a 5x5 world.
The left half of the arena is farming land, this land can be tilled, and seeds can be planted. it can be harvested once things have grown, if the things havent grown enough, it will destroy anything growing and yield nothing. (plants always grow at a steady rate)
The right half of the arena is grass land, it can be harvested. When harvested depending on the growth of the grass, it has a chance to yield seeds. grass always grows at a steady rate. (the arena starts with a random growh level for each grass tile)
The center strip is just pathway, unproductive land

The player can move up down left or right inside the arena and at each tile can perform the following actions
- harvest (on grass, yields seeds at a chance, on farm, if planted and not fully grown it destroys whats planted, if grown yields a grown plant)

The player starts in the middle with nothing and must acquire seeds and plant them.


After considering it, this might be  a more fun longer term game to do.

A simplification of this game that would work well for us would be to have a 3x3 tile game, 3 tiles of farm, 3x tiles of stone. No grass growth rate (always has a chance of yielding grass) and no tilling mechanic


The goal is to get this functionality up and running very quickly so that I can make a video on it quickly and dont spend forever iterating over something


How to encode the arena for fast access?
