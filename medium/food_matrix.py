
import math

# import random

# # Set the seed
# random.seed(42)

# # Generate 10 random numbers
# numbers = [random.random() for _ in range(10)]

# print(numbers)


# import numpy as np

# rng = np.random.default_rng(seed=42)
# numbers = rng.random(10)

# print(numbers)

import numpy as np
def get_random_matrix(size, n_foods, seed):
    assert type(size) is int
    assert type(seed) is int
    assert type(n_foods) is int

    matrix = np.zeros((size, size))

    food_positions = get_random_food_positions(size, n_foods, seed)
    
    # Great - now we can go ahead and add them to the matrix


# TODO - Randomize the matrix
def further_randomize_matrix():
    """
    Useful when we want to slighly randomize off an existing pattern
    """
    pass 


def get_random_food_positions(size, n_foods, seed):
    # Find the food
    np.random.seed(seed)
    food_positions = []
    for i in range(n_foods):
        # Generate a new food position
        # TODO - make sure the food dont appear on the very edge of the map
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        food_positions.append((x, y))
    return food_positions


def heading_from_dxdy(dx, dy):
    """
    Calculate the heading from a difference in dx and dy
    """
    angle = math.atan2(dx, dy)  # Returns angle in range [-π, π]
    if angle < 0:
        angle += 2 * math.pi     # Convert to [0, 2π)
    return angle


def distribute_signal_to_bins(angle, strength):
    angle = angle % (2 * math.pi)
    bins = [0.0] * 8

    # scale angle to bin space (0 to 8)
    index_f = angle / (math.pi / 4)  # 45° per bin
    lower_bin = int(index_f) % 8
    upper_bin = (lower_bin + 1) % 8
    frac = index_f - lower_bin

    # distribute signal proportionally
    bins[lower_bin] = (1 - frac) * strength
    bins[upper_bin] = frac * strength

    return bins


def compute_directional_signals(dx, dy, draw_dist):
    """
    Compute the directional signal for a spec
    """
    result = [0] * 8
    # 0 = up , 1 = ur, 2= r, 3 = dr, 4 = d, 5 = dl, 6 = l, 7 = ul

    # get the theta
    theta = heading_from_dxdy(dx, dy)

    # Get the strength
    dist = math.sqrt((dx**2) + (dy**2))
    strength = 1/(dist**.9) # signal decay

    # Perform Angular binning
    result = distribute_signal_to_bins(theta, strength)
    
    return result


