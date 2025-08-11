"""
Contains methods for generation of our
state space
"""

import math
import numpy as np
from reinforcement_app.simulation.spatial_methods import (
    find_x_y_delta,
    find_euclidean_distance,
)


def get_random_matrix(size, n_foods, seed):
    assert type(size) is int
    assert type(seed) is int
    assert type(n_foods) is int

    matrix = np.zeros((size, size))

    food_positions = get_random_food_positions(size, n_foods, seed)

    # Great - now we can go ahead and add them to the matrix

    return matrix, food_positions


# TODO - Randomize the matrix
def further_randomize_matrix():
    """
    Useful when we want to slighly randomize off an existing pattern
    """
    pass


def get_random_food_positions(size, min_spacing, n_foods, seed):
    """
    Build a list of
    """
    assert size > 5
    np.random.seed(seed)

    positions = []
    low_limit = int(size * 0.1)
    up_limit = int(size * 0.9)

    mid_range_lo = int(size * 0.4)
    mid_range_hi = int(size * 0.6)

    for i in range(n_foods):
        stop_condition = False
        attempts = 0
        while not stop_condition:
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)

            proposed_pos = (x, y)

            if proposed_pos in positions:
                pass
            elif min(proposed_pos) <= low_limit:
                pass
            elif max(proposed_pos) >= up_limit:
                pass
            elif (x >= mid_range_lo and x <= mid_range_hi) and (
                y >= mid_range_lo and y <= mid_range_hi
            ):
                pass
            else:
                failed = False
                # Check to see if its appropriate spacing
                for p in positions:
                    if find_euclidean_distance(p, proposed_pos) <= min_spacing:
                        failed = True
                if not failed:
                    positions.append(proposed_pos)
                    stop_condition = True

            attempts += 1
            if attempts > 1000:
                raise Exception("Error, could not converge")

    return positions


def get_random_danger_positions(size, min_spacing, n_dangers, foods, seed):
    """ """
    assert size > 5
    np.random.seed(seed)

    positions = []
    low_limit = int(size * 0.1)
    up_limit = int(size * 0.9)

    mid_range_lo = int(size * 0.4)
    mid_range_hi = int(size * 0.6)

    for i in range(n_dangers):
        stop_condition = False
        attempts = 0
        while not stop_condition:
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)

            proposed_pos = (x, y)

            if proposed_pos in positions:
                pass
            if proposed_pos in foods:
                pass
            elif min(proposed_pos) <= low_limit:
                pass
            elif max(proposed_pos) >= up_limit:
                pass
            elif (x >= mid_range_lo and x <= mid_range_hi) and (
                y >= mid_range_lo and y <= mid_range_hi
            ):
                pass
            else:
                failed = False
                # Check to see if its appropriate spacing
                for p in positions:
                    if find_euclidean_distance(p, proposed_pos) <= min_spacing:
                        failed = True
                for p in foods:
                    if find_euclidean_distance(p, proposed_pos) <= min_spacing:
                        failed = True
                if not failed:
                    positions.append(proposed_pos)
                    stop_condition = True

            attempts += 1
            if attempts > 1000:
                raise Exception("Error, could not converge")

    return positions


def heading_from_dxdy(dx, dy):
    """
    Calculate the heading from a difference in dx and dy
    """
    angle = math.atan2(dx, dy)  # Returns angle in range [-π, π]
    if angle < 0:
        angle += 2 * math.pi  # Convert to [0, 2π)
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


def compute_directional_signal(dx, dy, euclid):
    """
    Compute the directional signal for a specific food, given dy dx
    """
    result = [0] * 8
    # 0 = up , 1 = ur, 2= r, 3 = dr, 4 = d, 5 = dl, 6 = l, 7 = ul

    # get the theta
    theta = heading_from_dxdy(dx, dy)

    # Get the strength
    dist = euclid
    strength = 1 / (dist**0.9)  # signal decay

    # Perform Angular binning
    result = distribute_signal_to_bins(theta, strength)

    return result


def compute_directional_signals(player_pos, food_positions):
    """ """
    if len(food_positions) == 0:
        return [0] * 8
    signals = []
    for f in food_positions:
        dx, dy = find_x_y_delta(player_pos, f)
        euclid = find_euclidean_distance(player_pos, f)

        signal = compute_directional_signal(dx, dy, euclid)
        signals.append(signal)

    summed_signal = []
    for i in range(len(signals[0])):
        sum = 0
        for s in signals:
            sum += s[i]
        summed_signal.append(sum)

    # normalize the summed signal
    normalized_summed_signal = []
    for s in summed_signal:
        if s < 0:
            raise Exception
        elif s > 1:
            normalized_summed_signal.append(1)
        else:
            normalized_summed_signal.append(s)

    return normalized_summed_signal
