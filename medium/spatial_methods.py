"""
Some basic spatial methods that are helpful when
building the observation space and such
"""

import numpy as np


def normalize_delta(span, delta):
    ans = (delta / (2*span)) + (.5)
    if ans < 0:
        return 0
    elif ans > 1:
        return 1
    return ans


def find_euclidean_distance(point_a, point_b):
    """
    Find the euclidean distance between 2 points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


def find_x_y_delta(player_pos, food_pos):
    """
    Find the X and Y differences between 2 points
    """
    dx = food_pos[0] - player_pos[0]
    dy = food_pos[1] - player_pos[1]
    return dx, dy