"""
Testing script for the food matrix
"""
import math

from medium.food_matrix import (
    get_random_food_positions,
    heading_from_dxdy,
    distribute_signal_to_bins,
    compute_directional_signal,
)

def test_get_random_food_positions():
    """
    Test Getting the random food positions
    """

    ans = get_random_food_positions(10, 3, 42)
    assert type(ans) is list
    assert len(ans) == 3
    for a in ans:
        assert type(a) is tuple
        assert len(a) == 2

    ans2 = get_random_food_positions(10, 3, 42)
    for a1, a2 in zip(ans, ans2):
        assert a1 == a2


def test_heading_from_dxdy():
    """
    
    """
    dx = 0
    dy = 1
    ans = heading_from_dxdy(dx, dy)
    assert ans == 0

    dx = 0
    dy = 100
    ans = heading_from_dxdy(dx, dy)
    assert ans == 0

    dx = 0
    dy = -1
    ans = heading_from_dxdy(dx, dy)
    assert ans == math.pi

    dx = 1
    dy = 0
    ans = heading_from_dxdy(dx, dy)
    assert ans == math.pi/2

    dx = -1
    dy = 0
    ans = heading_from_dxdy(dx, dy)
    assert ans == 3*math.pi/2


def test_compute_directional_signals():
    """
    Compute the directional signal for a spec
    """
    dx = 1
    dy = 0
    draw = 10

    ans = compute_directional_signal(dx, dy, draw)
    assert type(ans) is list
    # 0 = up , 1 = ur, 2= r, 3 = dr, 4 = d, 5 = dl, 6 = l, 7 = ul
    # Check that there is signal only in the 2 index
    for i in range(8):
        if i == 2:
            assert ans[i] > 0
        else:
            assert ans[i] == 0


def test_distribute_signal_to_bins():
    ans = distribute_signal_to_bins(0, 1)
    assert ans == [1, 0, 0, 0, 0, 0, 0, 0]

    ans = distribute_signal_to_bins(math.pi/4, 1)
    assert ans == [0, 1, 0, 0, 0, 0, 0, 0]

    ans = distribute_signal_to_bins(math.pi/2, 1)
    assert ans == [0, 0, 1, 0, 0, 0, 0, 0]

    ans = distribute_signal_to_bins(3*math.pi/4, 1)
    assert ans == [0, 0, 0, 1, 0, 0, 0, 0]

    ans = distribute_signal_to_bins(math.pi, 1)
    assert ans == [0, 0, 0, 0, 1, 0, 0, 0]

    ans = distribute_signal_to_bins(5*math.pi/4, 1)
    assert ans == [0, 0, 0, 0, 0, 1, 0, 0]

    ans = distribute_signal_to_bins(3*math.pi/2, 1)
    assert ans == [0, 0, 0, 0, 0, 0, 1, 0]

    ans = distribute_signal_to_bins(7*math.pi/4, 1)
    assert ans == [0, 0, 0, 0, 0, 0, 0, 1]

    ans = distribute_signal_to_bins(1.9172391, 1)
    zero_count = 0
    for a in ans:
        if a == 0:
            zero_count += 1
    assert zero_count == 6
    assert sum(ans) == 1
    