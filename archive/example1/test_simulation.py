
from simulation import (
    pick_light_location,
    pick_spawn_point
)

def test_pick_light_location():
    for i in range(10):
        ans = pick_light_location()
        assert ans in ["left", "right"]


def test_pick_spawn_point():
    for i in range(10):
        ans = pick_spawn_point()
        assert ans in [3, 4, 5, 6, 7]