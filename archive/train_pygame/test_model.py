from model import get_random_foods, encode_foods, normalize_delta

import numpy as np


def test_get_random_foods():
    ans = get_random_foods(300, 300, 10)
    assert type(ans) is dict
    assert len(ans) == 10


def test_encode_foods():
    ans = get_random_foods(300, 300, 10)
    ans = encode_foods(150, 150, ans)
    assert len(ans) is 30
    assert type(ans) is np.ndarray
    assert ans.shape == (30,)


def test_normalize_delta():
    span = 100
    pos = -100

    update = normalize_delta(span, pos)
    assert update >= 0
    assert update < 0.1
    assert update <= 1

    span = 100
    pos = 100

    update = normalize_delta(span, pos)
    assert update >= 0
    assert update > 0.9
    assert update <= 1
