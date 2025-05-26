"""Test the signal_from_csv output."""

import pytest
import signal_from_csv_reverse_test  # import script

M = signal_from_csv_reverse_test.M  # get model handle
test_values = [  # result, reference value
    (M.D_b.DIC.c[839] * 1e6, 50.500),
]


# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-1  # +- 1 mu mol is good enough
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
