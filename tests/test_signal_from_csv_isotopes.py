"""Test the signal_from_csv output."""

import pytest
import signal_from_csv_isotopes_test  # import script

M = signal_from_csv_isotopes_test.M  # get model handle
test_values = [  # result, reference value
    (M.S_b.DIC.c[-1] * 1e6, 16.591),
    (M.D_b.DIC.c[-1] * 1e6, 24.406),
    (M.S_b.DIC.d[-1], -8.04709),
    (M.D_b.DIC.d[-1], -8.041501),
]


# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-1  # +- 1 mu mol is good enough
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
