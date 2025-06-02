"""Test isotope delta calculations."""

import signal_burial_epsilon_only_test as model  # import script
import pytest

M = model.M  # get model handle
tolerance = 1e-3
test_values = [  # result, reference value, tolerance
    (M.D_b.SO4.d[-1], 21.5987, tolerance, "M.D_b.SO4.d[-1] = 21.5979"),
    (M.D_b.SO4.c[-1] * 1000, 28, tolerance, "M.D_b.SO4.c[-1] = 28"),
]


# run tests
@pytest.mark.parametrize("test_input, expected, tolerance, message", test_values)
def test_values(test_input, expected, tolerance, message):
    """Test against known values."""
    try:
        assert (
            abs(expected) * (1 - tolerance)
            <= abs(test_input)
            <= abs(expected) * (1 + tolerance)
        )
    except AssertionError as e:
        raise Exception(f"{message} but got {test_input}, tol = {tolerance:.2e}") from e
