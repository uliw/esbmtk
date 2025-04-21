"""Test isotope delta calculations."""

import epsilon_regular_test as model  # import script
import pytest

M = model.M  # get model handle
tolerance = 1e-6
test_values = [  # result, reference value, tolerance
    (M.S_b.DIC.d[-1], 5, tolerance),
]


# run tests
@pytest.mark.parametrize("test_input, expected, tolerance", test_values)
def test_values(test_input, expected, tolerance):
    """Test against known values."""
    assert (
        abs(expected) * (1 - tolerance)
        <= abs(test_input)
        <= abs(expected) * (1 + tolerance)
    )
