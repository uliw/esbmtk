"""Test isotope delta calculations."""

import epsilon_scale_with_flux_test as model  # import script
import pytest

M = model.M  # get model handle
tolerance = 1e-1
test_values = [  # result, reference value, tolerance
    (M.S_b.DIC.d[-1], -2.5, tolerance, "M.S_b.DIC.d[-1] = -2.5"),
    (M.D_b.DIC.d[-1], 2.5, tolerance, "M.D_b.DIC.d[-1] = 2.5"),
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
