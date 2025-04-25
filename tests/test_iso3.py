import iso3_test  # import script
import pytest

M = iso3_test.M  # get model handle
test_values_input = [  # result, reference value
    (M.S_b.O2.c[-1], 0.00019999999999999998),
    (M.D_b.O2.c[-1], 0.00019999999999999998),
    (M.S_b.O2.d[-1], 2.4921132299215945),
    (M.D_b.O2.d[-1], -2.492088372772069),
]


# run tests
@pytest.mark.parametrize("test_input, expected", test_values_input)
def test_values(test_input, expected):
    """Test against test values."""
    t = 1e-2
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
