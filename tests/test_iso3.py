import pytest
import  iso3_test # import script

M = iso3_test.M  # get model handle
test_values = [  # result, reference value
    (M.S_b.O2.c[-1], 0.00019999999999999998),
    (M.D_b.O2.c[-1], 0.00019999999999999998),
    (M.S_b.O2.d[-1], 2.4921132299215945),
    (M.D_b.O2.d[-1], -2.492088372772069),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-6
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
