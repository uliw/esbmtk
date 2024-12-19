import pytest
import po4_2_test  # import script

M = po4_2_test.M  # get model handle 
test_values = [ # result, reference value
    (M.S_b.PO4.c[-1], 3.318233001125679e-05),
    (M.D_b.PO4.c[-1], 4.881170094975448e-05),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-12
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
