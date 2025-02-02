import pytest
import po4_2_test  # import script

M = po4_2_test.M  # get model handle 
test_values = [ # result, reference value
    (M.S_b.PO4.c[-1]*1e6, 31.55),
    (M.D_b.PO4.c[-1]*1e6, 46.40),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-1 # +- 1 mu mol is good enough 
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
