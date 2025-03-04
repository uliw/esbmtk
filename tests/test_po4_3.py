import pytest
import po4_3_test  # import script

M = po4_3_test.M  # get model handle 
test_values = [ # result, reference value
    (M.S_b.PO4.c[-1]*1e6, 14.99),
    (M.S_b.DIC.c[-1]*1e6, 1589.98),
    (M.D_b.PO4.c[-1]*1e6, 22.05),
    (M.D_b.DIC.c[-1]*1e6, 2338.18),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-1 # +- 1 mu mol is good enough 
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
