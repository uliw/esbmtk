import pytest
import po4_3_test  # import script

M = po4_3_test.M  # get model handle 
test_values = [ # result, reference value
    (M.S_b.PO4.c[-1], 1.4999891897165636e-05),
    (M.S_b.DIC.c[-1], 0.0015899885410995674),
    (M.D_b.PO4.c[-1], 2.2058362084399994e-05),
    (M.D_b.DIC.c[-1], 0.002338186380946409),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-12
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
