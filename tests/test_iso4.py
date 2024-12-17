import pytest
import  iso4_test # import script

M = iso4_test.M  # get model handle
test_values = [  # result, reference value
    (M.S_b.O2.c[-1], 0.0002306171887706928),
    (M.O2_At.c[-1], 0.20999483573786618),
    (M.S_b.O2.d[-1], 0.7204862035835163),
    (M.O2_At.d[-1], -0.00013345986215331608),
]
# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-6
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
