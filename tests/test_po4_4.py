import pytest
import po4_4_test  # import script

M = po4_4_test.M  # get model handle
test_values = [  # result, reference value
    (M.S_b.PO4.c[-1], 1.2921127934553388e-05),
    (M.D_b.PO4.c[-1], 1.899171422689367e-05),
    (M.S_b.DIC.c[-1], 0.0013716395610626638),
    (M.D_b.DIC.c[-1], 0.0020151217080507343),
    (M.S_b.DIC.d[-1], 9.03802309217078),
    (M.D_b.DIC.d[-1], -0.057647879068921326),
]

# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-12
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
