"""Test complex model calculations with isotopes.

Note, this is a copy of an early version of a 4 box
isotope model. It has not been tuned, and no attempt \
has been made to match them with observational values.
However,
they do serve as a benchmark and will detect if
numerical mistakes creep in.
"""

import boudreau_isotopes_test as bm
import pytest

run_time = "4 Myr"
debug = False
time_step = "1000 year"  # this is max timestep
rain_ratio = 0.3  # rain ratio CaCO3/OM
alpha = 0.6  # alpha RD
initial_mixing = "30 Sverdrup"  # High latitude mixing 280 ppm
M = bm.initialize_model_geometry(
    rain_ratio,
    alpha,
    run_time,
    time_step,
    initial_mixing,
    debug,
)

# M.debug_equations_file = True
M.run()
# M.plot([M.L_b.DIC, M.D_b.DIC])

tolerance = 1e-3
test_values = [  # result, reference value, tolerance, message
    (M.L_b.DIC.d[-1], 2.5, tolerance, "M.L_b.DIC.d[-1]"),
    (M.H_b.DIC.d[-1], 2.226, tolerance, "M.H_b.DIC.d[-1]"),
    (M.D_b.DIC.d[-1], 0.898, tolerance, "M.D_b.DIC.d[-1]"),
    # Atmosphere
    (M.CO2_At.d[-1], -6.65, tolerance, "M.CO2_At.d[-1]"),
    (M.CO2_At.c[-1] * 1e6, 267.8, tolerance, "M.CO2_At.d[-1]"),
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
        raise Exception(
            f"\n{message} should be {expected} but got {test_input}\n"
            f"diff = {expected - test_input:.2e}, tol = {tolerance:.2e}\n"
        ) from e
