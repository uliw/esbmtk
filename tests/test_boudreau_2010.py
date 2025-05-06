"""Test complex model calculations."""

import boudreau_2010_test as bd  # import script
import pytest

from esbmtk import carbonate_system_2_pp

tolerance = 1e-3
run_time = "1000 kyr"
time_step = "100 yr"  # this is max timestep
rain_ratio = 0.3
alpha = 0.6

# import the model definition
M = bd.initialize_model(rain_ratio, alpha, run_time, time_step)
# M.debug_equations_file = True
M.run(method="BDF")
# post process data
CaCO3_export = M.CaCO3_export.to(f"{M.f_unit}").magnitude
carbonate_system_2_pp(M.D_b, CaCO3_export, 200, 10999)


test_values = [  # result, reference value, tolerance, message
    (M.L_b.DIC.c[-1] * 1e6, 1940.91189, tolerance, "M.L_b.DIC.c[-1]"),
    (M.H_b.DIC.c[-1] * 1e6, 2152.08134, tolerance, "M.H_b.DIC.c[-1]"),
    (M.D_b.DIC.c[-1] * 1e6, 2294.96581, tolerance, "M.D_b.DIC.c[-1]"),
    # TA
    (M.L_b.TA.c[-1] * 1e6, 2282.13700, tolerance, "M.L_b.TA.c[-1]"),
    (M.H_b.TA.c[-1] * 1e6, 2348.50915, tolerance, "M.H_b.TA.c[-1]"),
    (M.D_b.TA.c[-1] * 1e6, 2403.81926, tolerance, "M.D_b.TA.c[-1]"),
    # pH
    (M.L_b.pH.c[-1], 8.268, tolerance, "M.L_b.pH.c[-1]"),
    (M.H_b.pH.c[-1], 8.231, tolerance, "M.H_b.pH.c[-1]"),
    (M.D_b.pH.c[-1], 7.912, tolerance, "M.D_b.pH.c[-1]"),
    # CO3
    (M.L_b.CO3.c[-1] * 1e6, 236.98378, tolerance, "M.L_b.CO3.c[-1]"),
    (M.H_b.CO3.c[-1] * 1e6, 140.47935, tolerance, "M.H_b.CO3.c[-1]"),
    (M.D_b.CO3.c[-1] * 1e6, 87.39742, tolerance, "M.D_b.CO3.c[-1]"),
    # characteristic depth intervals
    (M.D_b.zsnow.c[-1], 4812.9963, tolerance, "M.D_b.zsnow.c[-1]"),
    (M.D_b.zsat.c[-1], 3755, tolerance, "M.D_b.zsat.c[-1]"),
    (M.D_b.zcc.c[-1], 4812, tolerance, "M.D_b.zcc.c[-1]"),
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
