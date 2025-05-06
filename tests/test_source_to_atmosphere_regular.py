"""Test the various weathering calculations."""

import pytest

from esbmtk import (
    GasReservoir,
    Model,
    SourceProperties,
    Species2Species,
)

M = Model(
    stop="1 kyr",  # end time of model
    max_timestep="10 yr",  # time step
    element=[
        "Carbon",
    ],
    mass_unit="mol",
    volume_unit="liter",
    concentration_unit="mol/kg",
)
# Source definitions
SourceProperties(
    name="Fw",
    species=[M.DIC],
    delta={M.DIC: 0},
)

GasReservoir(
    name="CO2_At",
    species=M.CO2,
    species_ppm="280 ppm",
    delta=0,
)

# DIC input from crust
Species2Species(
    ctype="Regular",
    source=M.Fw.DIC,
    sink=M.CO2_At,  # source of flux
    rate="10 Tmol/year",
    delta=-7,
    id="volcanic_degassing_CO2",
)

# M.debug_equations_file = True
M.run()
# M.plot([M.CO2_At])

tolerance = 1e-5
test_values = [  # result, reference value, tolerance
    (M.CO2_At.d[-1], -1.1706287343610906, tolerance, "Atmosphere delta"),
    (M.CO2_At.c[-1] * 1e6, 336.2239964016, tolerance, "Atmosphere ppm"),
]


# run tests
@pytest.mark.parametrize("test_input, expected, tolerance, message", test_values)
def test_values(test_input, expected, tolerance, message):
    """Test against known values."""
    try:
        if expected == 0:
            # Special case for expected value being zero
            assert abs(test_input) <= tolerance, (
                f"{message}\n"
                f"expected {expected} but got {test_input}, tol = {tolerance:.2e}\n"
            )
        else:
            assert (
                abs(expected) * (1 - tolerance)
                <= abs(test_input)
                <= abs(expected) * (1 + tolerance)
            )
    except AssertionError as e:
        raise Exception(
            f"{message}\n"
            f"expected {expected} but got {test_input}, tol = {tolerance:.2e}\n"
        ) from e
