"""Test the various weathering calculations."""

import pytest

from esbmtk import (
    GasReservoir,
    Model,
    Species2Species,
    initialize_reservoirs,
)

M = Model(
    stop="1 kyr",  # end time of model
    max_timestep="10 yr",  # time step
    element=[
        "Carbon",
        "Oxygen",
    ],
    mass_unit="mol",
    volume_unit="liter",
    concentration_unit="mol/kg",
)

box_parameters: dict = {
    "b0": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b1": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b2": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b3": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b4": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b5": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "b6": {
        "c": {
            M.DIC: "0 umol/kg",
            M.TA: "0 umol/kg",
            M.O2: "0 umol/kg",
        },
        "g": [0, -10000, 1],
        "T": 2,
        "P": 240,
        "S": 35,
        "d": {M.DIC: 0},
    },
    "Fw": {
        "ty": "Source",
        "sp": [M.DIC, M.O2],
        "d": {M.DIC: 0},
    },
}
species_list = initialize_reservoirs(M, box_parameters)

GasReservoir(
    name="CO2_At_with_isotopes",
    species=M.CO2,
    species_ppm="280 ppm",
    delta=-7,
)

GasReservoir(
    name="CO2_At_with_isotopes_240",
    species=M.CO2,
    species_ppm="240 ppm",
    delta=-7,
)

GasReservoir(
    name="CO2_At_no_isotopes",
    species=M.CO2,
    species_ppm="280 ppm",
)

# ---- Test regular weathering flux O2 only ---- #
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.O2,  # source of flux
    sink=M.b1.O2,
    reservoir_ref=M.CO2_At_no_isotopes,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_no_no",
)

# ---- Test regular weathering flux where atmosphere as isotopes ---- #
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.O2,  # source of flux
    sink=M.b2.O2,
    reservoir_ref=M.CO2_At_with_isotopes,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_no",
)

# ---- Test where source and sink have isotopes
# # case 0, no isotope effects
# Species2Species(  # CaCO3 weathering as function of carbonate weathering
#     ctype="weathering",
#     source=M.Fw.DIC,  # source of flux
#     sink=M.b0.DIC,
#     reservoir_ref=M.CO2_At_with_isotopes,  # pCO2
#     scale=1,  # optional, defaults to 1
#     ex=0.4,  # exponent c
#     pco2_0="280 ppm",  # reference pCO2
#     rate="10 Tmol/yr",  # rate at pco2_0
#     id="weathering_yes_no",
# )

# ---- Test regular weathering flux where atmosphere and sink have isotopes ---- #
# case one: no isotope effects
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b1.DIC,
    reservoir_ref=M.CO2_At_with_isotopes,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_1",
)

# case two: constant delta
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b2.DIC,
    reservoir_ref=M.CO2_At_with_isotopes,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_2",
    delta=5,
)

# case three: constant epsilon
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b3.DIC,
    reservoir_ref=M.CO2_At_with_isotopes,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_3",
    epsilon=21,
)

# case 4: constant delta but different pCO2
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b4.DIC,
    reservoir_ref=M.CO2_At_with_isotopes_240,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_4_240",
    delta=5,
)

# case 5: epsilon  but different pCO2
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b5.DIC,
    reservoir_ref=M.CO2_At_with_isotopes_240,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_5_240",
    epsilon=21,
)

# case 6: isotopes, no effect but different pCO2
Species2Species(  # CaCO3 weathering as function of carbonate weathering
    ctype="weathering",
    source=M.Fw.DIC,  # source of flux
    sink=M.b6.DIC,
    reservoir_ref=M.CO2_At_with_isotopes_240,  # pCO2
    scale=1,  # optional, defaults to 1
    ex=0.4,  # exponent c
    pco2_0="280 ppm",  # reference pCO2
    rate="10 Tmol/yr",  # rate at pco2_0
    id="weathering_yes_yes_6_240",
)


# M.debug_equations_file = True
M.run()

# regular weathering flux
wnn_mass = M.b1.O2.volume * M.b1.swc.density / 1000 * M.b1.O2.c[-1]
wyn_mass = M.b2.O2.volume * M.b1.swc.density / 1000 * M.b1.O2.c[-1]
wyy0_mass = M.b1.DIC.volume * M.b1.swc.density / 1000 * M.b1.DIC.c[-1]
wyy1_mass = M.b1.DIC.volume * M.b1.swc.density / 1000 * M.b1.DIC.c[-1]
wyy2_mass = M.b2.DIC.volume * M.b2.swc.density / 1000 * M.b2.DIC.c[-1]
wyy3_mass = M.b3.DIC.volume * M.b3.swc.density / 1000 * M.b3.DIC.c[-1]
# wyy_d0 = M.b0.DIC.d[-1]
wyy_d1 = M.b1.DIC.d[-1]
wyy_d2 = M.b2.DIC.d[-1]
wyy_d3 = M.b3.DIC.d[-1]
wyy_d4 = M.b4.DIC.d[-1]
wyy_d5 = M.b5.DIC.d[-1]
wyy_d6 = M.b6.DIC.d[-1]

tolerance = 1e-5
test_values = [  # result, reference value, tolerance
    (wnn_mass.magnitude, 1e16, tolerance, "mass isotopes no no"),
    (wyn_mass.magnitude, 1e16, tolerance, "mass isotopes yes no"),
    #    (wyy0_mass.magnitude, 1e16, tolerance, "mass isotopes yes yes 0"),
    (wyy1_mass.magnitude, 1e16, tolerance, "mass isotopes yes yes 1"),
    (wyy2_mass.magnitude, 1e16, tolerance, "mass isotopes yes yes 2"),
    (wyy3_mass.magnitude, 1e16, tolerance, "mass isotopes yes yes 3"),
    (wyy_d1, 0, tolerance, "None isotopes yes yes 1"),
    (wyy_d2, 5, tolerance, "delta isotopes yes yes 2"),
    (wyy_d3, 21, tolerance, "delta isotopes yes yes 3"),
    (wyy_d4, 5, tolerance, "delta isotopes yes yes 4_240"),
    (wyy_d5, 21, tolerance, "delta isotopes yes yes 5_240"),
    (wyy_d6, 0, tolerance, "delta isotopes yes yes 6_240"),
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
