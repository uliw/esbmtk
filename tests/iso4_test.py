# This example is available as iso_4.py in ESBMTK Examples Repository
# https://github.com/uliw/ESBMTK-Examples/tree/main/Examples_from_the_manual
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    ConnectionProperties,  # the connection class
    GasReservoir,  # sink class
    Species2Species,
)
M = Model(
    stop="1 yr",  # end time of model
    max_timestep="1 month",  # upper limit of time step
    element=["Oxygen"],  # list of element definitions
)
GasReservoir(
    name="O2_At",
    species=M.O2,
    species_ppm="21 percent",
    delta=0,
)
Reservoir(
    name="S_b",  # box name
    geometry={"area": "2.85e14m**2", "volume": "3E16 m**3"},
    concentration={M.O2: "200 umol/l"},  # initial concentration
    delta={M.O2: 0},
    seawater_parameters={"T": 21.5, "P": 1, "S": 35},
)
Species2Species(  # Ocean to atmosphere F8
    source=M.O2_At,  # Reservoir Species
    sink=M.S_b.O2,  # Reservoir Species
    species=M.O2,
    piston_velocity="4.8 m/d",
    ctype="gasexchange",
    id="ex_O2",
)
M.run()
