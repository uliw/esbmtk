"""Test signal scdaling.

1. read signal from file
2. connect signal to pre-existing connection
3. modify connection scalingf factor after the fact.
"""

# import classes from the esbmtk library
from esbmtk import (
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
    Signal,
    SourceProperties,  # the source class
)

# define the basic model parameters
M = Model(
    stop="3 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
    debug_equations_file=False,
)

# boundary conditions
F_w = M.set_flux("45 Gmol", "year", M.C)  # P @280 ppm (Filipelli 2002)
# Source definitions
SourceProperties(
    name="weathering",
    species=[M.DIC],
)

# reservoir definitions
Reservoir(
    name="S_b",  # box name
    volume="3E16 m**3",  # surface box volume
    concentration={M.DIC: "0 umol/l"},  # initial concentration
)

Signal(
    name="CR",  # Signal name
    species=M.DIC,  # SpeciesProperties
    start="1 Myrs",
    filename="signal_from_csv_data.csv",
)

ConnectionProperties(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
    signal=M.CR,
    species=[M.DIC],
    ctype="regular",
    scale=1,
)

M.Conn_weathering_to_S_b_DIC_river.scale = 5

M.run()
# M.plot([M.CR, M.S_b.DIC])
