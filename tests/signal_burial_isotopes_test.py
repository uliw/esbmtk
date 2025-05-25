"""Test signal ccreation from csv file.

We use the same geometry and fluxes as om po4_2, but instead
of phosphate, we use carbon.
"""

# import classes from the esbmtk library
from esbmtk import (
    Q_,
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
    Signal,
    SinkProperties,  # sink class
    SourceProperties,  # the source class
)

# define the basic model parameters
M = Model(
    stop="3 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
)


F_w = M.set_flux("45 Gmol", "year", M.C)  # P @280 ppm (Filipelli 2002)
# Source definitions
SourceProperties(
    name="weathering",
    species=[M.DIC],
    delta={M.DIC: 12},
)
SinkProperties(
    name="burial",
    species=[M.DIC],
)
Reservoir(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.DIC: "1120 umol/l"},  # initial concentration
    delta={M.DIC: 12},
)
Signal(
    name="CR",  # Signal name
    species=M.DIC,  # SpeciesProperties
    filename="burial_signal_isotopes_from_csv_data.csv",
)
ConnectionProperties(
    id="river",  # connection id2
    ctype="regular",
    source=M.weathering,  # source of flux
    sink=M.D_b,  # target of flux
    rate=F_w,  # rate of flux
    species=[M.DIC],
    delta=12,
)
ConnectionProperties(
    ctype="regular",
    sink=M.burial,  # source of flux
    source=M.D_b,  # target of flux
    rate=0,  # rate of flux
    id="burial",  # connection id
    signal=M.CR,
    species=[M.DIC],
)

M.run()
# M.plot([M.CR, M.D_b.DIC])
