"""Test signal creation from csv file.

Here we use the isotopes_only signal type
"""

# import classes from the esbmtk library
from esbmtk import (
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
    element=["Sulfur"],  # list of element definitions
    # debug_equations_file=True,
)

# Source definitions
SourceProperties(
    name="weathering",
    species=[M.SO4],
    delta={M.SO4: -18},
)
SinkProperties(
    name="burial",
    species=[M.SO4],
)

Reservoir(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.SO4: "28000 umol/l"},  # initial concentration
    delta={M.SO4: 21},
)

ConnectionProperties(
    id="river",  # connection id2
    ctype="regular",
    source=M.weathering,  # source of flux
    sink=M.D_b,  # target of flux
    rate="12 Tmol/yr",
    species=[M.SO4],
    delta=-18,
)
ConnectionProperties(
    ctype="regular",
    id="burial",  # connection id
    sink=M.burial,  # source of flux
    source=M.D_b,  # target of flux
    rate="12 Tmol/yr",  # rate of flux
    species=[M.SO4],
    # epsilon=-39,
)

Signal(
    name="CR",  # Signal name
    species=M.SO4,  # SpeciesProperties
    filename="burial_signal_epsilon_only_from_csv_data.csv",
    stype="epsilon_only",
)

# M.connection_summary(filter_by="burial")
M.Conn_D_b_to_SO4_SO4_burial.signal = M.CR


M.run()
# M.plot([M.CR, M.D_b.SO4])
