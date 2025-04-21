"""Test isotope calculations for connections with a delta value."""

from esbmtk import (
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
    SourceProperties,
)

M = Model(
    stop="500 yr",  # end time of model
    max_timestep="10 yr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
)
# ---------------- Source to Reservoir rtegular flux --------------- #
SourceProperties(
    name="Fw",  # box name
    species=[M.DIC],
    delta={"DIC": 0},
)
Reservoir(
    name="S_b",  # box name
    volume="50E16 m**3",  # Surface box volume
    concentration={M.DIC: "0 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
ConnectionProperties(  # thermohaline upwelling
    source=M.Fw,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="regular",
    rate="5 Tmol/yr",
    delta=5,
    id="weathering",
)

# M.debug_equations_file = True
M.run()
# M.plot([M.S_b.DIC])
