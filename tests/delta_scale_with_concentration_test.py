"""Test isotope calculations for connections with a delta value."""

from esbmtk import (
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
)

M = Model(
    stop="4 kyr",  # end time of model
    max_timestep="1 yr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
)

# ---------------- Reservoir to Reservoir scale by concentration --------------- #
Reservoir(
    name="S_b",  # box name
    volume="1E16 m**3",  # Surface box volume
    concentration={M.DIC: "2000 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
Reservoir(
    name="D_b",  # box name
    volume="1E16 m**3",  # deeb box volume
    concentration={M.DIC: "1 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
ConnectionProperties(  # thermohaline upwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=1e16 * 0.1,
    delta=5,
    id="downwelling",
)

ConnectionProperties(  # thermohaline upwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=1e16 * 0.9,
    delta=5,
    id="downwelling2",
)

ConnectionProperties(  # thermohaline upwelling
    sink=M.S_b,  # source of flux
    source=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=1e16,
    id="upwelling",
)
# M.debug_equations_file = True
M.run()
# M.plot([M.S_b.DIC, M.D_b.DIC])
