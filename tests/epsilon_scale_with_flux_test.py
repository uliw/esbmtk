"""Test isotope calculations for connections with a delta value."""

from esbmtk import (
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
)

M = Model(
    stop="10 kyr",  # end time of model
    max_timestep="10 yr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
)

# ---------------- Reservoir to Reservoir scale by concentration --------------- #
"""Set up two reservoirs with fluxes between them"""
Reservoir(
    name="S_b1",  # box name
    volume="1E16 m**3",  # Surface box volume
    concentration={M.DIC: "2000 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
Reservoir(
    name="D_b1",  # box name
    volume="1E16 m**3",  # deeb box volume
    concentration={M.DIC: "1 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
ConnectionProperties(  # thermohaline upwelling
    ctype="scale_with_concentration",
    source=M.S_b1,  # source of flux
    sink=M.D_b1,  # target of flux
    scale=1e16,
    delta=5,
    id="downwelling_ref",
)
ConnectionProperties(  # thermohaline upwelling
    ctype="scale_with_concentration",
    source=M.D_b1,  # target of flux
    sink=M.S_b1,  # source of flux
    scale=1e16,
    id="upwelling_ref",
)

"""Now set up the reservoirs we use for the computation.
We use D_b with a scale with_flux connection, so this results in an
equation like F * l/c. This will fail if c == 0, so we need to set
and initial concentration.
"""

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
Fd = M.flux_summary(filter_by="downwelling_ref", return_list=True)[0]
Fu = M.flux_summary(filter_by="upwelling_ref", return_list=True)[0]

ConnectionProperties(
    ctype="scale_with_flux",
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ref_flux=Fd,
    scale=0.1,
    epsilon=5,
    id="downwelling",
)
ConnectionProperties(
    ctype="scale_with_flux",
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ref_flux=Fd,
    scale=0.9,
    epsilon=5,
    id="downwelling2",
)

ConnectionProperties(  # thermohaline upwelling
    ctype="scale_with_flux",
    source=M.D_b,  # target of flux
    sink=M.S_b,  # source of flux
    ref_flux=Fu,
    scale=1,
    id="upwelling",
)
# M.debug_equations_file = True
M.run(method="BDF")
# M.plot([M.S_b1.DIC, M.D_b1.DIC])
# M.plot([M.S_b.DIC, M.D_b.DIC])
