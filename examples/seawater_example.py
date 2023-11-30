from math import log10
from esbmtk import (
    Model,  # the model class
    ReservoirGroup,
)

# define the basic model parameters
M = Model(
    stop="6 Myr",  # end time of model
    timestep="1 kyr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
    concentration_unit="mol/kg",
    opt_k_carbonic=13,  # Use Millero 2006
    opt_pH_scale=1,  # 1:total, 3:free scale
    opt_buffers_mode=2,  # carbonate, borate water alkalinity only
)
ReservoirGroup(
    name="S_b",  # box name
    geometry=[-200, -800, 1],  # upper, lower, fraction
    concentration={M.DIC: "2220 umol/kg", M.TA: "2300 umol/kg"},  # species in box
    seawater_parameters={
        "T": 25,  # deg celsius
        "P": 0,  # bar
        "S": 35,  # PSU
    },
    register=M,  # this box will be available M.db
)
print(f"M.S_b.density = {M.S_b.swc.density:.2e} kg/m**3")
print(f"M.S_b.pk1 = {-log10(M.S_b.swc.K1):.2f}")

M.S_b.swc.show()
