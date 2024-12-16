# This example is available as iso_3.py in ESBMTK Examples Repository
# https://github.com/uliw/ESBMTK-Examples/tree/main/Examples_from_the_manual
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    ConnectionProperties,  # the connection class
    SourceProperties,  # the source class
    SinkProperties,  # sink class
)
M = Model(
    stop="3 kyr",  # end time of model
    max_timestep="100 yr",  # upper limit of time step
    element=["Oxygen"],  # list of element definitions
)
Reservoir(
    name="S_b",  # box name
    volume="50E16 m**3",  # surface box volume
    concentration={M.O2: "200 umol/l"},  # initial concentration
    delta={M.O2: 0},
)
Reservoir(
    name="D_b",  # box name
    volume="50E16 m**3",  # deeb box volume
    concentration={M.O2: "200 umol/l"},  # initial concentration
    delta={M.O2: 0},
)
ConnectionProperties(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale="20 Sv",
    id="downwelling",
)
ConnectionProperties(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale="20 Sv",
    id="upwelling",
    epsilon=5, # mUr
)
M.run()
