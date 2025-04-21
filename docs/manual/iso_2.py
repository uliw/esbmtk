"""Testcase for scale with flux isotope calculations.

Also available as iso_2.py in the ESBMTK Examples Repository. See
https://github.com/uliw/ESBMTK-Examples/tree/main/Examples_from_the_manual
"""

from esbmtk import (
    ConnectionProperties,  # the connection class
    Model,  # the model class
    Reservoir,  # the reservoir class
)

M = Model(
    stop="3 kyr",  # end time of model
    max_timestep="100 yr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
)
Reservoir(
    name="S_b",  # box name
    volume="50E16 m**3",  # surface box volume
    concentration={M.DIC: "2000 umol/l"},  # initial concentration
    delta={M.DIC: 0},
)
Reservoir(
    name="D_b",  # box name
    volume="50E16 m**3",  # deeb box volume
    concentration={M.DIC: "2000 umol/l"},  # initial concentration
    delta={M.DIC: 0},
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
)
Fd = M.flux_summary(filter_by="downwelling", return_list=True)[0]

ConnectionProperties(  # thermohaline upwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_flux",
    ref_flux=Fd,
    scale=0.2,
    epsilon=28,
    id="PIC",
)

M.run()
