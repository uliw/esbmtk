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

tau = Q_("100 years") * 12
# boundary conditions
F_w = M.set_flux("45 Gmol", "year", M.C)  # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # DIC residence time in surface box
F_b = 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup
# Source definitions
SourceProperties(
    name="weathering",
    species=[M.DIC],
)
SinkProperties(
    name="burial",
    species=[M.DIC],
)
# reservoir definitions
Reservoir(
    name="S_b",  # box name
    volume="3E16 m**3",  # surface box volume
    concentration={M.DIC: "0 umol/l"},  # initial concentration
)
Reservoir(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.DIC: "0 umol/l"},  # initial concentration
)
ConnectionProperties(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling_DIC",
)
ConnectionProperties(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling_DIC",
)
ConnectionProperties(  #
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
    species=[M.DIC],  # apply this only to PO4
)
ConnectionProperties(  #
    source=M.D_b,  # source of flux
    sink=M.burial,  # target of flux
    ctype="scale_with_flux",
    ref_flux="primary_production",
    scale=F_b,
    id="burial",
    species=[M.DIC],
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
)
M.run()
# M.plot([M.CR, M.S_b.DIC, M.D_b.DIC])
