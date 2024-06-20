# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    ConnectionProperties,  # the connection class
    SourceProperties,  # the source class
    SinkProperties,  # sink class
)
# define the basic model parameters
M = Model(
    stop="3 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Phosphor"],  # list of element definitions
)
# now try this
from esbmtk import Q_
tau = Q_("100 years")
tau * 12
# boundary conditions
F_w =  M.set_flux("45 Gmol", "year", M.P) # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface box
F_b = 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup
# Source definitions
SourceProperties(
    name="weathering",
    species=[M.PO4],
)
SinkProperties(
    name="burial",
    species=[M.PO4],
)
# reservoir definitions
Reservoir(
    name="S_b",  # box name
    volume="3E16 m**3",  # surface box volume
    concentration={M.PO4: "0 umol/l"},  # initial concentration
)
Reservoir(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.PO4: "0 umol/l"},  # initial concentration
)
ConnectionProperties(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
    ctype="regular",
)
ConnectionProperties(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling_PO4",
)
ConnectionProperties(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling_PO4",
)
ConnectionProperties(  #
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
    species=[M.PO4],  # apply this only to PO4
)
ConnectionProperties(  #
    source=M.D_b,  # source of flux
    sink=M.burial,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="primary_production",return_list=True)[0],
    scale=F_b,
    id="burial",
    species=[M.PO4],
)
from esbmtk import Signal

Signal(
    name="CR",  # Signal name
    species=M.PO4,  # SpeciesProperties
    start="1 Myrs",
    shape="pyramid",
    duration="1 Myrs",
    mass="45 Pmol",
)

ConnectionProperties(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
    signal=M.CR,
    species=[M.PO4],
    ctype="regular",
)
M.run()
