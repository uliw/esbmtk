# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    ConnectionProperties,  # the connection class
    SourceProperties,  # the source class
    SinkProperties,  # the sink class
    Q_, #for unit parsing
)

# define fundamental model parameters
M = Model(
    stop="3 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Phosphor"],  # list of element definitions
)
M = Model(
    stop="3 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Phosphor"],  # list of element definitions
    mass_unit="mol", #can be changed to another mass unit
    concentration_unit="mol/kg", #can be changed to another concentration unit
)
# try this:
from esbmtk import Q_
tau = Q_("100 years")
tau * 0.5 # Does not raise any error
thc = Q_("20 Sverdrup")
# boundary conditions
F_w =  M.set_flux("45 Gmol", "year", M.P) # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface box
F_b = 0.01  # About 1% of the exported P is buried in the deep ocean
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
Reservoir( #Surface Box
    name="S_b",  # box name 
    volume="3E16 m**3",  # surface box volume
    concentration={M.PO4: "0 umol/kg"},  # initial concentration
)

Reservoir( #Deep Box
    name="D_b",  # box name
    volume="100E16 m**3",  # deep box volume
    concentration={M.PO4: "0 umol/kg"},  # initial concentration
)
ConnectionProperties(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux 
    id="river",  # connection id
    ctype="regular", #connection type
)
ConnectionProperties(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc, #(in sverdrups, i.e. volumetric rate) 
    #volume / time * concentration (i.e. mass per unit volume) = flux (i.e. mass transfer per unit time)
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
    # volume / time * concentration (i.e. mass per unit volume) = flux (i.e mass transfer per unit time)
    id="primary_production",
    species=[M.PO4],  # apply this only to PO4
)
M.run()
