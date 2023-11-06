# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    Connection,  # the connection class
    Source,  # the source class
    Sink,  # sink class
    Q_,  # Quantity operator
)

# define the basic model parameters
M = Model(
    name="M",  # model name
    stop="3 Myr",  # end time of model
    timestep="1 kyr",  # upper limit of time step
    element=["Phosphor"],  # list of element definitions
)

# boundary conditions
F_w =  M.set_flux("45 Gmol", "year", M.P) # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface box
F_b = 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup

# Source definitions
Source(
    name="weathering",
    species=M.PO4,
    register=M,  # i.e., the instance will be available as M.weathering
)
Sink(
    name="burial",
    species=M.PO4,
    register=M,  #
)

# reservoir definitions
Reservoir(
    name="sb",  # box name
    species=M.PO4,  # species in box
    register=M,  # this box will be available as M.sb
    volume="3E16 m**3",  # surface box volume
    concentration="0 umol/l",  # initial concentration
)
Reservoir(
    name="db",  # box name
    species=M.PO4,  # species in box
    register=M,  # this box will be available M.db
    volume="100E16 m**3",  # deeb box volume
    concentration="0 umol/l",  # initial concentration
)

Connection(
    source=M.weathering,  # source of flux
    sink=M.sb,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
)

Connection(  # thermohaline downwelling
    source=M.sb,  # source of flux
    sink=M.db,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling_PO4",
    # ref_reservoirs=M.sb, defaults to the source instance
)
Connection(  # thermohaline upwelling
    source=M.db,  # source of flux
    sink=M.sb,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling_PO4",
)

Connection(  #
    source=M.sb,  # source of flux
    sink=M.db,  # target of flux
    ctype="scale_with_concentration",
    scale=M.sb.volume / tau,
    id="primary_production",
)

Connection(  #
    source=M.db,  # source of flux
    sink=M.burial,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="primary_production", return_list=True)[0],
    scale=F_b,
    id="burial",
)

M.run()
M.plot([M.sb, M.db])
M.save_data()
