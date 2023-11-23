# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    ReservoirGroup,  # the reservoir class
    ConnectionGroup,  # the connection class
    SourceGroup,  # the source class
    SinkGroup,  # sink class
    Q_,  # Quantity operator
    data_summaries,
)

# define the basic model parameters
M = Model(
    stop="6 Myr",  # end time of model
    timestep="1 kyr",  # upper limit of time step
    element=["Phosphor", "Carbon"],  # list of element definitions
)

# boundary conditions
F_w = M.set_flux("45 Gmol", "year", M.P)  # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface box
R_e = 1 - 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup

# Source definitions
SourceGroup(
    name="weathering",
    species=[M.PO4, M.DIC],
    register=M,  # i.e., the instance will be available as M.weathering
)
SinkGroup(
    name="burial",
    species=[M.PO4, M.DIC],
    register=M,  #
)

ReservoirGroup(
    name="S_b",
    volume="3E16 m**3",  # surface box volume
    concentration={M.DIC: "0 umol/l", M.PO4: "0 umol/l"},
    register=M,
)

ReservoirGroup(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.DIC: "0 umol/l", M.PO4: "0 umol/l"},  # species in box
    register=M,  # this box will be available M.db
)

ConnectionGroup(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    ctype="regular",  # required!
    id="river",  # connection id
)

ConnectionGroup(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling_PO4",
    # ref_reservoirs=M.sb, defaults to the source instance
)
ConnectionGroup(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling_PO4",
)

ConnectionGroup(  #
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
)

ConnectionGroup(  #
    source=M.D_b,  # source of flux
    sink=M.burial,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="primary_production PO4", return_list=True)[0],
    scale=1 - R_e,
    id="burial",
)

M.run()
pl = data_summaries(M, [M.DIC, M.PO4], [M.S_b, M.D_b], M)
M.plot(pl, fn="po4_2.png")
# M.save_state(directory="state_po42")
