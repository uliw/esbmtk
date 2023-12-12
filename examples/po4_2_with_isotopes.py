# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    Connection,
    ReservoirGroup,  # the reservoir class
    ConnectionGroup,  # the connection class
    SourceGroup,  # the source class
    SinkGroup,  # sink class
    Q_,  # Quantity operator
    data_summaries,
)

# define the basic model parameters
M = Model(
    stop="500 yr",  # end time of model
    timestep="1 yr",  # upper limit of time step
    element=["Phosphor", "Carbon"],  # list of element definitions
)

# Fixed ratios
Redfield = 130  # C:P ratio
R_e = 0.01  # About 1% of the exported P is buried in the deep ocean
tau = Q_("100 year")  # PO4 residence time in surface box
thc = "20*Sv"  # Thermohaline circulation in Sverdrup
F_w_OM = M.set_flux("5850 Gmol", "year", M.C)  # P @280 ppm (Filipelli 2002)
F_w_PO4 = F_w_OM / Redfield

# Source definitions
SourceGroup(
    register=M,  # i.e., the instance will be available as M.weathering
    name="weathering",
    species=[M.DIC, M.PO4],
    isotopes={M.DIC: True, M.PO4: False},
)
SinkGroup(
    register=M,  #
    name="burial",
    species=[M.DIC, M.PO4],
    isotopes={M.DIC: True, M.PO4: False},
)

ReservoirGroup(
    name="S_b",
    register=M,
    volume="3E16 m**3",  # surface box volume
    concentration={M.DIC: "1 umol/l", M.PO4: "0 umol/l"},
    isotopes={M.DIC: True},
    delta={M.DIC: 0},
)

ReservoirGroup(
    name="D_b",  # box name
    register=M,  # this box will be available M.db
    volume="100E16 m**3",  # deeb box volume
    concentration={M.DIC: "1 umol/l", M.PO4: "0 umol/l"},  # species in box
    isotopes={M.DIC: True},
    delta={M.DIC: 0},
)

ConnectionGroup(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate={M.DIC: F_w_OM, M.PO4: F_w_PO4},  # rate of flux
    delta={M.DIC: 0},
    ctype="regular",  # required!
    id="weathering",  # connection id
)

ConnectionGroup(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling",
)
ConnectionGroup(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling",
)

# Primary production as a function of P-concentration
Connection(  #
    source=M.S_b.DIC,  # source of flux
    sink=M.D_b.DIC,  # target of flux
    ref_reservoirs=M.S_b.PO4,
    ctype="scale_with_concentration",
    scale=Redfield * M.S_b.volume / tau,
    id="OM_production",
    alpha=-28,  # mUr
)

# POP export as a funtion of OM export
Connection(  #
    source=M.S_b.PO4,  # source of flux
    sink=M.D_b.PO4,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="OM_production", return_list=True)[0],
    scale=1 / Redfield,
    id="POP",
)

# P burial
Connection(  #
    source=M.D_b.PO4,  # source of flux
    sink=M.burial.PO4,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="POP", return_list=True)[0],
    scale=R_e,
    id="P_burial",
)

# OM burial
Connection(  #
    source=M.D_b.DIC,  # source of flux
    sink=M.burial.DIC,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="OM_production", return_list=True)[0],
    scale=R_e,
    id="OM_burial",
    # alpha=0,
)


M.run()
pl = data_summaries(
    M,
    [M.DIC],  # species
    [M.S_b, M.D_b],  # boxes
)
M.plot(pl, fn="po4_2_with_isotopes.png")
# M.save_state(directory="state_po42")
