from esbmtk import (
    Model,
    Reservoir,  # the reservoir class
    ConnectionProperties,  # the connection class
    SourceProperties,  # the source class
    SinkProperties,  # sink class
    data_summaries,
    Q_,
)
M = Model(
    stop="1 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Phosphor", "Carbon"],  # list of species definitions
)

# boundary conditions
F_w_PO4 =  M.set_flux("45 Gmol", "year", M.PO4) # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface boxq
F_b = 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup
Redfield = 106 # C:P
SourceProperties(
    name="weathering",
    species=[M.PO4, M.C],
    isotopes={M.C: True},
)
SinkProperties(
    name="burial",
    species=[M.PO4, M.C],
    isotopes={M.C: True},
)
Reservoir(
    name="S_b",
    volume="3E16 m**3",  # surface box volume
    concentration={M.C: "1 umol/l", M.PO4: "0 umol/l"},
    isotopes={M.C: True},
    delta={M.C: 0},
)
Reservoir(
    name="D_b",
    volume="100E16 m**3",  # deeb box volume
    concentration={M.C: "1 umol/l", M.PO4: "0 umol/l"},
    isotopes={M.C: True},
    delta={M.C: 0},
)
ConnectionProperties(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="thc_up",
)
ConnectionProperties(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="thc_down",
)
ConnectionProperties(  # weathering
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate={M.C: F_w_PO4 * Redfield, M.PO4: F_w_PO4},  # rate of flux
    ctype="regular",
    id="weathering",  # connection id
    delta={M.C: -28},
)
ConnectionProperties(  # P-uptake by photosynthesis
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
    species=[M.PO4],  # apply this only to PO4
)
ConnectionProperties(  # OM Primary production as a function of P-concentration
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ref_reservoirs=M.S_b.PO4,
    ctype="scale_with_concentration",
    scale=Redfield * M.S_b.volume / tau,
    species=[M.C],
    id="OM_production",
    epsilon=-28,  # mUr
)
ConnectionProperties(  # P burial
    source=M.D_b,  # source of flux
    sink=M.burial,  # target of flux
    ctype="scale_with_flux",
    ref_flux=M.flux_summary(filter_by="primary_production", return_list=True)[0],
    scale={M.PO4: F_b, M.C: F_b * Redfield},
    id="burial",
    epsilon={M.C: 0},
)

M.run()
pl = data_summaries(
    M,  # model instance 
    [M.C, M.PO4],  # SpeciesProperties list 
    [M.S_b, M.D_b],  # Reservoir list
)
M.plot(pl, fn="po4_4.png")

M.run()
