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
    stop="6 Myr",  # end time of model
    max_timestep="1 kyr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
    # debug_equations_file=True,
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
    delta={M.DIC: -10},
)
SinkProperties(
    name="burial",
    species=[M.DIC],
)
# reservoir definitions
Reservoir(
    name="S_b",  # box name
    volume="3E16 m**3",  # surface box volume
    concentration={M.DIC: "1 umol/l"},  # initial concentration
    delta={M.DIC: -10},
)
Reservoir(
    name="D_b",  # box name
    volume="100E16 m**3",  # deeb box volume
    concentration={M.DIC: "1 umol/l"},  # initial concentration
    delta={M.DIC: -10},
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
    filename="signal_from_csv_data_isotopes_reverse.csv",
    reverse_time=True,
)

ConnectionProperties(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
    signal=M.CR,
    species=[M.DIC],
    ctype="regular",
    delta=-10,
)
M.run()

# M.plot([M.S_b.DIC, M.D_b.DIC, M.CR])
# M.plot([M.CR])
