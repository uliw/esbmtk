from esbmtk import Model, Element, Species, Reservoir
from esbmtk import Signal, Connect, Source, Sink, Flux
from esbmtk import ExternalData
import matplotlib.pyplot as plt

# create model
Model(
    name="test",        # model name
    stop=1000,          # end time of model
    time_unit="yr",     # time units 
    dt=1,               # time step
)

# Element properties
Element(
    name="C",                  # Element Name
    model=test,                # Model handle
    mass_unit="mmol",          # base mass unit
    li_label="C^{12$S",        # Name of light isotope
    hi_label="C^{13}$S",       # Name of heavy isotope
    d_label="$\delta^{13}$C",       # Name of isotope delta
    d_scale="VPDB",            # Isotope scale. End of plot labels
    r=0.0112372,  # VPDB C13/C12 ratio https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf
)

# add species
Species(name="CO2", element=C)    # Name & element handle
Species(name="DIC", element=C)    # Name & element handle
Species(name="CaCO3", element=C)  # Name & element handle

Reservoir(
    name="Ocean",       # Name of reservoir
    species=DIC,        # Species handle
    delta=0,            # initial delta
    concentration=3,    # concentration 
    unit="mmol",        # mass unit
    volume=1.332E18,    # reservoir size (m^3)
)

Source(name="Carbonate_Weathering", species=CO2)
Sink(name="Carbonate_Burial", species=CaCO3)

Signal(
    name="P1",        # Name
    species=CO2,      # Element
    start=10,         # start
    duration=100,     # duration
    delta=1,          # isotope effect
    shape="pyramid",  # signal shape
    magnitude=1e17,   # magnitude or mass
)

# connect source to reservoir
Connect(
    source=Carbonate_Weathering,  # source of flux
    sink=Ocean,                   # target of flux
    rate=18E12,                   # flux rate
    pl=[P1]                       # process list
)


Connect(
    source=Ocean,          # source of flux
    sink=Carbonate_Burial, # target of flux
    kvalue = 1000,         # flux rate 
    C0 =3,                 # reference concentration
    rate = 18E12,           # flux rate
    alpha = 0,
)

# Run the model
test.run()

# plot the results
test.plot_data()
