from esbmtk import Model,  Reservoir
from esbmtk import Connect, Source, Sink, get_delta

Model(
    name="F",     # model name
    stop="10000 yrs",    # end time of model
    timestep=" 100 yr",   # base unit for time
    mass_unit = "mol",  # base unit for mass
    volume_unit = "l",  # base unit for volume
    element="Carbon",    # load default element and species definitions
    mtype = "both",     # calculate mass and isotopes
)
Source(name="SO1", species=CO2)
Sink(name="SI1", species=CO2)

Reservoir(
    name="O",                # Name of reservoir
    species=CO2,                 # Species handle
    delta=0,                     # initial delta
    concentration="2.6 mmol/l", # cocentration 
    volume="1.332E18 m**3",      # reservoir size (m^3)
)

Connect(
    source=SO1,  # source of flux
    sink=O,                   # target of flux
    rate="12.3E12 mol/yr",        # weathering flux in 
    delta=0,                      # isotope ratio
    plot="no",
)

Connect(
    source=O,  # source of flux
    sink=SI1,                   # target of flux
    rate="12.3E12 mol/yr",        # weathering flux in 
    alpha=12,                      # isotope ratio
    plot="yes",
)

F.run()
F.plot_data()
