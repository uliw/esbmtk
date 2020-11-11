from esbmtk import Model, Element, Species, Reservoir
from esbmtk import Signal, Connect, Source, Sink

# create model
Model(
    name="C_Cycle",     # model name
    stop="1000 yrs",    # end time of model
    timestep=" 1 yr",   # base unit for time
    mass_unit = "mol",  # base unit for mass
    volume_unit = "l",  # base unit for volume
    element="Carbon",    # load default element and species definitions
    offset="1751 yrs"   # map to external timescale
)

Signal(name = "ACR",              # Signal name
       species = CO2,             # Species
       filename = "emissions.csv" # filename
)

Source(name="Fossil_Fuel_Burning", species=CO2)
Source(name="Carbonate_Weathering", species=CO2)
Source(name="Organic_Weathering", species=CO2)
Source(name="Volcanic", species=CO2)
Sink(name="Carbonate_burial", species=CaCO3)
Sink(name="OM_burial", species=OM)

Reservoir(
    name="Ocean",                # Name of reservoir
    species=DIC,                 # Species handle
    delta=0,                     # initial delta
    concentration="2.6 mmol/l", # cocentration 
    volume="1.332E18 m**3",      # reservoir size (m^3)
)

# connect source to reservoir
Connect(
    source=Fossil_Fuel_Burning,  # source of flux
    sink=Ocean,                  # target of flux
    rate="0 mol/yr",             # weathering flux in 
    delta=0,                     # set a default flux
    pl=[ACR],                    # process list, here the anthropogenic carbon release
    scale=0.5                    # assume that the ocean uptke is half of the ACR
)

Connect(
    source=Carbonate_Weathering,  # source of flux
    sink=Ocean,                   # target of flux
    rate="12.3E12 mol/yr",        # weathering flux in 
    delta=0,                      # isotope ratio
    plot="no",
)

Connect(
    source=Organic_Weathering,  # source of flux
    sink=Ocean,                 # target of flux
    rate="4.0E12 mol/yr",       # flux rate
    delta=-20,                  # isotope ratio
    plot="no",
)

Connect(
    source=Volcanic,      # source of flux
    sink=Ocean,           # target of flux
    rate="6.0E12 mol/yr", # flux rate
    delta=-5,             # isotope ratio
    plot="no",
)

Connect(
    source=Ocean,          # source of flux
    sink=OM_burial,        # target of flux
    rate="4.2E12 mol/yr",  # burial rate
    alpha=-26.32,          # fractionation factor
)

Connect(
    source=Ocean,          # source of flux
    sink=Carbonate_burial, # target of flux
    rate="18.1E12 mol/yr", # burial rate
    ref_value="2.6 mmol/l",
    k_concentration = 1000,
    alpha=0,               # set the istope fractionation
)

# Run the model
C_Cycle.run()

# plot the results
C_Cycle.plot_data()
# save the results
C_Cycle.save_data()
