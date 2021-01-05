from esbmtk import Model, Element, Species, Reservoir
from esbmtk import Signal, Connect, Source, Sink, Flux
from esbmtk import ExternalData

# create model
Model(
    name="C_Cycle",     # model name
    stop="200 yrs",     # end time of model
    timestep=" 1 yr",   # time unit
    mass_unit = "mol",  # mass unit
    volume_unit = "l",  # volume unit
    mtype = "both",     # calculate mass and isotopes
)

# Element properties
Element(
    name="C",                  # Element Name
    model=C_Cycle,             # Model handle
    mass_unit="mmol",          # base mass unit
    li_label="C^{12$S",        # Name of light isotope
    hi_label="C^{13}$S",       # Name of heavy isotope
    d_label="$\delta^{13}$C",       # Name of isotope delta
    d_scale="VPDB",            # Isotope scale. End of plot labels
    r=0.0112372,  # VPDB C13/C12 ratio https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf
)

# add species
Species(name="CO2", element=C)  # Name & element handle
Species(name="DIC", element=C)
Species(name="OM", element=C)
Species(name="CaCO3", element=C)

Signal(name = "ACR",              # Signal name
       species = CO2,             # Species
       filename = "test-data.csv" # filename
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
    delta=2,                     # initial delta
    concentration="2.62 mmol/l", # cocentration 
    volume="1.332E18 m**3",      # reservoir size (m^3)
)

# connect source to reservoir
Connect(
    source=Fossil_Fuel_Burning,  # source of flux
    sink=Ocean,                  # target of flux
    rate="0 mol/yr",             # weathering flux in 
    delta=0,                     # set a default flux
    pl=[ACR],                    # process list, here the anthropogenic carbon release
)

Connect(
    source=Carbonate_Weathering,  # source of flux
    sink=Ocean,                   # target of flux
    rate="12.3E12 mol/yr",        # weathering flux in 
    delta=0,                      # isotope ratio
)

Connect(
    source=Organic_Weathering,  # source of flux
    sink=Ocean,                 # target of flux
    rate="4.0E12 mol/yr",       # flux rate
    delta=-20,                  # isotope ratio
)

Connect(
    source=Volcanic,      # source of flux
    sink=Ocean,           # target of flux
    rate="6.0E12 mol/yr", # flux rate
    delta=-5,             # isotope ratio
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
    alpha=0,               # set the isotope fractionation
)

# Run the model
C_Cycle.run()

# plot the results
C_Cycle.plot_data()
# save the results
C_Cycle.save_data()
