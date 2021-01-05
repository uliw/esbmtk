from esbmtk import Model, Reservoir, Signal
from esbmtk import Connect, Source, Sink

volume = "1 liter"

# create model
Model(
    name="equi4",  # model name
    stop="1000 s",        # end time of model
    timestep="0.1 s",            # time step
    element=["Carbon"],  # initialize carbon species
    mass_unit="mol",
    volume_unit="l",
    mtype = "both",     # calculate mass and isotopes
)

Reservoir(
    name="R1",    # Name of reservoir
    species=CO2,             # Species handle
    delta=10,                 # initial delta
    concentration=f"10 mol/l",       # concentration
    volume=volume,         # reservoir size (m^3)
)

Reservoir(
    name="R2",    # Name of reservoir
    species=CO2,             # Species handle
    delta=0,                 # initial delta
    concentration=f"11 mol/l",       # concentration
    volume=volume,         # reservoir size (m^3)
)

Connect(
    source=R1,   # target of flux
    sink=R2,    # source of flux
    rate="1 mol/s",           # flux rate
    ctype="flux_balance",
    k_value=1,
    left=[R1],
    right=[R2],
    #alpha=0,                # isotope fractionation
    plot="no",
)


# Run the model
equi4.run()

# plot the results
equi4.plot_data()
