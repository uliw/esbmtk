from esbmtk import Model, Reservoir, Signal
from esbmtk import Connect, Source, Sink

volume = "1 liter"

# create model
Model(
    name="equi4",  # model name
    stop="2 s",        # end time of model
    timestep="0.001 s",            # time step
    element=["Carbon"],  # initialize carbon species
    mass_unit="mol",
    volume_unit="l",
    m_type = "both",     # calculate mass and isotopes
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
    ctype="flux_balance",
    k_value=1,
    left=[R1],
    right=[R2],
    plot="no",
)


# Run the model
equi4.run()

# plot the results
equi4.plot_data()
