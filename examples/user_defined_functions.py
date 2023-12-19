# import classes from the esbmtk library
from esbmtk import (
    Model,  # the model class
    Reservoir,  # the reservoir class
    Connection,  # the connection class
    Source,  # the source class
    Sink,  # sink class
    Q_,  # Quantity operator
    register_user_function,
)

# define the basic model parameters
M = Model(
    stop="1000 yr",  # end time of model
    timestep="1 yr",  # upper limit of time step
    element=["Phosphor"],  # list of element definitions
    # parse_model=False,
)

# boundary conditions
F_w = M.set_flux("45 Gmol", "year", M.P)  # P @280 ppm (Filipelli 2002)
tau = Q_("100 year")  # PO4 residence time in surface box
R_e = 1 - 0.01  # About 1% of the exported P is buried in the deep ocean
thc = "20*Sv"  # Thermohaline circulation in Sverdrup

# Source definitions
Source(
    name="weathering",
    species=M.PO4,
    register=M,  # i.e., the instance will be available as M.weathering
)
Sink(
    name="burial",
    species=M.PO4,
    register=M,  #
)

# reservoir definitions
Reservoir(
    name="S_b",  # box name
    species=M.PO4,  # species in box
    register=M,  # this box will be available as M.S_b
    volume="3E16 m**3",  # surface box volume
    concentration="0 umol/l",  # initial concentration
)
Reservoir(
    name="D_b",  # box name
    species=M.PO4,  # species in box
    register=M,  # this box will be available M.D_b
    volume="100E16 m**3",  # deeb box volume
    concentration="0 umol/l",  # initial concentration
)

Connection(
    source=M.weathering,  # source of flux
    sink=M.S_b,  # target of flux
    rate=F_w,  # rate of flux
    id="river",  # connection id
    ctype="regular",
)

Connection(  # thermohaline downwelling
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="downwelling_PO4",
    # ref_reservoirs=M.S_b, defaults to the source instance
)
Connection(  # thermohaline upwelling
    source=M.D_b,  # source of flux
    sink=M.S_b,  # target of flux
    ctype="scale_with_concentration",
    scale=thc,
    id="upwelling_PO4",
)

Connection(  #
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
)

# --- Adding a user supplied function into the model --- #

""" To add a user supplied function, we need a local file
(e.g., my_functions.py) where we define our new function
(e.g.,  my_burial()). In addition we need some code that
describes the relationship of this function with the model
(e.g., add_my_function()).

Assuming that the functions my_burial() and
add_my_burial() exist in the local file my_functions.py,
we can register the module my_functions.py, and the
function my_burial() with the ESBMTK model using the aptly
named register_user_function() function. Note that the
function name can also be a list of several function names.
"""
register_user_function(M, "my_functions", "my_burial")

""" Next we import the import the function add_my_burial()
into this script, so that we connect it with the model.
Unlike my_burial() the add_my_burial() function can also be
defined in this script.
Note that unless add_my_burial() contains code to correctly
interpret quantities, you need to make sure that the parameters
passed to add_my_burial() are float numbers in model units
(e.g., liter, year, kg) etc. That is why the below code
uses M.D_b.volume.magnitude rather than  M.D_b.volume (which
is a quantity)
"""
from my_functions import add_my_burial

# Last but not least, apply the new function
add_my_burial(
    M.D_b,  # Source
    M.burial,  # Sink
    M.PO4,  # Species
    M.D_b.volume.magnitude / 2000.0,  # Scale
)

M.run()
M.plot([M.S_b, M.D_b], fn="po4_1.png")
# M.save_state(directory="state_po41")
