from __future__ import annotations

from pint import UnitRegistry

ureg = UnitRegistry(on_redefinition="ignore")
Q_ = ureg.Quantity

ureg.define("Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups")
ureg.define("Mol = 1 * mol / liter = M")

# import utility_functions
# import esbmtk

# from .base_class import esbmtkBase


# rom .species_definitions import carbon, sulfur, hydrogen, phosphor
from .esbmtk import *
from .extended_classes import *
from .connections import ConnectionGroup, Connection, Connect
from .utility_functions import *
from .sealevel import *
from .solver import *
