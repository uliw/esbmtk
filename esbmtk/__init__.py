from __future__ import annotations

from pint import UnitRegistry

ureg = UnitRegistry(on_redefinition="ignore")
Q_ = ureg.Quantity

ureg.define("Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups")
ureg.define("Mol = 1 * mol / liter = M")

# import utility_functions
# import esbmtk

# from .base_class import esbmtkBase
from .utility_functions import *
from .esbmtk import *
