from __future__ import annotations
import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry(on_redefinition="ignore")
ureg.enable_contexts("chemistry")
Q_ = ureg.Quantity
ureg.define("Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups")
ureg.define("Mol = 1 * mol / liter = M")
ureg.define("fraction = [] = frac")
ureg.define("percent = 1e-2 frac = pct")
ureg.define("permil = 1e-3 fraction")
ureg.define("ppm = 1e-6 fraction")
from .esbmtk_base import *
from .esbmtk import *
from .extended_classes import *
from .connections import *
from .utility_functions import *
from .sealevel import *
from .ode_backend import *
from .seawater import *
from .bio_pump_functions0.carbonate_chemistry import *
from .post_processing import *

np.seterr(invalid="ignore")
