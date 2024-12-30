"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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
from .carbonate_chemistry import *
from .connections import *
from .esbmtk import *
from .esbmtk_base import *
from .extended_classes import *
from .ode_backend import *
from .post_processing import *
from .processes import *
from .sealevel import *
from .seawater import *
from .utility_functions import *

np.seterr(invalid="ignore")
