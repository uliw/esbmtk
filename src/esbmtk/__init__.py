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

import numpy as np
import pkg_resources
from pint import UnitRegistry

__version__ = pkg_resources.get_distribution("esbmtk").version

# Units are needed for the subsequent imports
ureg = UnitRegistry(on_redefinition="ignore")
ureg.enable_contexts("chemistry")
Q_ = ureg.Quantity
ureg.define("Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups")
ureg.define("Mol = 1 * mol / liter = M")
ureg.define("fraction = [] = frac")
ureg.define("percent = 1e-2 frac = pct")
ureg.define("permil = 1e-3 fraction")
ureg.define("ppm = 1e-6 fraction")
np.seterr(invalid="ignore")

from .base_classes import (
    Flux,
    Sink,
    Source,
    Species,
    SpeciesError,
    SpeciesProperties,
)
from .carbonate_chemistry import (
    add_carbonate_system_1,
    add_carbonate_system_2,
    carbonate_system_1,
    carbonate_system_2,
    get_hplus,
    get_pco2,
)
from .connections import ConnectionProperties, Species2Species
from .esbmtk_base import esbmtkBase
from .extended_classes import (
    DataField,
    ExternalCode,
    ExternalData,
    GasReservoir,
    Reservoir,
    Signal,
    SinkProperties,
    SourceProperties,
    SpeciesNoSet,
    VectorData,
    VirtualSpecies,
)
from .model import Model
from .post_processing import (
    carbonate_system_1_pp,
    carbonate_system_2_pp,
    gas_exchange_fluxes,
)
from .processes import (
    gas_exchange,
    init_weathering,
    weathering_isotopes,
    weathering_isotopes_delta,
    weathering_isotopes_alpha,
    weathering_no_isotopes,
    weathering_ref_isotopes,
)
from .seawater import SeawaterConstants
from .utility_functions import (
    build_ct_dict,
    check_for_quantity,
    create_bulk_connections,
    data_summaries,
    initialize_reservoirs,
    phc,
    register_return_values,
    set_y_limits,
)

# __all__ = [
#     "ConnectionProperties",
#     "DataField",
#     "ExternalCode",
#     "ExternalData",
#     "Flux",
#     "GasReservoir",
#     "Model",
#     "Reservoir",
#     "SeawaterConstants",
#     "Signal",
#     "Sink",
#     "SinkProperties",
#     "Source",
#     "SourceProperties",
#     "Species",
#     "Species2Species",
#     "SpeciesError",
#     "SpeciesNoSet",
#     "SpeciesProperties",
#     "VectorData",
#     "VirtualSpecies",
#     "__version__",
#     "add_carbonate_system_1",
#     "add_carbonate_system_2",
#     "build_ct_dict",
#     "carbonate_system_1",
#     "carbonate_system_1_pp",
#     "carbonate_system_2",
#     "carbonate_system_2_pp",
#     "check_for_quantity",
#     "create_bulk_connections",
#     "data_summaries",
#     "esbmtkBase",
#     "gas_exchange",
#     "gas_exchange_fluxes",
#     "get_hplus",
#     "get_pco2",
#     "init_weathering",
#     "initialize_reservoirs",
#     "phc",
#     "register_return_values",
#     "set_y_limits",
#     "weathering_all_isotopes",
#     "weathering_no_isotopes",
#     "weathering_ref_isotopes",
# ]
