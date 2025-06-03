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


from . initialize_unit_registry import Q_, ureg
from .version import get_version
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
from .seawater import SeawaterConstants
from .processes import (
    gas_exchange,
    init_weathering,
    weathering_isotopes,
    weathering_isotopes_delta,
    weathering_isotopes_alpha,
    weathering_no_isotopes,
    weathering_ref_isotopes,
)
from .post_processing import (
    carbonate_system_1_pp,
    carbonate_system_2_pp,
    gas_exchange_fluxes,
)
from .model import Model
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
from .esbmtk_base import esbmtkBase
from .connections import ConnectionProperties, Species2Species
from .carbonate_chemistry import (
    add_carbonate_system_1,
    add_carbonate_system_2,
    carbonate_system_1,
    carbonate_system_2,
    get_hplus,
    get_pco2,
)
from .base_classes import (
    Flux,
    Sink,
    Source,
    Species,
    SpeciesError,
    SpeciesProperties,
)
import numpy as np
np.seterr(invalid="ignore")
