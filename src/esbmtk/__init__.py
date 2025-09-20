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


from . initialize_unit_registry import Q_ as Q_, ureg as ureg
from .version import get_version as get_version
from .utility_functions import (
    build_ct_dict as build_ct_dict,
    check_for_quantity as check_for_quantity,
    create_bulk_connections as create_bulk_connections,
    data_summaries as data_summaries,
    initialize_reservoirs as initialize_reservoirs,
    phc as phc,
    register_return_values as register_return_values,
    set_y_limits as set_y_limits,
)
from .seawater import SeawaterConstants as SeawaterConstants
from .processes import (
    gas_exchange as gas_exchange,
    init_weathering as init_weathering,
    weathering_isotopes as weathering_isotopes,
    weathering_isotopes_delta as weathering_isotopes_delta,
    weathering_isotopes_alpha as weathering_isotopes_alpha,
    weathering_no_isotopes as weathering_no_isotopes,
    weathering_ref_isotopes as weathering_ref_isotopes,
)
from .post_processing import (
    carbonate_system_1_pp as carbonate_system_1_pp,
    carbonate_system_2_pp as carbonate_system_2_pp,
    carbonate_system_3_pp as carbonate_system_3_pp,
    gas_exchange_fluxes as gas_exchange_fluxes,
)
from .model import Model as Model
from .extended_classes import (
    DataField as DataField,
    ExternalCode as ExternalCode,
    ExternalData as ExternalData,
    GasReservoir as GasReservoir,
    Reservoir as Reservoir,
    Signal as Signal,
    SinkProperties as SinkProperties,
    SourceProperties as SourceProperties,
    SpeciesNoSet as SpeciesNoSet,
    VectorData as VectorData,
    VirtualSpecies as VirtualSpecies,
)
from .esbmtk_base import esbmtkBase as esbmtkBase
from .connections import ConnectionProperties as ConnectionProperties, Species2Species as Species2Species
from .carbonate_chemistry import (
    add_carbonate_system_1 as add_carbonate_system_1,
    add_carbonate_system_2 as add_carbonate_system_2,
    add_carbonate_system_3 as add_carbonate_system_3,
    carbonate_system_1 as carbonate_system_1,
    carbonate_system_2 as carbonate_system_2,
    carbonate_system_3 as carbonate_system_3,
    get_hplus as get_hplus,
    get_pco2 as get_pco2,
)
from .carbonate_system_4 import (
    add_carbonate_system_4 as add_carbonate_system_4,
    carbonate_system_4 as carbonate_system_4,
)
from .base_classes import (
    Flux as Flux,
    Sink as Sink,
    Source as Source,
    Species as Species,
    SpeciesError as SpeciesError,
    SpeciesProperties as SpeciesProperties,
)
import numpy as np
np.seterr(invalid="ignore")
