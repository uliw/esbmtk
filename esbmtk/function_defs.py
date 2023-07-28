"""
functions_defs() is part of esbmtk, A general purpose Earth Science box model
toolkit Copyright (C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.


"""
from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from .extended_classes import ReservoirGroup


def init_carbonate_system_1(rg: ReservoirGroup):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: list = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Reservoir
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import ExternalCode, carbonate_system_1_ode

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_1_ode,
        ftype="cs1",
        # the vr_data_fields contains any data that is referenced inside the
        # function, rather than passed as argument, and all data that is
        # explicitly referenced by the model
        vr_datafields={
            "CO2aq": rg.swc.co2,  # 4
        },
        function_input_data=list(),
        function_params=list(),
        register=rg,
        return_values={"Hplus": rg.swc.hplus},
    )

    return ec



