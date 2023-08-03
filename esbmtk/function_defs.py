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
import numpy as np

if tp.TYPE_CHECKING:
    from .extended_classes import ReservoirGroup, Flux


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
        fname="carbonate_system_1_ode",
        ftype="cs1",
        # this needs cleanup as it should be handled by the return_values
        # list.
        vr_datafields={
            "CO2aq": rg.swc.co2,  # 4
        },
        # function_input_data=[rg.DIC, rg.TA, rg.Hplus],
        function_input_data=[rg.swc, rg.DIC, rg.TA, "Hplus"],
        register=rg,
        # name and initial value pairs
        # return_values={"Hplus": rg.swc.hplus},
        return_values=[
            {"Hplus": rg.swc.hplus},
            {"CO2aq": rg.swc.co2aq},
        ],
        # return_values=["Hplus", "CO2aq"],
    )

    return ec


def init_carbonate_system_2(
    rg: ReservoirGroup,
    export_flux: Flux,
    dic_burial_flux: Flux,
    ta_burial_flux: Flux,
    r_sb: ReservoirGroup,
    r_db: ReservoirGroup,
    area_table: np.ndarray,
    area_dz_table: np.ndarray,
    Csat_table: np.ndarray,
    AD: float,
    kwargs: dict,
):

    from esbmtk import ExternalCode, carbonate_system_2_ode

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_2_ode,
        fname="carbonate_system_2_ode",
        ftype="cs2",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        # the vr_data_fields contains any data that is referenced inside the
        # function, rather than passed as argument, and all data that is
        # explicitly referenced by the model
        vr_datafields={
            "depth_area_table": area_table,
            "area_dz_table": area_dz_table,
            "Csat_table": Csat_table,
        },
        function_input_data=[
            rg,  # 0
            export_flux,  # 1
            r_db.DIC,  # 2
            r_db.TA,  # 3
            r_sb.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
            kwargs["Ksp0"],  # 7
            float(kwargs["kc"]),  # 8
            float(AD),  # 9
            float(abs(kwargs["zsat0"])),  # 10
            float(kwargs["I_caco3"]),  # 11
            float(kwargs["alpha"]),  # 12
            float(abs(kwargs["zsat_min"])),  # 13
            float(abs(kwargs["zmax"])),  # 14
            float(abs(kwargs["z0"])),  # 15
        ],
        return_values=[
            dic_burial_flux,
            ta_burial_flux,
            {"Hplus": rg.swc.hplus},
            {"zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=rg,
    )

    return ec
