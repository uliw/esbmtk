"""
esbmtk: A general purpose Earth Science box model toolkit Copyright
(C), 2020 Ulrich G.  Wortmann

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

import typing as tp
import numpy as np

from esbmtk import (
    photosynthesis,
    remineralization,
    gas_exchange_with_isotopes_2,
    gas_exchange_no_isotopes_2,
    carbonate_system_3,
    ExternalCode,
)

if tp.TYPE_CHECKING:
    from esbmtk import Flux, ReservoirGroup


def init_photosynthesis(rg, productivity):
    """Setup photosynthesis instances"""

    M = rg.mo
    ec = ExternalCode(
        name="ps",
        species=rg.mo.Oxygen.O2,
        fname="photosynthesis",
        function=photosynthesis,
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        vr_datafields={
            "POM": 0.0,
            "POM_l": 0.0,
            "CO2aq": rg.swc.co2,
        },
        function_input_data=[
            rg.O2,
            rg.TA,
            rg.DIC,
            rg.PO4,
            rg.SO4,
            rg.H2S,
            "Hplus",
            "CO2aq",
            productivity,
        ],
        function_params=(
            rg.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.PUE,
            M.rain_rate,
            M.OM_frac / 1000.0 + 1.0,
            M.alpha,
            rg.swc.K1,
            rg.swc.K1K1,
            rg.swc.K1K2,
            rg.swc.KW,
            rg.swc.KB,
            rg.swc.boron,
            M.Carbon.r
        ),
        register=rg,
        return_values=[
            {"Hplus": rg.swc.hplus},
            {"CO2aq": rg.swc.co2aq},
            {"F_rg.O2": "photosynthesis"},
            {"F_rg.TA": "photosynthesis"},
            {"F_rg.PO4": "photosynthesis"},
            {"F_rg.SO4": "photosynthesis"},
            {"F_rg.H2S": "photosynthesis"},
            {"F_rg.DIC": "photosynthesis"},
            {"F_rg.POM": "photosynthesis"},
            {"F_rg.PIC": "photosynthesis"},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def init_remineralization(
    rg: ReservoirGroup,
    pom_fluxes: list[Flux],
    pom_fluxes_l: list[Flux],
    pic_fluxes: list[Flux],
    pic_fluxes_l: list[Flux],
    pom_remin_fractions: float | list[float],
    pic_remin_fractions: float | list[float],
    CaCO3_reactions: bool,
):
    """ """

    if not isinstance(pom_remin_fractions, list):
        pom_remin_fractions = list(pom_remin_fractions)
    if not isinstance(pic_remin_fractions, list):
        pic_remin_fractions = list(pic_remin_fractions)

    M = rg.mo
    ec = ExternalCode(
        name="rm",
        species=rg.mo.Carbon.CO2,
        function=remineralization,
        fname="remineralization",
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        # hplus is not used but needed in post processing
        vr_datafields={"Hplus": rg.swc.hplus},
        function_input_data=[
            pom_fluxes,
            pom_fluxes_l,
            pic_fluxes,
            pic_fluxes_l,
            pom_remin_fractions,
            pic_remin_fractions,
            rg.H2S,
            rg.SO4,
            rg.O2,
            rg.PO4,
            rg.volume.magnitude,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.alpha,
            CaCO3_reactions,
        ],
        register=rg,
        return_values=[
            {"F_rg.DIC": "remineralization"},
            {"F_rg.TA": "remineralization"},
            {"F_rg.H2S": "remineralization"},
            {"F_rg.SO4": "remineralization"},
            {"F_rg.O2": "remineralization"},
            {"F_rg.PO4": "remineralization"},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)
    return ec


def init_carbonate_system_3(
    rg: ReservoirGroup,
    pic_export_flux: Flux,
    r_sb: ReservoirGroup,
    r_db: ReservoirGroup,
    area_table: np.ndarray,
    area_dz_table: np.ndarray,
    Csat_table: np.ndarray,
    AD: float,
    kwargs: dict,
):
    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_3,
        fname="carbonate_system_3",
        ftype="cs2",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        vr_datafields={
            "depth_area_table": area_table,
            "area_dz_table": area_dz_table,
            "Csat_table": Csat_table,
        },
        function_input_data=[
            rg,  # 0
            pic_export_flux,  # 1
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
            {"F_rg.DIC": "db_remineralization"},
            {"F_rg.TA": "db_remineralization"},
            {"Hplus": rg.swc.hplus},
            {"zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=rg,
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def init_gas_exchange_no_isotopes(
    r_gas,
    r_liquid,
    r_reference,
    species,
    piston_velocity,
    solubility,
):
    """Setup GasExchange instances"""

    # convert pv into model units
    pv = piston_velocity.to("meter/yr").magnitude

    ec = ExternalCode(
        name=f"{r_liquid.parent.name}_{r_liquid.name}_gexo2",
        species=species,
        fname="gas_exchange_no_isotopes_2",
        function=gas_exchange_no_isotopes_2,
        ftype="cs1",  # cs1 is independent of fluxes, cs2 is not
        vr_datafields={},
        function_input_data=[
            r_gas,
            r_liquid,
            r_reference,
        ],
        function_params=(
            r_liquid.parent.area,
            solubility,
            pv,
            r_liquid.parent.swc.p_H2O,
        ),
        register=r_gas,
        return_values=[
            {f"F_rg.{species.name}": "gex"},
        ],
    )
    # print(f"return value = {ec.return_values}")
    r_liquid.mo.lpc_f.append(ec.fname)

    return ec


def init_gas_exchange_with_isotopes(
    r_gas,
    r_liquid,
    r_reference,
    species,
    piston_velocity,
    solubility,
):
    """Setup GasExchange instances"""

    # convert pv into model units
    pv = piston_velocity.to("meter/yr").magnitude

    ec = ExternalCode(
        name=f"{r_liquid.parent.name}_{r_liquid.name}_gexco2",
        species=species,
        fname="gas_exchange_with_isotopes_2",
        function=gas_exchange_with_isotopes_2,
        ftype="cs1",  # cs1 is independent of fluxes, cs2 is not
        vr_datafields={},
        function_input_data=[
            r_gas,
            r_liquid,
            r_reference,
        ],
        function_params=(
            r_liquid.parent.area,
            solubility,
            pv,
            r_liquid.parent.swc.p_H2O,
            r_liquid.parent.swc.a_db,
            r_liquid.parent.swc.a_dg,
            r_liquid.parent.swc.a_u,
        ),
        register=r_gas,
        return_values=[
            {f"F_rg.{species.name}": "gexwi"},
        ],
    )
    r_liquid.mo.lpc_f.append(ec.fname)

    return ec
