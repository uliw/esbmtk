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
from esbmtk import ExternalCode
from esbmtk.bio_pump_functions1.bio_geochemical_reactions import (
    photosynthesis,
    OM_remineralization,
    gas_exchange_with_isotopes_2,
    gas_exchange_no_isotopes_2,
    carbonate_system_3,
)

if tp.TYPE_CHECKING:
    from esbmtk import Flux, ReservoirGroup


def init_photosynthesis(
    rg,
    productivity,
    piston_velocity,
    O2_At,
    CO2_At,
    CaCO3_reactions,
):
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
            O2_At,
            CO2_At,
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
            rg.area,
            rg.sed_area,
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
            M.Carbon.r,
            rg.swc.p_H2O,
            piston_velocity,
            rg.swc.SA_o2,
            rg.swc.SA_co2,
            rg.swc.a_db,
            rg.swc.a_dg,
            rg.swc.a_u,
            CaCO3_reactions,
        ),
        register=rg,
        return_values=[
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
            {f"R_{rg.full_name}.CO2aq": rg.swc.co2aq},
            {f"F_{rg.full_name}.O2": "photosynthesis"},
            {f"F_{rg.full_name}.TA": "photosynthesis"},
            {f"F_{rg.full_name}.PO4": "photosynthesis"},
            {f"F_{rg.full_name}.SO4": "photosynthesis"},
            {f"F_{rg.full_name}.H2S": "photosynthesis"},
            {f"F_{rg.full_name}.DIC": "photosynthesis"},
            {f"F_M.O2_At": "photosynthesis"},
            {f"F_M.CO2_At": "photosynthesis"},
            {f"F_{rg.full_name}.POM": "photosynthesis"},
            {f"F_{rg.full_name}.PIC": "photosynthesis"},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def init_OM_remineralization(
    rg: ReservoirGroup,
    pom_fluxes: list[Flux],
    pom_fluxes_l: list[Flux],
    pic_fluxes: list[Flux],
    pic_fluxes_l: list[Flux],
    pom_remin_fractions: list[float],
    pic_remin_fractions: list[float],
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
        function=OM_remineralization,
        fname="OM_remineralization",
        ftype="cs2",  # cs1 is independent of fluxes, cs2 is not
        # hplus is not used but needed in post processing
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
            rg.DIC,
            rg.TA,
            "Hplus",
        ],
        function_params=(
            rg.volume.magnitude,
            rg.area,
            rg.sed_area,
            M.PC_ratio,
            M.NC_ratio,
            M.O2C_ratio,
            M.P_burial,
            M.alpha,
            rg.swc.K1,
            rg.swc.K1K1,
            rg.swc.K1K2,
            rg.swc.KW,
            rg.swc.KB,
            rg.swc.boron,
            rg.DIC.species.r,
            M.OM_d,
            CaCO3_reactions,
        ),
        register=rg,
        return_values=[
            {f"F_{rg.full_name}.DIC": "OM_remineralization"},
            {f"F_{rg.full_name}.TA": "OM_remineralization"},
            {f"F_{rg.full_name}.H2S": "OM_remineralization"},
            {f"F_{rg.full_name}.SO4": "OM_remineralization"},
            {f"F_{rg.full_name}.O2": "OM_remineralization"},
            {f"F_{rg.full_name}.PO4": "OM_remineralization"},
            {"F_M.O2_At": "OM_weathering_O2"},
            {"F_M.CO2_At": "OM_weathering_CO2"},
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
        ],
    )
    rg.mo.lpc_f.append(ec.fname)
    return ec


def init_carbonate_system_3(
    rg: ReservoirGroup,
    pic_export_flux: Flux,
    r_sb: ReservoirGroup,
    r_ib: ReservoirGroup,
    r_db: ReservoirGroup,
    kwargs: dict,
):
    # s = r_db.swc
    # p = (s.k1, s.k2, s.k1k2, s.KW, s.KB, s.ca2, s.boron s.zsat0, s.zsat_min, s.zmax, s.z0)
    # tables = (s.depth_area_table, s.area_dz_table, s.Csat_table)

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_3,
        fname="carbonate_system_3",
        ftype="cs2",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        function_input_data=[
            pic_export_flux,  # 1
            r_db.DIC,  # 2
            r_db.TA,  # 3
            r_sb.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
            "CO3",
        ],
        function_params=(
            kwargs["Ksp0"],  # 7
            float(kwargs["kc"]),  # 8
            float(abs(kwargs["zsat0"])),  # 10
            float(kwargs["I_caco3"]),  # 11
            float(kwargs["alpha"]),  # 12
            float(abs(kwargs["zsat_min"])),  # 13
            float(abs(kwargs["zmax"])),  # 14
            float(abs(kwargs["z0"])),  # 15
            rg.swc.K2,
            rg.swc.K1K2,
            rg.swc.ca2,
            r_ib.area,  # area at top of intermediate water
            r_ib.mo.hyp.oa,  # total ocean area
            "area_table",
            "area_dz_table",
            "Csat_table",
        ),
        return_values=[
            {f"F_{rg.full_name}.DIC": "db_OM_remineralization"},
            {f"F_{rg.full_name}.TA": "db_OM_remineralization"},
            {f"R_{rg.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
            {f"R_{rg.full_name}.CO3": rg.swc.co3},
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

    breakpoint()
    ec = ExternalCode(
        name=f"{r_liquid.parent.name}_{r_liquid.name}_exchange",
        species=species,
        fname="gas_exchange_no_isotopes_2",
        function=gas_exchange_no_isotopes_2,
        ftype="cs1",  # cs1 is independent of fluxes, cs2 is not
        vr_datafields={},
        function_input_data=[
            r_gas,
            r_liquid,
        ],
        function_params=(
            r_liquid.parent.area,
            solubility,
            pv,
            r_liquid.parent.swc.p_H2O,
        ),
        register=r_gas,
        return_values=[
            {f"F_{r_gas.full_name}": "exchange"},
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
        name=f"{r_liquid.parent.name}_{r_liquid.name}_exchange",
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
            {f"F_{r_gas.full_name}": "exchange"},
        ],
    )
    r_liquid.mo.lpc_f.append(ec.fname)

    return ec
