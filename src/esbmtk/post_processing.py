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

import typing as tp

import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from .esbmtk import GasReservoir, Reservoir, Species, SpeciesGroup

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]


def carbonate_system_1_pp(box_names: SpeciesGroup) -> None:
    """Calculate carbonate species.

    Based on previously calculated Hplus, TA, and DIC concentrations.

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    :param rg: A reservoirgroup object with initialized carbonate system
    """
    from esbmtk import VectorData

    if not isinstance(box_names, list):
        box_names = [box_names]

    for rg in box_names:
        k1 = rg.swc.K1  # K1
        k2 = rg.swc.K2  # K2
        k1k2 = rg.swc.K1K2
        hplus = rg.Hplus.c
        dic = rg.DIC.c

        VectorData(
            name="HCO3",
            register=rg,
            species=rg.mo.HCO3,
            data=dic / (1 + hplus / k1 + k2 / hplus),
            label="HCO3-",
            plt_units=rg.mo.c_unit,
        )

        VectorData(
            name="CO3",
            register=rg,
            species=rg.mo.CO3,
            data=dic / (1 + hplus / k2 + hplus**2 / k1k2),
            label="CO32-",
            plt_units=rg.mo.c_unit,
        )

        VectorData(
            name="pH",
            register=rg,
            species=rg.mo.pH,
            data=-np.log10(hplus),
            label="pH",
            plt_units="total scale",
        )

        VectorData(
            name="Omega",
            register=rg,
            species=rg.mo.pH,
            data=rg.swc.ca2 * rg.CO3.c / rg.swc.Ksp_ca,
            label=r"$\Omega$-Calcite",
            plt_units="",
        )


def carbonate_system_2_pp(
    bn: Reservoir | list,  # 2 Reservoir handle
    export_fluxes: float | list,  # 3 CaCO3 export flux as DIC
    zsat_min: float = 200,
    zmax: float = 10000,
) -> None:
    """Calculate the fraction of carbonate that is dissolved.

    :param rg: Reservoir, e.g., M.D_b
    :param export: export flux in mol/year
    :param zsat_min: depth of mixed layer
    :param zmax: depth of lookup table

    This function then saves the data as

    M.box_name.Hplus
    M.box_name.CO3
    M.box_name.CO2aq
    M.box_name.pH
    M.box_name.zsat # top of lysocline
    M.box_name.zcc # bottom of lysocline
    M.box_name.zsnow # snow line
    M.box_name.Fburial # The CaCO3 burial flux (Export - dissolution)
    M.box_name.Fdiss  # The CaCO3 dissolution flux

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    """
    from math import log

    from esbmtk import VectorData

    # ensure that all objects are lists
    if not isinstance(bn, list):
        bn = [bn]
    if not isinstance(export_fluxes, list):
        export_fluxes = [export_fluxes]

    # loop over boxes
    for i, rg in enumerate(bn):
        p = rg.cs2.function_params
        sp, cp, area_table, area_dz_table, Csat_table = p
        ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
        k1, k2, k1k2, KW, KB, ca2, boron, isotopes = sp
        hplus: NDArrayFloat = rg.Hplus.c
        dic: NDArrayFloat = rg.DIC.c
        zsnow: NDArrayInt = rg.zsnow.c.astype(int)
        export_data = export_fluxes[i]
        # make sure we have a vector
        if isinstance(export_data, float | int):
            export: NDArrayFloat = dic * 0 + export_data
        else:
            export = export_data

        # hco3 = dic / (1 + hplus / k1 + k2 / hplus)
        co3: NDArrayFloat = dic / (1 + hplus / k2 + hplus**2 / k1k2)
        co2aq: NDArrayFloat = dic / (1 + k1 / hplus + k1k2 / hplus**2)
        zsat: NDArrayInt = np.clip(
            zsat0 * np.log(ca2 * co3 / ksp0),
            zsat_min,
            zmax,
        ).astype(int)

        B_AD: NDArrayFloat = export / AD
        Fdiss: NDArrayFloat = co3 * 0
        Fburial: NDArrayFloat = co3 * 0
        zcc: NDArrayInt = co3.astype(int) * 0

        for i, z in enumerate(zsat):
            zcc[i] = int(
                zsat0 * log(export[i] * ca2 / (ksp0 * AD * kc) + ca2 * co3[i] / ksp0)
            )  # eq3
            if zcc[i] > zmax:
                zcc[i] = zmax
                print(
                    f"Warning zcc > zmax, i = {i}, co3 = {co3[i] * 1e6} umol/kg, export = {export[i] / 1e12:.2f} Tmol/y"
                )
            elif zcc[i] < z0:
                zcc[i] = z0

            A_z0_zsat: float = area_table[z0] - area_table[z]
            A_zsat_zcc: float = area_table[z] - area_table[zcc[i]]
            A_zcc_zmax: float = area_table[zcc[i]] - area_table[zmax]
            BCC: float = A_zcc_zmax * B_AD[i]
            BNS: float = alpha * A_z0_zsat * B_AD[i]
            diff_co3: float = Csat_table[z : zcc[i]] - co3[i]
            area_p: NDArrayFloat = area_dz_table[z : zcc[i]]
            try:
                BDS_under: float = kc * area_p.dot(diff_co3)
            except:
                breakpoint()

            BDS_resp: float = alpha * (A_zsat_zcc * B_AD[i] - BDS_under)
            BDS: float = BDS_under + BDS_resp

            """ Note that we do not recalculate zsnow in post processing
            since it is already known from the original run
            """
            if zsnow[i] <= zcc[i]:  # reset zsnow
                # dzdt_zsnow: int = abs(zsnow - zcc)
                # zsnow[i] = zcc[i]
                BPDC: float = 0.0
            else:  # integrate saturation difference over area
                # if zsnow[i] > zmax:
                # zsnow[i] = zmax
                # integrate saturation difference over area
                diff: float = Csat_table[zcc[i] : zsnow[i]] - co3[i]
                area_p_snow: NDArrayFloat = area_dz_table[zcc[i] : zsnow[i]]
                BPDC = max(0, kc * area_p_snow.dot(diff))
                # dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

            Fdiss[i] = BDS + BCC + BNS + BPDC
            Fburial[i] = export[i] - Fdiss[i]

        VectorData(
            name="Fburial",
            register=rg,
            species=rg.mo.Fburial,
            data=Fburial,
            label="Fburial",
            plt_units=rg.mo.f_unit,
        )
        VectorData(
            name="Fdiss",
            register=rg,
            species=rg.mo.Fdiss,
            data=Fdiss,
            label="Fdiss",
            plt_units=rg.mo.f_unit,
        )
        VectorData(
            name="CO3",
            register=rg,
            species=rg.mo.CO3,
            data=co3,
            label="CO32-",
            plt_units=rg.mo.c_unit,
        )

        VectorData(
            name="CO2aq",
            register=rg,
            species=rg.mo.CO2aq,
            data=co2aq,
            label="CO2aq",
            plt_units=rg.mo.c_unit,
        )
        VectorData(
            name="pH",
            register=rg,
            species=rg.mo.pH,
            data=-np.log10(hplus),
            label="pH",
            plt_units="total scale",
        )
        VectorData(
            name="zsat",
            register=rg,
            species=rg.mo.zsat,
            data=zsat,
            label="zsat",
            plt_units="m",
        )
        VectorData(
            name="zcc",
            register=rg,
            species=rg.mo.zcc,
            data=zcc,
            label="zcc",
            plt_units="m",
        )
        VectorData(
            name="CaCO3_export",
            register=rg,
            species=rg.mo.DIC,
            data=export,
            label="CaCO3_export",
            plt_units="mol/year",
        )


def gas_exchange_fluxes(
    liquid_reservoir: Species,
    gas_reservoir: GasReservoir,
    pv: str,
):
    """Calculate gas exchange fluxes for a given reservoir.

    :param liquid_reservoir: Species handle
    :param gas_reservoir:  Species handle
    :param pv: piston velocity as string e.g., "4.8 m/d"

    :returns:

    """
    from esbmtk import Q_, gas_exchange

    if isinstance(pv, str):
        pv = Q_(pv).to("meter/yr").magnitude
    elif isinstance(pv, Q_):
        pv = pv.to("meter/yr").magnitude
    else:
        raise ValueError("pv must be quantity or string")

    scale = liquid_reservoir.register.area * pv
    gas_c = gas_reservoir

    if liquid_reservoir.species.name == "DIC":
        solubility = liquid_reservoir.register.swc.SA_co2
        g_c_aq = liquid_reservoir.register.CO2aq
        a_db = liquid_reservoir.register.swc.co2_a_db
        a_dg = liquid_reservoir.register.swc.co2_a_dg
        a_u = liquid_reservoir.register.swc.co2_a_u
    elif liquid_reservoir.species.name == "O2":
        solubility = liquid_reservoir.register.swc.SA_o2
        g_c_aq = liquid_reservoir.register.O2
        a_db = liquid_reservoir.register.swc.o2_a_db
        a_dg = liquid_reservoir.register.swc.o2_a_dg
        a_u = liquid_reservoir.register.swc.o2_a_u
    else:
        raise ValueError("flux calculation is only supported for DIC and O2")

    p = (
        scale,
        solubility,
        a_db,
        a_dg,
        a_u,
        gas_reservoir.isotopes,
        liquid_reservoir.isotopes,
    )

    return gas_exchange(gas_c.c, liquid_reservoir.c, g_c_aq.c, p)
