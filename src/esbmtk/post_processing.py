from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from .esbmtk import Reservoir, Species, GasReservoir

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


def carbonate_system_1_pp(box_names: SpeciesGroup) -> None:
    """Calculates and returns various carbonate species based on previously calculated
    Hplus, TA, and DIC concentrations.

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
    """Calculates and returns the fraction of the carbonate rain that is
    dissolved an returned back into the ocean.

    :param rg: Reservoir, e.g., M.D_b
    :param export: export flux in mol/year
    :param zsat_min: depth of mixed layer
    :param zmax: depth of lookup table

    returns:

    DIC_burial, DIC_burial_l, Hplus, zsnow

    Additionally, it calculates  the following critical depth intervals:

    zsat: top of lysocline
    zcc: carbonate compensation depth

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    """

    from esbmtk import VectorData

    # ensure that all objects are lists
    if not isinstance(bn, list):
        bn = [bn]
    if not isinstance(export_fluxes, list):
        export_fluxes = [export_fluxes]

    # loop over boxes
    for i, rg in enumerate(bn):

        p = rg.cs.function_params
        sp, cp, area_table, area_dz_table, Csat_table = p
        ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
        k1, k2, k1k2, KW, KB, ca2, boron, isotopes = sp
        hplus = rg.Hplus.c
        dic = rg.DIC.c
        zsnow = rg.zsnow.c.astype(int)
        area_table = rg.model.area_table
        area_dz_table = rg.model.area_dz_table
        Csat_table = rg.model.Csat_table
        export = export_fluxes[i]
        
        # test the type of export flux information
        if isinstance(export, float):
            # ensure we have a vector
            export = dic * 0 + export
        elif not isinstance(export, np.ndarray):
            export = dic * 0 + export.c

        # hco3 = dic / (1 + hplus / k1 + k2 / hplus)
        co3 = dic / (1 + hplus / k2 + hplus**2 / k1k2)
        co2aq = dic / (1 + k1 / hplus + k1k2 / hplus**2)
        zsat = np.clip(zsat0 * np.log(ca2 * co3 / ksp0), zsat_min, zmax).astype(int)
        zcc = (
            zsat0 * np.log(export * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0)
        ).astype(int)

        B_AD = export / AD

        A_z0_zsat = area_table[z0] - area_table[zsat]
        A_zsat_zcc = area_table[zsat] - area_table[zcc]
        # FIXME: Is  A_zsat_zcc a vector?
        A_zcc_zmax = area_table[zcc] - area_table[zmax]

        Fdiss = zsat * 0
        Fburial = zsat * 0

        for i, e in enumerate(zsat):
            BCC = A_zcc_zmax[i] * B_AD[i]
            BNS = alpha * A_z0_zsat[i] * B_AD[i]
            diff_co3 = Csat_table[zsat[i] : zcc[i]] - co3[i]
            area_p = area_dz_table[zsat[i] : zcc[i]]
            BDS_under = kc * area_p.dot(diff_co3)
            BDS_resp = alpha * (A_zsat_zcc[i] * B_AD[i] - BDS_under)
            BDS = BDS_under + BDS_resp

            if zsnow[i] <= zcc[i]:  # reset zsnow
                zsnow[i] = zcc[i]
                BPDC = 0
            else:  # integrate saturation difference over area
                if zsnow[i] > zmax:
                    zsnow[i] = zmax
                # integrate saturation difference over area
                diff = Csat_table[zcc[i] : zsnow[i]] - co3[i]
                area_p = area_dz_table[zcc[i] : zsnow[i]]
                BPDC = max(0, kc * area_p.dot(diff))

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


def gas_exchange_fluxes(
    liquid_reservoir: Species,
    gas_reservoir: GasReservoir,
    pv: str,
):
    """Calculate gas exchange fluxes for a given reservoir

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
    gas_c = gas_reservoir.c
    p_H2O = liquid_reservoir.register.swc.p_H2O

    if liquid_reservoir.species.name == "DIC":
        solubility = liquid_reservoir.register.swc.SA_co2
        g_c_aq = liquid_reservoir.register.CO2aq.c
    elif liquid_reservoir.species.name == "O2":
        solubility = liquid_reservoir.register.swc.SA_o2
        g_c_aq = liquid_reservoir.register.O2.c
    else:
        raise ValueError("flux calculation is only supported for DIC and O2")

    c = liquid_reservoir.register
    swc = liquid_reservoir.register.swc
    p = (
        scale,
        p_H2O,
        solubility,
        swc.a_db,
        swc.a_dg,
        swc.a_u,
        liquid_reservoir.isotopes,
    )

    return gas_exchange(gas_c, liquid_reservoir.c, g_c_aq, p)
