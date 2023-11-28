from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from .esbmtk import ReservoirGroup, Reservoir, GasReservoir

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


def carbonate_system_1_pp(rg: ReservoirGroup) -> None:
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

    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    k1k2 = rg.swc.K1K2
    hplus = rg.Hplus.c
    dic = rg.DIC.c

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
    # rg.omega = rg.swc.ca2 * rg.CO3 / rg.swc.Ksp
    # rg.CO2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus**2)))


def carbonate_system_2_pp(
    rg: ReservoirGroup,  # 2 Reservoir handle
    export: float,  # 3 CaCO3 export flux as DIC
    zsat_min: float,
    zmax: float,
) -> None:
    """Calculates and returns the fraction of the carbonate rain that is
    dissolved an returned back into the ocean.

    :param rg: ReservoirGroup, e.g., M.D_b
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

    p = rg.cs.function_params
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
    k1, k2, k1k2, KW, KB, ca2, boron = sp
    hplus = rg.Hplus.c
    dic = rg.DIC.c
    zsnow = rg.zsnow.c.astype(int)
    area_table = rg.model.area_table
    area_dz_table = rg.model.area_dz_table
    Csat_table = rg.model.Csat_table

    # hco3 = dic / (1 + hplus / k1 + k2 / hplus)
    co3 = dic / (1 + hplus / k2 + hplus**2 / k1k2)
    co2aq = dic / (1 + k1 / hplus + k1k2 / hplus**2)
    zsat = np.clip(zsat0 * np.log(ca2 * co3 / ksp0), zsat_min, zmax).astype(int)
    zcc = (zsat0 * np.log(export * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0)).astype(
        int
    )
    B_AD = export / AD

    A_z0_zsat = area_table[z0] - area_table[zsat]
    A_zsat_zcc = area_table[zsat] - area_table[zcc]
    A_zcc_zmax = area_table[zcc] - area_table[zmax]

    # must be loop
    Fdiss = zsat * 0
    Fburial = zsat * 0

    for i, e in enumerate(zsat):
        BCC = A_zcc_zmax[i] * B_AD
        BNS = alpha * A_z0_zsat[i] * B_AD
        diff_co3 = Csat_table[zsat[i] : zcc[i]] - co3[i]
        area_p = area_dz_table[zsat[i] : zcc[i]]
        BDS_under = kc * area_p.dot(diff_co3)
        BDS_resp = alpha * (A_zsat_zcc[i] * B_AD - BDS_under)
        BDS = BDS_under + BDS_resp
        if zsnow[i] > zmax:
            zsnow[i] = zmax
        # integrate satu ration difference over area
        diff = Csat_table[zcc[i] : zsnow[i]] - co3[i]
        area_p = area_dz_table[zcc[i] : zsnow[i]]
        BPDC = kc * area_p.dot(diff)
        BPDC = max(BPDC, 0)  # prevent negative values
        Fdiss[i] = BDS + BCC + BNS + BPDC
        Fburial[i] = export - Fdiss[i]

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
    liquid_reservoir: Reservoir,
    gas_reservoir: GasReservoir,
    pv: str,
):
    """Calculate gas exchange fluxes for a given reservoir

    :param liquid_reservoir: Reservoir handle
    :param gas_reservoir:  Reservoir handle
    :param pv: piston velocity as string e.g., "4.8 m/d"

    :returns:

    """
    from esbmtk import Q_, gas_exchange_ode

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

    return gas_exchange_ode(scale, gas_c, p_H2O, solubility, g_c_aq)
