from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from .esbmtk import ReservoirGroup

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
        name="HCO3",
        register=rg,
        species=rg.mo.HCO3,
        data=dic / (1 + (hplus / k1) + (k2 / hplus)),
    )
    VectorData(
        name="CO3",
        register=rg,
        species=rg.mo.CO3,
        data=dic / (1 + hplus / k2 + hplus * hplus / k1k2),
    )
    rg.CO3.c[rg.CO3.c < 0] = 0

    VectorData(
        name="pH",
        register=rg,
        species=rg.mo.pH,
        data=-np.log10(hplus),
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

    # Parameters
    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    k1k2 = rg.swc.K1K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron
    hplus = rg.Hplus.c
    ta = rg.TA.c
    dic = rg.DIC.c
    p = rg.cs.function_input_data
    ksp0 = p[7]
    ca2 = rg.swc.ca2  # Ca2+
    kc = p[8]
    AD = p[9]
    zsat0 = int(abs(p[10]))

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4

    VectorData(
        name="CA",
        register=rg,
        species=rg.mo.CA,
        data=ta + fg,
    )
    VectorData(
        name="HCO3",
        register=rg,
        species=rg.mo.HCO3,
        data=dic / (1 + (hplus / k1) + (k2 / hplus)),
    )
    VectorData(
        name="CO3",
        register=rg,
        species=rg.mo.CO3,
        data=dic / (1 + hplus / k2 + hplus * hplus / k1k2),
    )
    rg.CO3.c[rg.CO3.c < 0] = 0

    VectorData(
        name="CO2aq",
        register=rg,
        species=rg.mo.CO2aq,
        data=dic / (1 + (k1 / hplus) + (k1k2 / (hplus * hplus))),
    )
    VectorData(
        name="pH",
        register=rg,
        species=rg.mo.pH,
        data=-np.log10(hplus),
    )
    VectorData(
        name="zsat",
        register=rg,
        species=rg.mo.zsat,
        data=np.clip(zsat0 * np.log(ca2 * rg.CO3.c / ksp0), zsat_min, zmax),
    )
    VectorData(
        name="zcc",
        register=rg,
        species=rg.mo.zcc,
        data=zsat0 * np.log(export * ca2 / (ksp0 * AD * kc) + ca2 * rg.CO3.c / ksp0),
    )


def gas_exchange_fluxes(
    liquid_reservoir: Reservoir,
    gas_reservoir: GasReservoir,
    pv: str,
):
    """Calculate gas exchange fluxes for a given Reservoir

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
