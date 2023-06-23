from __future__ import annotations
import typing as tp
import numpy as np

if tp.TYPE_CHECKING:
    from .esbmtk import ReservoirGroup


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

    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron
    hplus = rg.Hplus.c
    dic = rg.DIC.c
    ta = rg.TA.c

    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    rg.cs.CA: float = ta + fg
    rg.cs.HCO3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    rg.cs.CO3: float = (rg.cs.CA.ca - rg.cs.CA.hco3) / 2


def carbonate_system_2_pp(
    rg: ReservoirGroup,  # 2 Reservoir handle
    Bm: float,  # 3 CaCO3 export flux as DIC
) -> None:
    """Calculates and returns the fraction of the carbonate rain that is
    dissolved an returned back into the ocean. This functions returns:

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

    # Parameters
    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron
    hplus = rg.Hplus
    ta = rg.TA.c
    dic = rg.DIC.c

    p = rg.cs.function_params

    ksp0 = p[5]
    ca2 = rg.swc.ca2  # Ca2+
    kc = p[6]
    AD = p[8]
    zsat0 = int(abs(p[9]))
    zsat_min = int(abs(p[14]))
    zmax = int(abs(p[15]))

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    rg.cs.CA: float = ta + fg
    rg.cs.HCO3: float = dic / (1 + (hplus / k1) + (k2 / hplus))

    rg.cs.CO3: float = (rg.cs.CA - rg.cs.HCO3) / 2
    if rg.cs.CO3 <= 0:  # abvoid zero values
        co3 = 1e-16

    rg.cs.co2aq = dic - rg.cs.CO3 - rg.cs.HCO3
    rg.cs.zsat = int(max((zsat0 * np.log(ca2 * rg.cs.CO3 / ksp0)), zsat_min))  # eq2
    rg.cs.zsat = min(rg.cs.zsat, zmax)  # limit values to depth floor
    rg.cs.zsat = max(rg.cs.zsat, zsat_min)  # limit values to top of box
    rg.cs.zcc = int(
        zsat0 * np.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0)
    )  # eq3
