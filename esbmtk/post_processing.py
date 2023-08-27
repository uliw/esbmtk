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
    rg.CA: float = ta + fg
    rg.HCO3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    rg.CO3: float = (rg.CA - rg.HCO3) / 2
    rg.omega = rg.swc.ca2 * rg.CO3 / rg.swc.Ksp
    rg.pH: float = -np.log10(hplus)
    rg.CO2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus**2)))


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
    hplus = rg.Hplus.c
    ta = rg.TA.c
    dic = rg.DIC.c
    p = rg.cs.function_input_data
    ksp0 = p[7]
    ca2 = rg.swc.ca2  # Ca2+
    kc = p[8]
    AD = p[9]
    zsat0 = int(abs(p[10]))
    zsat_min = int(abs(p[13]))
    zmax = int(abs(p[14]))

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    rg.CA: float = ta + fg
    rg.HCO3: float = dic / (1 + (hplus / k1) + (k2 / hplus))

    rg.CO3: float = (rg.CA - rg.HCO3) / 2
    rg.CO3[rg.CO3 < 0] = 0
    rg.CO2aq = dic - rg.CO3 - rg.HCO3
    rg.omega = rg.swc.ca2 * rg.CO3 / rg.swc.Ksp
    rg.pH: float = -np.log10(hplus)
    rg.zsat = np.clip(zsat0 * np.log(ca2 * rg.CO3 / ksp0), zsat_min, zmax)
    rg.zcc = zsat0 * np.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * rg.CO3 / ksp0)  # eq3
    # rg.zsnow = rg.zsnow.c
