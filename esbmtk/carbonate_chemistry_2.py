"""esbmtk: A general purpose Earth Science box model toolkit Copyright
     (C), 2020-2021 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or
     modify it under the terms of the GNU General Public License as
     published by the Free Software Foundation, either version 3 of
     the License, or (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
     General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see
     <https://www.gnu.org/licenses/>.

"""

import typing as tp
import numpy as np
from esbmtk import ReservoirGroup, Flux

if tp.TYPE_CHECKING:
    from .esbmtk import ReservoirGroup, Flux


def carbonate_system_1_ode(
    rg: ReservoirGroup, dic: float, ta: float, hplus: float
) -> tuple:
    """Calculates and returns the carbonate concentrations and saturation state
     for the given reservoirgroup

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    Author: M. Niazi & T. Tsan, 2021, with modifications by U. Wortmann 2022
    """

    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron

    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # hplus
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    # hco3 and co3
    """ Since CA = [hco3] + 2[co3], can the below expression can be simplified
    """
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # co2 (aq)
    """DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    Let's test this once we have a case where pco2 is calculated from co2aq
    """
    #  co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    co2aq: float = dic - hco3 - co3

    # rg.cs.H = hplus
    # rg.cs.CA = ca
    # rg.cs.HCO3 = hco3
    # rg.cs.CO3 = co3
    # rg.cs.CO2a = co2aq
    # rg.cs.OH = oh
    # rg.cs.BOH4 = boh4

    return hplus, ca, hco3, co3, co2aq, oh, boh4


def carbonate_system_2_ode(
    t, rg: ReservoirGroup, Bm: Flux, dic: float, ta: float
) -> float:
    """Calculates and returns the carbonate concentrations and carbonate compensation
    depth (zcc) at the ith time-step of the model.


    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004


    Author: M. Niazi & T. Tsan, 2021
    """

    # Parameters
    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    ksp0 = rg.swc.Ksp0
    ca2 = rg.swc.ca2  # Ca2+
    boron = rg.swc.boron  # boron

    # concentration
    hplus = rg.cs.H  # hplus from last timestep

    # still missing parameters
    last_t = rg.cs.last_t
    kc = rg.cs.kc
    AD = rg.cs.AD
    zsat0 = rg.cs.zsat0
    I_caco3 = rg.cs.I_caco3
    alpha = rg.cs.alpha
    zsat_min = int(abs(rg.cs.zsat_min))
    zmax = int(abs(rg.cs.zmax))
    z0 = int(abs(rg.cs.z0))
    zsnow = rg.cs.zsnow  # previous zsnow
    depth_area_table = rg.cs.depth_area_table
    area_dz_table = rg.cs.area_dz_table
    Csat_table = rg.cs.Csat_table

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    # small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    co2aq: float = dic - co3 - hco3
    # co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    # omega: float = (ca2 * co3) / ksp

    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables

    # prevent co3 from becoming zero
    if co3 <= 0:
        co3 = 1e-16

    zsat = int(max((zsat0 * np.log(ca2 * co3 / ksp0)), zsat_min))  # eq2
    if zsat < zsat_min:
        zsat = int(zsat_min)

    zcc = int(zsat0 * np.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3

    # ---- Get fractional areas
    B_AD = Bm / AD

    if zcc > zmax:
        zcc = int(zmax)
    if zcc < zsat_min:
        zcc = zsat_min

    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]

    # ------------------------Calculate Burial Fluxes------------------------------------
    # BCC = (A(zcc, zmax) / AD) * B, eq 7
    BCC = A_zcc_zmax * B_AD

    # BNS = alpha_RD * ((A(z0, zsat) * B) / AD) eq 8
    BNS = alpha * A_z0_zsat * B_AD

    # BDS_under = kc int(zcc,zsat) area' Csat(z,t) - [CO3](t) dz, eq 9a
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]

    BDS_under = kc * area_p.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp

    # BPDC =  kc int(zsnow,zcc) area' Csat(z,t) - [CO3](t) dz, eq 10
    if zcc < zsnow:  # zcc cannot
        if zsnow > zmax:  # zsnow cannot exceed ocean depth
            zsnow = zmax

        diff = Csat_table[zcc : int(zsnow)] - co3
        area_p = area_dz_table[zcc : int(zsnow)]
        BPDC = kc * area_p.dot(diff)
        # eq 4 dzsnow/dt = Bpdc(t) / (a'(zsnow(t)) * ICaCO3
        zsnow = zsnow - BPDC / (area_dz_table[int(zsnow)] * I_caco3) * t - last_t

    else:  # zcc > zsnow
        # there is no carbonate below zsnow, so BPDC = 0
        zsnow = zcc
        BPDC = 0

    # BD & F_burial
    BD: float = BDS + BCC + BNS + BPDC
    Fburial = Bm - BD
    # Fburial12 = Fburial * input_data[1][i - 1] / input_data[0][i - 1]
    # diss12 = (B12 - Fburial12) * dt  # dissolution flux light isotope

    # copy results into datafields
    rg.cs.H = hplus  # 0
    rg.cs.CA = ca  # 1
    rg.cs.HCO3 = hco3  # 2
    rg.cs.CO3 = co3  # 3
    rg.cs.CO2aq = co2aq  # 4
    rg.cs.zsat = zsat  # 5
    rg.cs.zcc = zcc  # 6
    rg.cs.zsnow = zsnow  # 7
    rg.cs.last_t = t

    return Fburial
