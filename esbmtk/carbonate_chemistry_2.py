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


def get_hplus(
    rg: ReservoirGroup,
    dic: float,
    ta: float,
    hplus: float,
) -> tuple:

    """Calculates and returns the H+ and carbonate alkalinity concentrations
     for the given reservoirgroup

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    """

    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron
    hplus_0 = hplus

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

    return hplus - hplus_0, ca


def carbonate_system_1_ode(
    rg: ReservoirGroup,
    dic: float,
    ta: float,
    hplus: float,  # H1 from get_hplus
    ca: float,  # carbonate alkalinity from get_hplus
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
   
    return hco3, co3, co2aq


def carbonate_system_2_ode(
    t,
    rg: ReservoirGroup,
    Bm: Flux,
    dic: float,
    ta: float,
    i: float,
    max_i: float,
    last_t: float,
) -> float:
    """Calculates and returns the carbonate burial flux and carbonate compensation
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

    # concentration
    hplus = rg.cs.H[i - 1]  # hplus from last timestep

    # Parameters
    k1 = rg.swc.K1  # K1
    k2 = rg.swc.K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    boron = rg.swc.boron  # boron

    p = rg.cs.function_params

    ksp0 = p[5]
    ca2 = rg.swc.ca2  # Ca2+
    kc = p[6]
    AD = p[8]
    zsat0 = int(abs(p[9]))
    I_caco3 = p[12]
    alpha = p[13]
    zsat_min = int(abs(p[14]))
    zmax = int(abs(p[15]))
    z0 = int(abs(p[16]))
    zsnow = int(abs(p[19]))  # previous zsnow
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

    # print(
    #     f"i = {i}, zsat0 = {zsat0:.1f}, ca= {ca:.2e}, co3 = {co3:.2e}, ksp0 = {ksp0:.2e}, zsat_min = {zsat_min:.1f}"
    # )
    # print(f"zsat = {zsat:.1f}\n")
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
    diss = Bm - Fburial
    # Fburial12 = Fburial * input_data[1][i - 1] / input_data[0][i - 1]
    # diss12 = (B12 - Fburial12) * dt  # dissolution flux light isotope

    if i > max_i:
        rg.cs.H.extend([hplus])
        rg.cs.CA.extend([ca])
        rg.cs.HCO3.extend([hco3])
        rg.cs.CO3.extend([co3])
        rg.cs.CO2aq.extend([co2aq])
        rg.cs.zsat.extend([zsat])
        rg.cs.zcc.extend([zcc])
        rg.cs.zsnow.extend([zsnow])
        rg.cs.Fburial.extend([Fburial])
    else:
        rg.cs.H[i] = hplus  #
        rg.cs.CA[i] = ca  # 1
        rg.cs.HCO3[i] = hco3  # 2
        rg.cs.CO3[i] = co3  # 3
        rg.cs.CO2aq[i] = co2aq  # 4
        rg.cs.zsat[i] = zsat  # 5
        rg.cs.zcc[i] = zcc  # 6
        rg.cs.zsnow[i] = zsnow  # 7
        rg.cs.Fburial[i] = Fburial

    return -diss


def gas_exchange_ode(scale, gas_c, p_H2O, solubility, co2aq):

    # set flux
    # note that the sink delta is co2aq as returned by the carbonate VR
    # this equation is for mmol but esbmtk uses mol, so we need to
    # multiply by 1E3

    f = scale * (  # area in m^2
        gas_c  # Atmosphere
        * (1 - p_H2O)  # p_H2O
        * solubility  # SA_co2 = mol/(m^3 atm)
        - co2aq * 1000  # [CO2]aq mol
    )

    return -f
