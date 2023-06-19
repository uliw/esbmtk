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
from numba import njit
from esbmtk import ReservoirGroup, Flux

if tp.TYPE_CHECKING:
    from .esbmtk import ReservoirGroup, Flux


def get_hplus(
    rg: ReservoirGroup,
    dic: float,
    ta: float,
    hplus: float,
) -> float:

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

    return hplus - hplus_0


#@njit(parallel=False, fastmath=True, error_model="numpy")
def carbonate_system_1_ode(
    rg: ReservoirGroup,
    dic: float,
    ta: float,
    hplus: float,
    i: float,
    max_i: float,
) -> float:

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

    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # co2 (aq)
    """DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    Let's test this once we have a case where pco2 is calculated from co2aq
    """
    #  co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    co2aq: float = dic - hco3 - co3

    # diff = hplus - rg.cs.H.extend
    # save in state for co2aq, hco3, co3
    if i >= max_i:
        rg.cs.H = np.append(rg.cs.H, hplus)
        rg.cs.CA = np.append(rg.cs.CA, ca)
        rg.cs.HCO3 = np.append(rg.cs.HCO3, hco3)
        rg.cs.CO3 = np.append(rg.cs.CO3, co3)
        rg.cs.CO2aq = np.append(rg.cs.CO2aq, co2aq)
    else:
        rg.cs.H[i] = hplus  # 1
        rg.cs.CA[i] = ca  # 1
        rg.cs.HCO3[i] = hco3  # 2
        rg.cs.CO3[i] = co3  # 3
        rg.cs.CO2aq[i] = co2aq  # 4

    diff = hplus - hplus_0

    return diff, co2aq

# @njit(parallel=False, fastmath=True, error_model="numpy")
def carbonate_system_2_ode(
    t,  # 1: time step
    rg: ReservoirGroup,  # 2 Reservoir handle
    Bm: float,  # 3 CaCO3 export flux as DIC
    dic_db: float,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_sb: float,  # 6 [DIC] in the surface box
    dic_sb_l: float,  # 7 [DIC_l] in the surface box
    hplus: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    i: float,  # 10  current index
    max_i: float,  # 11, max length of vr vectors
    last_t: float,  # 12 time of the last calls to cs2
) -> tuple:
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

    Author: M. Niazi & T. Tsan, 2021
    """

    hplus_0 = hplus  # hplus from last timestep

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
    # zsnow = int(abs(p[19]))  # previous zsnow
    depth_area_table = rg.cs.depth_area_table
    area_dz_table = rg.cs.area_dz_table
    Csat_table = rg.cs.Csat_table

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)

    # print(f"t = {t}")
    # print(f"RG = {rg.full_name}")
    # print(f"BM = {Bm}")
    # print(f"dic_db = {dic_db}"),
    # print(f"ta_db = {ta_db}"),
    # print(f"dic_sb = {dic_sb}, dic_sb_l = {dic_sb_l}, zsnow = {zsnow}")
    # print(f"i = {i}, hplus_0 = {hplus_0}, hplus = {hplus}, oh = {oh}, boh4 = {boh4}")
    fg: float = hplus - oh - boh4
    # print(f"ta_db = {ta_db}, fg = {fg}")
    ca: float = ta_db + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic_db / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic_db / (1 + (hplus / k1) + (k2 / hplus))
    # print(f"ca = {ca}, hco3 = {hco3}")
    co3: float = (ca - hco3) / 2
    # DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    # small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    co2aq: float = dic_db - co3 - hco3
    # co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    # omega: float = (ca2 * co3) / ksp

    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables

    # prevent co3 from becoming zero
    if co3 <= 0:
        co3 = 1e-16

    # print(f"i={i}, ca2={ca2}, co3={co3}, ksp0 = {ksp0}")
    zsat = int(max((zsat0 * np.log(ca2 * co3 / ksp0)), zsat_min))  # eq2
    zsat = min(zsat, zmax)
    zsat = max(zsat, zsat_min)
    # print(
    #     f"i = {i}, zsat0 = {zsat0:.1f}, ca= {ca:.2e}, co3 = {co3:.2e}, ksp0 = {ksp0:.2e}, zsat_min = {zsat_min:.1f}"
    # )
    # print(f"zsat = {zsat:.1f}\n")

    zcc = int(zsat0 * np.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3

    # ---- Get fractional areas
    B_AD = Bm / AD

    zcc = min(zcc, zmax)
    zcc = max(zcc, zsat_min)
    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]

    # ------------------------Calculate Burial Fluxes----------------------------- #
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
    # if zcc < zsnow:  # zsnow is deeper than zcc, so we need to dissolve
    #     if zsnow > zmax:  # zsnow cannot exceed ocean depth
    #         zsnow = zmax
    # sedimentary CaCO3
    # if zsnow > zmax:  # zsnow cannot exceed ocean depth
    #     zsnow = zmax
    # moved to equations.py

    # get saturation difference per depth interval
    diff: np.ndarray = Csat_table[zcc : int(zsnow)] - co3

    # get table of depth intervals
    # print(f"zcc = {zcc}, zsnow = {zsnow}")
    area_p: np.ndarray = area_dz_table[zcc : int(zsnow)]

    # integrate saturation difference over area
    BPDC = kc * area_p.dot(diff)

    BPDC = max(BPDC, 0)
    #     d_zsnow = 0
    # print(f"BPDC = {BPDC:.2e}")
    # eq 4 dzsnow/dt = Bpdc(t) / (a'(zsnow(t)) * ICaCO3
    # print(f"area_dz_table[int(zsnow)] = {area_dz_table[int(zsnow)]:.2e}")
    # print(f"I_caco3 = {I_caco3}, dt = {(t - last_t):.2e}")

    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    # elif zcc >= zsnow:  # e.g. 5000 > 4750
    #     # there is no carbonate below 4750, so BPDC = 0
    #     d_zsnow = 0
    #     BPDC = 0

    # get isotope ratio in reservoir
    # BD & F_burial
    BD: float = BDS + BCC + BNS + BPDC
    Fburial = Bm - BD
    """Bm is the flux of CaCO3 into the box. However, the model should
    use the bypass option and leave all flux calculations to the
    cs_code.  As such, we simply add the fraction of the input flux
    that dissolves, and ignore the fraction that is buried.  

    The isotope ratio of the dissolutio flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    DB_r = dic_sb_l / dic_sb
    BD_l = BD * dic_sb_l / dic_sb
    Fburial_l = Fburial * DB_r

    dH = hplus - hplus_0

    if i > max_i:
        rg.cs.H = np.append(rg.cs.H, hplus)
        rg.cs.CA = np.append(rg.cs.CA, ca)
        rg.cs.HCO3 = np.append(rg.cs.HCO3, hco3)
        rg.cs.CO3 = np.append(rg.cs.CO3, co3)
        rg.cs.CO2aq = np.append(rg.cs.CO2aq, co2aq)
        rg.cs.zsat = np.append(rg.cs.zsat, zsat)
        rg.cs.zcc = np.append(rg.cs.zcc, zcc)
        rg.cs.Fburial = np.append(rg.cs.Fburial, Fburial)
        rg.cs.Fburial_l = np.append(rg.cs.Fburial, Fburial)
    else:
        rg.cs.H[i] = hplus  #
        rg.cs.CA[i] = ca  # 1
        rg.cs.HCO3[i] = hco3  # 2
        rg.cs.CO3[i] = co3  # 3
        rg.cs.CO2aq[i] = co2aq  # 4
        rg.cs.zsat[i] = zsat  # 5
        rg.cs.zcc[i] = zcc  # 6
        rg.cs.Fburial[i] = Fburial
        rg.cs.Fburial_l[i] = Fburial_l

    return -BD, -BD_l, dH, d_zsnow


def gas_exchange_ode(scale, gas_c, p_H2O, solubility, c_aq) -> float:
    """Calculate the gas exchange flux across the air sea interface

    Parameters:
     scale: surface area in m^2
     gas_c: species concentration in atmosphere
     p_H2O: water vapor partial pressure
     solubility: species solubility  mol/(m^3 atm)
     c_aq: concentration of the dissolved gas in water
    """

    f = scale * (  # area in m^2
        gas_c  # Atmosphere
        * (1 - p_H2O)  # p_H2O
        * solubility  # SA_co2 = mol/(m^3 atm)
        - c_aq * 1000  # [CO2]aq mol
    )

    return -f


def gas_exchange_ode_with_isotopes(
    scale,  # surface area in m^2
    gas_c,  # species concentration in atmosphere
    gas_c_l,  # same but for the light isotope
    liquid_c,  # c of the reference species (e.g., DIC)
    liquid_c_l,  # same but for the light isotope
    p_H2O,  # water vapor pressure
    solubility,  # solubility constant
    c_aq,  # Gas concentration in liquid phase
    a_db,  # fractionation factor between dissolved CO2aq and HCO3-
    a_dg,  # fractionation between CO2aq and CO2g
    a_u,  # kinetic fractionation during gas exchange
) -> tuple:
    """Calculate the gas exchange flux across the air sea interface
    for co2 incliding isotope effects.

    Note that the sink delta is co2aq as returned by the carbonate VR
    this equation is for mmol but esbmtk uses mol, so we need to
    multiply by 1E3
    """

    # equilibrium concentration of CO2 in water based on pCO2
    eco2_at = gas_c * (1 - p_H2O) * solubility  # p Atmosphere  # p_H2O
    # equilibrium concentration of CO2 in water based on CO2aq
    eco2_aq = c_aq * 1000

    # total flux
    f = scale * (eco2_at - eco2_aq)

    # get heavy isotope concentration in the respective reservoirs
    c13g = gas_c - gas_c_l  #
    c13aq = liquid_c - liquid_c_l
    # get 13C CO2 equlibrium concentration  CO2 in water based on pCO2
    eco2_at_13 = gas_c * c13g * (1 - p_H2O) * solubility * a_dg  # p_H2O

    # get 13C equilibrium  CO2 in water based on DIC m & l
    eco2_aq_13 = a_db * eco2_aq * c13aq

    # 13C flux
    f13 = scale * a_u * (eco2_at_13 - eco2_aq_13)
    f12 = f - f13

    return -f, -f12
