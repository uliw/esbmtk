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
from numba.typed import List

if tp.TYPE_CHECKING:
    from .esbmtk import Reservoir, Model
    from .extended_classes import Reservoir


def carbonate_system_1_ode(i, input_data: list, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations and saturation state
     at the ith time-step of the model.

    The function assumes that vr_data will be in the following order:
        [H+, CA, HCO3, CO3, CO2(aq), omega]

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    Author: M. Niazi & T. Tsan, 2021, with modifications by U. Wortmann 2022
    """

    k1 = params[0]  # K1
    k2 = params[1]  # K2
    KW = params[2]  # KW
    KB = params[3]  # KB
    boron = params[4]  # boron
    ca2 = params[5]  # Ca2+
    ksp = params[6]  # Ksp
    hplus: float = params[7]  # hplus from last timestep

    dic: float = input_data[0]
    ta: float = input_data[1]
    hplus: float = input_data[2]

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
    # omega: float = ca2 * co3 / ksp

    # this may not be necessary to keep all data. maybe move to params
    vr_data[0][i] = hplus
    vr_data[1][i] = ca
    vr_data[2][i] = hco3
    vr_data[3][i] = co3
    vr_data[4][i] = co2aq
    vr_data[5][i] = oh
    vr_data[5][i] = boh4
    params[7] = hplus  # preserve state of hplus
    # vr_data[5][i] = omega


def carbonate_system_2_ode(i: int, input_data: List, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations and carbonate compensation
    depth (zcc) at the ith time-step of the model.

    The function assumes that vr_data will be in the following order:
        [H+, CA, HCO3, CO3, CO2(aq), zsat, zcc, zsnow, Fburial,
        B, BNS, BDS_under, BDS_resp, BDS, BCC, BPDC, BD,omega]

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004

    See add_carbonate_system_2 in utility_functions.py on how to call this function.
    The input data is a follows

        reservoir DIC.m,  # 0
        reservoir DIC.l,  # 1
        reservoir DIC.c,  # 2
        reservoir TA.m,  # 3 TA mass
        reservoir.TA.c,  # 4 TA concentration
        Export_flux.fa,  # 5
        area_table,  # 6
        area_dz_table,  # 7
        Csat_table,  # 8
        reservoir.DIC.v,  # 9 reservoir volume

    Author: M. Niazi & T. Tsan, 2021
    """

    # Parameters
    k1 = params[0]
    k2 = params[1]
    KW = params[2]
    KB = params[3]
    boron = params[4]
    ksp0 = params[5]
    kc = params[6]
    volume = params[7]
    AD = params[8]
    zsat0 = int(abs(params[9]))
    ca2 = params[10]
    dt = params[11]
    I_caco3 = params[12]
    alpha = params[13]
    zsat_min = int(abs(params[14]))
    zmax = int(abs(params[15]))
    z0 = int(abs(params[16]))
    ksp = params[17]
    hplus = params[18]  # previous h+ concentrartion
    zsnow = params[19]  # previous zsnow

    # Data
    dic: float = input_data[2]  # DIC concentration [mol/kg]
    ta: float = input_data[4]  # TA concentration [mol/kg]
    Bm: float = input_data[5]  # Carbonate Export Flux [mol/yr]
    B12: float = input_data[5]  # Carbonate Export Flux light isotope
    v: float = input_data[9]  # volume
    # lookup tables
    depth_area_table: np.ndarray = input_data[6]  # depth look-up table
    area_dz_table: np.ndarray = input_data[7]  # area_dz table
    Csat_table: np.ndarray = input_data[8]  # Csat table

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
        zsnow = zsnow - BPDC / (area_dz_table[int(zsnow)] * I_caco3) * dt

    else:  # zcc > zsnow
        # there is no carbonate below zsnow, so BPDC = 0
        zsnow = zcc
        BPDC = 0

    # BD & F_burial
    BD: float = BDS + BCC + BNS + BPDC
    Fburial = Bm - BD
    # Fburial12 = Fburial * input_data[1][i - 1] / input_data[0][i - 1]
    diss = (Bm - Fburial) * dt  # dissolution flux
    # diss12 = (B12 - Fburial12) * dt  # dissolution flux light isotope

    # # print("{Fburial}.format(")
    # print(Bm)
    # print(Fburial)
    # print(diss)
    # print()
    # # print('df ={:.2e}\n'.format(diss/dt))

    """ Now that the fluxes are known we need to update the reservoirs.
    The concentration in the in the DIC (and TA) of this box are
    DIC.m[i] + Export Flux - Burial Flux, where the isotope ratio
    the Export flux is determined by the overlying box, and the isotope ratio
    of the burial flux is determined by the isotope ratio of this box
    

    # Update DIC in the deep box
    input_data[0][i] = input_data[0][i] + diss  # DIC
    input_data[1][i] = input_data[1][i] + diss12  # 12C
    input_data[2][i] = input_data[0][i] / v  # DIC concentration

    # Update TA in deep box
    input_data[3][i] = input_data[3][i] + 2 * diss  # TA
    input_data[4][i] = input_data[3][i] / v  # TA concentration

    we needto return the relevant fluxes, so that they can be used in the equation
    systems
    """

    # copy results into datafields
    vr_data[0][i] = hplus  # 0
    vr_data[1][i] = ca  # 1
    vr_data[2][i] = hco3  # 2
    vr_data[3][i] = co3  # 3
    vr_data[4][i] = co2aq  # 4
    vr_data[5][i] = zsat  # 5
    vr_data[6][i] = zcc  # 6
    vr_data[7][i] = zsnow  # 7
    vr_data[8][i] = Fburial  # 8
    # vr_data[9][i] = Fburial12  # 9
    vr_data[10][i] = diss / dt  # 9
    vr_data[11][i] = Bm  # 9

    params[18] = hplus  # previous h+ concentrartion
    params[19] = zsnow  # previous zsnow

    return Fburial, 2 * Fburial
