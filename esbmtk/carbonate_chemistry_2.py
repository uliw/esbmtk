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
from __future__ import annotations
import typing as tp
import numpy as np
import math
from esbmtk import ReservoirGroup

# if tp.TYPE_CHECKING:
#     from .esbmtk import ReservoirGroup, Flux


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
    co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus**2)))
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

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004
    """

    # Parameters
    hplus_0 = hplus  # hplus from last timestep
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
    depth_area_table = rg.cs.depth_area_table
    area_dz_table = rg.cs.area_dz_table
    Csat_table = rg.cs.Csat_table

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta_db + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic_db / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))
    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy**0.5))
    co3 = max(dic_db / (1 + hplus / k2 + hplus * hplus / (k1 * k2)), 3.7e-05)
    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables
    zsat = int(zsat0 * math.log(ca2 * co3 / ksp0))
    zsat = np.clip(zsat, zsat_min, zmax)
    zcc = int(zsat0 * math.log(Bm * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3
    zcc = np.clip(zcc, zsat_min, zmax)
    # get fractional areas
    B_AD = Bm / AD
    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD
    BNS = alpha * A_z0_zsat * B_AD
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]
    # if len(diff_co3) != len(area_p):
    #     breakpoint()
    BDS_under = kc * area_p.dot(diff_co3)

    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp
    # get saturation difference per depth interval
    diff: np.ndarray = Csat_table[zcc : int(zsnow)] - co3
    area_p: np.ndarray = area_dz_table[zcc : int(zsnow)]
    # integrate saturation difference over area
    BPDC = kc * area_p.dot(diff)
    BPDC = max(BPDC, 0)  # prevent negative values
    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)
    # get isotope ratio in reservoir
    BD: float = BDS + BCC + BNS + BPDC
    """Bm is the flux of CaCO3 into the box. However, the model should
    use the bypass option and leave all flux calculations to the
    cs_code.  As such, we simply add the fraction of the input flux
    that dissolves, and ignore the fraction that is buried.  

    The isotope ratio of the dissolutio flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    BD_l = BD * dic_sb_l / dic_sb
    # BD_h = BD - BD_l
    # print(f"13C burial = {get_delta(BD_l, BD_h,  0.0112372):.2f}")

    dH = hplus - hplus_0

    return -BD, -BD_l, dH, d_zsnow


def gas_exchange_ode(scale, gas_c, p_H2O, solubility, g_c_aq) -> float:
    """Calculate the gas exchange flux across the air sea interface

    Parameters:
    scale: surface area in m^2
    gas_c: species concentration in atmosphere
    p_H2O: water vapor partial pressure
    solubility: species solubility  mol/(m^3 atm)
    gc_aq: concentration of the dissolved gas in water
    """
    beta = solubility * (1 - p_H2O)
    f = scale * (gas_c * beta - g_c_aq * 1e3)

    return -f


def gas_exchange_ode_with_isotopes(
    scale,  # surface area in m^2 * piston velocity
    gas_c,  # species concentration in atmosphere
    gas_c_l,  # same but for the light isotope
    liquid_c,  # c of the reference species (e.g., DIC)
    liquid_c_l,  # same but for the light isotopeof DIC
    p_H2O,  # water vapor pressure
    solubility,  # solubility constant
    gas_c_aq,  # Gas concentration in liquid phase
    a_db,  # fractionation factor between dissolved CO2aq and HCO3-
    a_dg,  # fractionation between CO2aq and CO2g
    a_u,  # kinetic fractionation during gas exchange
) -> tuple(float, float):
    """Calculate the gas exchange flux across the air sea interface
    for co2 including isotope effects.

    Note that the sink delta is co2aq as returned by the carbonate VR
    this equation is for mmol but esbmtk uses mol, so we need to
    multiply by 1E3
    """

    # Solibility with correction for pH2O
    beta = solubility * (1 - p_H2O)
    """total flux across interface dpends on the difference in either
    concentration or pressure the atmospheric pressure is known, as gas_c, and
    we can calculate the equilibrium pressure that corresponds to the dissolved
    gas in the water as [CO2]aq/beta.

    Conversely, we can convert the the pCO2 into the amount of dissolved CO2 =
    pCO2 * beta
    """
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_c_aq * 1e3)

    # h/c ratio in HCO3 estimated via h/c in DIC. Zeebe writes C12/C13 ratio
    # but that does not work. the C13/C ratio results however in -8 permil
    # offset, which is closer to observations
    Rt = (liquid_c - liquid_c_l) / liquid_c

    # get heavy isotope concentrations in atmosphere
    gas_c_h = gas_c - gas_c_l  # gas heavy isotope concentration

    f_h = scale * a_u * (a_dg * gas_c_h * beta - Rt * a_db * gas_c_aq * 1e3)
    # print(f"gas_c = {gas_c:.2e}, gas_c_l {gas_c_l:.2e}, gas_c_h {gas_c_h:.2e}")
    # print(f"liquid_c = {liquid_c*1000:.2f}, Rd = {Rd:.2e}")
    # print(
    #     f"p13CO2 atmosphere = {1000 * beta * gas_c_h:.2f}, p13CO2 water = {1000 * Rd * liquid_c:.2f}"
    # )
    f_l = f - f_h

    return -f, -f_l
