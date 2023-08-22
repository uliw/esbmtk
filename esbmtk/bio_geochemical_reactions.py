"""
esbmtk: A general purpose Earth Science box model toolkit Copyright
(C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import typing as tp
from math import log, sqrt
import numpy as np
from numba import njit
from esbmtk import get_new_ratio_from_alpha

if tp.TYPE_CHECKING:
    from esbmtk import ReservoirGroup


@njit()
def photosynthesis(
    o2,
    ta,
    dic,
    dic_l,
    po4,
    so4,
    h2s,
    hplus,
    co2aq,
    productivity,  # actually P flux
    p: tuple[float],  # parameters
) -> tuple:
    """Calculate the effects of photosynthesis in the surface boxes"""
    """O2 in surface box as result of photosynthesis equals the primary
    productivity export flux of organic C times the O2:C ratio
    TA increases because of nitrate uptake during photosynthesis

    Note that this functions returns fluxes, so we need to calculate
    dMdt, not dCdt
    """

    # unpack parameters
    (
        volume,
        PC_ratio,
        NC_ratio,
        O2C_ratio,
        PUE,
        rain_rate,
        om_fractionation_factor,
        alpha,
        k1,
        k1k1,
        k1k2,
        KW,
        KB,
        boron,
    ) = p

    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    hplus_0 = hplus
    co2aq_0 = co2aq
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # hplus
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1k1 - k1k2 * (4 - 8 * gamm)
    hplus: float = 0.5 * ((gamm - 1) * k1 + sqrt(dummy))
    # co2aq is calcualted anew for each t, and used as is, so no need to get dMdt
    co2aq: float = dic / (1 + (k1 / hplus) + (k1k2 / (hplus * hplus)))
    # for H1 we need the state a t-1, so we need to return dMdt
    dMdt_H = hplus - hplus_0
    dMdt_co2aq = co2aq - co2aq_0

    # POM formation
    dMdt_po4 = -productivity * PUE  # remove PO4 into POM
    POM_F = -dMdt_po4 * PC_ratio  # mass of newly formed POM
    r = get_new_ratio_from_alpha(dic, dic_l, om_fractionation_factor)
    POM_F_l = POM_F * r  # mass of POM_l
    dMdt_dic = -POM_F  # remove DIC by POM formation
    dMdt_dic_l = -POM_F_l
    dMdt_ta = POM_F * NC_ratio  # add TA from nitrate uptake into POM

    # CaCO3 formation
    alpha = 1
    PIC_F = POM_F * alpha / rain_rate  # newly formed CaCO3
    PIC_F_l = PIC_F * dic_l / dic
    dMdt_dic += -PIC_F  # dic removed
    dMdt_dic_l += -PIC_F
    dMdt_ta += 2 * -PIC_F  # TA removed

    # sulfur reactions, assuming that there is alwways enough O2
    dMdt_h2s = -h2s * volume  # H2S oxidation
    dMdt_so4 = dMdt_h2s  # add S to the sulfate pool
    dMdt_ta += 2 * dMdt_so4  # adjust Alkalinity
    # add O2 from photosynthesis - h2s oxidation
    dCdt_o = POM_F * O2C_ratio - 2 * h2s * volume

    return (  # note that these are returned as fluxes
        dMdt_H,
        dMdt_co2aq,
        dCdt_o,
        dMdt_ta,
        dMdt_po4,
        dMdt_so4,
        dMdt_h2s,
        dMdt_dic,
        dMdt_dic_l,
        POM_F,
        POM_F_l,
        PIC_F,
        PIC_F_l,
    )


@njit()
def remineralization(
    pom_fluxes: list,  # POM export fluxes
    pom_fluxes_l: list,  # POM_l export fluxes
    pic_fluxes: list,
    pic_fluxes_l: list,
    pom_remin_fractions: list,  # list of remineralization fractions
    pic_remin_fractions: float,
    h2s: float,  # concentration
    so4: float,  # concentration
    o2: float,  # o2 concentration
    po4: float,  # po4 concentration
    volume: float,  # box volume
    PC_ratio: float,
    NC_ratio: float,
    O2C_ratio: float,
    alpha: float,
    CaCO3_reactions=True,
    # burial: float,
) -> float:
    """Reservoirs can have multiple sources of POM with different
    remineralization efficiencies, e.g., low latidtude POM flux, vs
    high latitude POM flux. We only add the part that is remineralized.
    Note: The CaCO3 fluxes are handled below
    """
    pom_flux = 0
    pom_flux_l = 0
    # sum all POM and dic fluxes
    for i, f in enumerate(pom_fluxes):
        pom_flux += f * pom_remin_fractions[i]
        pom_flux_l += pom_fluxes_l[i] * pom_remin_fractions[i]

    # remove Alkalinity and add dic and po4 from POM remineralization
    # this happens irrespective of oxygen levels
    dMdt_po4 = pom_flux / PC_ratio  # return PO4
    dMdt_ta = -pom_flux * NC_ratio  # remove Alkalinity from NO3
    dMdt_dic = pom_flux  # add DIC from POM
    dMdt_dic_l = pom_flux_l
    m_h2s = h2s * volume

    m_o2 = o2 * volume
    # how much O2 is needed to oxidize all POM and H2S
    m_o2_eq = pom_flux * O2C_ratio + 2 * m_h2s

    if m_o2 > m_o2_eq:  # box has enough oxygen
        dMdt_o2 = -m_o2_eq  # consume O2
        dMdt_h2s = -m_h2s  # consume all h2s
        dMdt_so4 = -m_h2s  # add sulfate

    else:  # box has not enough oxygen
        dMdt_o2 = -m_o2  # remove all available oxygen
        # calculate how much POM is left to oxidize
        pom_flux = pom_flux - m_o2 / O2C_ratio
        # oxidize the remaining POM via sulfate reduction
        dMdt_so4 = -pom_flux / 2  # one SO4 oxidizes 2 carbon, and add 2 mol to TA
        dMdt_h2s = -dMdt_so4  # move S to reduced reservoir
        dMdt_ta += 2 * -dMdt_so4  # adjust Alkalinity for changes in sulfate

    if CaCO3_reactions:
        pic_flux = 0.0
        dic_flux_l = 0.0
        for i, f in enumerate(pic_fluxes):
            pic_flux += f * pic_remin_fractions[i] * alpha
            dic_flux_l += pic_fluxes_l[i] * pic_remin_fractions[i] * alpha

        # add Alkalinity and DIC from CaCO3 dissolution. Note that
        dMdt_dic += pic_flux
        dMdt_dic_l += dic_flux_l
        dMdt_ta += 2 * pic_flux

    # note, these are returned as fluxes
    return [dMdt_dic, dMdt_dic_l, dMdt_ta, dMdt_h2s, dMdt_so4, dMdt_o2, dMdt_po4]


def carbonate_system_3(
    rg: ReservoirGroup,  # 2 Reservoir handle
    pic_f: float,  # 3 CaCO3 export flux as DIC
    dic_db: float,  # 4 DIC in the deep box
    dic_db_l: float,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_sb: float,  # 6 [DIC] in the surface box
    dic_sb_l: float,  # 7 [DIC_l] in the surface box
    hplus_0: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    ksp0,
    kc,
    AD,
    zsat0,
    I_caco3,
    alpha,
    zsat_min,
    zmax,
    z0,
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
    k1 = rg.swc.K1  # K1
    k1k1 = rg.swc.K1K1
    k2 = rg.swc.K2  # K2
    k1k2 = rg.swc.K1K2  # K2
    KW = rg.swc.KW  # KW
    KB = rg.swc.KB  # KB
    ca2 = rg.swc.ca2  # Ca2+
    boron = rg.swc.boron  # boron
    zsat0 = int(abs(zsat0))
    zsat_min = int(abs(zsat_min))
    zmax = int(abs(zmax))
    z0 = int(abs(z0))
    depth_area_table = rg.cs.depth_area_table
    area_dz_table = rg.cs.area_dz_table
    Csat_table = rg.cs.Csat_table
    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus_0
    boh4: float = boron * KB / (hplus_0 + KB)
    fg: float = hplus_0 - oh - boh4
    ca: float = ta_db + fg

    # calculate carbon speciation
    # The following equations are after Follows et al. 2006
    gamm: float = dic_db / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1k1 - k1k2 * (4 - 8 * gamm)
    hplus: float = 0.5 * ((gamm - 1) * k1 + sqrt(dummy))
    co3 = max(dic_db / (1 + hplus / k2 + hplus * hplus / k1k2), 3.7e-05)
    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    zsat = np.clip(zsat, zsat_min, zmax)
    zcc = int(zsat0 * log(pic_f * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3
    zcc = np.clip(zcc, zsat_min, zmax)
    # get fractional areas
    B_AD = pic_f / AD
    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD  # CCD dissolution
    BNS = alpha * A_z0_zsat * B_AD  # water column dissolution
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]
    BDS_under = kc * area_p.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp  # respiration
    if zsnow > zmax:
        zsnow = zmax
    diff: np.ndarray = Csat_table[zcc : int(zsnow)] - co3
    area_p: np.ndarray = area_dz_table[zcc : int(zsnow)]
    # integrate saturation difference over area
    BPDC = kc * area_p.dot(diff)  # diss if zcc < zsnow
    BPDC = max(BPDC, 0)  # prevent negative values
    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    """CaCO3_export is the flux of particulate CaCO3 into the box.
    The fraction that is buried does not affect the water chemistry,
    so here we only consider the carbonate that is dissolved.
    The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    # add dic and TA from dissolution
    dMdt_dic = BDS + BCC + BNS + BPDC
    dMdt_ta = 2 * dMdt_dic
    dMdt_dic_l = dMdt_dic * dic_sb_l / dic_sb
    dH = hplus - hplus_0

    return dMdt_dic, dMdt_dic_l, dMdt_ta, dH, d_zsnow


@njit
def gas_exchange_with_isotopes_2(
    gas_c,  # species concentration in atmosphere
    gas_c_l,  # same but for the light isotope
    liquid_c,  # c of the reference species (e.g., DIC)
    liquid_c_l,  # same but for the light isotopeof DIC
    gas_c_aq,  # Gas concentration in liquid phase
    p,  # parameters
) -> tuple(float, float):
    """Calculate the gas exchange flux across the air sea interface
    for co2 including isotope effects.

    Note that the sink delta is co2aq as returned by the carbonate VR
    this equation is for mmol but esbmtk uses mol, so we need to
    multiply by 1E3

    The Total flux across interface dpends on the difference in either
    concentration or pressure the atmospheric pressure is known, as gas_c, and
    we can calculate the equilibrium pressure that corresponds to the dissolved
    gas in the water as [CO2]aq/beta.

    Conversely, we can convert the the pCO2 into the amount of dissolved CO2 =
    pCO2 * beta

    The h/c ratio in HCO3 estimated via h/c in DIC. Zeebe writes C12/C13 ratio
    but that does not work. the C13/C ratio results however in -8 permil
    offset, which is closer to observations
    """

    area, solubility, piston_velocity, p_H2O, a_db, a_dg, a_u = p
    scale = area * piston_velocity

    # Solibility with correction for pH2O
    beta = solubility * (1 - p_H2O)
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_c_aq * 1e3)
    # isotope ratio of DIC
    Rt = (liquid_c - liquid_c_l) / liquid_c
    # get heavy isotope concentrations in atmosphere
    gas_c_h = gas_c - gas_c_l  # gas heavy isotope concentration
    # get exchange of the heavy isotope
    f_h = scale * a_u * (a_dg * gas_c_h * beta - Rt * a_db * gas_c_aq * 1e3)
    f_l = f - f_h  # the corresponding flux of the light isotope

    return f, f_l


@njit
def gas_exchange_no_isotopes_2(
    gas_c,  # species concentration in atmosphere
    liquid_c,  # c of the reference species (e.g., DIC)
    gas_c_aq,  # Gas concentration in liquid phase
    p,  # parameters
) -> tuple(float, float):
    """Calculate the gas exchange flux across the air sea interface"""

    (
        area,
        solubility,
        piston_velocity,
        p_H2O,
    ) = p

    scale = area * piston_velocity
    # Solibility with correction for pH2O
    beta = solubility * (1 - p_H2O)
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_c_aq * 1e3)
    return f
