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

from math import log, sqrt
import numpy as np
import numpy.typing as npt
from numba import njit

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


@njit(fastmath=True)
def gas_exchange_no_isotopes_2(
    gas_c,  # species concentration in atmosphere
    gas_c_aq,  # Gas concentration in liquid phase
    p,  # parameters
) -> float:
    """ Calculate the gas exchange flux across the air sea interface
    Note that this equation is for mmol but esbmtk uses mol, so we need to
    multiply by 1E3
    """

    area, solubility, piston_velocity, p_H2O = p
    scale = area * piston_velocity
    # Solibility with correction for pH2O
    beta = solubility * (1 - p_H2O)
    # f as afunction of solubility difference
    f = scale * (beta * gas_c - gas_c_aq * 1e3)
    return f


@njit(fastmath=True)
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

    Note that this equation is for mmol but esbmtk uses mol, so we need to
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


@njit(fastmath=True)
def photosynthesis(
    o2_atmosphere,
    co2_atmosphere,
    co2_atmosphere_l,
    o2,
    ta,
    dic,
    dic_l,
    po4,
    so4,
    h2s,
    hplus,
    co2aq,
    po4_upwelling_flux,  # actually P flux
    p: tuple[float],  # parameters
) -> tuple:
    """Calculate the effects of photosynthesis in the surface boxes
    Note that this functions returns fluxes, so we need to calculate
    dMdt, not dCdt
    """

    # unpack parameters
    (
        volume,
        surface_area,
        sed_area,
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
        r_carbon,
        pH2O,
        pv,
        o2_solubility,
        co2_solubility,
        a_db,
        a_dg,
        a_u,
        CaCO3_reactions,
    ) = p

    # save data from previous time step
    hplus_0 = hplus
    co2aq_0 = co2aq
    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
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
    # Note that these are not used in Reservoir calcuations, so we can return
    # dCdt instead of dMdt
    dCdt_H = hplus - hplus_0
    dCdt_co2aq = co2aq - co2aq_0

    # Gas Exchange O2
    o2_ex = gas_exchange_no_isotopes_2(
        o2_atmosphere,  # species concentration in atmosphere
        o2,  # c of the reference species (e.g., DIC)
        (surface_area, o2_solubility, pv, pH2O),
    )
    # Gase exchange CO2
    co2_ex, co2_ex_l = gas_exchange_with_isotopes_2(
        co2_atmosphere,  # species concentration in atmosphere
        co2_atmosphere_l,
        dic,
        dic_l,
        co2aq,  # Gas concentration in liquid phase
        (surface_area, co2_solubility, pv, pH2O, a_db, a_dg, a_u),
    )

    # check signs!
    dMdt_o2_at = -o2_ex
    dMdt_o2 = o2_ex
    # print(f"dMdt_o2 gex = {dMdt_o2:.2e}")

    # check signs!
    dMdt_co2_at = -co2_ex
    dMdt_co2_at_l = -co2_ex_l
    dMdt_dic = co2_ex
    dMdt_dic_l = co2_ex_l

    # POM formation
    dMdt_po4 = -po4_upwelling_flux * PUE  # remove PO4 into POM
    POM_F = po4_upwelling_flux * PUE * PC_ratio  # mass of newly formed POM
    r = get_new_ratio_from_alpha(dic, dic_l, om_fractionation_factor)
    POM_F_l = POM_F * r  # mass of POM_l
    dMdt_dic += -POM_F  # remove DIC by POM formation
    dMdt_dic_l += -POM_F_l
    dMdt_ta = POM_F * NC_ratio  # add TA from nitrate uptake into POM

    # CaCO3 formation
    if CaCO3_reactions:
        # newly formed CaCO3
        PIC_F = POM_F / rain_rate
        PIC_F_l = PIC_F * dic_l / dic  # same as water
        # Account for sedimentary respiration
        # diss = PIC_F * alpha * sed_area / surface_area
        # diss_l = PIC_F_l * alpha * sed_area / surface_area
        # dMdt_dic += diss
        # dMdt_dic_l += diss_l
        # dMdt_ta += 2 * diss
        # PIC_F -= diss
        # PIC_F_l -= diss_l
        # dic & ta removed by photosynthesis
        dMdt_dic += -PIC_F
        dMdt_dic_l += -PIC_F_l
        dMdt_ta += 2 * -PIC_F
    else:
        PIC_F = 0.0
        PIC_F_l = 0.0
        
    # sulfur reactions, assuming that there is alwways enough O2
    dMdt_h2s = 0  # -h2s * volume  # H2S oxidation
    dMdt_so4 = dMdt_h2s  # add S to the sulfate pool
    dMdt_ta += 2 * dMdt_so4  # adjust Alkalinity

    # add O2 from photosynthesis - h2s oxidation
    dMdt_o2 += POM_F * O2C_ratio - 2 * dMdt_h2s

    return (  # note that these are returned as fluxes
        dCdt_H,
        dCdt_co2aq,
        dMdt_o2,
        dMdt_ta,
        dMdt_po4,
        dMdt_so4,
        dMdt_h2s,
        dMdt_dic,
        dMdt_dic_l,
        dMdt_o2_at,
        dMdt_co2_at,
        dMdt_co2_at_l,
        POM_F,
        POM_F_l,
        PIC_F,
        PIC_F_l,
    )


@njit(fastmath=True)
def OM_remineralization(
    pom_fluxes: list,  # POM export fluxes
    pom_fluxes_l: list,  # POM_l export fluxes
    pic_fluxes: list,
    pic_fluxes_l: list,
    pom_remin_fractions: list,  # list of remineralization fractions
    pic_remin_fractions: float,  # can these g on to the parameter list?
    h2s: float,  # concentration
    so4: float,  # concentration
    o2: float,  # o2 concentration
    po4: float,  # po4 concentration
    dic: float,
    dic_l: float,
    ta: float,
    hplus: float,
    p: tuple,  # parameter list
    # burial: float,
) -> float:
    (
        volume,
        area,
        sed_area,
        PC_ratio,
        NC_ratio,
        O2C_ratio,
        P_burial,
        alpha,
        k1,
        k1k1,
        k1k2,
        KW,
        KB,
        boron,
        r_carbon,  # carbon reference value
        omwd,  # delta of weathering organic carbon
        CaCO3_reactions,
    ) = p

    # save data from previous time step
    hplus_0 = hplus
    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg
    # hplus
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1k1 - k1k2 * (4 - 8 * gamm)
    hplus: float = 0.5 * ((gamm - 1) * k1 + sqrt(dummy))
    # Note that these are not used in Reservoir calcuations, so we can return
    # dCdt instead of dMdt
    dCdt_H = hplus - hplus_0

    """Reservoirs can have multiple sources of POM with different
    remineralization efficiencies, e.g., low latidtude POM flux, vs
    high latitude POM flux. We only add the part that is remineralized.
    """
    pom_flux = 0
    pom_flux_l = 0
    # sum all POM and dic fluxes
    for i, f in enumerate(pom_fluxes):
        pom_flux += pom_fluxes[i] * pom_remin_fractions[i]
        pom_flux_l += pom_fluxes_l[i] * pom_remin_fractions[i]

    if True:
        # get_om_burial_fluxes. These are in isotopic equilibrium
        # with the organic matter rain
        om_burial = pom_flux * P_burial
        om_burial_l = pom_flux_l * P_burial
        # burial must match weathering and CO2 in Atmosphere. The Isotope
        # ratio depends however on the isotope ratio of the weathered OM
        # dMdt_co2_At = 1.06 * om_burial
        # dMdt_co2_At_l = 1.06 * om_burial * 1000 / ((omwd + 1000) * r_carbon + 1000)

        # weathering consumes atmospheric O2. Here we force to be equal to burial
        dMdt_o2_At = -om_burial * O2C_ratio

        # Reduce POM rain by the fraction that is buried
        pom_flux -= om_burial
        pom_flux_l -= om_burial_l

        # Remove Alkalinity and add dic and po4 from POM remineralization
        # Assume that all OM is always fully metabolized,
        # irrespective of oxygen levels
        dMdt_po4 = pom_flux / PC_ratio

        # if we do burial compensation locally
        pom_flux += om_burial
        pom_flux_l += om_burial * 1000 / ((omwd + 1000) * r_carbon + 1000)
        dMdt_co2_At = 0
        dMdt_co2_At_l = 0
        dMdt_o2_At = 0

    else:
        dMdt_po4 = (1 - P_burial) * pom_flux / PC_ratio
        dMdt_co2_At = 0
        dMdt_co2_At_l = 0
        dMdt_o2_At = 0

    dMdt_dic = pom_flux  # add DIC from POM
    dMdt_dic_l = pom_flux_l

    # Calculate O2 requirement. Note that O2 & H2S are concentrations
    m_o2 = o2 * volume
    m_h2s = h2s * volume
    m_o2_eq = pom_flux * O2C_ratio + 2 * m_h2s
    # reduce Alkalinity from NO3. This happens irrespective of O2
    dMdt_ta = -pom_flux * NC_ratio
    
    if m_o2 > m_o2_eq:  # box has enough oxygen
        dMdt_o2 = -m_o2_eq  # consume O2
        dMdt_h2s = -m_h2s  # consume all h2s
        dMdt_so4 = dMdt_h2s  # add sulfate from h2s
        dMdt_ta -= dMdt_so4 # reduce Alkalinity from sulfate addition
        
    else:  # box has not enough oxygen
        dMdt_o2 = -m_o2  # remove all available oxygen
        # calculate how much POM is left to oxidize
        pom_flux = pom_flux - m_o2 / O2C_ratio
        # # oxidize the remaining POM via sulfate reduction
        dMdt_so4 = -pom_flux / 2  # one SO4 oxidizes 2 carbon, and add 2 mol to TA
        dMdt_h2s = -dMdt_so4  # move S to reduced reservoir
        dMdt_ta += 2 * -dMdt_so4  # adjust Alkalinity for changes in sulfate

    if CaCO3_reactions:
        """photosynthesis calculates the total PIC production and removes DIC and TA
        from the surface box accordingly. Most of the PIC flux either ends up as
        sediment on the slope, or sinks further into the deep box. However, a fraction is
        remineralized through aerobic respiration in the sediment and contributes DIc and TA
        to the intermediate waters. See eq 8 in Boudreau 2010
        Move to CS3
        """
        pic_flux = 0.0
        pic_flux_l = 0.0
        for i, f in enumerate(pic_fluxes):
            pic_flux += pic_fluxes[i]
            pic_flux_l += pic_fluxes_l[i]

        # add Alkalinity and DIC from CaCO3 dissolution. Note that
        dMdt_dic += alpha * pic_flux * sed_area / area
        dMdt_dic_l += alpha * pic_flux_l * sed_area / area
        dMdt_ta += 2 * alpha * pic_flux * sed_area / area

    return [
        dMdt_dic,
        dMdt_dic_l,
        dMdt_ta,
        dMdt_h2s,
        dMdt_so4,
        dMdt_o2,
        dMdt_po4,
        dMdt_o2_At,
        dMdt_co2_At,
        dMdt_co2_At_l,
        dCdt_H,
    ]


@njit(fastmath=True)
def carbonate_system_3(
    # rg: ReservoirGroup,  # 2 Reservoir handle
    pic_f: float,  # 3 CaCO3 export flux as DIC
    dic_db: float,  # 4 DIC in the deep box
    dic_db_l: float,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_sb: float,  # 6 [DIC] in the surface box
    dic_sb_l: float,  # 7 [DIC_l] in the surface box
    hplus: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    co3: float,
    p: tuple,
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

    (  # unpack parameters
        ksp0,
        kc,
        zsat0,
        I_caco3,
        alpha,
        zsat_min,
        zmax,
        z0,
        k2,
        k1k2,
        ca2,
        box_area,
        ocean_area,
        depth_area_table,
        area_dz_table,
        Csat_table,
    ) = p

    zsat0 = int(abs(zsat0))
    zsat_min = int(abs(zsat_min))
    zmax = int(abs(zmax))
    z0 = int(abs(z0))

    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    co3_0 = co3  # save value from t-1
    co3 = max(dic_db / (1 + hplus / k2 + hplus * hplus / k1k2), 3.7e-05)
    dCdt_co3 = co3 - co3_0
    # all depths will be positive to facilitate the use of lookup_tables
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    zsat = min(zmax, max(zsat_min, zmax))
    zcc = int(
        zsat0 * log(pic_f * ca2 / (ksp0 * box_area * kc) + ca2 * co3 / ksp0)
    )  # eq3
    zcc = min(zmax, max(zsat_min, zcc))

    # the below tables are for the entire ocean, so we needto scale them
    # to the respective box area, in this case the top of intermediate box
    fractional_area = box_area / ocean_area
    A_z0_zsat = (depth_area_table[z0] - depth_area_table[zsat]) * fractional_area
    A_zsat_zcc = (depth_area_table[zsat] - depth_area_table[zcc]) * fractional_area
    A_zcc_zmax = (depth_area_table[zcc] - depth_area_table[zmax]) * fractional_area
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC: float = A_zcc_zmax * pic_f / box_area  # CCD dissolution
    BNS: float = alpha * A_z0_zsat * pic_f / box_area  # water column dissolution
    diff_co3: NDArrayFloat = Csat_table[zsat:zcc] - co3
    area_p: NDArrayFloat = area_dz_table[zsat:zcc]
    BDS_under: float = kc * area_p.dot(diff_co3) * fractional_area
    BDS_resp: float = alpha * (A_zsat_zcc * pic_f / box_area - BDS_under)
    BDS: float = BDS_under + BDS_resp  # respiration
    if zsnow > zmax:
        zsnow = zmax
    diff: NDArrayFloat = Csat_table[zcc : int(zsnow)] - co3
    area_p: NDArrayFloat = area_dz_table[zcc : int(zsnow)]
    # integrate saturation difference over area
    BPDC: float = kc * area_p.dot(diff) * fractional_area  # diss if zcc < zsnow
    BPDC = max(BPDC, 0)  # prevent negative values
    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3 * fractional_area)

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

    return dMdt_dic, dMdt_dic_l, dMdt_ta, d_zsnow, dCdt_co3
