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
import numpy.typing as npt
from math import log, sqrt
from .utility_functions import (
    __checkkeys__,
    __addmissingdefaults__,
    __checktypes__,
    register_return_values,
)

if tp.TYPE_CHECKING:
    from .esbmtk import SeawaterConstants, ReservoirGroup, Flux

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

"""
Carbonate System 1 setup requires 3 steps: First we define the actual function,
carbonate_system_1_ode().  In the second step we create a wrapper
init_carbonate_system_1() that defines how to integrate this function into
esbmtk.  In the third step we create a functiom that uses
init_carbonate_system_1() to associates cs1 instances with the respective
reservoirs.

The process for cs2 is analogous
"""


def carbonate_system_1_ode(
    swc: any,
    dic: float,
    dic_l: float,
    ta: float,
    hplus: float,
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

    k1 = swc.K1  # K1
    k2 = swc.K2  # K1
    k1k1 = swc.K1K1  # K1 * K1
    k1k2 = swc.K1K2  # K1 * K2
    KW = swc.KW  # KW
    KB = swc.KB  # KB
    boron = swc.boron  # boron
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
    co2aq: float = dic / (1 + (k1 / hplus) + (k1k2 / (hplus * hplus)))
    co3 = max(dic / (1 + hplus / k2 + hplus * hplus / k1k2), 3.7e-05)
    diff = hplus - hplus_0

    return diff, co2aq, co3


def init_carbonate_system_1(rg: ReservoirGroup):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: list = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Reservoir
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import ExternalCode, carbonate_system_1_ode

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_1_ode,
        fname="carbonate_system_1_ode",
        ftype="cs1",
        function_input_data=[rg.swc, rg.DIC, rg.TA, "Hplus"],
        register=rg,
        # name and initial value pairs
        # return_values={"Hplus": rg.swc.hplus},
        return_values=[
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
            {f"R_{rg.full_name}.CO2aq": rg.swc.co2aq},
            {f"R_{rg.full_name}.CO3": rg.swc.co3},
            # {"Hplus": rg.swc.hplus},
            # {"CO2aq": rg.swc.co2aq},
        ],
        # return_values=["Hplus", "CO2aq"],
    )
    rg.mo.lpc_f.append(ec.fname)

    return ec


def add_carbonate_system_1(rgs: list):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: list = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Reservoir
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import init_carbonate_system_1, Reservoir

    for rg in rgs:
        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            pass
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")

        ec = init_carbonate_system_1(rg)
        register_return_values(ec, rg)
        rg.has_cs1 = True


def carbonate_system_2_ode(
    rg: ReservoirGroup,  # 2 Reservoir handle
    CaCO3_export: float,  # 3 CaCO3 export flux as DIC
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
    zcc = int(
        zsat0 * log(CaCO3_export * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0)
    )  # eq3
    zcc = np.clip(zcc, zsat_min, zmax)
    # get fractional areas
    B_AD = CaCO3_export / AD
    A_z0_zsat = depth_area_table[z0] - depth_area_table[zsat]
    A_zsat_zcc = depth_area_table[zsat] - depth_area_table[zcc]
    A_zcc_zmax = depth_area_table[zcc] - depth_area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD
    BNS = alpha * A_z0_zsat * B_AD
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_p = area_dz_table[zsat:zcc]
    BDS_under = kc * area_p.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp
    if zsnow > zmax:
        zsnow = zmax
    diff: NDArrayFloat = Csat_table[zcc : int(zsnow)] - co3
    area_p: NDArrayFloat = area_dz_table[zcc : int(zsnow)]
    # integrate saturation difference over area
    BPDC = kc * area_p.dot(diff)
    BPDC = max(BPDC, 0)  # prevent negative values
    d_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)
    BD: float = BDS + BCC + BNS + BPDC

    """CACO3_export is the flux of CaCO3 into the box. However, the model should
    use the bypass option and leave all flux calculations to the
    cs_code.  As such, we simply add the fraction of the input flux
    that dissolves, and ignore the fraction that is buried.  

    The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    BD_l = BD * dic_sb_l / dic_sb
    dH = hplus - hplus_0
    # F_DIC, F_DIC_l, F_TA, dH, d_zsnow
    return BD, BD_l, 2 * BD, dH, d_zsnow, co3


def gas_exchange_ode(scale, gas_c, p_H2O, solubility, g_c_aq) -> float:
    """Calculate the gas exchange flux across the air sea interface

    Parameters:
    scale: surface area in m^2 * piston_velocity
    gas_c: species concentration in atmosphere
    p_H2O: water vapor partial pressure
    solubility: species solubility  mol/(m^3 atm)
    gc_aq: concentration of the dissolved gas in water
    """

    beta = solubility * (1 - p_H2O)
    f = scale * (gas_c * beta - g_c_aq * 1e3)
    return -f


def init_carbonate_system_2(
    rg: ReservoirGroup,
    export_flux: Flux,
    r_sb: ReservoirGroup,
    r_db: ReservoirGroup,
    area_table: NDArrayFloat,
    area_dz_table: NDArrayFloat,
    Csat_table: NDArrayFloat,
    AD: float,
    kwargs: dict,
):
    from esbmtk import ExternalCode, carbonate_system_2_ode

    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_2_ode,
        fname="carbonate_system_2_ode",
        ftype="cs2",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        vr_datafields={
            "depth_area_table": area_table,
            "area_dz_table": area_dz_table,
            "Csat_table": Csat_table,
        },
        function_input_data=[
            rg,  # 0
            export_flux,  # 1
            r_db.DIC,  # 2
            r_db.TA,  # 3
            r_sb.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
            kwargs["Ksp0"],  # 7
            float(kwargs["kc"]),  # 8
            float(AD),  # 9
            float(abs(kwargs["zsat0"])),  # 10
            float(kwargs["I_caco3"]),  # 11
            float(kwargs["alpha"]),  # 12
            float(abs(kwargs["zsat_min"])),  # 13
            float(abs(kwargs["zmax"])),  # 14
            float(abs(kwargs["z0"])),  # 15
        ],
        return_values=[
            {f"F_{rg.full_name}.DIC": "db_cs2"},
            {f"F_{rg.full_name}.TA": "db_cs2"},
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
            {f"R_{rg.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
            {f"R_{rg.full_name}.CO3": rg.swc.co3},
        ],
        register=rg,
    )

    return ec


def add_carbonate_system_2(**kwargs) -> None:
    """Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        r_sb: list of ReservoirGroup objects in the surface layer
        r_db: list of ReservoirGroup objects in the deep layer
        carbonate_export_fluxes: list of flux objects which must match the
        list of ReservoirGroup objects.
        zsat_min = depth of the upper boundary of the deep box
        z0 = upper depth limit for carbonate burial calculations
        typically zsat_min

    Optional Parameters:

        zsat = initial saturation depth (m)
        zcc = initial carbon compensation depth (m)
        zsnow = initial snowline depth (m)
        zsat0 = characteristic depth (m)
        Ksp0 = solubility product of calcite at air-water interface (mol^2/kg^2)
        kc = heterogeneous rate constant/mass transfer coefficient for calcite dissolution (kg m^-2 yr^-1)
        Ca2 = calcium ion concentration (mol/kg)
        pc = characteristic pressure (atm)
        pg = seawater density multiplied by gravity due to acceleration (atm/m)
        I = dissolvable CaCO3 inventory
        co3 = CO3 concentration (mol/kg)
        Ksp = solubility product of calcite at in situ sea water conditions (mol^2/kg^2)

    """

    from esbmtk import Reservoir, init_carbonate_system_2

    # list of known keywords
    lkk: dict = {
        "r_db": list,  # list of deep reservoirs
        "r_sb": list,  # list of corresponding surface reservoirs
        "carbonate_export_fluxes": list,
        "AD": float,
        "zsat": int,
        "zsat_min": int,
        "zcc": int,
        "zsnow": int,
        "zsat0": int,
        "Ksp0": float,
        "kc": float,
        "Ca2": float,
        "pc": (float, int),
        "pg": (float, int),
        "I_caco3": (float, int),
        "alpha": float,
        "zmax": (float, int),
        "z0": (float, int),
        "Ksp": (float, int),
        # "BM": (float, int),
    }
    # provide a list of absolutely required keywords
    lrk: list[str] = [
        "r_db",
        "r_sb",
        "carbonate_export_fluxes",
        "zsat_min",
        "z0",
    ]

    # we need the reference to the Model in order to set some
    # default values.

    reservoir = kwargs["r_db"][0]
    model = reservoir.mo
    # list of default values if none provided
    lod: dict = {
        "r_sb": [],  # empty list
        "zsat": -3715,  # m
        "zcc": -4750,  # m
        "zsnow": -5000,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "AD": model.hyp.area_dz(-200, -6000),
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/m
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  # dissolveable CaCO3 in mol/m^2
        "zmax": -6000,  # max model depth
        "Ksp": reservoir.swc.Ksp,  # mol^2/kg^2
    }

    # make sure all mandatory keywords are present
    __checkkeys__(lrk, lkk, kwargs)

    # add default values for keys which were not specified
    kwargs = __addmissingdefaults__(lod, kwargs)

    # test that all keyword values are of the correct type
    __checktypes__(lkk, kwargs)

    # establish some shared parameters
    # depths_table = np.arange(0, 6001, 1)
    depths: NDArrayFloat = np.arange(0, 6002, 1, dtype=float)
    r_db = kwargs["r_db"]
    r_sb = kwargs["r_sb"]
    ca2 = r_db[0].swc.ca2
    pg = kwargs["pg"]
    pc = kwargs["pc"]
    z0 = kwargs["z0"]
    Ksp0 = kwargs["Ksp0"]
    # test if corresponding surface reservoirs have been defined
    if len(r_sb) == 0:
        raise ValueError(
            "Please update your call to add_carbonate_system_2 and add the list of of corresponding surface reservoirs"
        )

    # C saturation(z) after Boudreau 2010
    Csat_table: NDArrayFloat = (Ksp0 / ca2) * np.exp((depths * pg) / pc)
    area_table = model.hyp.get_lookup_table(0, -6002)  # area in m^2(z)
    area_dz_table = model.hyp.get_lookup_table_area_dz(0, -6002) * -1  # area'
    AD = model.hyp.area_dz(z0, -6000)  # Total Ocean Area

    for i, rg in enumerate(r_db):  # Setup the virtual reservoirs
        ec = init_carbonate_system_2(
            rg,
            kwargs["carbonate_export_fluxes"][i],
            r_sb[i],
            r_db[i],
            area_table,
            area_dz_table,
            Csat_table,
            AD,
            kwargs,
        )

        register_return_values(ec, rg)
        rg.has_cs2 = True


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
    # print(f"scale = {scale:2e}")
    # print(f"beta = {beta:2e}")
    # print(f"gas_c = {gas_c:2e}")
    # print(f"gas_c_aq = {gas_c_aq:2e}")
    # print(f"liquid_c = {liquid_c:2e}")
    # print(f"liquid_c_l = {liquid_c_l:2e}")
    # print(f"gas_c_h = {gas_c_h:2e}")
    # print(f"f = {f:2e}")
    # print(f"f_l = {f_l:2e}")
    # breakpoint()

    return -f, -f_l


def get_pco2(SW) -> float:
    """Calculate the concentration of pCO2"""

    dic_c: float = SW.dic
    hplus_c: float = SW.hplus
    k1: float = SW.K1
    k2: float = SW.K2
    co2: NDArrayFloat = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c**2)))
    pco2: NDArrayFloat = co2 / SW.K0 * 1e6
    return pco2


def calc_pCO2(
    dic,  # see above why no type hints
    hplus,
    SW,
) -> NDArrayFloat:
    """
    Calculate the concentration of pCO2 as a function of DIC,
    H+, K1 and k2 and returns a numpy array containing
    the pCO2 in uatm at each timestep. Calculations are based off
    equations from Follows, 2006. doi:10.1016/j.ocemod.2005.05.004
    dic: Reservoir  = DIC concentrations in mol/kg
    hplus: Reservoir = H+ concentrations in mol/kg
    SW: Seawater = Seawater object for the model
    it is typically used with a DataField object, e.g.
    pco2 = calc_pCO2(dic,h,SW)

     DataField(name = "SurfaceWaterpCO2",
                       associated_with = reservoir_handle,
                       y1_data = pco2,
                       y1_label = r"pCO_{2}",
                       y1_legend = r"pCO_{2}",
                       )
                       
    Author: T. Tsan

    """

    dic_c: NDArrayFloat = dic.c
    hplus_c: NDArrayFloat = hplus.c

    k1: float = SW.K1
    k2: float = SW.K2

    co2: NDArrayFloat = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c**2)))

    pco2: NDArrayFloat = co2 / SW.K0 * 1e6

    return pco2


def calc_pCO2b(
    dic: NDArrayFloat,
    hplus: NDArrayFloat,
    SW: SeawaterConstants,
) -> NDArrayFloat:
    """
    Same as calc_pCO2, but accepts values/arrays rather than Reservoirs.
    Calculate the concentration of pCO2 as a function of DIC,
    H+, K1 and k2 and returns a numpy array containing
    the pCO2 in uatm at each timestep. Calculations are based off
    equations from Follows, 2006. doi:10.1016/j.ocemod.2005.05.004
    dic:  = DIC concentrations in mol/kg
    hplus: = H+ concentrations in mol/kg
    SW: Seawater = Seawater object for the model
    it is typically used with a DataField object, e.g.
    pco2 = calc_pCO2b(dic,h,SW)
    
    DataField(name = "SurfaceWaterpCO2",
                      associated_with = reservoir_handle,
                      y1_data = pco2b,
                      y1_label = r"pCO_{2}",
                      y1_legend = r"pCO_{2}",
                      )
                       
    """

    dic_c: NDArrayFloat = dic

    hplus_c: NDArrayFloat = hplus

    k1: float = SW.K1
    k2: float = SW.K2

    co2: NDArrayFloat = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c**2)))

    pco2: NDArrayFloat = co2 / SW.K0 * 1e6

    return pco2


# define a transform function to display the Hplus concentration as pH
def phc(m: float) -> float:
    """the reservoir class accepts a plot transform. here we use this to
    display the H+ concentrations as pH. After import, you can use it
    with like this in the reservoir definition

     plot_transform_c=phc,

    """
    import numpy as np

    pH = -np.log10(m)
    return pH


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
    hplus: float = 0.5 * ((gamm - 1) * k1 + sqrt(dummy))

    return hplus - hplus_0
