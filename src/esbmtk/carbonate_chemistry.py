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
from math import log, sqrt
import numpy as np
import numpy.typing as npt
from numba import njit
from esbmtk import ExternalCode, Reservoir, Flux, SeawaterConstants
from esbmtk.utility_functions import (
    __checkkeys__,
    __addmissingdefaults__,
    __checktypes__,
    register_return_values,
    check_for_quantity,
)


if tp.TYPE_CHECKING:
    from .esbmtk import Q_

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


# @njit(fastmath=True)
def get_hplus(dic, ta, h0, boron, K1, K1K2, KW, KB) -> float:
    """Calculate H+ concentration based on a previous estimate
    [H+]. After Follows et al. 2006,
    doi:10.1016/j.ocemod.2005.05.004

    :param dic: DIC in mol/kg
    :param ta: TA in mol/kg
    :param h0: initial guess for H+ mol/kg
    :param boron: boron concentration
    :param K1: Ksp1
    :param K1K2: Ksp1 * Ksp2
    :param KW: K_water
    :param KB: K_boron

    :returns H: new H+ concentration in mol/kg
    """
    oh = KW / h0
    boh4 = boron * KB / (h0 + KB)
    fg = h0 - boh4 - oh
    cag = ta + fg
    gamm = dic / cag
    dummy = (1 - gamm) ** 2 * K1**2 - 4.0 * K1K2 * (1.0 - 2.0 * gamm)

    return 0.5 * ((gamm - 1.0) * K1 + sqrt(dummy))


# @njit(fastmath=True)
def carbonate_system_1(dic, ta, hplus_0, co2aq_0, p) -> tuple:
    """Calculates and returns the H+ and carbonate alkalinity concentrations
     for the given reservoirgroup

    :param dic: float with the dic concentration
    :param ta: float with the ta concentration
    :param hplus_0: float with the H+ concentration
    :param co2aq_0: float with the [CO2]aq concentration
    :param p: tuple with the parameter list
    :returns:  dCdt_Hplus, dCdt_co2aq

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654
    Follows, 2006, doi:10.1016/j.ocemod.2005.05.004
    """

    k1, k2, k1k2, KW, KB, boron, isotopes = p
    if isotopes:  # dic = (x1, x2)
        dic = dic[0]

    hplus = get_hplus(dic, ta, hplus_0, boron, k1, k1k2, KW, KB)
    co2aq = dic / (1 + k1 / hplus + k1k2 / hplus**2)
    dCdt_Hplus = hplus - hplus_0
    dCdt_co2aq = co2aq - co2aq_0

    return dCdt_Hplus, dCdt_co2aq


def init_carbonate_system_1(rg: Reservoir):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: tp.List = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Species
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import ExternalCode

    p = (
        rg.swc.K1,
        rg.swc.K2,
        rg.swc.K1K2,
        rg.swc.KW,
        rg.swc.KB,
        rg.swc.boron,
        rg.DIC.isotopes,
    )
    ec = ExternalCode(
        name="cs",
        species=rg.mo.Carbon.CO2,
        function=carbonate_system_1,
        fname="carbonate_system_1",
        ftype="std",
        function_input_data=[rg.DIC, rg.TA, "Hplus", "CO2aq"],
        function_params=p,
        register=rg,
        return_values=[
            {f"R_{rg.full_name}.Hplus": rg.swc.hplus},
            {f"R_{rg.full_name}.CO2aq": rg.swc.co2aq},
        ],
    )
    # rg.mo.lpc_f.append(ec.fname)
    rg.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend

    return ec


def add_carbonate_system_1(rgs: tp.List):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: tp.List = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Species
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """
    from esbmtk import init_carbonate_system_1

    for rg in rgs:
        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            ec = init_carbonate_system_1(rg)
            register_return_values(ec, rg)
            rg.has_cs1 = True
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")


# @njit(fastmath=True)
def carbonate_system_2(
    CaCO3_export: float,  # 3 CaCO3 export flux as DIC
    dic_t_db: float | tuple,  # 4 DIC in the deep box
    ta_db: float,  # 5 TA in the deep box
    dic_t_sb: float | tuple,  # 6 [DIC] in the surface box
    hplus_0: float,  # 8 hplus in the deep box at t-1
    zsnow: float,  # 9 snowline in meters below sealevel at t-1
    p,
) -> tuple:
    """Calculates and returns the fraction of the carbonate rain that is
    dissolved an returned back into the ocean. This functions returns:

    DIC_burial, DIC_burial_l, Hplus, zsnow

    LIMITATIONS:
    - Assumes all concentrations are in mol/kg
    - Assumes your Model is in mol/kg

    Calculations are based off equations from:
    Boudreau et al., 2010, https://doi.org/10.1029/2009GB003654

    """
    sp, cp, area_table, area_dz_table, Csat_table = p
    ksp0, kc, AD, zsat0, I_caco3, alpha, zsat_min, zmax, z0 = cp
    k1, k2, k1k2, KW, KB, ca2, boron, isotopes = sp

    if isotopes:
        dic_db, dic_db_l = dic_t_db
        dic_sb, dic_sb_l = dic_t_sb
    else:
        dic_db = dic_t_db
        dic_sb = dic_t_sb

    hplus = get_hplus(dic_db, ta_db, hplus_0, boron, k1, k1k2, KW, KB)
    co3 = max(dic_db / (1 + hplus / k2 + hplus**2 / k1k2), 3.7e-05)

    """ --- Compute critical depth intervals eq after  Boudreau (2010) ---
   All depths will be positive to facilitate the use of lookup_tables.
   Note that these tables are different than the hyspometry data tables
   that expect positive and negative numbers.
    """
    zsat = int(zsat0 * log(ca2 * co3 / ksp0))
    zsat = min(zmax, max(zsat_min, zsat))
    zcc = int(
        zsat0 * log(CaCO3_export * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0)
    )  # eq3
    zcc = min(zmax, max(zsat_min, zcc))

    B_AD = CaCO3_export / AD  # get fractional areas
    A_z0_zsat = area_table[z0] - area_table[zsat]
    A_zsat_zcc = area_table[zsat] - area_table[zcc]
    A_zcc_zmax = area_table[zcc] - area_table[zmax]
    # ------------------------Calculate Burial Fluxes----------------------------- #
    BCC = A_zcc_zmax * B_AD
    BNS = alpha * A_z0_zsat * B_AD
    diff_co3 = Csat_table[zsat:zcc] - co3
    area_sat_cc = area_dz_table[zsat:zcc]
    BDS_under = kc * area_sat_cc.dot(diff_co3)
    BDS_resp = alpha * (A_zsat_zcc * B_AD - BDS_under)
    BDS = BDS_under + BDS_resp

    # sediment dissolution if zcc is deeper than snowline
    if zsnow <= zcc:  # reset zsnow
        dzdt_zsnow = abs(zsnow - zcc)
        BPDC = 0
        zsnow = zcc
    else:  # integrate saturation difference over area
        if zsnow > zmax:  # limit zsnow to ocean depth
            zsnow = zmax

        diff: NDArrayFloat = Csat_table[zcc : int(zsnow)] - co3
        area_cc_snow: NDArrayFloat = area_dz_table[zcc : int(zsnow)]
        BPDC = max(0, kc * area_cc_snow.dot(diff))
        dzdt_zsnow = -BPDC / (area_dz_table[int(zsnow)] * I_caco3)

    """ CACO3_export is the flux of CaCO3 into the box.
    Boudreau's orginal approach is as follows.
    CACO3_export = B_diss + Fburial
    However, the model should use the bypass option and leave all flux
    calculations to the carbonate_system code. As such, ignore the burial flux
    (since it was never added), and only add the fraction of the input flux
    that dissolves back into the box
    """
    F_diss = BDS + BCC + BNS + BPDC
    dCdt_Hplus = hplus - hplus_0

    """ The isotope ratio of the dissolution flux is determined by the delta
    value of the sediments we are dissolving, and the delta of the carbonate rain.
    The currrent code, assumes that both are the same.
    """
    if isotopes:
        F_diss_l = F_diss * dic_sb_l / dic_sb
        rv = (F_diss, F_diss_l, F_diss * 2, dCdt_Hplus, dzdt_zsnow)
    else:
        rv = (F_diss, F_diss * 2, dCdt_Hplus, dzdt_zsnow)

    return rv


def init_carbonate_system_2(
    export_flux: Flux,
    r_sb: Reservoir,  # Surface box
    r_db: Reservoir,  # deep box
    kwargs: dict,
):
    """Initialize a carbonate system 2 instance.
    Note that the current implmentation assumes that the export flux is
    the total export flux over surface area of the mixed layer, i.e.,
    the sediment area between z0 and zmax

    Parameters
    ----------
    export_flux : Flux
        CaCO3 export flux from the surface box
    r_sb : Reservoir
        Reservoir instance of the surface box
    box r_db : Reservoir
        Reservoir instance of the deep box
    kwargs : dict
        dictionary of keyword value pairs


    """

    AD = r_sb.mo.hyp.area_dz(kwargs["z0"], kwargs["zmax"])
    s = r_db.swc
    sp = (s.K1, s.K2, s.K1K2, s.KW, s.KB, s.ca2, s.boron, r_sb.DIC.isotopes)
    cp = (
        kwargs["Ksp0"],  # 7
        float(kwargs["kc"]),  # 8
        AD,  # 9
        int(abs(kwargs["zsat0"])),  # 10
        kwargs["I_caco3"],  # 11
        kwargs["alpha"],  # 12
        int(abs(kwargs["zsat_min"])),  # 13
        int(abs(kwargs["zmax"])),  # 14
        int(abs(kwargs["z0"])),  # 15
    )

    ec = ExternalCode(
        name="cs",
        species=r_sb.mo.Carbon.CO2,
        function=carbonate_system_2,
        fname="carbonate_system_2",
        ftype="needs_flux",
        r_s=r_sb,  # source (RG) of CaCO3 flux,
        r_d=r_db,  # sink (RG) of CaCO3 flux,
        function_input_data=[
            export_flux,  # 1
            r_db.DIC,  # 2
            r_db.TA,  # 3
            r_sb.DIC,  # 4
            "Hplus",  # 5
            "zsnow",  # 6
        ],
        function_params=(
            sp,
            cp,
            r_db.mo.area_table,
            r_db.mo.area_dz_table,
            r_db.mo.Csat_table,
        ),
        return_values=[
            {f"F_{r_db.full_name}.DIC": "db_cs2"},
            {f"F_{r_db.full_name}.TA": "db_cs2"},
            {f"R_{r_db.full_name}.Hplus": r_db.swc.hplus},
            {f"R_{r_db.full_name}.zsnow": float(abs(kwargs["zsnow"]))},
        ],
        register=r_db,
    )
    r_db.mo.lpc_f.append(ec.fname)  # list of function to be imported in ode backend

    return ec


def add_carbonate_system_2(**kwargs) -> None:
    """Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        r_sb: tp.List of Reservoir objects in the surface layer
        r_db: tp.List of Reservoir objects in the deep layer
        carbonate_export_fluxes: tp.List of flux objects which must match the
        list of Reservoir objects.
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

    # list of known keywords
    lkk: dict = {
        "r_db": tp.List,  # list of deep reservoirs
        "r_sb": tp.List,  # list of corresponding surface reservoirs
        "carbonate_export_fluxes": tp.List,
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
    lrk: tp.List[str] = [
        "r_db",
        "r_sb",
        "carbonate_export_fluxes",
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
        "zsnow": -4750,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/km
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  # dissolveable CaCO3 in mol/m^2
        "zmax": -10999,  # max model depth
        "Ksp": reservoir.swc.Ksp_ca,  # mol^2/kg^2
    }
    __checkkeys__(lrk, lkk, kwargs)
    kwargs = __addmissingdefaults__(lod, kwargs)
    __checktypes__(lkk, kwargs)

    if "zsat_min" not in kwargs:
        kwargs["zsat_min"] = kwargs["z0"]

    r_db = kwargs["r_db"]
    r_sb = kwargs["r_sb"]
    pg = kwargs["pg"]
    pc = kwargs["pc"]
    zmax = abs(int(kwargs["zmax"]))
    # test if corresponding surface reservoirs have been defined
    if len(r_sb) == 0:
        raise ValueError(
            "Please update your call to add_carbonate_system_2 and add\
            the list of corresponding surface reservoirs"
        )

    # check if we already have the hypsometry and saturation tables
    if not hasattr(model, "area_table"):
        depth_range = np.arange(0, zmax, 1, dtype=float)  # mbsl
        model.area_table = model.hyp.get_lookup_table_area()  # area in m^2(z)
        model.area_dz_table = model.hyp.get_lookup_table_area_dz() * -1  # area_dz
        model.Csat_table = (reservoir.swc.Ksp0 / reservoir.swc.ca2) * np.exp(
            (depth_range * pg) / pc
        )

    for i, rg in enumerate(r_db):  # Setup the virtual reservoirs
        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            rg.swc.update_parameters()
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")

        ec = init_carbonate_system_2(
            kwargs["carbonate_export_fluxes"][i],
            r_sb[i],
            r_db[i],
            kwargs,
        )

        register_return_values(ec, rg)
        rg.has_cs2 = True


def get_pco2(SW) -> float:
    """Calculate the concentration of pCO2"""

    dic_c: float = SW.dic
    hplus_c: float = SW.hplus
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
