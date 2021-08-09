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

from numbers import Number
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
from copy import deepcopy, copy
from time import process_time
from numba import njit
from numba.typed import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time
import builtins
from .esbmtk import esbmtkBase, Model, Reservoir, VirtualReservoir, ReservoirGroup, Flux


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


class SeawaterConstants(esbmtkBase):
    """Provide basic seawater properties as a function of T and Salinity.
    Pressure may come at a later stage

    Example:

    Seawater(name="SW",
             model=,
             temperature = optional in C, defaults to 25,
             salinity  = optional in psu, defaults to 35,
             pressure = optional, defaults to 0 bars = 1atm,
             pH = optional, defaults to 8.1,
             units = "mol/l" or "mol/kg"
            )

    useful methods:

    SW.show() will list all known values

    After initialization this class provides access to each value the following way

    instance_name.variable_name

    """

    def __init__(self, **kwargs: Dict[str, str]):

        import math
        from esbmtk import Q_

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "model": Model,
            "salinity": (int, float),
            "temperature": (int, float),
            "pH": (int, float),
            "pressure": Number,
            "register": any,
            "units": (any),
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "units"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "salinity": 35.0,
            "temperature": 25.0,
            "pH": 8.1,
            "pressure": 0,
            "register": "None",
        }

        # validate input and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        u1 = Q_("mol/liter").units
        u2 = Q_("mol/kg").units
        if self.units != u1 and self.units != u2:
            raise ValueError(f"units must be {u1} or {u2}")

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        self.mo: Model = self.model
        self.hplus = 10 ** -self.pH
        self.constants: list = ["K0", "K1", "K2", "KW", "KB", "Ksp", "Ksp0"]
        self.species: list = [
            "dic",
            "ta",
            "ca",
            "co2",
            "hco3",
            "co3",
            "boron",
            "boh4",
            "boh3",
            "oh",
            "ca2",
            "so4",
            "hplus",
        ]

        self.update()

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_name__()

    def update(self, **kwargs: dict) -> None:
        """Update values if necessary"""

        from math import log10
        from esbmtk import Q_

        if kwargs:
            self.lrk: list = []
            self.__validateandregister__(kwargs)

        # update K values and species concentrations according to P, S, and T
        self.__init_std_seawater__()
        self.__init_carbon__()
        self.__init_boron__()
        self.__init_water__()
        self.__init_gasexchange__()
        self.__init_calcite__()
        self.__init_c_fractionation_factors__()

        # get total alkalinity
        self.ca = self.hco3 + 2 * self.co3
        self.ta = self.ca + self.boh4 + self.oh - self.hplus

        # convert to mol/liter if necessary
        if self.units == Q_("1 mole/liter").units:
            cf = (1000 + self.salinity) / 1000

            # constants and species are just names, so we need to
            # retrieve the actual variable first
            for n in self.constants:
                v = getattr(self, n)
                setattr(self, n, v * cf)
            for n in self.species:
                v = getattr(self, n)
                setattr(self, n, v * cf)
        else:
            "\n Constants are mol/kg! \n"

        # update pk values
        for n in self.constants:
            v = getattr(self, n)
            pk = f"p{n.lower()}"
            setattr(self, pk, -log10(v))

    def show(self) -> None:
        """Printout pK values. """

        from math import log10

        for n in self.species:
            v = getattr(self, n)
            print(f"{n} = {v * 1E6:.2f} nmol/l")

        print(f"pH = {-log10(self.hplus):.2f}\n")
        print(f"salinity = {self.salinity:.2f}")
        print(f"temperature = {self.temperature:.2f}\n")

        for n in self.constants:
            K = getattr(self, n)
            pk = getattr(self, f"p{n.lower()}")
            print(f"{n} = {K:.2e}, p{n} = {pk:.2f}")

    def __init_std_seawater__(self) -> None:
        """Provide values for standard seawater. Data after Zeebe and Gladrow
        all values in mol/kg. All values after Zeebe and Gladrow 2001

        """

        self.dic = 0.00204
        self.boron = 0.00042
        self.oh = 0.00001
        self.so4 = 2.7123 / 96
        self.ca2 = 0.01028
        self.Ksp0 = 4.29e-07  # after after Boudreau et al 2010

    def __init_gasexchange__(self) -> None:
        """Initialize constants for gas-exchange processes"""

        self.water_vapor_partial_pressure()
        self.co2_solubility_constant()
        self.o2_solubility_constant()

    def __init_carbon__(self) -> None:
        """Calculate the carbon equilibrium values as function of
        temperature T and salinity S

        """

        from math import exp, log, log10

        T = 273.15 + self.temperature
        S = self.salinity

        # After Weiss 1974
        lnK0: float = (
            93.4517 * 100 / T
            - 60.2409
            + 23.3585 * log(T / 100)
            + S * (0.023517 - 0.023656 * T / 100 + 0.0047036 * (T / 100) ** 2)
        )

        lnk1: float = (
            -2307.1266 / T
            + 2.83655
            - 1.5529413 * log(T)
            + S ** 0.5 * (-4.0484 / T - 0.20760841)
            + S * 0.08468345
            + S ** (3 / 2) * -0.00654208
            + log(1 - 0.001006 * S)
        )

        lnk2: float = (
            -9.226508
            - 3351.6106 / T
            - 0.2005743 * log(T)
            + (-0.106901773 - 23.9722 / T) * S ** 0.5
            + 0.1130822 * S
            - 0.00846934 * S ** 1.5
            + log(1 - 0.001006 * S)
        )

        self.K0: float = exp(lnK0)
        self.K1: float = exp(lnk1)
        self.K2: float = exp(lnk2)

        self.K1 = self.__pressure_correction__("K1", self.K1)
        self.K2 = self.__pressure_correction__("K2", self.K2)

        self.co2 = self.dic / (
            1 + self.K1 / self.hplus + self.K1 * self.K2 / self.hplus ** 2
        )
        self.hco3 = self.dic / (1 + self.hplus / self.K1 + self.K2 / self.hplus)
        self.co3 = self.dic / (
            1 + self.hplus / self.K2 + self.hplus ** 2 / (self.K1 * self.K2)
        )

    def __init_boron__(self) -> None:
        """Calculate the boron equilibrium values as function of
        temperature T and salinity S

        """

        from math import exp, log

        T = 273.15 + self.temperature
        S = self.salinity

        lnkb = (
            (
                -8966.9
                - 2890.53 * S ** 0.5
                - 77.942 * S
                + 1.728 * S ** 1.5
                - 0.0996 * S ** 2
            )
            / T
            + 148.0248
            + 137.1942 * S ** 0.5
            + 1.62142 * S
            - (24.4344 + 25.085 * S ** 0.5 + 0.2474 * S) * log(T)
            + 0.053105 * S ** 0.5 * T
        )

        self.KB = exp(lnkb)
        self.KB = self.__pressure_correction__("KB", self.KB)

        self.boh4 = self.boron * self.KB / (self.hplus + self.KB)
        self.boh3 = self.boron - self.boh4

    def __init_water__(self) -> None:
        """Calculate the water equilibrium values as function of
        temperature T and salinity S

        """

        from math import exp, log

        T = 273.15 + self.temperature
        S = self.salinity

        lnKW = (
            148.96502
            - 13847.27 / T
            - 23.6521 * log(T)
            + (118.67 / T - 5.977 + 1.0495 * log(T)) * S ** 0.5
            - 0.01615 * S
        )
        self.KW = exp(lnKW)
        self.KW = self.__pressure_correction__("KW", self.KW)
        self.oh = self.KW / self.hplus

    def __pressure_correction__(self, n: str, K: float) -> float:
        """Correct K-values for pressure. After Zeebe and Wolf Gladrow 2001

        name = name of K-value, i.e. "K1"
        K = uncorrected value
        T = temperature in Deg C
        P = pressure in atm
        """

        from math import exp, log

        R: float = 83.131
        Tc: float = self.temperature
        T: float = 273.15 + Tc
        P: float = self.pressure
        RT: float = R * T

        A: dict = {}
        A["K1"]: list = [25.50, 0.1271, 0.0, 3.08, 0.0877]
        A["K2"]: list = [15.82, -0.0219, 0.0, -1.13, -0.1475]
        A["KB"]: list = [29.48, 0.1622, -2.6080, 2.84, 0.0]
        A["KW"]: list = [25.60, 0.2324, -3.6246, 5.13, 0.0794]
        A["KS"]: list = [18.03, 0.0466, 0.3160, 4.53, 0.0900]
        A["KF"]: list = [9.780, -0.0090, -0.942, 3.91, 0.054]
        A["Kca"]: list = [48.76, 0.5304, 0.0, 11.76, 0.3692]
        A["Kar"]: list = [46.00, 0.5304, 0.0, 11.76, 0.3692]

        a: list = A[n]

        DV: float = -a[0] + (a[1] * Tc) + (a[2] / 1000 * Tc ** 2)
        DK: float = -a[3] / 1000 + (a[4] / 1000 * Tc) + (0 * Tc ** 2)

        # print(f"DV = {DV}")
        # print(f"DK = {DK}")
        # print(f"log k= {log(K)}")

        lnkp: float = -(DV / RT) * P + (0.5 * DK / RT) * P ** 2 + log(K)
        # print(lnkp)

        return exp(lnkp)

    def water_vapor_partial_pressure(self) -> None:
        """Calculate the water vapor partial pressure at sealevel (1 atm) as
        a function of temperature and salinity. Eq. Weiss and Price 1980
        doi:10.1016/0304-4203(80)90024-9

        Since we assume that we only use this expression at sealevel,
        we drop the pressure term

        The result is in p/1atm (i.e., a percentage)
        """

        T = self.temperature + 273.15
        S = self.salinity

        self.p_H2O = np.exp(
            24.4543 - 67.4509 * (100 / T) - 4.8489 * np.log(T / 100) - 0.000544 * S
        )

    def co2_solubility_constant(self) -> None:
        """Calculate the solubility of CO2 at a given temperature and salinity.
        Coefficients after Sarmiento and Gruber 2006 which includes
        corrections for CO2 to correct for non ideal gas behavior

        Parameters Ai & Bi from Tab 3.2.2 in  Sarmiento and Gruber 2006

        The result is in mol/(m^3 * atm)
        """

        # Calculate the volumetric solubility function F_A in mol/l
        S = self.salinity  # unitless
        T = 273.15 + self.temperature  # C
        A1 = -160.7333
        A2 = 215.4152
        A3 = 89.892
        A4 = -1.47759
        B1 = 0.029941
        B2 = -0.027455
        B3 = 0.0053407

        # F in mol/(l * atm)
        F = self.calc_solubility_term(S, T, A1, A2, A3, A4, B1, B2, B3)

        # correct for water vapor partial pressure
        self.SA_co2 = F / (1 - self.p_H2O)  # mol/(m^3 * atm)

    def o2_solubility_constant(self) -> None:
        """Calculate the solubility of CO2 at a given temperature and salinity. Coefficients
        after Sarmiento and Gruber 2006 which includes corrections for CO2 to correct for non
        ideal gas behavior

        Parameters Ai & Bi from Tab 3.2.2 in  Sarmiento and Gruber 2006

        The result is in mol/(m^3 atm)
        """

        # Calculate the volumetric solubility function F_A in mol/l/m^3
        S = self.salinity  # unit less
        T = 273.15 + self.temperature  # in C
        A1 = -58.3877
        A2 = 85.8079
        A3 = 23.8439
        A4 = 0
        B1 = -0.034892
        B2 = 0.015568
        B3 = -0.0019387

        b = self.calc_solubility_term(S, T, A1, A2, A3, A4, B1, B2, B3)

        # and convert from bunsen coefficient to solubility
        VA = 22.4136  # after Sarmiento & Gruber 2006
        self.SA_o2 = b / VA

    def calc_solubility_term(self, S, T, A1, A2, A3, A4, B1, B2, B3) -> float:
        ln_F = (
            A1
            + A2 * (100 / T)
            + A3 * np.log(T / 100)
            + A4 * (T / 100) ** 2
            + S * (B1 + B2 * (T / 100) + B3 * (T / 100) ** 2)
        )
        F = np.exp(ln_F) * 1000  # to get mol/(m^3 atm)

        return F

    def __init_calcite__(self) -> None:
        """Calculate Calcite solubility as a function of pressure following
        Fig 1 in in Boudreau et al, 2010, https://doi.org/10.1029/2009gl041847

        Note that this equation assumes an idealized ocean temperature profile.
        So it cannot be applied to a warm ocean

        """

        self.Ksp = 4.3513e-7 * np.exp(0.0019585 * self.pressure)

    def __init_c_fractionation_factors__(self):
        """Calculate the fractionation factors for the various carbon species transitions.
        After Zeebe and Gladrow, 2001, CHapter 3.2.3

        e = (a -1) * 1E3

        and

        a =  1 + e / 1E3

        where the subscripts denote:

        g = gaseous CO2
        d = dissolved CO2
        b = bicarbonate ion
        c = carbonate ion

        """

        T = 273.15 + self.temperature

        # CO2g versus HCO3
        self.e_gb: float = -9483 / T + 23.89
        self.a_gb: float = 1 + self.e_gb / 1000

        # CO2aq versus CO2g
        self.e_dg: float = -373 / T + 0.19
        self.a_dg: float = 1 + self.e_dg / 1000

        # CO2aq versus HCO3
        self.e_db: float = -9866 / T + 24.12
        self.a_db: float = 1 + self.e_db / 1000

        # CO32- versus HCO3
        self.e_cb: float = -867 / T + 2.52
        self.a_cb: float = 1 + self.e_cb / 1000

        # kinetic fractionation during gas exchange
        # parameters after Zhang et. al.1995
        # https://doi.org/10.1016/0016-7037(95)91550-D
        m = 0.14 / 16
        c = m * 5 + 0.95
        self.e_u: float = self.temperature * m - c
        self.a_u: float = 1 + self.e_u / 1000


def calc_pCO2(
    dic: Union[Reservoir, VirtualReservoir],
    hplus: Union[Reservoir, VirtualReservoir],
    SW: SeawaterConstants,
) -> Union[NDArray, Float]:

    """
    Calculate the concentration of pCO2 as a function of DIC,
    H+, K1 and k2 and returns a numpy array containing
    the pCO2 in uatm at each timestep. Calculations are based off
    equations from Follows, 2006. doi:10.1016/j.ocemod.2005.05.004
    dic: Reservoir  = DIC concentrations in mol/liter
    hplus: Reservoir = H+ concentrations in mol/liter
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

    dic_c: [NDArray, Float] = dic.c
    hplus_c: [NDArray, Float] = hplus.c

    k1: float = SW.K1
    k2: float = SW.K2

    co2: [NDArray, Float] = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c ** 2)))

    pco2: [NDArray, Float] = co2 / SW.K0 * 1e6

    return pco2


def calc_pCO2b(
    dic: Union[float, NDArray],
    hplus: Union[float, NDArray],
    SW: SeawaterConstants,
) -> Union[NDArray, Float]:

    """
    Same as calc_pCO2, but accepts values/arrays rather than Reservoirs.
    Calculate the concentration of pCO2 as a function of DIC,
    H+, K1 and k2 and returns a numpy array containing
    the pCO2 in uatm at each timestep. Calculations are based off
    equations from Follows, 2006. doi:10.1016/j.ocemod.2005.05.004
    dic:  = DIC concentrations in mol/liter
    hplus: = H+ concentrations in mol/liter
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

    dic_c: [NDArray, Float] = dic

    hplus_c: [NDArray, Float] = hplus

    k1: float = SW.K1
    k2: float = SW.K2

    co2: [NDArray, Float] = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c ** 2)))

    pco2: [NDArray, Float] = co2 / SW.K0 * 1e6

    return pco2


@njit(parallel=False, fastmath=True, error_model="numpy")
def calc_carbonates_1(i: int, input_data: List, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations with the format of
    [d1, d2, d3, d4, d5] where each variable corresponds to
    [H+, CA, HCO3, CO3, CO2(aq)], respectively, at the ith time-step of the model.

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/L

    Calculations are based off equations from Follows, 2006.
    doi:10.1016/j.ocemod.2005.05.004

    Example: see carbonate_system_1 in utility_function.py

    Author: M. Niazi & T. Tsan, 2021

    """
    dic: float = input_data[0][i - 1]
    ta: float = input_data[1][i - 1]

    # calculates carbonate alkalinity (ca) based on H+ concentration from the
    # previous time-step
    # hplus: float = input_data[2][i - 1]
    hplus: float = vr_data[0][i - 1]

    k1 = params[0]
    k2 = params[1]
    KW = params[2]
    KB = params[3]
    boron = params[4]
    # hplus = params[5 ]
    ca2 = params[6]
    ksp = params[7]
    ksp0 = params[8]

    # ca
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg
    # hplus
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy ** 0.5))
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

    # co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    co2aq: float = dic - hco3 - co3
    omega: float = ca2 * co3 / ksp

    vr_data[0][i] = hplus
    vr_data[1][i] = ca
    vr_data[2][i] = hco3
    vr_data[3][i] = co3
    vr_data[4][i] = co2aq
    vr_data[5][i] = omega


@njit(fastmath=True, error_model="numpy")
def calc_carbonates_2(i: int, input_data: List, vr_data: List, params: List) -> None:
    """Calculates and returns the carbonate concentrations and carbonate compensation
    depth (zcc) at the ith time-step of the model.

    The function assumes that vr_data will be in the following order:
        [H+, CA, HCO3, CO3, CO2(aq), zsat, zcc, zsnow]

    LIMITATIONS:
    - This in used in conjunction with ExternalCode objects!
    - Assumes all concentrations are in mol/L
    - Assumes your Model is in mol/L ! Otherwise, DIC and TA updating will not
    be correct.

    Calculations are based off equations from Follows, 2006 and
    Boudreau et al 2010,
    doi:10.1016/j.ocemod.2005.05.004
    http://dx.doi.org/10.1029/2009GB003654

    See carbon_system_2 in utility_functions.py on how to call this function

    Author: M. Niazi & T. Tsan, 2021

    """

    # get constants
    k1 = params[0]
    k2 = params[1]
    KW = params[2]
    KB = params[3]
    boron = params[4]
    ksp0 = params[5]
    kc = params[6]
    sa = params[7]
    volume = params[8]
    AD = params[9]
    zsat0 = int(abs(params[10]))
    ca2 = params[11]
    dt = params[12]
    pc = params[13]
    pg = params[14]
    I_caco3 = params[15]
    alpha = params[16]
    zsat_min = int(abs(params[17]))
    zmax = int(abs(params[18]))
    z0 = int(abs(params[19]))
    ksp = params[20]

    # get lookup up tables
    depth_area_table: NDArray = input_data[7]  # depth look-up table
    area_dz_table: NDArray = input_data[8]  # area_dz table
    Csat_table: NDArray = input_data[9]  # Csat table

    # get data from previous time steps
    # input data is data which calculated elsewhere
    # vr_data is data which is calculated in this module
    dic: float = input_data[3][i - 1]  # DIC concentration [mol/l]
    ta: float = input_data[5][i - 1]  # TA concentration [mol/l]
    B: float = input_data[6][i - 1]  # Carbonate Export Flux [mol/yr]
    hplus: float = vr_data[0][i - 1]  # H+ concentration [mol/l]
    zsat = vr_data[5][i - 1]
    zcc = vr_data[6][i - 1]
    zsnow = vr_data[7][i - 1]

    # calc carbonate alkalinity based t-1
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)
    fg: float = hplus - oh - boh4
    ca: float = ta + fg

    # calculate carbon speciation at t
    # The following equations are after Follows et al. 2006
    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))

    hplus: float = 0.5 * ((gamm - 1) * k1 + (dummy ** 0.5))
    # co3: float = dic / (1 + (hplus / k2) + ((hplus ** 2) / (k1 * k2)))
    hco3: float = dic / (1 + (hplus / k1) + (k2 / hplus))
    co3: float = (ca - hco3) / 2
    # DIC = hco3 + co3 + co2 + H2CO3 The last term is however rather
    # small, so it may be ok to simply write co2aq = dic - hco3 + co3.
    co2aq: float = dic - co3 - hco3
    # co2aq: float = dic / (1 + (k1 / hplus) + (k1 * k2 / (hplus ** 2)))
    omega: float = (ca2 * co3) / ksp

    # ---------- compute critical depth intervals eq after  Boudreau (2010)
    # all depths will be positive to facilitate the use of lookup_tables

    # prevent co3 from becoming zero
    if co3 <= 0:
        co3 = 1e-16

    zsat = int(max((zsat0 * np.log(ca2 * co3 / ksp0)), zsat_min))  # eq2
    if zsat < zsat_min:
        zsat = int(zsat_min)

    zcc = int(zsat0 * np.log(B * ca2 / (ksp0 * AD * kc) + ca2 * co3 / ksp0))  # eq3

    # ---- Get fractional areas
    B_AD = B / AD

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
        # print(zcc, zsnow)
        zsnow = zsnow - BPDC / (area_dz_table[int(zsnow)] * I_caco3) * dt
        # print(zcc, zsnow)
        # print()
    else:  # zcc > zsnow
        # there is no carbonate below zsnow, so  BPDC = 0
        zsnow = zcc
        # dummy values for testing purposes; will be removed later
        zsnow_dt = 0
        BPDC = 0

    # BD & F_burial
    BD: float = BDS + BCC + BNS + BPDC
    Fburial = B - BD

    # ----------------------Update DIC and TA Reservoirs ---------------
    # note that DIC and TA have already been computed. So we use the
    # i, rather than i -1
    # TA mass [mol]
    input_data[4][i] = input_data[4][i] - 2 * Fburial * dt
    # Ta concentration [mol/l]
    input_data[5][i] = input_data[4][i] / volume

    # DIC isotopes assuming no fractionation, so no need to update delta
    r = input_data[1][i] / input_data[0][i]  # C12/C ratio
    input_data[0][i] = input_data[0][i] - Fburial * dt  # DIC
    input_data[1][i] = input_data[1][i] - Fburial * dt * r  # 12C
    input_data[2][i] = input_data[0][i] - input_data[1][i]  # 13C
    input_data[3][i] = input_data[0][i] / volume  # DIC concentration

    # copy results into datafields
    vr_data[0][i] = hplus
    vr_data[1][i] = ca
    vr_data[2][i] = hco3
    vr_data[3][i] = co3
    vr_data[4][i] = co2aq
    vr_data[5][i] = zsat
    vr_data[6][i] = zcc
    vr_data[7][i] = zsnow
    vr_data[8][i] = Fburial
    # temporary added values for testing
    vr_data[9][i] = B
    vr_data[10][i] = BNS
    vr_data[11][i] = BDS_under
    vr_data[12][i] = BDS_resp
    vr_data[13][i] = BDS
    vr_data[14][i] = BCC
    vr_data[15][i] = BPDC
    vr_data[16][i] = BD
    vr_data[17][i] = 0.0
    vr_data[18][i] = 0.0
    vr_data[19][i] = omega
