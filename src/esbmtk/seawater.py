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
from .esbmtk_base import esbmtkBase

if tp.TYPE_CHECKING:
    from .esbmtk import Reservoir, Model
    from .extended_classes import ReservoirGroup


# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]
    
class SeawaterConstants(esbmtkBase):
    """Provide basic seawater properties as a function of T, P and Salinity.
    This module does not reca

    Example:

    Seawater(name="SW",
             register=M # model handle
             temperature = optional in C, defaults to 25,
             salinity  = optional in psu, defaults to 35,
             pressure = optional, defaults to 0 bars = 1atm,
             pH = optional, defaults to 8.1,
            )

    Results are always in mol/kg

    Acess the values "dic", "ta", "ca", "co2", "hco3",
    "co3", "boron", "boh4", "boh3", "oh", "ca2", "so4","hplus",
    as SW.co3 etc.

    This method also provides "K0", "K1", "K2", "KW", "KB", "Ksp",
    "Ksp0", "KS", "KF" and their corresponding pK values, as well
    as the density for the given (P/T/S conditions)

    useful methods:

    SW.show() will list all known values

    After initialization this class provides access to each value the following way

    instance_name.variable_name

    """

    def __init__(self, **kwargs: dict[str, str]):

        from esbmtk import Model, Reservoir, ReservoirGroup

        self.defaults: dict[list[any, tuple]] = {
            "name": ["None", (str)],
            "salinity": [35.0, (int, float)],
            "temperature": [25.0, (int, float)],
            "pH": [8.1, (int, float)],
            "pressure": [0, (int, float)],
            "register": ["None", (Model, Reservoir, ReservoirGroup)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        # self.mo: Model = self.model
        self.hplus = 10**-self.pH
        self.constants: list = ["K0", "K1", "K2", "KW", "KB", "Ksp", "Ksp0", "KS", "KF"]
        self.species: list = [
            "dic",
            "ta",
            "ca",
            "co2",
            "co2aq",
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

        self.update_parameters()

        if self.register != "None":
            self.__register_name_new__()

    def update_parameters(self, **kwargs: dict) -> None:
        """Update values if necessary"""

        if kwargs:
            self.__initialize_keyword_variables__(kwargs)

        # update K values and species concentrations according to P, S, and T
        self.__get_density__()
        self.__init_std_seawater__()
        self.__init_bisulfide__()
        self.__init_hydrogen_floride__()
        self.__init_carbon__()
        self.__init_boron__()
        self.__init_water__()
        self.__init_gasexchange__()
        self.__init_calcite__()
        self.__init_c_fractionation_factors__()

        # get total alkalinity
        self.ca = self.hco3 + 2 * self.co3
        self.ta = self.ca + self.boh4 + self.oh - self.hplus

    def show(self) -> None:
        """Printout constants"""

        from math import log10

        for n in self.species:
            v = getattr(self, n)
            print(f"{n} = {v * 1E6:.2f} nmol/kg")

        print()
        # print(f"pCO2 = {get_pco2(self):.2e}")
        print(f"pH = {-log10(self.hplus):.2f}")
        print(f"salinity = {self.salinity:.2f}")
        print(f"temperature = {self.temperature:.2f}\n")

        for n in self.constants:
            K = getattr(self, n)  # get K value
            pk = f"p{n.lower()}"  # get K name
            print(f"{n} = {K:.2e}, {pk} = {-log10(K):.2f}")

    def __get_density__(self):
        """Calculate seawater density as function of temperature,
        pressure and salinity in kg/m^3. Shamelessy copied
        from R. Zeebes equic.m mathlab file.

        TC = temp in C
        P = pressure
        S = salinity
        """

        TC = self.temperature
        P = self.pressure
        S = self.salinity
        # density of pure water
        rhow = (
            999.842594
            + 6.793952e-2 * TC
            - 9.095290e-3 * TC**2
            + 1.001685e-4 * TC**3
            - 1.120083e-6 * TC**4
            + 6.536332e-9 * TC**5
        )

        # density of of seawater at 1 atm, P=0
        A = (
            8.24493e-1
            - 4.0899e-3 * TC
            + 7.6438e-5 * TC**2
            - 8.2467e-7 * TC**3
            + 5.3875e-9 * TC**4
        )
        B = -5.72466e-3 + 1.0227e-4 * TC - 1.6546e-6 * TC**2
        C = 4.8314e-4
        rho0 = rhow + A * S + B * S ** (3 / 2) + C * S**2

        """Secant bulk modulus of pure water is the average change in
        pressure divided by the total change in volume per unit of
        initial volume.
        """
        Ksbmw = (
            19652.21
            + 148.4206 * TC
            - 2.327105 * TC**2
            + 1.360477e-2 * TC**3
            - 5.155288e-5 * TC**4
        )
        # Secant bulk modulus of seawater at 1 atm
        Ksbm0 = (
            Ksbmw
            + S * (54.6746 - 0.603459 * TC + 1.09987e-2 * TC**2 - 6.1670e-5 * TC**3)
            + S ** (3 / 2) * (7.944e-2 + 1.6483e-2 * TC - 5.3009e-4 * TC**2)
        )
        # Secant modulus of seawater at S,T,P
        Ksbm = (
            Ksbm0
            + P
            * (3.239908 + 1.43713e-3 * TC + 1.16092e-4 * TC**2 - 5.77905e-7 * TC**3)
            + P * S * (2.2838e-3 - 1.0981e-5 * TC - 1.6078e-6 * TC**2)
            + P * S ** (3 / 2) * 1.91075e-4
            + P * P * (8.50935e-5 - 6.12293e-6 * TC + 5.2787e-8 * TC**2)
            + P**2 * S * (-9.9348e-7 + 2.0816e-8 * TC + 9.1697e-10 * TC**2)
        )
        # Density of seawater at S,T,P in kg/m^3
        self.density = rho0 / (1.0 - P / Ksbm)

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

    def __init_hydrogen_floride__(self) -> None:
        """Bisulfide ion concentration after Dickson 1994, cf.
        Zeebe and Gladrow 2001, p 260

        """

        import numpy as np

        T = 273.15 + self.temperature
        S = self.salinity
        I = (19.924 * S) / (1000 - 1.005 * S)

        lnKF = (
            1590.2 / T
            - 12.641
            + 1.525 * I**0.5
            + np.log(1 - 0.001005 * S)
            + np.log(1 + self.ST / self.KS)
        )

        self.KF = np.exp(lnKF)
        self.FT = 7e-5 * self.salinity / 35

    def __init_bisulfide__(self) -> None:
        """Bisulfide ion concentration after Dickson 1994, cf.
        Zeebe and Gladrow 2001, p 260

        """

        import numpy as np

        T = 273.15 + self.temperature
        S = self.salinity
        I = (19.924 * S) / (1000 - 1.005 * S)
        lnKS = (
            -4276.1 / T
            + 141.328
            - 23.093 * np.log(T)
            + (-13856 / T + 324.57 - 47.986 * np.log(T)) * I**0.5
            + (35474 / T - 771.54 + 114.723 * np.log(T)) * I
            - 2698 / T * I**1.5
            + 1776 / T * I**2
            + np.log(1 - 0.001005 * S)
        )

        self.KS = np.exp(lnKS)
        self.ST = self.so4 * self.salinity / 35

    def __init_gasexchange__(self) -> None:
        """Initialize constants for gas-exchange processes"""

        self.water_vapor_partial_pressure()
        self.co2_solubility_constant()
        self.o2_solubility_constant()

    def __init_carbon__(self) -> None:
        """Calculate the carbon equilibrium values as function of
        temperature T and salinity S

        """

        from math import exp, log

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
            + S**0.5 * (-4.0484 / T - 0.20760841)
            + S * 0.08468345
            + S ** (3 / 2) * -0.00654208
            + log(1 - 0.001006 * S)
        )

        lnk2: float = (
            -9.226508
            - 3351.6106 / T
            - 0.2005743 * log(T)
            + (-0.106901773 - 23.9722 / T) * S**0.5
            + 0.1130822 * S
            - 0.00846934 * S**1.5
            + log(1 - 0.001006 * S)
        )

        self.K0: float = exp(lnK0)
        self.K1: float = exp(lnk1)
        self.K2: float = exp(lnk2)

        self.K1 = self.__pressure_correction__("K1", self.K1)
        self.K2 = self.__pressure_correction__("K2", self.K2)

        self.K1K1: float = self.K1 * self.K1
        self.K1K2: float = self.K1 * self.K2

        # self.K_l : list = [self.K0, self.K1, self.K2, self.K1K1, self.K1K2]

        self.co2 = self.dic / (
            1 + self.K1 / self.hplus + self.K1 * self.K2 / self.hplus**2
        )
        self.hco3 = self.dic / (1 + self.hplus / self.K1 + self.K2 / self.hplus)
        self.co3 = self.dic / (
            1 + self.hplus / self.K2 + self.hplus**2 / (self.K1 * self.K2)
        )
        self.co2aq = self.dic / (
            1 + (self.K1 / self.hplus) + (self.K1 * self.K2 / (self.hplus**2))
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
                - 2890.53 * S**0.5
                - 77.942 * S
                + 1.728 * S**1.5
                - 0.0996 * S**2
            )
            / T
            + 148.0248
            + 137.1942 * S**0.5
            + 1.62142 * S
            - (24.4344 + 25.085 * S**0.5 + 0.2474 * S) * log(T)
            + 0.053105 * S**0.5 * T
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
            + (118.67 / T - 5.977 + 1.0495 * log(T)) * S**0.5
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

        DV: float = -a[0] + (a[1] * Tc) + (a[2] / 1000 * Tc**2)
        DK: float = -a[3] / 1000 + (a[4] / 1000 * Tc) + (0 * Tc**2)

        # print(f"DV = {DV}")
        # print(f"DK = {DK}")
        # print(f"log k= {log(K)}")

        lnkp: float = -(DV / RT) * P + (0.5 * DK / RT) * P**2 + log(K)
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
        After Zeebe and Gladrow, 2001, Chapter 3.2.3, page 186

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

        # CO2g versus HCO3, e = epsilon, a = alpha
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


"""
    dic: tp.Union[Reservoir, esbmtk.extended_classes.VirtualReservoir],
    hplus: tp.Union[Reservoir, esbmtk.extended_classes.VirtualReservoir],
    SW: SeawaterConstants,

"""


