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
import PyCO2SYS as pyco2
import warnings

from .esbmtk_base import esbmtkBase

if tp.TYPE_CHECKING:
    from .esbmtk import Reservoir, Model
    from .extended_classes import ReservoirGroup


# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class SeawaterConstants(esbmtkBase):
    """Provide basic seawater properties as a function of T, P and Salinity.
    Since we cannot know if TA and DIC have already been specified, creating
    the instance uses standard seawater composition. Updating/Setting TA & DIC
    does not recalculate these values after initialization, unless
    you explicitly call the update_parameters() method.

    Example::

        Seawater(name="SW",
                 register=M # model handle
                 temperature = optional in C, defaults to 25,
                 salinity  = optional in psu, defaults to 35,
                 pressure = optional, defaults to 0 bars = 1atm,
                 pH = 8.1, # optional
                )

    Results are always in mol/kg

    Acess the values "dic", "ta", "ca", "co2", "hco3",
    "co3", "boron", "boh4", "boh3", "oh", "ca2", "so4","hplus",
    as SW.co3 etc.

    This method also provides "K0", "K1", "K2", "KW", "KB", "Ksp",
    "Ksp0", "KS", "KF" and their corresponding pK values, as well
    as the density for the given (P/T/S conditions)

    useful methods:

    SW.show() will list values

    After initialization this class provides access to each value the following way

    instance_name.variable_name

    Since this class is just a frontend to PyCO2SYS, it is easy to add parameters
    that are supported in PyCO2SYS. See the update_parameter() method.
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
            "dic": ["None", (str, float)],
            "ta": ["None", (str, float)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        # self.mo: Model = self.model
        self.hplus = 10**-self.pH
        self.constants: list = [
            "K0",
            "K1",
            "K2",
            "KW",
            "KB",
            "Ksp_ca",
            "Ksp_b",
            "Ksp0",
            "KS",
            "KF",
        ]
        self.species: list = [
            "dic",
            "ta",
            "ca",
            "co2aq",
            "hco3",
            "co3",
            "boron",
            "boh4",
            "boh3",
            "oh",
            "ca2",
            "mg2",
            "k",
            "so4",
            "hplus",
        ]

        if self.dic == "None" or self.ta == "None":
            self.__init_std_seawater__()
            warnings.warn(f"Initializing {self.name} with default seawater")

        self.update_parameters()

        if self.register != "None":
            self.__register_name_new__()

    def update_parameters(self, **kwargs: dict) -> None:
        """Update values if necessary"""

        if 'pos' in kwargs:
            pos = kwargs['pos']
        else:
            pos = -1

        if hasattr(self.register, "TA"):
            self.ta = self.register.TA.c[pos] * 1e6

        if hasattr(self.register, "DIC"):
            self.dic = self.register.DIC.c[pos] * 1e6
            
        results = pyco2.sys(
            salinity=self.salinity,
            temperature=self.temperature,
            pressure=self.pressure * 10, # in deci bar!
            par1_type=1,  # "1" =  "alkalinity"
            par1=self.ta,
            par2_type=2,  # "1" = dic
            par2=self.dic,
            opt_k_carbonic=self.register.model.opt_k_carbonic,
            opt_pH_scale=self.register.model.opt_pH_scale,
            opt_buffers_mode=self.register.model.opt_buffers_mode,
        )
        # update K values and species concentrations according to P, S, and T
        self.density = self.get_density(self.salinity, self.temperature, self.pressure)

        self.KF = results["k_fluoride"]
        self.FT = 7e-5 * self.salinity / 35
        self.KW = results["k_water"]
        self.KB = results["k_borate"]
        self.KS = results["k_bisulfate"]
        self.K0 = results["k_CO2"]
        self.K1 = results["k_carbonic_1"]
        self.K2 = results["k_carbonic_2"]
        self.Ksp_ca = float(results["k_calcite"])
        self.Ksp_ar = float(results["k_aragonite"])
        self.K1K1 = self.K1**2
        self.K1K2 = self.K1 * self.K2
        self.oh = results["OH"] * 1e-6
        self.co3 = results["CO3"] * 1e-6
        self.co2aq = results["aqueous_CO2"] * 1e-6
        self.boron = results["total_borate"] * 1e-6
        self.boh3 = results["BOH3"] * 1e-6
        self.boh4 = results["BOH4"] * 1e-6
        self.pH_free = results["pH_free"]
        self.pH_total = results["pH_total"]
        self.hplus = 10**-self.pH_total
        self.ca2 = results["total_calcium"] * 1e-6
        self.so4 = results["total_sulfate"] * 1e-6
        self.ST = self.so4 * self.salinity / 35
        self.pCO2 = results["pCO2"]
        self.fCO2 = results["fCO2"]
        self.Ksp0 = 4.29e-07  # after after Boudreau et al 2010
        self.__init_gasexchange__()
        self.__init_c_fractionation_factors__()

    def show(self) -> None:
        """Printout constants. Units are mol/kg or
        (mol**2/kg for doubly charged ions"""

        from math import log10

        print(f"\nSeawater constants for {self.register.full_name}")
        print(f"T = {self.temperature} [C]")
        print(f"P = {self.pressure} [bar]")
        print(f"S = {self.salinity} [PSU]")
        print(f"density = {self.density:.4f} [kg/m**3]\n")

        for n in self.species:
            v = getattr(self, n)
            print(f"{n} = {v:.5f} mol/kg")

        print()
        # print(f"pCO2 = {get_pco2(self):.2e}")
        print(f"pH = {-log10(self.hplus):.2f}")
        print(f"salinity = {self.salinity:.2f}")
        print(f"temperature = {self.temperature:.2f}\n")

        print("Units are mol/kg or mol^2/kg\n")
        for n in self.constants:
            K = getattr(self, n)  # get K value
            pk = f"p{n.lower()}"  # get K name
            print(f"{n} = {K:.2e}, {pk} = {-log10(K):.4f}")

        print()

    def get_density(self, S, TC, P) -> float:
        """Calculate seawater density as function of
        temperature, salinity and pressure

        :param S: salinity in PSU
        :param TC:  temp in C
        :param P: pressure in bar

        :returns rho: in kg/m**3
        """

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
        return rho0 / (1.0 - P / Ksbm)

    def __init_std_seawater__(self) -> None:
        """Provide values for standard seawater. Data after Zeebe and Gladrow
        all values in mol/kg. All values after Zeebe and Gladrow 2001

        This only used so that we can call pyco2SYS in order to get the
        equilibrium constants. We can drop this once the program logic
        initializes swc after the reservoir concentrations have been set.
        """

        self.co2aq = 0.00001
        self.hco3 = 0.00177
        self.co3 = 0.00026
        self.boron = 0.000416  #
        self.oh = 0.00001
        self.so4 = 2.7123 / 96
        self.ca2 = 0.01028
        self.mg2 = 0.05282
        self.k = 0.01021
        self.dic = self.co2aq + self.hco3 + self.co3
        self.ta = self.hco3 + 2 * self.co3  # estimate
        

    def __init_gasexchange__(self) -> None:
        """Initialize constants for gas-exchange processes"""

        self.water_vapor_partial_pressure()
        self.co2_solubility_constant()
        self.o2_solubility_constant()

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
        self.F = self.calc_solubility_term(S, T, A1, A2, A3, A4, B1, B2, B3)

        # correct for water vapor partial pressure
        self.SA_co2 = self.F / (1 - self.p_H2O)  # mol/(m^3 * atm)

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
