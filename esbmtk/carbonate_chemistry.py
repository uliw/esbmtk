"""
     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020-2021 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
from .esbmtk import esbmtkBase, Model, Reservoir, VirtualReservoir, ReservoirGroup

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
            )

    useful methods:

    SW.show() will list all known values

    After initialization this class provides access to each value the following way

    instance_name.variable_name

    """

    def __init__(self, **kwargs: Dict[str, str]):

        import math

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "model": Model,
            "salinity": (int, float),
            "temperature": (int, float),
            "pH": (int, float),
            "pressure": Number,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "salinity": 35.0,
            "temperature": 25.0,
            "pH": 8.1,
            "pressure": 0,
        }

        # validate input and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        self.mo: Model = self.model
        self.hplus = 10 ** -self.pH
        self.constants: list = ["K0", "K1", "K2", "KW", "KB"]
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
        ]

        self.update()
        self.__register_name__()

    def update(self, **kwargs: dict) -> None:
        """Update values if necessary"""

        from math import log10

        if kwargs:
            self.lrk: list = []
            self.__validateandregister__(kwargs)

        # update K values and species concentrations according to P, S, and T
        self.__init_std_seawater__()
        self.__init_carbon__()
        self.__init_boron__()
        self.__init_water__()

        # get total alkalinity

        self.ca = self.hco3 + 2 * self.co3
        self.ta = self.ca + self.boh4 + self.oh - self.hplus

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
        all values in mol/kg. To convert to seawater these values need to be
        multiplied by sw

        """

        S = self.salinity
        swc = (1000 + S) / 1000
        self.dic = 0.00204 * swc
        self.boron = 0.00042 * swc
        self.oh = 0.00001 * swc

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


@njit
def calc_H(
    i: int,
    a1: NDArray[Float[64]],  # carbonate alkalinity
    a2: NDArray[Float[64]],  # dic
    a3: NDArray[Float[64]],
    a4: NDArray[Float[64]],
):

    """

    This function will calculate the H+ concentration at t=i
    time step. Returns a tuple in the form of [m, l, h] which pertains to
    the mass, and respective isotopes of the element. l and h will
    default to 1. Calculations are based off equations from Follows et al., 2006.
    doi:10.1016/j.ocemod.2005.05.004

    a1 = carbonate alkalinity concentrations
    a2 = dic concentrations
    a3 = [ list of SeawaterConstants]

    i = index of current timestep
    a1 to a3 = optional fcn parameters. These must be present
    even if your function will not use it. These will default to 0.

    Limitations: Assumes concentrations are in mol/L

    This function can then be used in conjunction with a VirtualReservoir, e.g.,

    VirtualReservoir(
         name="V_H",
         species=Hplus,
         concentration=f"{SW.hplus*1000} mmol/l",
         volume=volume,
         plot_transform_c=phc,
         legend_left="pH",
         function=calc_H,
         a1=V_CA.c,
         a2=DIC.c,
         a3=[SW constants],
         a4=[array] # muust be provided, but can be empty
    )

    Author: M. Niazi & T. Tsan, 2021

    The function will not return a value, bur rather write directly to ref!

    """

    # from esbmtk import phc

    ca: float = a1[i - 1]  # mol/L
    dic: float = a2[i - 1]  # mol/L
    k1 = a3[0]
    k2 = a3[1]
    volume = a3[2]

    gamm: float = dic / ca
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))
    c: float = 0.5 * ((gamm - 1) * k1 + (dummy ** 0.5))
    m: float = c * volume

    # print(f"DIC = {dic*1000}, ca = {ca*1000}")
    # print(f"new pH = {phc(c)}")

    return m, 1.0, 1.0, 1.0, c


@njit
def calc_CA(
    i: int,
    a1: NDArray[Float[64]],  # Total Alkalinity
    a2: NDArray[Float[64]],  # Hplus
    a3: NDArray[Float[64]],
    a4: NDArray[Float[64]],
):

    """
    This function will calculate the carbonate alkalinity concentration
    at the ith time step. Returns a tuple in the form of [m, l, h]
    which pertains to the mass, and respective isotopes. For carbonate
    alkalinity, m will equal to the amount of carbonate alkalinity in
    mol/L and l and h will default to 1.  Calculations are based off
    equations from Follows et al., 2006.
    doi:10.1016/j.ocemod.2005.05.004


    a1 = total alkalinity concentration in model units
    a2 = H+ concentrations in model units
    a3 = [list of SeawaterConstants]

    i = index of current timestep
    a1 to a3 = optional fcn parameters. These must be present
    even if your function will not use it

    Limitations: Assumes concentrations are in mol/L

    This function can then be used in conjunction with a VirtualReservoir, e.g.,

    VirtualReservoir(
         name="V_H",
         species=Hplus,
         concentration=f"{SW.hplus*1000} mmol/l",
         volume=volume,
         plot_transform_c=phc,
         legend_left="pH",
         function=calc_H,
         a1=TA.c,
         a2=H+.c,
         a3=[swc.KW, swc.KB, swc.boron]
         a4=[] # array, must be provided but can be empty
    )

    Author: M. Niazi & T. Tsan, 2021

    """

    ta: float = a1[i - 1]  # mol/L
    hplus: float = a2[i - 1]  # mol/L

    KW = a3[0]
    KB = a3[1]
    boron = a3[2]
    volume = a3[3]

    # print(f"KW = {KW:.2e}, KB={KB:.2e}")
    oh: float = KW / hplus
    boh4: float = boron * KB / (hplus + KB)

    fg: float = hplus - oh - boh4  # mol/L

    # print(f"ta = {ta*1000} mmol, fg ={fg*1000} mmol, hplus = {phc(hplus)}")
    c: float = ta + fg
    m: float = c * volume

    # print(f" volume = {volume:.2e}")
    # print(f"CA m = {m:.2e}, c= {c*1000:.2e} mmol")

    return m, 1.0, 1.0, 1.0, c


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


def carbonate_system(
    ca_con: float,
    hplus_con: float,
    volume: float,
    swc: SeawaterConstants,
    rg: ReservoirGroup = "None",
) -> tuple:

    """Setup the virtual reservoirs for carbonate alkalinity and H+

    ca_con: initial carbonate concentration. Must be a quantity
    hplus_con: initial H+ concentration. Must be a quantity
    volume: volume : Must be a quantity for reservoir definition but when  used
    as argumment to the functionn it muts be converted to magnitude

    swc : a seawater constants object
    rg: optional, must be a reservoir group. If present, the below reservoirs
        will be registered with this group.

    Returns the reservoir handles to VCA and VH

    All list type objects must be converted to numba Lists, if the function is to be used with
    the numba solver.
    """

    from esbmtk import VirtualReservoir, phc, calc_CA, calc_H

    v1 = VirtualReservoir(
        name="VCA",
        species=CA,
        concentration=ca_con,
        volume=volume,
        plot="no",
        function=calc_CA,
        register=rg,
    )

    v2 = VirtualReservoir(
        name="VH",
        species=Hplus,
        concentration=hplus_con,
        volume=volume,
        plot_transform_c=phc,
        legend_left="pH",
        plot="yes",
        function=calc_H,
        a1=getattr(rg, "VCA").c,
        a2=getattr(rg, "DIC").c,
        a3=List([swc.K1, swc.K2, volume.magnitude]),
        a4=np.zeros(3),
        register=rg,
    )

    v1.update(
        a1=getattr(rg, "TA").c,
        a2=getattr(rg, "VH").c,
        a3=List([swc.KW, swc.KB, swc.boron, volume.magnitude]),
        a4=np.zeros(3),
    )

    return v1, v2
