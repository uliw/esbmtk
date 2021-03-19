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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time
import builtins
from .esbmtk import esbmtkBase, Model, Reservoir, VirtualReservoir

class SeawaterConstants(esbmtkBase):
    """Provide basic seawater properties as a function of T and Salinity.
    Pressure may come at a later stage

    Example:

    Seawater(name="SW",
             model=
             temperature = optional in C, defaults to 25
             salinity  = optional in psu, defaults to 35
             pressure = optional, defaults to 0 bars = 1atm
             pH = optional, defaults to 8.1
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
        A["KB"]: list = [29.48, 0.1622, 2.6080, 2.84, 0.0]
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

def calc_H(
    i: int,
    a1: Union[Reservoir, VirtualReservoir],
    a2: Union[Reservoir, VirtualReservoir],
    a3: SeawaterConstants,
    a4=0,
    a5=0,
    a6=0,
) -> tuple:

    """

    This function will calculate the H+ concentration at t=i
    time step. Returns a tuple in the form of [m, l, h] which pertains to
    the mass, and respective isotopes of the element. l and h will
    default to 1. Calculations are based off equations from Follows et al., 2006.
    doi:10.1016/j.ocemod.2005.05.004

    a1 = carbonate alkalinity reservoir object
    a2 = dic reservoir object
    a3 = SeawaterConstants object

    i = index of current timestep
    a1 to a6 = optional fcn parameters. These must be present
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
         a1=V_CA,
         a2=DIC,
         a3=SW,
    )

    Author: M. Niazi & T. Tsan, 2021

    """

    CA: float = a1.c[i - 1]  # mol/L
    DIC: float = a2.c[i - 1]  # mol/L
    SW: SeawaterConstants = a3  #

    k1: float = SW.K1
    k2: float = SW.K2

    gamm: float = DIC / CA
    dummy: float = (1 - gamm) * (1 - gamm) * k1 * k1 - 4 * k1 * k2 * (1 - (2 * gamm))
    m: float = (0.5 * ((gamm - 1) * k1 + (dummy ** 0.5)))
    l: float = 1.0
    h: float = 1.0

    return [m, l, h]


def calc_CA(
    i: int,
    a1: Union[Reservoir, VirtualReservoir],
    a2: Union[Reservoir, VirtualReservoir],
    a3: SeawaterConstants,
    a4=0,
    a5=0,
    a6=0,
) -> tuple:

    """
    This function will calculate the carbonate alkalinity concentration
    at the ith time step. Returns a tuple in the form of [m, l, h]
    which pertains to the mass, and respective isotopes. For carbonate
    alkalinity, m will equal to the amount of carbonate alkalinity in
    mol/L and l and h will default to 1.  Calculations are based off
    equations from Follows et al., 2006.
    doi:10.1016/j.ocemod.2005.05.004


    a1 = total alkalinity reservoir object
    a2 = H+ reservoir reservoir object
    a3 = SeawaterConstants object

    i = index of current timestep
    a1 to a6 = optional fcn parameters. These must be present
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
         a1=V_CA,
         a2=DIC,
         a3=SW,
    )

    Author: M. Niazi & T. Tsan, 2021

    """

    ta: float = a1.c[i - 1]  # mol/L
    hplus: float = a2.c[i - 1]  # mol/L
    SW: SeawaterConstants = a3

    oh: float = SW.KW / hplus
    boh4: float = SW.boron * SW.KB / (hplus + SW.KB)

    fg: float = hplus - oh - boh4  # mol/L

    m: float = ta + fg
    l: float = 1
    h: float = 1

    return [m, l, h]

def calc_pCO2(
    dic: Union[Reservoir, VirtualReservoir],
    hplus: Union[Reservoir, VirtualReservoir],
    SW: SeawaterConstants,
) -> [NDArray, Float]:

    """
    Calculate the concentration of pCO2 as a function of DIC,
    H+, K1 and k2 and returns a numpy array containing
    the pCO2 in uatm at each timestep. Calculations are based off
    equations from Follows, 2006. doi:10.1016/j.ocemod.2005.05.004

    DIC: Reservoir  = DIC concentrations in mol/liter
    Hplus: Reservoir = H+ concentrations in mol/liter
    SW: Seawater = Seawater object for the model

    Author: T. Tsan

    """

    dic_c: [NDArray, Float] = dic.c
    hplus_c: [NDArray, Float] = hplus.c

    k1: float = SW.K1
    k2: float = SW.K2

    co2: [NDArray, Float] = dic_c / (1 + (k1 / hplus_c) + (k1 * k2 / (hplus_c ** 2)))

    pco2: [NDArray, Float] = co2 / SW.K0 * 1E6

    return pco2
