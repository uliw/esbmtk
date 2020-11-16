# from pint import UnitRegistry
from numbers import Number
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
from pandas import DataFrame
from copy import deepcopy, copy
from time import process_time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
import time
import builtins
set_printoptions(precision=4)
from .utility_functions import *
from .esbmtk import *

class Carbon(esbmtkBase):
    """ Some often used definitions
    
    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """


        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "model": Model,
            "name": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["model", "name"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({
            "model": "a model handle",
        })
        self.__validateandregister__(kwargs)  # initialize keyword values

        eh = Element(
            name="C",  # Element Name
            model=self.model,  # Model handle
            mass_unit="mmol",  # base mass unit
            li_label="C^{12}$S",  # Name of light isotope
            hi_label="C^{13}$S",  # Name of heavy isotope
            d_label="$\delta^{13}$C",  # Name of isotope delta
            d_scale="VPDB",  # Isotope scale. End of plot labels
            r=0.0112372,  # VPDB C13/C12 ratio https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf
        )

        # add species
        Species(name="CO2", element=eh)  # Name & element handle
        Species(name="DIC", element=eh)
        Species(name="OM", element=eh)
        Species(name="CaCO3", element=eh)
        Species(name="HCO3", element=eh)
        Species(name="CO3", element=eh)
        Species(name="OH", element=eh)


class Sulfur(esbmtkBase):
    """ Some often used definitions
    
    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """


        self.lkk: Dict[str, any] = {
            "model": Model,
            "name": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["model", "name"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({
            "model": "a model handle",
        })
        self.__validateandregister__(kwargs)  # initialize keyword values

        eh = Element(
            name="S",
            model=self.model,  # model handle
            mass_unit="mmol",  # base mass unit
            li_label="$^{32}$S",  # Name of light isotope
            hi_label="$^{34}$S",  # Name of heavy isotope
            d_label="$\delta^{34}$S",  # Name of isotope delta
            d_scale="VCDT",  # Isotope scale. End of plot labels
            r=0.044162589,  # isotopic abundance ratio for species
        )

        # add species
        Species(name="SO4", element=eh)  # Name & element handle
        Species(name="SO3", element=eh)
        Species(name="SO2", element=eh)
        Species(name="HS", element=eh)
        Species(name="H2S", element=eh)
        Species(name="FeS", element=eh)
        Species(name="FeS2", element=eh)
        Species(name="S0", element=eh)

class Hydrogen(esbmtkBase):
    """ Some often used definitions
    
    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """


        self.lkk: Dict[str, any] = {
            "model": Model,
            "name": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["model", "name"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({
            "model": "a model handle",
        })
        self.__validateandregister__(kwargs)  # initialize keyword values

        eh = Element(
            name="H",
            model=self.Model,  # model handle
            mass_unit="mmol",  # base mass unit
            li_label="$^{1$}H",  # Name of light isotope
            hi_label="$^{2}$H",  # Name of heavy isotope
            d_label="$\delta^{2}$H",  # Name of isotope delta
            d_scale="VSMOV",  # Isotope scale. End of plot labels  # needs verification 
            r=0.00015575 ,  # isotopic abundance ratio for species # needs verification 
        )

        # add species
        Species(name="H+", element=eh)  # Name & element handle
