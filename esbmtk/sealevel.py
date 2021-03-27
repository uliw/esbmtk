"""

     esbmtk.connections

     Classes which handle the connections and fluxes between esbmtk objects
     like Reservoirs, Sources, and Sinks.

     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

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

# from pint import UnitRegistry
from __future__ import annotations
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
from .utility_functions import map_units
from .processes import *
from .esbmtk import esbmtkBase, Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class hypsometry(esbmtkBase):
    """A class to provide hypsometric data for the depth interval between -6000 to 5000 meter
    The data is derived from etopo 5, but internally represented by a spline approximation

    """

    def __init__(self, **kwargs):
        """Initialize a hypsometry object"""

        # allowed keywords
        self.lkk: Dict[str, any] = {
            "name": str,
            "register": (Model, str),
            "model": Model,
        }

        # required keywords
        self.lrk: list = [
            "name",
            "model",
        ]
        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "register": "None",
        }

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)

        # legacy variables
        self.pfn = "spline_paramaters.txt"
        self.hfn = "Hypsometric_Curve_05m.csv"
        self.sa = 510067420E6 # in square meters, http://www.physicalgeography.net/fundamentals/8o.html
        self.mo = self.model
        self.__register_name__()
        self.__init_curve__()

    def area(self, depth: list) -> float:
        """Calculate the area between two elevation datums
        depth = [lower, upper]

        """

        a: list = interpolate.splev(depth, self.tck)
        area: float = a[0] - a[1]

        return area

    def __init_curve__(self):
        """Initialize Spline Parameters. See  __bootstrap_curve__ if you want
        to change the default parameters

        """
        t = [
            -6000.0,
            -6000.0,
            -6000.0,
            -6000.0,
            -5250.0,
            -4500.0,
            -3750.0,
            -3000.0,
            -1500.0,
            -1120.0,
            -750.0,
            -560.0,
            -370.0,
            -180.0,
            -90.0,
            0.0,
            380.0,
            750.0,
            1500.0,
            2250.0,
            3000.0,
            5990.0,
            5990.0,
            5990.0,
            5990.0,
        ]
        c = [
            0.01018464,
            0.00825062,
            0.08976178,
            0.26433525,
            0.44127754,
            0.5799517,
            0.59791548,
            0.6263245,
            0.63035567,
            0.63978284,
            0.64800198,
            0.6501602,
            0.68030866,
            0.75133294,
            0.86590303,
            0.92052208,
            0.96111183,
            0.97330001,
            0.99966578,
            0.99759724,
            1.00067306,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        k = 3

        self.tck = (t, c, k)

    def __bootstrap_curve__(self):
        """Regenerate the spline data based on the hypsometric data in
        Hypsometric_Curve_05m.csv,

        """
        df = pd.read_csv(
            "Hypsometric_Curve_05m.csv",
            float_precision="high",
            nrows=1200,
            skiprows=300,
        )
        area = df.iloc[:, 2].to_numpy()  # get area as numpy arrat
        elevation = df.iloc[:, 1].to_numpy()  # get area as numpy arrat

        tck = interpolate.splrep(
            elevation,
            area,
            s=0.001,
        )
        print(f"t = {tck[0].__repr__()}")
        print(f"c = {tck[1].__repr__()}")
        print(f"k = {tck[2].__repr__()}")

        depth = np.linspace(-6000, 1000, 50)
        a = interpolate.splev(depth, tck)

        plt.style.use(["ggplot"])
        fig = plt.figure()  # Create a figure instance called fig
        ax = plt.subplot()  # Create a plot instance called ax
        ax.plot(elevation, area)  # create a line plot
        ax.plot(depth, a)  # create a line plot
        plt.show()  # display figure
