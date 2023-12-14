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
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.interpolate
from .esbmtk_base import esbmtkBase

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class hypsometry(esbmtkBase):
    """A class to provide hypsometric data for the depth interval between -6000 to 1000
    meter (relative to sealevel)
    The data is derived from etopo 2, but internally represented by a spline approximation

    Invoke as:
               hyspometry(name="hyp")


    """

    def __init__(self, **kwargs):
        """Initialize a hypsometry object
        User facing methods:

          hyp.area (z) return the ocean area at a given depth in m^2

          hyp.area_dz(0,-200)::

               will return the surface area between 0 and -200 mbsl (i.e.,
               the contintal shelves). Note that this is the sediment area
               projected to the ocean surface, not the actual surface area
               of the sediment.

               This number has a small
               error since we exclude the areas below 6000 mbsl. The error
               is however a constant an likely within the uncertainty of
               the total surface area. The numbers obtained from
               http://www.physicalgeography.net/fundamentals/8o.html for
               total ocean area vary between 70.8% for all water covered
               areas and 72.5% for ocean surface. This routine returns
               70.5%

          hyp.sa = Earth total surface area in m^2

          hyp.volume(0,-200)

          hyp.get_lookup_table(min_depth: int, max_depth: int)::

                  Generate a vector which contains the area(z) in 1 meter intervals
                  Note that the numbers are area_percentage. To get actual area, you need to
                  multiply with the total surface area (hyp.sa)

        """
        from esbmtk import Model

        # allowed keywords
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "register": ["None", (Model, str)],
            "model": ["None", (str, Model)],
        }

        # required keywords
        self.lrk: list = [
            "name",
        ]

        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None" and self.model != "None":
            self.register = self.model

        self.parent = self.register

        # legacy variables
        self.pfn = "spline_paramaters.txt"
        self.hfn = "Hypsometric_Curve_05m.csv"
        # total surface area in square meters, http://www.physicalgeography.net/fundamentals/8o.html
        self.sa = 510067420e6
        self.mo = self.model
        self.__register_name_new__()
        self.__init_curve__()
        self.oa = self.area_dz(0, -6000)  # total ocean area

    def volume(self, u: float, l: float) -> float:
        """Calculate the area between two elevation datums

        u = upper limit (e.g., -10)
        l = lower limit (e.g., -100)

        returns the volume in cubic meters
        """

        u = abs(u)
        l = abs(l)
        if l < u:
            raise ValueError(f"hyp.volume: {l} must be higher than {u}")

        return np.sum(self.hypdata[u:l]) * self.sa

    def area(self, depth: int) -> float:
        """Calculate the ocean area at a given depth

        depth must be an integer between 0 and 6000 mbsl, or a
        numpy array of integers between 0 and 6000 mbsl

        """

        depth = np.abs(depth).astype(int)

        if np.max(depth) > 6001:
            raise ValueError("area() is only defined to a depth of 6001 mbsl")

        return np.take(self.hypdata, depth) * self.sa

    def area_dz(self, u: float, l: float) -> float:
        """Calculate the area between two elevation datums

        u = upper limit
        l = lower limit

        the interpolation function returns a numpy array with
        cumulative area percentages do the difference between the
        lowest and highest value is the area contained between
        both limits. The difference between the upper and lower
        bounds is the area percentage contained between both depths.

        The function returns this value multiplied by total surface area,
        i.e., in square meters.

        """

        if l < -6002:
            raise ValueError("area_dz() is only defined to a depth of 6000 mbsl")

        a: NDArrayFloat = sp.interpolate.splev([u, l], self.tck)

        return (a[0] - a[-1]) * self.sa

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

        self.hypdata = sp.interpolate.splev(np.arange(1000, -6001, -1), self.tck)

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
        area = df.iloc[:, 2].to_numpy()  # get area as numpy array
        elevation = df.iloc[:, 1].to_numpy()  # get area as numpy array

        tck = sp.interpolate.splrep(
            elevation,
            area,
            s=0.001,
        )
        print(f"t = {tck[0].__repr__()}")
        print(f"c = {tck[1].__repr__()}")
        print(f"k = {tck[2].__repr__()}")

        depth = np.linspace(-6000, 1000, 50)
        a = sp.interpolate.splev(depth, tck)

        plt.style.use(["ggplot"])
        plt.figure()  # Create a figure instance called fig
        ax = plt.subplot()  # Create a plot instance called ax
        ax.plot(elevation, area)  # create a line plot
        ax.plot(depth, a)  # create a line plot
        plt.show()  # display figure

    def get_lookup_table(self, min_depth: int, max_depth: int) -> NDArrayFloat:
        """Generate a vector which contains the area(z) in 1 meter intervals
        The numbers are given in m^2 which represent the actual area.

        The calculations multiply the area_percentage by the total surface area (hyp.sa)
        """

        if not -6002 <= min_depth <= 0:
            raise ValueError("min_depth must be <= 0 and >= -6000")

        if not -6002 <= max_depth <= min_depth:
            raise ValueError("max_depth must be <= 0 and >= -6000")

        return (
            sp.interpolate.splev(np.arange(min_depth, max_depth, -1), self.tck)
            * self.sa
        )

    def get_lookup_table_area_dz(self, min_depth: int, max_depth: int) -> NDArrayFloat:
        """Generate a vector which contains the first derivative of area(z) in 1 meter intervals
        Note that the numbers are in m^2

        """

        return np.diff(self.get_lookup_table(min_depth, max_depth))


def get_box_geometry_parameters(box, fraction=1) -> None:
    """
    Calculate box volume and area from the data in box.

    :param box: list or dict with the geometry parameters
    :fraction: 0 to 1 to specify a fractional part (i.e., Atlantic)

    If box is a list the first entry is the upper
    depth datum, the second entry is the lower depth datum, and the
    third entry is the total ocean area.  E.g., to specify the upper
    200 meters of the entire ocean, you would write:

    geometry=[0,-200,3.6e14]

    the corresponding ocean volume will then be calculated by the
    calc_volume method in this case the following instance variables
    will also be set:

     - self.volume in model units (usually liter)
     - self.are:a surface area in m^2 at the upper bounding surface
     - self.sed_area: area of seafloor which is intercepted by this box.
     - self.area_fraction: area of seafloor which is intercepted by
       this relative to the total ocean floor area

    It is also possible to specify volume and area explicitly. In this
    case provide a dictionary like this::

        box = {"area": "1e14 m**2", # surface area in m**2
               "volume": "3e16 m**3", # box volume in m**3
               "ta": "4e16 m**2", # reference area
              }

    """
    from esbmtk import Q_

    box.geometry_unset = True

    if isinstance(box.geometry, list):
        # Calculate volume and area as a function of box geometry
        top = box.geometry[0]
        bottom = box.geometry[1]
        fraction = box.geometry[2]
        volume = f"{box.mo.hyp.volume(top, bottom) * fraction} m**3"
        box.volume = Q_(volume)
        box.volume = box.volume.to(box.mo.v_unit)
        box.area = Q_(f"{box.mo.hyp.area(top) * fraction} m**2")
        box.sed_area = box.mo.hyp.area_dz(top, bottom) * fraction
        box.sed_area = Q_(f"{box.sed_area} m**2")

    elif isinstance(box.geometry, dict):
        box.volume = Q_(box.geometry["volume"]).to(box.mo.v_unit)
        box.area = Q_(box.geometry["area"]).to(box.mo.a_unit).magnitude
        box.sed_area = box.area
    else:
        raise ValueError("You need to provide volume or geometry!")

    # box.area_fraction = box.sed_area / box.reference_area
    box.area_dz = box.sed_area
    box.geometry_unset = False
