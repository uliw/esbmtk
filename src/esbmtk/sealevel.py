"""

     esbmtk.sealevel

     Classes which provide access to hypsometric data

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

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from .esbmtk_base import esbmtkBase

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class hypsometry(esbmtkBase):
    """A class to provide hypsometric data for the depth interval between -6000 to 1000
    meter (relative to sealevel)

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
                  Note that the numbers are area_percentage. To get actual area, you
                  need to multiply with the total surface area (hyp.sa)

        get_lookup_table_area_dz(0, -6002

        """
        from esbmtk import Model

        # allowed keywords
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "register": ["None", (Model, str)],
            "model": ["None", (str, Model)],
            "max_elevation": [1000, int],
            "max_depth": [-11000, int],
            "basin": ["global", str],
            "hyp_data_fn": ["Hypsometric_Curve_05m", str],
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
        # total surface area in square meters,
        # http://www.physicalgeography.net/fundamentals/8o.html
        self.sa = 510067420e6  # area in m^2
        self.mo = self.model
        self.__register_name_new__()
        self.read_data()
        self.oa = 361e12  #  m^2 https://en.wikipedia.org/wiki/Ocean

    def read_data(self) -> None:
        """Read the hypsometry data from a pickle file.
        If the pickle file is missing, create it from the
        csv data

        save the hypsometry data as a numpy array with
        elevation, area, and area_dz in self.hypdata
        """
        from importlib import resources as impresources
        import pathlib as pl

        fn_csv = impresources.files("esbmtk") / f"{self.hyp_data_fn}.csv"
        fqfn_csv: pl.Path = pl.Path(fn_csv)
        fn_pickle: str = impresources.files("esbmtk") / f"{self.hyp_data_fn}.pickle"
        fqfn_pickle: pl.Path = pl.Path(fn_pickle)

        if fqfn_pickle.exists():  # check if pickle file exist
            # get creation date of pickle file
            pickle_date = pl.Path(fn_pickle).stat().st_ctime
        else:
            pickle_date = 0

        csv_date = pl.Path(fn_csv).stat().st_ctime
        if csv_date < pickle_date:  # pickle file is newer
            df = pd.read_pickle(fn_pickle)
        else:  # pickle file is older
            if fqfn_csv.exists():
                print(
                    "pickle file is older/missing recreating hypsography pickle",
                    f"from {self.hyp_data_fn}.csv",
                )
                df = pd.read_csv(fn_csv, float_precision="high")
                df.to_pickle(fqfn_pickle)
            else:
                raise FileNotFoundError(f"Cannot find file {fqfn_csv}")

        max_el = df.iloc[0, 0] - self.max_elevation
        max_de = df.iloc[-1, 0] - self.max_elevation - self.max_depth

        elevation = df.iloc[max_el:max_de, 0].to_numpy()
        area = df.iloc[max_el:max_de, 1].to_numpy() * self.sa

        # create lookup table with area and area_dz
        self.hypdata = np.column_stack(
            (
                elevation[:-1],
                area[:-1],
                np.diff(area),
            )
        )

    def get_lookup_table_area(self) -> NDArrayFloat:
        """Return the area values between 0 and max_depth
        as 1-D array
        """
        return self.hypdata[self.max_elevation :, 1]

    def get_lookup_table_area_dz(self) -> NDArrayFloat:
        """Return the are_dz values between 0 and max_depth
        as 1-D array
        """

        return self.hypdata[self.max_elevation :, 2]

    def area(self, elevation: int) -> float:
        """Calculate the ocean area at a given depth

        Parameters
        ----------
        elevation : int
            Elevation datum in meters

        Returns
        -------
        float
            area in m^2

        """

        # calculate index
        i = int(self.max_elevation - elevation)

        if (elevation > self.max_elevation) or (elevation < self.max_depth):
            raise ValueError(
                (
                    f"hyp.area: {elevation} must be between"
                    f"{self.max_elevation} and {self.min_depth}"
                )
            )

        return self.hypdata[i,1]

    def area_dz(self, u: float, l: float) -> float:
        """calculate the area between two elevation datums

        Parameters
        ----------
        u : float
            upper elevation datum in meters (relative to sealevel)
        l : float
            lower elevation datum relative to sealevel

        Returns
        -------
        float
            area in m^2

        Raises
        ------
        ValueError
            if elevation datums are outside the defined interval

        """
        if (u > self.max_elevation) or (l < self.max_depth):
            raise ValueError(
                (
                    f"hyp.area: {u} must be < {self.max_elevation}"
                    f"and {l} > {self.min_depth}"
                )
            )

        u = self.max_elevation - int(u)
        l = self.max_elevation - int(l)

        return self.hypdata[u, 1] - self.hypdata[l, 1]

    def volume(self, u: float, l: float) -> float:
        """Calculate the area between two elevation datums

        Parameters
        ----------
        u : float
            upper elevation datum in meters (relative to sealevel)
        l : float
            lower elevation datum relative to sealevel

        Returns
        -------
        float
            volume in m^3

        Raises
        ------
        ValueError
            if elevation datums are outside the defined interval

        """

        if (u > self.max_elevation) or (l < self.max_depth):
            raise ValueError(
                (
                    f"hyp.area: {u} must be < {self.max_elevation}"
                    f"and {l} > {self.min_depth}"
                )
            )

        u = self.max_elevation - int(u)
        l = self.max_elevation - int(l)

        return np.sum(self.hypdata[u:l])

    def show_data(self):
        """Provide a diagnostic graph that shows the hypsometric data
        use by ESBMTK
        """

        elevation = self.hypdata[:, 0]
        area = self.hypdata[:, 1] / self.sa

        import matplotlib.pyplot as plt

        plt.style.use("uli")

        fig, ax = plt.subplots()
        ax.plot(100 - area * 100, elevation, color="C0")
        ax.set_title("ESBMTK Hypsometric Data")
        ax.set_xlabel("Cumulative Area [%]")
        ax.set_ylabel("Elevation [m]")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.grid(True, which="both")
        fig.tight_layout()
        fig.savefig("Hysography_debug.pdf")
        plt.show()


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
