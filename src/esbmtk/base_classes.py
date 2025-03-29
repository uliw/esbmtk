"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G. Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
import os
import typing as tp
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from . import Q_
from .esbmtk_base import esbmtkBase
from .model import Model
from .utility_functions import get_l_mass

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from .connections import Species2Species
    from .extended_classes import DataField, ExternalData
    from .processes import Process


class ReservoirError(Exception):
    """Custom Error Class for reservoir-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxError(Exception):
    """Custom Error Class for flux-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ScaleError(Exception):
    """Custom Error Class for unit scale-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class SpeciesError(Exception):
    """Custom Error Class for species-related errors."""

    def __init__(self, message):
        """Initialize Error Instance with formatted message."""
        message = f"\n\n{message}\n"
        super().__init__(message)


def deprecated_keyword(message):
    """Issue a deprecation warning with the provided message."""
    warnings.warn(message, DeprecationWarning, stacklevel=2)


class ElementProperties(esbmtkBase):
    r"""Each model, can have one or more elements.

    This class sets element specific properties

    Example::

        ElementProperties(name      = "S "           # the element name
                model     = Test_model     # the model handle
                mass_unit =  "mol",        # base mass unit
                li_label  =  "$^{32$S",    # Label of light isotope
                hi_label  =  "$^{34}S",    # Label of heavy isotope
                d_label   =  r"$\delta^{34}$S",  # Label for delta value
                d_scale   =  "VCDT",       # Isotope scale
                r         = 0.044162589,   # isotopic abundance ratio for element
                reference = "https link or citation",
              )
    """

    # set element properties
    def __init__(self, **kwargs) -> any:
        """Initialize all instance variables.

        Defaults are as follows::

            self.defaults: dict[str, tp.List[any, tuple]] = {
               "name": ["M", (str)],
               "model": ["None", (str, Model)],
               "register": ["None", (str, Model)],
               "full_name": ["None", (str)],
               "li_label": ["None", (str)],
               "hi_label": ["None", (str)],
               "d_label": ["None", (str)],
               "d_scale": ["None", (str)],
               "r": [1, (float, int)],
               "mass_unit": ["mol", (str, Q_)],
               "parent": ["None", (str, Model)],
               "reference": ["None", (str)],

        }

        Required keywords: "name", "model", "mass_unit"
        """
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["M", (str)],
            "model": ["None", (str, Model)],
            "register": ["None", (str, Model)],
            "full_name": ["None", (str)],
            "li_label": ["None", (str)],
            "hi_label": ["None", (str)],
            "d_label": ["None", (str)],
            "d_scale": ["None", (str)],
            "r": [1, (float, int)],
            "mass_unit": ["", (str, Q_)],
            "parent": ["None", (str, Model)],
            "reference": ["None", (str)],
        }

        # list of absolutely required keywords
        self.lrk: list = ["name", "model", "mass_unit"]
        self.__initialize_keyword_variables__(kwargs)

        self.parent = self.model
        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.mo: Model = self.model  # model handle
        self.mu: str = self.mass_unit  # display name of mass unit
        self.ln: str = self.li_label  # display name of light isotope
        self.hn: str = self.hi_label  # display name of heavy isotope
        self.dn: str = self.d_label  # display string for delta
        self.ds: str = self.d_scale  # display string for delta scale
        self.lsp: list = []  # list of species for this element.
        self.mo.lel.append(self)

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_with_parent__()

    def list_species(self) -> None:
        """List all species which are predefined for this element."""
        for e in self.lsp:
            print(e.n)

    def __register_species_with_model__(self) -> None:
        """Bit of hack, but makes model code more readable."""
        for s in self.lsp:
            setattr(self.model, s.name, s)


class SpeciesProperties(esbmtkBase):
    """Each model, can have one or more species.

    This class sets species specific properties

    Example::

        SpeciesProperties(name = "SO4",
                element = S,

    )

    Defaults::

        self.defaults: dict[any, any] = {
            "name": ["None", (str)],
            "element": ["None", (ElementProperties, str)],
            "display_as": [kwargs["name"], (str)],
            "m_weight": [0, (int, float, str)],
            "register": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "parent": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "flux_only": [False, (bool)],
            "logdata": [False, (bool)],
            "scale_to": ["None", (str)],
            "stype": ["concentration", (str)],
        }

    Required keywords: "name", "element"
    """

    # set species properties
    def __init__(self, **kwargs) -> None:
        """Initialize all instance variables."""
        from esbmtk import GasReservoir

        # provide a list of all known keywords
        self.defaults: dict[any, any] = {
            "name": ["None", (str)],
            "element": ["None", (ElementProperties, str)],
            "display_as": [kwargs["name"], (str)],
            "m_weight": [0, (int, float, str)],
            "register": [
                kwargs["element"],
                (Model, ElementProperties, Species, GasReservoir),
            ],
            "parent": ["None", (Model, ElementProperties, Species, GasReservoir)],
            "flux_only": [False, (bool)],
            "logdata": [False, (bool)],
            "scale_to": ["mmol", (str)],
            "stype": ["concentration", (str)],
        }

        # provide a list of absolutely required keywords
        self.lrk = ["name", "element"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        if "display_as" not in kwargs:
            self.display_as = self.name

        # legacy names
        self.n = self.name  # display name of species
        self.mass_unit = self.element.mass_unit
        self.mu = self.mass_unit  # display name of mass unit
        self.ln = self.element.ln  # display name of light isotope
        self.hn = self.element.hn  # display name of heavy isotope
        self.dn = self.element.dn  # display string for delta
        self.ds = self.element.ds  # display string for delta scale
        self.r = self.element.r  # ratio of isotope standard
        self.mo = self.element.mo  # model handle
        self.eh = self.element.n  # element name
        self.e = self.element  # element handle
        self.dsa = self.display_as  # the display string.
        # self.mo.lsp.append(self)   # register self on the list of model objects
        self.e.lsp.append(self)  # register this species with the element

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_with_parent__()


class SpeciesBase(esbmtkBase):
    """Base class for all Species objects."""

    def __init__(self, **kwargs) -> None:
        """Instantiate Class."""
        raise NotImplementedError(
            "SpeciesBase should never be used. Use the derived classes"
        )

    def __set_legacy_names__(self, kwargs) -> None:
        """Move the below out of the way."""
        from esbmtk.sealevel import get_box_geometry_parameters

        self.atol: list[float] = [1.0, 1.0]  # tolerances
        self.lof: list[Flux] = []  # flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  # flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[ElementProperties] = []  # list of elements in thiis reservoir
        self.doe: dict[SpeciesProperties, Flux] = {}  # species flux pairs
        self.loc: set[Species2Species] = set()  # set of connection objects
        self.ldf: list[DataField] = []  # list of datafield objects
        # list of processes which calculate reservoirs
        self.lpc: list[Process] = []
        self.ef_results = False  # Species has external function results

        # legacy names
        self.n: str = self.name  # name of reservoir
        # if "register" in self.kwargs:
        if self.register == "None":
            self.pt = self.name
        else:
            self.pt: str = f"{self.register.name}_{self.n}"
            self.groupname = self.register.name
            # self.full_name = f"{self.register.name}.{self.n}"
        # else:
        #   self.pt = self.name

        self.sp: SpeciesProperties = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle
        self.model = self.mo
        self.rvalue = self.sp.r
        self.m_unit = self.model.m_unit
        self.v_unit = self.model.v_unit
        self.c_unit = self.model.c_unit

        # right y-axis label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"
        self.xl: str = self.mo.xl  # set x-axis lable to model time

        if self.legend_left == "None":
            self.legend_left = self.species.dsa

        self.legend_right = f"{self.species.dn} [{self.species.ds}]"
        # legend_left is in __init__ !

        # decide whether we use isotopes
        if self.mo.m_type == "both":
            self.isotopes = True
        elif self.mo.m_type == "mass_only":
            self.isotopes = False

        if self.geometry != "None" and self.geometry_unset:
            get_box_geometry_parameters(self)

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.parent = self.register

    def __setitem__(self, i: int, value: float):
        """Create a placeholder setitem function."""
        return self.__set_data__(i, value)

    def __call__(self) -> None:  # what to do when called as a function ()
        """Return self when called as a function."""
        return self

    def __getitem__(self, i: int) -> NDArrayFloat:
        """Get flux data by index."""
        return np.array([self.m[i], self.l[i], self.c[i]])

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """Set values when isotope data is present.

        :param i: index
        :param value: array of [mass, li, hi, d]

        """
        self.m[i]: float = value[0]
        # update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on on both variables.
        self.c[i]: float = value[0] / self.v[i]  # update concentration
        self.l[i]: float = value[1] / self.v[i]  # update concentration

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """Set values when no isotope data is present.

        :param i: index
        :param value: array of [mass]

        """
        self.m[i]: float = value[0]
        self.c[i]: float = self.m[i] / self.v[i]  # update concentration

    def __update_mass__() -> None:
        """Place holder function."""
        raise NotImplementedError("__update_mass__ is not yet implmented")

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """Write data to file.

        This function is called by the write_data() and save_state() methods

        :param prefix:
        :param start:
        :param stop:
        :param stride:
        :param append:
        :param directory:
        """
        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp  # species handle
        mo = self.sp.mo  # model handle

        # smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        # fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"

        # sdn = self.sp.dn  # delta name
        # sds = self.sp.ds  # delta scale
        rn = self.full_name  # reservoir name
        mn = self.sp.mo.n  # model name
        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{mn}_{rn}.csv"  # file name
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{rn}.csv"  # file name
        else:
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{rn} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        # df[f"{rn} {sn} [{smu}]"] = self.m.to(self.mo.m_unit).magnitude[start:stop:stride]  # mass
        if self.isotopes:
            # print(f"rn = {rn}, sp = {sp.name}")
            df[f"{rn} {sp.ln} [{cmu}]"] = self.l[start:stop:stride]  # light isotope
        df[f"{rn} {sn} [{cmu}]"] = self.c[start:stop:stride]  # concentration

        file_path = Path(fn)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)
        return df

    def __sub_sample_data__(self, stride) -> None:
        """Subsample the results before saving processing."""
        self.m = self.m[2:-2:stride]
        self.l = self.l[2:-2:stride]
        self.c = self.c[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation.

        beginning so that a new run will start with these values

        save the current results into the temp fields
        """
        # print(f"Reset data with {len(self.m)}, stride = {self.mo.reset_stride}")
        self.mc = np.append(self.mc, self.m[0 : -2 : self.mo.reset_stride])
        # self.dc = np.append(self.dc, self.d[0 : -2 : self.mo.reset_stride])
        self.cc = np.append(self.cc, self.c[0 : -2 : self.mo.reset_stride])

        # copy last result into first field
        self.m[0] = self.m[-2]
        self.l[0] = self.l[-2]
        # self.h[0] = self.h[-2]
        # self.d[0] = self.d[-2]
        self.c[0] = self.c[-2]

    def __merge_temp_results__(self) -> None:
        """Replace the data fields with saved values."""
        self.m = self.mc
        self.c = self.cc
        # self.d = self.dc

    def __read_state__(self, directory: str, prefix="state_") -> None:
        """Read data from csv-file into a dataframe.

        The CSV file must have the following columns
            - Model Time t
            - Species_Name m
            - Species_Name l
            - Species_Name h
            - Species_Name d
            - Species_Name c
            - Flux_name m
            - Flux_name l etc etc.
        """
        read: set = set()
        curr: set = set()

        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{self.mo.n}_{self.full_name}.csv"
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{self.full_name}.csv"
        else:
            raise SpeciesError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Flux {fn} does not exist in Species {self.full_name}"
            )

        self.df: pd.DataFrame = pd.read_csv(fn)
        self.headers: list = list(self.df.columns.values)
        df = self.df
        headers = self.headers

        # the headers contain the object name for each data in the
        # reservoir or flux thus, we must reduce the list to unique
        # object names first. Note, we must preserve order
        header_list: list = []
        for x in headers:
            n = x.split(" ")[0]
            if n not in header_list:
                header_list.append(n)

        # loop over all columns
        col: int = 1  # we ignore the time column
        for n in header_list:
            name = n.split(" ")[0]
            logging.debug(f"Looking for {name}")
            # this finds the reservoir name
            if name == self.full_name:
                # logging.debug(f"found reservoir data for {name}")
                col = self.__assign_reservoir_data__(self, df, col, True)
            else:
                raise SpeciesError(f"Unable to find Flux {n} in {self.full_name}")

        # test if we missed any fluxes
        for f in list(curr.difference(read)):
            warnings.warn(
                f"\nDid not find values for {f}\n in saved state", stacklevel=2
            )

    def __assign_reservoir_data__(
        self, obj: any, df: pd.DataFrame, col: int, res: bool
    ) -> int:
        """Assign the third last entry data to all values in reservoir.

        :param obj: # Species
        :param df: pd.dataframe
        :param col: int # index into column position
        :param res: True # indicates whether obj is reservoir

        :returns: int # index into last column
        """
        if obj.isotopes:
            obj.l[:] = df.iloc[-1, col]  # get last row
            col += 1
            obj.c[:] = df.iloc[-1, col]
            col += 1
        else:
            # v = df.iloc[-1, col]
            obj.c[:] = df.iloc[-1, col]
            col += 1

        return col

    def get_plot_format(self):
        """Return concentrat data in plot units."""
        from pint import Unit

        if isinstance(self.plt_units, Q_):
            unit = f"{self.plt_units.units:~P}"
        elif isinstance(self.plt_units, Unit):
            unit = f"{self.plt_units:~P}"
        else:
            unit = f"{self.plt_units}"

        y1_label = f"{self.legend_left} [{unit}]"

        if self.display_as == "mass":
            y1 = (self.m * self.mo.m_unit).to(self.plt_units).magnitude
        elif self.display_as == "ppm":
            y1 = self.c * 1e6
            y1_label = "ppm"
        elif self.display_as == "length":
            y1 = (self.c * self.mo.l_unit).to(self.plt_units).magnitude
        else:
            y1 = (self.c * self.mo.c_unit).to(self.plt_units).magnitude

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise SpeciesError("Plot transform must be a function")

        return y1, y1_label, unit

    def __plot__(self, M: Model, ax) -> None:
        """Plot Model data.

        :param M: Model
        :param ax: # graph axes handle
        """
        from esbmtk.utility_functions import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude

        y1, y1_label, _unit = self.get_plot_format()

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=y1_label)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(y1_label)

        # add any external data if present
        for i, d in enumerate(self.led):
            leg = f"{self.lm} {d.legend}"
            ax.scatter(d.x[1:-2], d.y[1:-2], color=f"C{i + 2}", label=leg)

        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()
        set_y_limits(ax, self)

        if self.isotopes:
            axt = ax.twinx()
            y2 = self.d  # no conversion for isotopes
            axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.ld)
            set_y_limits(axt, self)
            ax.spines["top"].set_visible(False)
            # set combined legend
            handler2, label2 = axt.get_legend_handles_labels()
            axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(6)
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

        ax.set_title(self.full_name)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Optional arguments are:

        :param index: int = 0 # this will show data at the given index
        :param indent: int = 0 # print indentation
        """
        off: str = "  "
        # index = kwargs.get("index", 0)
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this reservoir
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        # show_data(self, index=index, indent=indent)

        print(f"\n{ind}Connnections:")
        for p in sorted(self.loc):
            print(f"{off}{ind}{p.full_name}.info()")

        print(f"\n{ind}Fluxes:")

        # m = Q_("1 Sv").to("l/a").magnitude
        for i, f in enumerate(self.lof):
            print(f"{off}{ind}{f.full_name}: {self.lodir[i] * f.m[-2]:.2e}")

        print()
        print("Use the info method on any of the above connections")
        print("to see information on fluxes and processes")


class Species(SpeciesBase):
    """Species specific information data fields.

    Example::

        Species(name = "foo",      # Name of reservoir
                  species = S,          # SpeciesProperties handle
                  delta = 20,           # initial delta - optional (defaults  to 0)
                  mass/concentration = "1 unit"  # species concentration or mass
                  volume/geometry = "1E5 l",      # reservoir volume (m^3)
                  plot = "yes"/"no", defaults to yes
                  plot_transform_c = a function reference, optional (see below)
                  legend_left = str, optional, useful for plot transform
                  display_precision = number, optional, inherited from Model
                  register = Model instance
                  isotopes = True/False otherwise use Model.m_type
                  seawater_parameters= dict, optional
                  )

    You must either give mass or concentration.  The result will
    always be displayed as concentration though.

    You must provide either the volume or the geometry keyword.  In
    the latter case provide a list where the first entry is the upper
    depth datum, the second entry is the lower depth datum, and the
    third entry is the total ocean area.  E.g., to specify the upper
    200 meters of the entire ocean, you would write:

    geometry=[0,-200,3.6e14]

    the corresponding ocean volume will then be calculated by the
    calc_volume method in this case the following instance variables
    will also be set:

    self.volume in model units (usually liter) self.are:a surface area
    in m^2 at the upper bounding surface self.sed_area: area of
    seafloor which is intercepted by this box.  self.area_fraction:
    area of seafloor which is intercepted by this relative to the
    total ocean floor area

    It is also possible to specify volume and area explicitly. In this
    case provide a dictionary like this::

        geometry = {"area": "1e14 m**2", # surface area
                    "volume": "3e16 m**3", # box volume
                   }

    Adding seawater_properties:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    If this optional parameter is specified, a SeaWaterConstants instance will
    be registered for this Species as Species.swc See the
    SeaWaterConstants class for details how to specify the parameters,
    e.g.:

    .. code-block:: python

            seawater_parameters = {"temperature": 2,
                                   "pressure": 240,
                                   "salinity" : 35,
                                  }

    Using a transform function:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    In some cases, it is useful to transform the reservoir
    concentration data before plotting it.  A good example is the H+
    concentration in water which is better displayed as pH.  We can do
    this by specifying a function to convert the reservoir
    concentration into pH units::

    .. code-block:: python

        def phc(c :float) -> float:
            # Calculate concentration as pH. c can be a number or numpy array
            import numpy as np
            pH :float = -np.log10(c)
            return pH

    this function can then be added to a reservoir as:

    hplus.plot_transform_c = phc

    You can modify the left legend to suit the transform via the
    legend_left keyword

    Note, at present the plot_transform_c function will only take one
    argument, which always defaults to the reservoir concentration.
    The function must return a single argument which will be
    interpreted as the transformed reservoir concentration.

    Accesing Species Data:
    ~~~~~~~~~~~~~~~~~~~~~~~~

    You can access the reservoir data as:

        - Name.m # mass

        - Name.d # delta

        - Name.c # concentration

    Useful methods include:

        - Name.write_data() # save data to file

        - Name.info() # info Species

    """

    def __init__(self, **kwargs) -> None:
        """Initialize a reservoir.

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
              "name": ["None", (str)],
              "species": ["None", (str, SpeciesProperties)],
              "delta": ["None", (int, float, str)],
              "concentration": ["None", (str, Q_, float)],
              "mass": ["None", (str, Q_)],
              "volume": ["None", (str, Q_)],
              "geometry": ["None", (list, dict, str)],
              "plot_transform_c": ["None", (any)],
              "legend_left": ["None", (str)],
              "plot": ["yes", (str)],
              "groupname": ["None", (str)],
              "rtype": ["regular", (str)],
              "function": ["None", (str, col.Callable)],
              "display_precision": [0.01, (int, float)],
              "register": [
                  "None",
                  (SourceProperties, SinkProperties, Reservoir, ConnectionProperties, Model, str),
              ],
              "parent": [
                  "None",
                  (SourceProperties, SinkProperties, Reservoir, ConnectionProperties, Model, str),
              ],
              "full_name": ["None", (str)],
              "seawater_parameters": ["None", (dict, str)],
              "isotopes": [False, (bool)],
              "ideal_water": ["None", (str, bool)],
              "has_cs1": [False, (bool)],
              "has_cs2": [False, (bool)],

        }

        Required Keywords::

            self.lrk: tp.List = [
              "name",
              "species",
              "register",
              ["volume", "geometry"],
              ["mass", "concentration"],

        ]
        """
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
            phc,
        )
        from esbmtk.sealevel import get_box_geometry_parameters

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": ["None", (int, float, str)],
            "concentration": ["None", (str, Q_, float)],
            "mass": ["None", (str, Q_)],
            "volume": ["None", (str, Q_)],
            "geometry": ["None", (list, dict, str)],
            "geometry_unset": [True, (bool)],
            "plot_transform_c": ["None", (any)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "rtype": ["regular", (str)],
            "function": ["None", (str, callable)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
            ],
            "parent": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
            ],
            "full_name": ["None", (str)],
            "seawater_parameters": ["None", (dict, str)],
            "isotopes": [False, (bool)],
            "ideal_water": ["None", (str, bool)],
            "has_cs1": [False, (bool)],
            "has_cs2": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
            ["volume", "geometry"],
            ["mass", "concentration"],
        ]

        self.__initialize_keyword_variables__(kwargs)

        if isinstance(self.register, Model):
            self.model = self.register
        else:
            self.model = self.register.model
        self.parent = self.register
        self.c = np.zeros(len(self.model.time))
        self.l = np.zeros(len(self.model.time))
        self.m = np.zeros(len(self.model.time))
        self.__set_legacy_names__(kwargs)

        if self.delta != "None":
            self.isotopes = True

        # geoemtry information
        if self.volume == "None":
            get_box_geometry_parameters(self)
        else:
            self.volume = Q_(self.volume).to(self.mo.v_unit)

        # append reservoir volume to list of toc's
        self.model.toc = (*self.model.toc, self.volume.to(self.model.v_unit).magnitude)
        self.v_index = self.model.gcc
        self.model.gcc = self.model.gcc + 1
        self.c_unit = self.model.c_unit
        # This should probably be species specific?
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx

        if self.sp.stype == "concentration":
            if self.mass == "None":
                if isinstance(self.concentration, str | Q_):
                    cc = Q_(self.concentration)
                    # concentration can be mol/kg or mol/l
                    _sm, sc = str(cc.units).split(" / ")  # get
                    _mm, mc = str(self.mo.c_unit).split(" / ")  # model
                    if mc == "liter" and sc == "kilogram":
                        cc = Q_(f"{cc.magnitude} {str(self.mo.c_unit)}")
                        warnings.warn(
                            "\nConvert mol/kg to mol/liter assuming density = 1\n",
                            stacklevel=2,
                        )
                    elif sc != mc:
                        raise ScaleError(
                            f"no transformation for {cc.units} to {self.mo.c_unit}"
                        )
                    self._concentration = cc.to(self.mo.c_unit)
                    self.plt_units = self.mo.c_unit
                else:
                    cc = self.concentration
                    self.plt_units = self.mo.c_unit
                    self._concentration = cc

                self.mass = (
                    self.concentration.to(self.mo.c_unit).magnitude
                    * self.volume.to(self.mo.v_unit).magnitude
                )
                self.mass = Q_(f"{self.mass} {self.mo.c_unit}")
                self.display_as = "concentration"

                # fixme: c should be dimensionless, not sure why this happens
                self.c = self.c.to(self.mo.c_unit).magnitude

                if self.species.scale_to != "None":
                    _c, m = str(self.mo.c_unit).split(" / ")
                    self.plt_units = Q_(f"{self.species.scale_to} / {m}")
            elif self.concentration == "None":
                m = Q_(self.mass)
                self.plt_units = self.mo.m_unit
                self.mass: int | float = m.to(self.mo.m_unit).magnitude
                self.concentration = self.massto(self.mo.m_unit) / self.volume.to(
                    self.mo.v_unit
                )
                self.display_as = "mass"
            else:
                raise SpeciesError("You need to specify mass or concentration")

        elif self.sp.stype == "length":
            self.plt_units = self.mo.l_unit
            self.c = (
                np.zeros(self.mo.number_of_datapoints + 1)
                + Q_(self.concentration).magnitude
            )
            self.display_as = "length"
        self.state = 0

        # save the unit which was provided by the user for display purposes
        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"

        # initialize mass vector
        if self.mass == "None":
            self.m: NDArrayFloat = np.zeros(self.mo.number_of_datapoints + 1)
        else:
            self.m: NDArrayFloat = np.zeros(self.species.mo.steps) + self.mass
        self.l: NDArrayFloat = np.zeros(self.mo.number_of_datapoints + 1)
        # self.c: NDArrayFloat = np.zeros(self.mo.steps)
        self.v: NDArrayFloat = (
            np.zeros(self.mo.number_of_datapoints + 1)
            + self.volume.to(self.mo.v_unit).magnitude
        )  # reservoir volume

        if self.delta != "None":
            self.l = get_l_mass(self.c, self.delta, self.species.r)

        # create temporary memory if we use multiple solver iterations
        if self.mo.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.dc = np.empty(0)

        self.mo.lor.append(self)  # add this reservoir to the model
        if self.rtype != "flux_only":
            self.mo.lic.append(self)  # reservoir type object list

        # register instance name in global name space
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_with_parent__()

        # decide which setitem functions to use
        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

        if self.species.name == "Hplus":
            self.plot_transform_c = phc
        # any auxilliary init - normally empty, but we use it here to extend the
        # reservoir class in virtual reservoirs
        self.__aux_inits__()

    @property
    def concentration(self) -> float:
        """Concentration Setter."""
        return self._concentration

    @property
    def delta(self) -> float:
        """Delta Setter."""
        return self._delta

    @property
    def mass(self) -> float:
        """Mass Setter."""
        return self._mass

    # @property
    # def volume(self) -> float:
    #     return self._volume

    # @volume.setter
    # def volume(self) -> None:
    #     self.volume = self._volume.to(self.register.v_unit)

    @concentration.setter
    def concentration(self, c) -> None:
        if self.update and c != "None":
            breakpoint()
            # this requires unit screening
            # then conversion into model units and mganitude
            # followed by updates to c and m
            self._concentration = c.to(self.mo.c_unit)
            self.mass = (
                self._concentration * self.volume * self.density / 1000
            )  # caculate mass
            self.c = self.c * 0 + self._concentration.magnitude
            self.m = self.m * 0 + self.mass

    @delta.setter
    def delta(self, d: float) -> None:
        if self.update and d != "None":
            self._delta: float = d
            self.isotopes = True
            self.l = get_l_mass(self.c, d, self.species.r)

    @mass.setter
    def mass(self, m: float) -> None:
        if self.update and m != "None":
            self._mass: float = m
            """ problem: m_unit can be mole, but data can be in liter * mole /kg
            this should not happen and results in an error converting to magnitide
            """
            self.m = np.zeros(self.species.mo.number_of_datapoints + 1) + m
            self.c = self.m / self.volume.to(self.mo.v_unit).magnitude


class Flux(esbmtkBase):
    """A class which defines a flux object.

    Flux objects contain
    information which links them to an species, describe things like
    the mass and time unit, and store data of the total flux rate at
    any given time step.  Similarly, they store the flux of the light
    and heavy isotope flux, as well as the delta of the flux.  This is
    typically handled through the Species2Species object.  If you set it up
    manually

    Example::

        Flux = (name = "Name" # optional, defaults to _F
             species = species_handle,
             delta = any number,
             rate  = "12 mol/s" # must be a string
             display_precision = number, optional, inherited from Model

    )

    You can access the flux data as

        - Name.m # mass
        - Name.d # delta
        - Name.c # same as Name.m since flux has no concentration
    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize a flux.

        Arguments are the species name the flux
        rate (mol/year), the delta value and unit

        Example::

        Flux = (name = "Name" # optional, defaults to _F
             species = species_handle,
             delta = any number,
             rate  = "12 mol/s" # must be a string
             display_precision = number, optional, inherited from Model

        )

        You can access the flux data as:

        - Name.m # mass
        - Name.d # delta
        - Name.c # same as Name.m since flux has no concentration

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
              "name": ["None", (str)],
              "species": ["None", (str, SpeciesProperties)],
              "delta": [0, (str, int, float)],
              "rate": ["None", (str, Q_, int, float)],
              "plot": ["yes", (str)],
              "display_precision": [0.01, (int, float)],
              "isotopes": [False, (bool)],
              "register": [
                  "None",
                  (
                      str,
                      Species,
                      GasReservoir,
                      Species2Species,
                      Species2Species,
                      Signal,
                  ),
              ],
              "save_flux_data": [False, (bool)],
              "id": ["None", (str)],
              "ftype": ["None", (str)],

        }

        Required Keywords: "species", "rate", "register"
        """
        from esbmtk import (
            Q_,
            ExternalCode,
            GasReservoir,
            Signal,
            Species,
            Species2Species,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": [0, (str, int, float)],
            "rate": ["None", (str, Q_, int, float)],
            "plot": ["yes", (str)],
            "display_precision": [0.01, (int, float)],
            "isotopes": [False, (bool)],
            "register": [
                "None",
                (
                    str,
                    Species,
                    GasReservoir,
                    Species2Species,
                    Species2Species,
                    Signal,
                ),
            ],
            "save_flux_data": [False, (bool)],
            "id": ["None", (str)],
            "ftype": ["None", (str)],
            "computed_by": ["None", (str, ExternalCode)],
            "serves_as_input": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["species", "rate", "register"]
        self.__initialize_keyword_variables__(kwargs)
        self.parent = self.register

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: SpeciesProperties = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle
        self.rvalue = self.sp.r

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        # model units
        self.plt_units = Q_(self.rate).units
        self.mu: str = f"{self.species.mu}/{self.mo.t_unit}"

        # and convert flux into model units
        if isinstance(self.rate, str):
            self.rate: float = Q_(self.rate).to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, Q_):
            self.rate: float = self.rate.to(self.mo.f_unit).magnitude
        elif isinstance(self.rate, int | float):
            self.rate: float = self.rate

        li = get_l_mass(self.rate, self.delta, self.sp.r) if self.delta else 0
        self.fa: NDArrayFloat = np.asarray([self.rate, li])

        # in case we want to keep the flux data
        if self.save_flux_data:
            self.m: NDArrayFloat = (
                np.zeros(self.model.number_of_datapoints + 1) + self.rate
            )  # add the flux

            if self.isotopes:
                self.l: NDArrayFloat = np.zeros(self.model.number_of_datapoints + 1)
                if self.rate != 0:
                    self.l = get_l_mass(self.m, self.delta, self.species.r)
                    self.fa[1] = self.l[0]

            if self.mo.number_of_solving_iterations > 0:
                self.mc = np.empty(0)
                self.dc = np.empty(0)

        else:
            # setup dummy variables to keep existing numba data structures
            self.m = np.zeros(2)
            self.l = np.zeros(2)

            if self.rate != 0:
                self.fa[1] = get_l_mass(self.fa[0], self.delta, self.species.r)

        self.lm: str = f"{self.species.n} [{self.mu}]"  # left y-axis a label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"  # right y-axis a label

        self.legend_left: str = self.species.dsa
        self.legend_right: str = f"{self.species.dn} [{self.species.ds}]"

        self.xl: str = self.model.xl  # se x-axis label equal to model time
        self.lop: list[Process] = []  # list of processes
        self.lpc: list = []  # list of external functions
        self.led: list[ExternalData] = []  # list of ext data
        self.source: str = ""  # Name of reservoir which acts as flux source
        self.sink: str = ""  # Name of reservoir which acts as flux sink

        if self.name == "None":
            if isinstance(self.parent, (Species2Species)):
                self.name = f"_F{self.id}"
                self.n = self.name
            else:
                self.name = f"{self.id}_F"

        self.__register_with_parent__()
        self.mo.lof.append(self)  # register with model flux list

        # decide which setitem functions to use
        # decide whether we use isotopes
        if self.mo.m_type == "both":
            self.isotopes = True
        elif self.mo.m_type == "mass_only":
            self.isotopes = False

        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
            # self.__get_data__ = self.__get_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__
            # self.__get_data__ = self.__get_without_isotopes__

    # setup a placeholder setitem function
    def __setitem__(self, i: int, value: NDArrayFloat):
        """Setitem function."""
        return self.__set_data__(i, value)

    def __getitem__(self, i: int) -> NDArrayFloat:
        """Get data by index."""
        # return self.__get_data__(i)
        return self.fa

    def __set_with_isotopes__(self, i: int, value: NDArrayFloat) -> None:
        """Write data by index."""
        self.m[i] = value[0]
        self.l[i] = value[1]
        self.fa = value[:4]

    def __set_without_isotopes__(self, i: int, value: NDArrayFloat) -> None:
        """Write data by index."""
        self.fa = [value[0], 0]
        self.m[i] = value[0]

    # FIXME: this does nothing, do we still need it?
    # def __call__(self) -> None:  # what to do when called as a function ()
    #     """"""
    #     pass
    #     return

    def __add__(self, other):
        """Add two fluxes.

        FIXME: adding two fluxes works for the masses, but not for delta
        """
        self.fa = self.fa + other.fa
        self.m = self.m + other.m
        self.l = self.l + other.l

    def __sub__(self, other):
        """Substract two fluxes.

        FIXME: This works for the masses, but not for delta
        """
        self.fa = self.fa - other.fa
        self.m = self.m - other.m
        self.l = self.l - other.l

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.

        Optional arguments are:

        :param index: int = 0 this will show data at the given index
        :param indent: int = 0 indentation
        """
        # index = 0 if "index" not in kwargs else kwargs["index"]
        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this object
        print(f"{ind}{self.__str__(kwargs)}")
        print(f"{ind}Data sample:")
        # show_data(self, index=index, indent=indent)

        if len(self.lop) > 0:
            self._extracted_from_info_27(ind)
        else:
            print("There are no processes for this flux")

    # FIXME Rename this here and in `info`
    def _extracted_from_info_27(self, ind):
        print(f"\n{ind}Process(es) acting on this flux:")
        off: str = "  "
        for p in self.lop:
            print(f"{off}{ind}{p.__repr__()}")

        print("")
        print(
            "Use help on the process name to get an explanation what this process does"
        )
        if self.register == "None":
            print(f"e.g., help({self.lop[0].n})")
        else:
            print(f"e.g., help({self.register.name}.{self.lop[0].name})")

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.

        :param M: Model
        :param ax: matplotlib axes handle
        """
        from esbmtk import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude
        y1 = (self.m * M.m_unit).to(self.plt_units).magnitude

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                y1 = self.plot_transform_c(self.c)
            else:
                raise FluxError("Plot transform must be a function")

        # plot first axis
        ax.plot(x[1:-2], y1[1:-2], color="C0", label=self.legend_left)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(self.legend_left)
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        # plot second axis
        if self.isotopes:
            axt = ax.twinx()
            # FIXME: y2 and ln2 are never used
            # y2 = self.d  # no conversion for isotopes
            # ln2 = axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.data.ld)
            set_y_limits(axt, M)
            ax.spines["top"].set_visible(False)
            # set combined legend
            _handler2, _label2 = axt.get_legend_handles_labels()
            # FIXME: legend is never used
            # legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
            #     6
            # )
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
        ax.set_title(self.full_name)

    def __sub_sample_data__(self, stride) -> None:
        """Subsample the results before saving, or processing."""
        if self.save_flux_data:
            self.m = self.m[2:-2:stride]
            self.l = self.m[2:-2:stride]

    def __reset_state__(self) -> None:
        """Copy the result of the last computation."""
        if self.save_flux_data:
            self.mc = np.append(self.mc, self.m[0 : -2 : self.mo.reset_stride])
            # copy last element to first position
            self.m[0] = self.m[-2]
            self.l[0] = self.l[-2]

    def __merge_temp_results__(self) -> None:
        """Replace the data fields with saved values."""
        self.m = self.mc


class SourceSink(esbmtkBase):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Sink(name = "Pyrite",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """

    def __init__(self, **kwargs) -> None:
        """Initialize class instance.

        Defaults::

            self.defaults: dict[str, tp.List[any, tuple]] = {
               "name": ["None", (str)],
               "species": ["None", (str, SpeciesProperties)],
               "display_precision": [0.01, (int, float)],
               "register": [
                   "None",
                   (
                       SourceProperties,
                       SinkProperties,
                       Reservoir,
                       ConnectionProperties,
                       Model,
                       str,
                   ),
               ],
               "delta": ["None", (str, int, float)],
               "isotopes": [False, (bool)],

        Required Keywords: "name", "species"
        """
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
        )

        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    Model,
                    str,
                ),
            ],
            "delta": ["None", (str, int, float)],
            "epsilon": ["None", (str, int, float)],
            "isotopes": [False, (bool)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":  # use a sensible default
            self.register = self.species.element.register

        self.loc: set[Species2Species] = set()  # set of connection objects

        # legacy names
        # if self.register != "None":
        #    self.full_name = f"{self.name}.{self.register.name}"
        self.parent = self.register
        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.model = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.t_unit)
        self.lio: list = []
        self.m = 1  # set default mass and concentration values
        self.c = 1
        self.mo.lic.append(self)  # add source to list of res type objects

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        elif self.register == "None":
            self.pt = self.name
        else:
            self.pt: str = f"{self.register.name}_{self.n}"
            self.groupname = self.register.name

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_with_parent__()
        self.mo.lic.remove(self)

    @property
    def delta(self):
        """Delta Setter."""
        return self._delta

    @delta.setter
    def delta(self, d):
        """Set/Update delta."""
        if d != "None":
            self._delta = d
            self.isotopes = True
            self.m = 1
            self.c = 1
            self.l = get_l_mass(self.c, d, self.species.r)
            # self.c = self.l / (self.m - self.l)
            # self.provided_kwargs.update({"delta": d})


class Sink(SourceSink):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Sink(name = "Pyrite",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """


class Source(SourceSink):
    """Meta class to setup a Source/Sink objects.

    These are
    not actual reservoirs, but we stil need to have them as objects
    Example::

        Ssource(name = "weathering",
            species = SO4,
            display_precision = number, optional, inherited from Model
            delta = number or str. optional defaults to "None"
            register = Model handle
        )
    """
