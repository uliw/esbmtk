"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G.  Wortmann

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

import copy as cp
import logging
import math
import os
import typing as tp
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

if tp.TYPE_CHECKING:
    from .base_classes import Flux, Model, Species2Species

from .base_classes import Species, SpeciesBase, SpeciesProperties
from .esbmtk_base import esbmtkBase
from .utility_functions import get_delta, get_imass, get_l_mass

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class ReservoirError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class SourceSinkPropertiesError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class SignalError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class DataFieldError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ESBMTKFunctionError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class ExternalDataError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class GasReservoirError(Exception):
    """Custom Error Class."""

    def __init__(self, message):
        """Initialize Error Instance."""
        message = f"\n\n{message}\n"
        super().__init__(message)


class Reservoir(esbmtkBase):
    """Create a group of reservoirs.

    which share
    a common volume, and potentially connections. E.g., if we have twoy
    reservoir groups with the same reservoirs, and we connect them
    with a flux, this flux will apply to all reservoirs in this group.

    A typical examples might be ocean water which comprises several
    species.  A reservoir group like ShallowOcean will then contain
    sub-reservoirs like DIC in the form of ShallowOcean.DIC

    Example::

        Reservoir(name = "ShallowOcean",        # Name of reservoir group
                    volume/geometry = "1E5 l",       # see below
                    delta   = {DIC:0, TA:0, PO4:0]  # dict of delta values
                    mass/concentration = {DIC:"1 unit", TA: "1 unit"}
                    plot = {DIC:"yes", TA:"yes"}  defaults to yes
                    isotopes = {DIC: True/False} see Species class for details
                    seawater_parameters = dict, optional, see below
                    register= model handle, required
               )

    Notes: The subreservoirs are derived from the keys in the concentration or mass
           dictionary. Toward this end, the keys must be valid species handles and
           -- not species names -- !

    Connecting two reservoir groups requires that the names in both
    group match, or that you specify a dictionary which delineates the
    matching.

    Most parameters are passed on to the Species class. See the reservoir class
    documentation for details

    The geometry keyword specifies the upper depth interval, the lower
    depth interval, and the fraction of the total ocean area inhabited by the reservoir

    If the geometry parameter is supplied, the following instance variables will be
    computed:

     - self.volume: in model units (usually liter)
     - self.area: surface area in m^2 at the upper bounding surface
     - self.sed_area: area of seafloor which is intercepted by this box.
     - self.area_fraction: area of seafloor which is intercepted by this relative to the total ocean floor area

    seawater_parameters:
    ~~~~~~~~~~~~~~~~~~~~

    If this optional parameter is specified, a SeaWaterConstants instance will
    be registered for this Species as Species.swc
    See the  SeaWaterConstants class for details how to specify the parameters, e.g.::

        seawater_parameters = {"temperature": 2,
                           "pressure": 240,
                           "salinity" : 35,
                           },

    """

    def __init__(self, **kwargs) -> None:
        """Initialize a new reservoir group."""
        from esbmtk import (
            Q_,
            Model,
            SeawaterConstants,
            SpeciesProperties,
        )
        from esbmtk.sealevel import get_box_geometry_parameters
        from esbmtk.utility_functions import dict_alternatives

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "delta": ["None", (dict, str)],
            "concentration": ["None", (dict, str)],
            "mass": ["None", (str, dict)],
            "volume": ["None", (str, Q_)],
            "geometry": ["None", (str, list, dict)],
            "plot": ["None", (str, dict)],
            "isotopes": [False, (dict, bool)],
            "seawater_parameters": ["None", (dict, str)],
            "carbonate_system": [False, (bool)],
            "register": ["None", (str, Model)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            ["volume", "geometry"],
        ]

        if "concentration" in kwargs:
            self.species: list = list(kwargs["concentration"].keys())
        elif "mass" in kwargs:
            self.species: list = list(kwargs["mass"].keys())
        else:
            raise ReservoirError("You must provide either mass or concentration")

        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.species[0].mo
        # legacy variable
        self.set_area_warning = False
        self.n = self.name
        self.mo = self.species[0].mo
        self.model = self.species[0].mo
        self.parent = self.register
        self.has_cs1 = False
        self.has_cs2 = False

        # geoemtry information
        if self.volume == "None":
            get_box_geometry_parameters(self)
        elif isinstance(self.volume, str):
            self.volume = Q_(self.volume).to(self.mo.v_unit)
        elif not isinstance(self.volume, Q_):
            raise ReservoirError("Volume must be string or quantity")

        self.__register_with_parent__()

        # if len(self.species) == 1:
        la = ["mass", "concentration", "delta", "plot", "isotopes"]
        for a in la:
            if not isinstance(getattr(self, a), dict):
                setattr(self, a, {self.species[0]: getattr(self, a)})

        self.lor: list = []  # list of reservoirs in this group.
        # loop over all entries in species and create the respective reservoirs
        for s in self.species:
            if not isinstance(s, SpeciesProperties):
                raise ReservoirError(f"{s.n} needs to be a valid species name")
            rtype = "flux_only" if s.flux_only else "regular"

            a = Species(
                name=f"{s.name}",
                register=self,
                species=s,
                delta=self.delta.get(s, "None"),
                concentration=self.concentration.get(s, "0 mol/kg"),
                isotopes=self.isotopes.get(s, False),
                plot=self.plot.get(s, "None"),
                volume=self.volume,
                groupname=self.name,
                rtype=rtype,
            )
            # register as part of this group
            self.lor.append(a)

        # register a seawater_parameter instance if necessary
        if self.seawater_parameters != "None":
            temp = dict_alternatives(self.seawater_parameters, "temperature", "T")
            sal = dict_alternatives(self.seawater_parameters, "salinity", "S")
            bar = dict_alternatives(self.seawater_parameters, "pressure", "P")

            if hasattr(self, "DIC") and hasattr(self, "DIC"):
                SeawaterConstants(
                    name="swc",
                    temperature=temp,
                    pressure=bar,
                    salinity=sal,
                    register=self,
                    ta=self.TA.c[0],
                    dic=self.DIC.c[0],
                )
            else:
                SeawaterConstants(
                    name="swc",
                    temperature=temp,
                    pressure=bar,
                    salinity=sal,
                    register=self,
                    ta=0.002,
                    dic=0.002,
                )
                warnings.warn(
                    f"\n\nUsing SeawaterConstants without provinding DIC "
                    f"and TA values for {self.name}\n\n",
                    stacklevel=2,
                )
        self.register.lrg.append(self)


class SourceSinkProperties(esbmtkBase):
    """Create Source/Sink Groups.

    These are not actual reservoirs, but we stil need to have them as objects
    Example::

           SinkProperties(name = "Pyrite",
                species = [SO42, H2S],
                )

    where the first argument is a string, and the second is a reservoir handle
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Class Instance."""
        from esbmtk import Model, Sink, Source, SpeciesProperties

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, list)],
            "delta": [{}, dict],
            "isotopes": [False, (dict, bool)],
            "register": ["None", (str, Model)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        self.__initialize_keyword_variables__(kwargs)

        # register this object
        self.mo = self.species[0].mo  # get model handle
        self.model = self.mo
        if self.register == "None":
            self.register = self.mo
        # legacy variables
        self.n = self.name
        self.parent = self.register
        self.loc: set[Species2Species] = set()  # set of connection objects
        self.__register_with_parent__()
        self.lor: list = []  # list of sub reservoirs in this group

        # input variables can be either dictionaries, single values
        if isinstance(self.species, list):
            la = ["delta", "isotopes"]
            for a in la:
                if not isinstance(getattr(self, a), dict):
                    setattr(self, a, {self.species[0]: getattr(self, a)})

        # loop over species names and setup sub-objects
        for _i, s in enumerate(self.species):
            if isinstance(s, str) and s != "None":
                raise ValueError(f"{s} need to be a species object, not a string")

            if not isinstance(s, SpeciesProperties):
                raise SourceSinkPropertiesError(
                    f"{s.n} needs to be a valid species name"
                )
            if type(self).__name__ == "SourceProperties":
                a = Source(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=self.delta.get(s, "None"),
                )
            elif type(self).__name__ == "SinkProperties":
                a = Sink(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    isotopes=self.isotopes.get(s, False),
                )
            else:
                raise SourceSinkPropertiesError(
                    f"{type(self).__name__} is not a valid class type"
                )
            # register in list of reservoirs
            self.lor.append(a)


class SinkProperties(SourceSinkProperties):
    """A wrapper to setup a Sink object.

    Example::

           SinkProperties(name = "Burial",
                species = [SO42, H2S],
                delta = {"SO4": 10}
                )

    """


class SourceProperties(SourceSinkProperties):
    """A wrapper to setup a Source object.

    Example::

        SourceProperties(name = "weathering",
                species = [SO42, H2S],
                delta = {"SO4": 10}
                )

    """


class Signal(esbmtkBase):
    """Create a signal.

    which is described by its startime (relative to the model time), it's size
    (as mass) and duration, or as
    duration and magnitude. Furthermore, we can presribe the signal shape
    (square, pyramid, bell, file )and whether the signal will repeat. You can
    also specify whether the event will affect the delta value.
    Alternatively, signal can be read from a CSV file. The file needs to
    specify
    Time [unit], Flux [unit/time unit], delta value

    The delta value column is optional.The units must be of similar
    dimensions as the model dimensions (e.g., mol/l or mol/kg). Data
    will be automatically interpolated.

    Example::

          Signal(name = "Name",
                 species = SpeciesProperties handle,
                 start = "0 yrs",     # optional
                 duration = "0 yrs",  #
                 reverse: [False, (bool)], # optional
                 delta = 0,           # optional
                 stype = "addition"   # or multiplication
                 shape = "square/pyramid/bell/filename"
                 mass/magnitude/filename  # give one
                 offset = '0 yrs',     #
                 scale = 1, optional,  #
                 offset = option #
                 reservoir = r-handle # optional, see below
                 source = s-handle optional, see below
                 display_precision = number, optional, inherited from Model
                 register,
                )

    Signals are cumulative, i.e., complex signals ar created by
    adding one signal to another (i.e., Snew = S1 + S2)

    The optional scaling argument will only affect the y-column data of
    external data files

    Signals are registered with a flux during flux creation,
    i.e., they are passed on the process list when calling the
    connector object.

    if the filename argument is used, you can provide a filename which
    contains the data to be used in csv format. The data will be
    interpolated to the model domain, and added to the already existing data.
    The external data need to be in the following format

      Time, Rate, delta value
      0,     10,   12

      i.e., the first row needs to be a header line

    All time data in the csv file will be treated as realative time
    (i.e., the start time will be mapped to zero). The reverse
    keyword can be used to invert a signal that is read from a csv
    file.

    Last but not least, you can provide an optional reservoir name. In
    this case, the signal will create a source as (signal_name_source)
    and the connection to the specified reservoir. If you build a
    complex signal do this as the last step. If you additionally
    provide a source name the connection will be made between the
    provided source (this can be useful if you use source groups).

    This class has the following methods

      Signal.repeat()
      Signal.plot()
      Signal.info()

    The signal class provides the following data fields

        self.data.m which contains the interpolated signal
                    also available as self.m
        self.data.l which contain the interpolated isotope
                    data in the form of the light isotope
                    also availavle as self.l
                    If no isotope data is given, it is 0

        self.ed is the object reference for the externaldata
                   instance in cases wher data is read from
                   a csv file
    """

    def __init__(self, **kwargs) -> None:
        """Parse and initialize variables."""
        from esbmtk import Q_, Model, Sink, Source, Species, SpeciesProperties

        # provide a list of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "start": ["0 yrs", (str, Q_)],
            "duration": ["None", (str, Q_)],
            "species": ["None", (SpeciesProperties)],
            "delta": [0, (int, float)],
            "stype": ["addition", (str)],
            "reverse": [False, (bool)],
            "shape": ["None", (str)],
            "filename": ["None", (str)],
            "mass": ["None", (str, Q_)],
            "magnitude": ["None", (str, Q_)],
            "offset": ["0 yrs", (str, Q_)],
            "plot": ["no", (str)],
            "scale": [1, (int, float)],
            "display_precision": [0.01, (int, float)],
            "reservoir": ["None", (Source, Sink, Species, str)],
            "source": ["None", (Source, Sink, Species, str)],
            "legend_right": ["None", (str)],
            "register": ["None", (str, Model)],
            "isotopes": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = [
            "name",
            ["duration", "filename"],
            "species",
            ["shape", "filename"],
            ["magnitude", "mass", "filename"],
        ]

        self.__initialize_keyword_variables__(kwargs)

        # list of signals we are based on
        self.los: list[Signal] = []

        # convert units to model units
        self.st: int | float = int(
            Q_(self.start).to(self.species.mo.t_unit).magnitude
        )  # start time

        if "mass" in self.kwargs:
            self.mass = Q_(self.mass).to(self.species.mo.m_unit).magnitude
        elif "magnitude" in self.kwargs:
            self.magnitude = Q_(self.magnitude).to(self.species.mo.f_unit).magnitude

        if "duration" in self.kwargs:
            self.duration = int(Q_(self.duration).to(self.species.mo.t_unit).magnitude)
            if self.duration / self.species.mo.dt < 10:
                warnings.warn(
                    """\n\n Your signal duration is covered by less than 10
                    Intergration steps. This may not be what you want
                    Consider adjusting the max_step parameter of the model object\n""",
                    stacklevel=2,
                )

        self.offset = Q_(self.offset).to(self.species.mo.t_unit).magnitude

        # legacy name definitions
        self.full_name = ""
        self.l: int = self.duration
        self.n: str = self.name  # the name of the this signal
        self.sp: SpeciesProperties = self.species  # the species
        self.mo: Model = self.species.mo  # the model handle
        self.model = self.mo
        if self.register == "None":
            self.register = self.mo
        self.parent = self.register
        self.ty: str = self.stype  # type of signal
        self.sh: str = self.shape  # shape the event
        self.d: float = self.delta  # delta value offset during the event
        self.kwd: dict[str, any] = self.kwargs  # list of keywords
        self.led: list = []

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.s_m, self.s_l = self.__init_signal_data__()
        self.signal_data = self.__map_signal__()
        """ self.__map_signal__() returns a Flux object, so we need to remove this
        from the list of model Fluxes, with
        self.mo.lof.remove(self.data)
        since we do not use is as a Flux. Probably
        better to just create a vector object instead. FIXME sometime
        """
        self.mo.lof.remove(self.signal_data)
        self.m = self.signal_data.m
        if self.isotopes:
            self.l = self.signal_data.l

        self.signal_data.n: str = (
            self.name + "_data"
        )  # update the name of the signal data
        self.legend_left = self.signal_data.legend_left
        self.legend_right = self.signal_data.legend_right
        # update isotope values
        # self.data.li = get_l_mass(self.data.m, self.data.d, self.sp.r)

        self.__register_with_parent__()
        self.mo.los.append(self)  # register with model

        # in case we deal with a sink or source signal
        if self.reservoir != "None":
            self.__apply_signal__()

    def __init_signal_data__(self) -> None:
        """1. Create a vector which contains the signal data.

        The vector length
           can exceed the modelling domain.
        2. Trim the signal vector in such a way that it fits within the
           modelling domain
        3. Create an empty flux and replace it with the signal vector data.

        Note that this flux will then be added to an existing flux.
        """
        # these are signal times, not model time
        if self.duration != "None":
            self.length: int = int(round(self.duration / self.mo.dt))
        # create signal vector
        if self.sh == "square":
            self.__square__(0, self.length)
        elif self.sh == "pyramid":
            self.__pyramid__(0, self.length)
        elif self.sh == "bell":
            self.__bell__(0, self.length)
        elif "filename" in self.kwargs:  # use an external data set
            self.length = self.__int_ext_data__()
        else:
            raise SignalError(
                "argument needs to be either square/pyramid, or an ExternalData object. "
            )

        return self.s_m, self.s_l

    def __map_signal__(self) -> Flux:
        """Map signal to model domain.

        s.t. signal data fits the time grid of model.
        Returns mapped data.
        """
        from esbmtk import Flux

        # Create a dummy flux we can act up
        mapped_signal_data: Flux = Flux(
            name=self.n + "_data",
            species=self.sp,
            rate=f"0 {self.sp.mo.f_unit}",
            delta=0,
            isotopes=self.isotopes,
            save_flux_data=True,
            register=self,
        )
        # Creating signal time array
        dt = self.mo.dt  # model time step
        signal_start = self.st
        signal_end = self.st + self.duration if self.filename == "None" else self.et
        model_time = self.mo.time  # model time array
        # imitate signal time array with same size as interpolated signal data (self.s_m)
        signal_time = np.arange(signal_start, signal_end, dt)

        # Create initial results, which are all nan Check with self.stype
        # whether it's addition, then 0 as default, or multiplication then 1 as
        # default (instead of nan)
        mapped_time = np.full_like(model_time, np.nan, dtype=float)
        if self.stype == "addition":
            mapped_m = np.full_like(model_time, 0, dtype=float)
        elif self.stype == "multiplication":
            mapped_m = np.full_like(model_time, 1, dtype=float)
        else:
            # in case something is wrong with stype, still create mapped data
            mapped_m = np.full_like(model_time, np.nan, dtype=float)

        mapped_l = np.full_like(
            model_time, np.nan, dtype=float
        )  # keep it as is for isotopes until clarified

        # Filter signal time which exists in model time
        # If signal time in model time, return True in mask
        # Every time element in model time flagged with True is
        # collected in mapped_time (array is only used within this method)
        mask = np.isin(model_time, signal_time)
        mapped_time[mask] = model_time[mask]

        # Go through mapped_time to check where there was a match between model
        # and signal times. Collect signal data for where times matched
        for i, t in enumerate(mapped_time):
            if t >= 0:
                signal_index = np.searchsorted(signal_time, t)
                mapped_m[i] = self.s_m[signal_index]
                if self.isotopes:
                    mapped_l[i] = self.s_l[
                        signal_index
                    ]  # TODO: for future thinking how to calculate isotope fluxes

        if self.reverse:
            mapped_signal_data.m = np.flip(mapped_m)
            mapped_signal_data.l = np.flip(mapped_l)
        else:
            mapped_signal_data.m = mapped_m
            mapped_signal_data.l = mapped_l

        return mapped_signal_data

    def __square__(self, s, e) -> None:
        """Create Square Signal."""
        self.s_m: NDArrayFloat = np.zeros(e - s)
        self.s_d: NDArrayFloat = np.zeros(e - s)

        if "mass" in self.kwd:
            h = self.mass / self.duration  # get the height of the square
            self.magnitude = h

        elif "magnitude" in self.kwd:
            h = self.magnitude
            self.mass = h * self.duration
        else:
            raise SignalError("You must specify mass or magnitude of the signal")

        self.s_m = self.s_m + h  # add this to the section
        self.s_d = self.s_d + self.d  # add the delta offset
        self.s_l = get_l_mass(self.s_m, self.s_d, self.sp.r)

    def __pyramid__(self, s, e) -> None:
        """Create pyramid type Signal.

        s = start index
        e = end index
        """
        if "mass" in self.kwd:
            h = 2 * self.mass / self.duration  # get the height of the pyramid

        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise SignalError("You must specify mass or magnitude of the signal")

        # create pyramid
        c: int = int(round((e - s) / 2))  # get the center index for the peak
        x: NDArrayFloat = np.array([0, c, e - s])  # setup the x coordinates
        y: NDArrayFloat = np.array([0, h, 0])  # setup the y coordinates

        d: NDArrayFloat = np.array([0, self.d, 0])  # setup the d coordinates
        xi = np.arange(0, e - s)  # setup the points at which to interpolate

        self.s_m: NDArrayFloat = np.interp(xi, x, y)  # interpolate flux
        self.s_d: NDArrayFloat = np.interp(xi, x, d)  # interpolate delta
        self.s_l = get_l_mass(self.s_m, self.s_d, self.sp.r)

    def __bell__(self, s, e) -> None:
        """Create a bell curve type signal.

        s = start index
        e = end index

        Note that the area under the curve equals one.
        So we can scale the result simply with mass
        """
        import sys

        c: int = int(round((e - s) / 2))  # get the center index for the peak
        x: NDArrayFloat = np.arange(-c, c + 1, 1)
        e: float = math.e
        pi: float = math.pi
        mu: float = 0
        phi: float = c / 4

        print(f"mu = {mu} ,phi = {phi}")
        print(f"x[0] = {x[0]}, x[-1] = {x[-1]}")
        print(sys.float_info)

        a = -((x - mu) ** 2) / (2 * phi**2)
        # get bell curve
        self.s_m = 1 / (phi * math.sqrt(2 * pi)) * e**a
        self.s_d = self.s_m * self.delta / max(self.s_m)
        self.s_l = self.s_m

        if "mass" in self.kwargs:
            self.s_m = self.s_m * self.mass
        elif "magnitude" in self.kwargs:
            self.s_m = self.s_m * self.magnitude / max(self.s_m)
        else:
            raise SignalError("Bell type signal require either mass or magnitude")

    def __int_ext_data__(self) -> None:
        """Interpolate External data as a signal.

        Unlike the other signals,
        this will replace the values in the flux with those read from the
        external data source. The external data need to be in the following format

        Time [units], Rate [units], delta value [units]
        0,     10,   12

        i.e., the first row needs to be a header line

        """
        self.ed = ExternalData(
            name=f"{self.name}_ed",
            filename=self.filename,
            register=self,
            legend=f"{self.name}_ed",
            disp_units=False,  # we need the data in model units
        )

        self.s_time = self.ed.x
        self.s_data = self.ed.y * self.scale
        self.st: float = self.s_time[0]  # signal start time
        self.et: float = self.s_time[-1]  # signal end time
        signal_duration = self.et - self.st
        model_time_step = self.mo.dt
        # Calculate how many data points are needed to interpolate signal duration with model time step
        num_steps = int(signal_duration / model_time_step)
        # setup the points at which to interpolate
        xi = np.linspace(self.st, self.et, num_steps + 1)

        self.s_m: NDArrayFloat = np.interp(
            xi, self.s_time, self.s_data
        )  # interpolate flux
        if self.ed.zh:
            self.s_delta = self.ed.d
            self.s_d: NDArrayFloat = np.interp(xi, self.s_time, self.s_delta)
            self.s_l = get_l_mass(self.s_m, self.s_d, self.sp.r)
        else:
            self.s_l: NDArrayFloat = np.zeros(num_steps)

        return num_steps

    def __apply_signal__(self) -> None:
        """Create a source, and connect signal, source and reservoir.

        Maybe this logic should be me moved elsewhere?
        """
        from esbmtk import Source, Species2Species

        if self.source == "None":
            self.source = Source(name=f"{self.name}_Source", species=self.sp)

        Species2Species(
            source=self.source,  # source of flux
            sink=self.reservoir,  # target of flux
            rate="0 mol/yr",  # flux rate
            signal=self,  # list of processes
            plot="no",
        )

    def __add__(self, other):
        """Allow the addition of two signals and return a new signal.

        FIXME: this requires cleanup
        """
        new_signal = cp.deepcopy(self)
        new_signal.m = self.m + other.m
        # get delta of self
        if self.isotopes:
            this_delta = get_delta(self.l, self.m - self.l, self.signal_data.sp.r)
            other_delta = get_delta(other.l, other.m - other.l, other.data.sp.r)
            new_signal.l = get_l_mass(
                new_signal.m, this_delta + other_delta, new_signal.signal_data.sp.r
            )
            # new_signal.l = max(self.l, other.l)

        new_signal.name: str = self.name + "_and_" + other.name
        # print(f"adding {self.n} to {other.n}, returning {ns.n}")
        # new_signal.data.n: str = self.n + "_and_" + other.n + "_data"
        new_signal.st = min(self.st, other.st)

        new_signal.sh = "compound"
        new_signal.los.append(self)
        new_signal.los.append(other)

        return new_signal

    def repeat(self, start, stop, offset, times) -> None:
        """Create a new signal by repeating an existing signal.

        Example::

            new_signal = signal.repeat(start,   # start time of signal slice to be repeated
                                       stop,    # end time of signal slice to be repeated
                                       offset,  # offset between repetitions
                                       times,   # number of time to repeat the slice
                                       )

        """
        ns: Signal = cp.deepcopy(self)
        ns.n: str = self.n + f"_repeated_{times}_times"
        ns.signal_data.n: str = self.n + f"_repeated_{times}_times_data"
        start: int = int(start / self.mo.dt)  # convert from time to index
        stop: int = int(stop / self.mo.dt)
        offset: int = int(offset / self.mo.dt)
        ns.start: float = start
        ns.stop: float = stop
        ns.offset: float = stop - start + offset
        ns.times: float = times
        ns.ms: NDArrayFloat = self.signal_data.m[
            start:stop
        ]  # get the data slice we are using
        ns.ds: NDArrayFloat = self.signal_data.d[start:stop]

        diff = 0
        for _ in range(times):
            start += ns.offset
            stop += ns.offset
            if start > len(self.signal_data.m):
                break
            elif stop > len(self.signal_data.m):  # end index larger than data size
                diff: int = stop - len(self.signal_data.m)  # difference
                stop -= diff
                lds: int = len(ns.ds) - diff
            else:
                lds: int = len(ns.ds)

            ns.signal_data.m[start:stop]: NDArrayFloat = (
                ns.signal_data.m[start:stop] + ns.ms[:lds]
            )
            ns.signal_data.d[start:stop]: NDArrayFloat = (
                ns.signal_data.d[start:stop] + ns.ds[:lds]
            )

        # and recalculate li and hi
        ns.signal_data.l: NDArrayFloat
        ns.signal_data.h: NDArrayFloat
        [ns.signal_data.l, ns.signal_data.h] = get_imass(
            ns.signal_data.m, ns.signal_data.d, ns.signal_data.sp.r
        )
        return ns

    def __register_with_flux__(self, flux) -> None:
        """Register this signal with a flux.

        This should probably be done
        through a process!

        """
        self.fo: Flux = flux  # the flux handle
        self.sp: SpeciesProperties = flux.sp  # the species handle
        # list of processes
        flux.lop.append(self)

    def __call__(self, t) -> list[float, float]:
        """Return Signal value at time t.

        (mass and mass for light
        isotope). This will work as long a t is a multiple of dt, and i = t.
        may extend this by addding linear interpolation but that will
        be costly

        """
        import numpy as np

        m = np.interp(t, self.mo.time, self.m)
        lm = np.interp(t, self.mo.time, self.l) if self.isotopes else 0

        return [m, lm]

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.

        M: Model
        ax: matplotlib axes handle
        """
        from esbmtk import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude
        y1 = (self.signal_data.m * M.f_unit).to(self.signal_data.plt_units).magnitude
        legend = f"{self.legend_left} [{M.f_unit}]"

        # # test for plt_transform
        # if self.plot_transform_c != "None":
        #     if Callable(self.plot_transform_c):
        #         y1 = self.plot_transform_c(self.c)
        #     else:
        #         raise ValueError("Plot transform must be a function")

        # plot first axis
        ln1 = ax.plot(x[1:-2], y1[1:-2], color="C0", label=self.legend_left)
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(legend)
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        # plot second axis
        if self.isotopes:
            axt = ax.twinx()
            y2 = self.d  # no conversion for isotopes
            ln2 = axt.plot(x[1:-2], y2[1:-2], color="C1", label=self.legend_right)
            axt.set_ylabel(self.signal_data.ld)
            set_y_limits(axt, M)
            x.spines["top"].set_visible(False)
            # set combined legend
            handler2, label2 = axt.get_legend_handles_labels()
            legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
                6
            )
        else:
            ax.legend()
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

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

        This function is called by the
        write_data() and save_state() methods

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

        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"

        rn = self.full_name  # reservoir name
        mn = self.sp.mo.n  # model name
        fn = f"{directory}/{prefix}{rn}.csv"  # file name

        # build the dataframe
        df: pd.dataframe = DataFrame()

        # breakpoint()
        df[f"{rn} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        # df[f"{rn} {sn} [{smu}]"] = self.m.to(self.mo.m_unit).magnitude[start:stop:stride]  # mass
        if self.isotopes:
            # print(f"rn = {rn}, sp = {sp.name}")
            df[f"{rn} {sp.ln} [{cmu}]"] = self.l[start:stop:stride]  # light isotope
        df[f"{rn} {sn} [{cmu}]"] = self.signal_data.m[
            start:stop:stride
        ]  # concentration

        file_path = Path(fn)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)
        return df

    def __init_signal_function__(self):
        """Initialize a an external code instance that can be called by the solver."""
        if self.isotopes:
            p = (self.mo.time, self.m)
            function_name = "signal_with_istopes"
        else:
            p = (self.mo.time, self.m, self.l)
            function_name = "signal_no_istopes"

        ec = ExternalCode(
            name=f"ec_signal_{self.name}",
            fname=function_name,
            ftype="std",
            species=self.species,
            function_input_data=["t"],
            function_params=p,
            register=self.model,
            return_values=[
                {"N_one": "None"},
            ],
        )
        self.mo.lpc_f.append(ec.fname)
        return ec


class VectorData(esbmtkBase):
    """A simple container for 1-dimensional data.

    Typically used for results obtained by postprocessing.
    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize Class."""
        from pint import Unit

        self.defaults: dict[str, list(str, tuple)] = {
            "name": ["None", (str,)],
            "register": ["None", (str, Reservoir)],
            "species": ["None", (str, SpeciesProperties)],
            "data": ["None", (str, np.ndarray, float)],
            "isotopes": [False, (bool,)],
            "plt_units": ["None", (str, Unit)],
            "label": ["None", (str, bool)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "register",
            "species",
            "data",
            "label",
            "plt_units",
        ]
        self.__initialize_keyword_variables__(kwargs)

        self.n = self.name
        self.parent = self.register
        self.sp = self.species
        self.mo = self.species.mo
        self.model = self.species.mo
        self.c = self.data
        self.x = self.c
        self.label = self.name
        self.register.model.lvd.append(self)
        self.__register_with_parent__()

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

        from esbmtk import SpeciesError

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp  # species handle
        mo = self.sp.mo  # model handle

        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
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

    def get_plot_format(self):
        """Return concentrat data in plot units."""
        from pint import Unit

        from esbmtk import Q_

        if isinstance(self.plt_units, Q_):
            unit = f"{self.plt_units.units:~P}"
        elif isinstance(self.plt_units, Unit):
            unit = f"{self.plt_units:~P}"
        else:
            unit = f"{self.plt_units}"

        y1_label = f"{self.label} [{unit}]"

        y1 = self.c

        return y1, y1_label, unit


class DataField(esbmtkBase):
    """DataField Class.

    Datafields can be used to plot data which is computed after
    the model finishes in the overview plot windows. Therefore, datafields will
    plot in the same window as the reservoir they are associated with.
    Datafields must share the same x-axis is the model, and can have up to two
    y axis.

    Example::

             DataField(name = "Name"
                       register = Model handle,
                       x1_data =  ["None", (np.ndarray, list)], defaults to model time
                       y1_data = NDArrayFloat or list of arrays
                       y1_label = Data label(s)
                       y1_legend = Y-Axis Label
                       y1_type = "plot", | "scatter"
                       y2_data = NDArrayFloat    # optional
                       y2_legend = Y-Axis label # optional
                       y2_label = Data legend(s) # optional
                       y2_type = "plot", | "scatter"
                       common_y_scale = "no",  #optional, default "no"
                       display_precision = number, optional, inherited from Model
                       )

    All y1 and y2 keywords can be either a single value or a list of values.
    Note that Datafield data is not mapped to model units. Care must be taken
    that the data units match the model units.

    The instance provides the following data

    Name.x    = X-axis = model X-axis
    Name.y1_data
    Name.y1_label
    Name.y1_legend

    Similarly for y2

    You can specify more than one data set, and be explicit about color and
    linestyle choices.

    Example::

            DataField(
                    name="df_pH",
                    x1_data=[M.time, M.time, M.time, M.ef_hplus_l.x, M.ef_hplus_h.x, M.ef_hplus_d.x],
                    y1_data=[
                    -np.log10(M.L_b.Hplus.c),
                    -np.log10(M.H_b.Hplus.c),
                    -np.log10(M.D_b.Hplus.c),
                    -np.log10(M.ef_hplus_l.y),
                    -np.log10(M.ef_hplus_h.y),
                    -np.log10(M.ef_hplus_d.y),
                    ],
                    y1_label="Low latitude, High latitude, Deep box, d_L, d_H, d_D".split(", "),
                    y1_color="C0 C1 C2 C0 C1 C2".split(" "),
                    y1_style="solid solid solid dotted dotted dotted".split(" "),
                    y1_legend="pH",
                    register=M,
                    )

    """

    def __init__(self, **kwargs: dict[str, any]) -> None:
        """Initialize this instance."""
        from . import ExternalCode, Model, SpeciesNoSet, VirtualSpecies

        # dict of all known keywords and their type
        self.defaults: dict[str, list(str, tuple)] = {
            "name": ["None", (str)],
            "register": [
                "None",
                (
                    Model,
                    Species,
                    Reservoir,
                    SpeciesNoSet,
                    VirtualSpecies,
                    ExternalCode,
                ),
            ],
            "associated_with": [
                "None",
                (
                    Model,
                    Species,
                    Reservoir,
                    SpeciesNoSet,
                    VirtualSpecies,
                    ExternalCode,
                ),
            ],
            "y1_data": ["None", (np.ndarray, list)],
            "x1_data": ["None", (np.ndarray, list, str)],
            "x1_as_time": [False, (bool)],
            "y1_label": ["Not Provided", (str, list)],
            "y1_legend": ["Not Provided", (str)],
            "y1_type": ["plot", (str, list)],
            "y1_color": ["None", (str, list)],
            "y1_style": ["None", (str, list)],
            "y2_data": ["None", (str, np.ndarray, list)],
            "x2_data": ["None", (np.ndarray, list, str)],
            "x2_as_time": [False, (bool)],
            "y2_label": ["Not Provided", (str, list)],
            "y2_legend": ["Not Provided", (str)],
            "y2_type": ["plot", (str, list)],
            "y2_color": ["None", (str, list)],
            "y2_style": ["None", (str, list)],
            "common_y_scale": ["no", (str)],
            "display_precision": [0.01, (int, float)],
            "title": ["None", (str)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", ["register", "associated_with"], "y1_data"]

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initialize_keyword_variables__(kwargs)

        if self.register == "None":
            self.register = self.associated_with

        # set legacy variables
        self.legend_left = self.y1_legend
        self.isotopes = False
        self.parent = self.register

        # if self.associated_with == "None":
        #     self.associated_with = self.mo.lor[0]

        # self.mo = self.associated_with.mo
        self.mo = self.register.mo
        self.model = self.mo
        if self.associated_with == "None":
            if isinstance(self.register, Model):
                self.associated_with = self.register.lor[0]
            elif isinstance(self.register, Species):
                self.associated_with = self.register
            else:
                raise DataFieldError(
                    "Set associated_with or register to a reservoir name"
                )

        if isinstance(self.associated_with, Species):
            self.plt_units = self.associated_with.plt_units
        elif isinstance(self.associated_with, Reservoir):
            self.plt_units = self.associated_with.lor[0].plt_units
        else:
            raise ValueError("This needs fixing")

        if self.y1_color == "None":
            self.y1_color = []
            for i, _d in enumerate(self.y1_data):
                self.y1_color.append(f"C{i}")

        if self.y1_style == "None":
            self.y1_style = []
            for _i, _d in enumerate(self.y1_data):
                self.y1_style.append("solid")

        if self.y2_color == "None":
            self.y2_color = []
            for i, _d in enumerate(self.y2_data):
                self.y2_color.append(f"C{i}")

        if self.y2_style == "None":
            self.y2_style = []
            for _i, _d in enumerate(self.y2_data):
                self.y2_style.append("solid")

        self.n = self.name
        self.led = []

        # register with reservoir
        # self.associated_with.ldf.append(self)
        # register with model. needed for print_reservoirs
        self.register.ldf.append(self)
        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_with_parent__()

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """Write_data and save_state.

        This method is called by the respective write_data and save_state
        functions.
        """
        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        # some short hands
        mo = self.mo  # model handle

        mtu = f"{mo.t_unit:~P}"
        rn = self.n  # reservoir name
        mn = self.mo.n  # model name

        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{mn}_{rn}.csv"  # file name
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{rn}.csv"  # file name
        else:
            raise DataFieldError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{self.n} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        df[f"{self.n} {self.y1_legend}"] = self.y1_data[start:stop:stride]  # y1 data

        if self.y2_data != "None":
            df[f"{self.n} {self.y2_legend}"] = self.y2_data[
                start:stop:stride
            ]  # y2_data

        file_path = Path(fn)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)
        return df

    def __unify_data__(self, M: Model, x, y, label) -> tuple(list, list):
        """Format data so that it can be used by the __plot__ method.

        The input data for the DataField Class can be either missing, be a
        single array, or a list of arrays.  The module analyzes the data and
        modifies is in such a way that it can be used by the __plot__ method
        without further adjustments.

        :param x: str, array, list, withe the x-data
        :param y: str, array, list, withe the y-data

        :return x,y: as list
        """
        import numpy as np

        # print(f"ud0: type x = {type(x)}, len = {len(x)}")
        # print(f"ud1: type y = {type(x)}, len = {len(y)}")
        # consider the y-axis first
        if isinstance(y, str):
            ValueError("Provide at least one data-set")
        elif isinstance(y, np.ndarray):
            y = [y]
            y_l = 1
        elif isinstance(y, list):
            y_l = len(y)
        else:
            raise DataFieldError("Y data needs to be array, numpy array or list")

        # consider the x-axis next
        if isinstance(x, str):  # no x-data has been provided
            if y_l == 1:  # single y data
                x = [M.time]
                # print(f" no x-data y_l = {y_l}")
            else:  # mutiple y data
                # print(f" no x-data mutiple y-data y_l = {y_l}")
                x = []
                for _e in range(y_l):
                    x.append(M.time)

        elif isinstance(x, np.ndarray):
            # print(f"np_array, y_l = {y_l}")
            xx = []
            if y_l == 1:  # single y data
                xx.append(x)
            else:  # mutiple y data
                # print(f" no x-data mutiple y-data y_l = {y_l}")
                for _e in range(y_l):
                    xx.append(x)
            x = xx
            # print(f"after: {type(x)}")
        elif isinstance(x, list):  # assume that lists match
            if len(x) != len(y):
                raise DataFieldError(f"Y data needs to match x data for {label}")
        else:
            raise DataFieldError(
                f"Y data needs to be array, numpy array or list for {label}"
            )

        if y_l == 1:
            if not isinstance(label, list):
                label = [label]
        elif y_l > 1 and not isinstance(label, list):
            ll = []
            # FIXME: Add test that this is list otherwise error
            for _e in y_l:
                ll.append(label)
            label = ll

        return x, y, label

    def __plot_data__(self, ax, x, y, t, label, i, color, style) -> None:
        """Plot data either as line or scatterplot.

        :param ax: axis handle
        :param x: x data
        :param y: y data
        :param t: plot type
        :param l: label str
        :param i: index

        """
        if t == "plot":
            # ax.plot(x, y, color=f"C{i}", label=l)
            ax.plot(x, y, color=color[i], linestyle=style[i], label=label)
        else:
            ax.scatter(x, y, color=color[i], label=label)

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.

        M: Model
        ax: matplotlib axes handle
        """
        # plot external data first
        for i, d in enumerate(self.led):
            time = (d.x * M.t_unit).to(M.d_unit).magnitude
            # yd = (d.y * M.c_unit).to(self.plt_units).magnitude
            leg = f"{d.legend}"
            ax.scatter(time[1:-2], d.y[1:-2], color=f"C{i}", label=leg)

        self.x1_data, self.y1_data, self.y1_label = self.__unify_data__(
            M,
            self.x1_data,
            self.y1_data,
            self.y1_label,
        )

        ymin = list()
        ymax = list()
        for i, _d in enumerate(self.y1_data):  # loop over datafield list
            if self.x1_as_time:
                x1 = (self.x1_data[i] * M.t_unit).to(M.d_unit).magnitude
            else:
                x1 = self.x1_data[i]
                # y1 = (self.y * M.c_unit).to(self.plt_units).magnitude
                # 1 = self.y
                # y1_label = f"{self.legend_left} [{self.plt_units:~P}]"

            ptype = self.y1_type[i] if isinstance(self.y1_type, list) else self.y1_type

            self.__plot_data__(
                ax,
                x1,
                self.y1_data[i],
                ptype,
                self.y1_label[i],
                i,
                self.y1_color,
                self.y1_style,
            )
            last_i = i
            u, v = ax.get_ylim()
            ymin.append(u)
            ymax.append(v)
            # set_y_limits(ax, self)
        # add any external data if present
        ymin = min(ymin)
        ymax = max(ymax)
        ax.set_ylim([ymin, ymax])

        last_i = i
        ax.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
        ax.set_ylabel(self.y1_legend)
        # remove unnecessary frame species
        ax.spines["top"].set_visible(False)
        handler1, label1 = ax.get_legend_handles_labels()

        # test if independeny y_data is present
        ymin = list()
        ymax = list()
        if not isinstance(self.y2_data, str):
            self.x2_data, self.y2_data, self.y2_label = self.__unify_data__(
                M,
                self.x2_data,
                self.y2_data,
                self.y2_label,
            )
            axt = ax.twinx()
            for i, _d in enumerate(self.y2_data):  # loop over datafield list
                if self.x2_as_time:
                    x2 = (self.x1_data[i] * M.t_unit).to(M.d_unit).magnitude
                else:
                    x2 = self.x1_data[i]
                self.__plot_data__(
                    axt,
                    x2,
                    self.y2_data[i],
                    self.y2_type,
                    self.y2_label[i],
                    i + last_i + 1,
                    self.y2_color,
                    self.y2_style,
                )
            u, v = axt.get_ylim()
            ymin.append(u)
            ymax.append(v)
            ymin = min(ymin)
            ymax = max(ymax)
            axt.set_ylim([ymin, ymax])

            axt.set_xlabel(f"{M.time_label} [{M.d_unit:~P}]")
            axt.set_ylabel(self.y2_legend)
            # remove unnecessary frame species
            axt.spines["top"].set_visible(False)
            handler2, label2 = axt.get_legend_handles_labels()
            legend = axt.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(
                6
            )
            axt.legend()
        else:
            ax.legend(handler1, label1)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
        if self.title != "None":
            ax.set_title(self.title)
        else:
            ax.set_title(self.register.full_name)


class SpeciesNoSet(SpeciesBase):
    """class that makes no assumptions about the type of data.

    I.e., all data will be left alone. The original class will calculate delta and
    concentration from mass an d and h and l. Since we want to use this class without a
    priory knowledge of how the reservoir arrays are being used we overwrite the data
    generated during initialization with the values provided in the keywords
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Class."""
        from esbmtk import (
            ConnectionProperties,
            Reservoir,
            SinkProperties,
            SourceProperties,
            SpeciesProperties,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "plot_transform_c": ["None", (str, tp.Callable)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "function": ["None", (str, tp.Callable)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    ConnectionProperties,
                    str,
                ),
            ],
            "full_name": ["None", (str)],
            "isotopes": [False, (bool)],
            "volume": ["None", (str, int, float)],
            "vr_datafields": [{}, (dict)],
            "function_input_data": (list, str),
            "function_params": [list(), (list, str)],
            "geometry": ["None", (list, str)],
            "alias_list": ["None", (list, str)],
            "ref_flux": ["None", (list, str)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
        ]

        self.__validateandregister__(kwargs)
        self._initialize_legacy_attributes(kwargs)

        self.isotopes = False
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx
        self.plt_units = self.mo.c_unit
        # save the unit which was provided by the user for display purposes

        # ------------------------------------------

        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"

        self.mo.lor.append(self)  # add this reservoir to the model
        # self.mo.lor.remove(self)
        # but lets keep track of  virtual reservoir in lvr.

        # this should be done in aux-inits of a derived class if necessary
        # self.mo.lvr.append(self)
        # print(f"added {self.name} to lvr 1")
        # register instance name in global name space
        self.__register_with_parent__()

        self.__aux_inits__()
        self.state = 0


class ExternalCode(SpeciesNoSet):
    """Implement user-provided functions.

    The data inside an ExternalCode instance will only change in response to a
    user-provided function but will otherwise remain unaffected. That is, it is
    up to the user-provided function to manage changes in response to external fluxes.

    An ExternalCode instance is declared in the following way::

        ExternalCode(
            name="cs",                  # instance name
            species=M.CO2,                # must be Speciesproperties instance
            vr_datafields={             # the vr_datafields contain any data that is referenced inside the
                "Hplus": self.swc.hplus, # function, rather than passed as an argument, and all data
                "Beta": 0.0             # that is explicitly referenced by the model
            },
            function=calc_carbonates,   # function reference, see below
            fname="function name as string",
            function_input_data="DIC TA",
            function_params=(float),    # Note that parameters must be individual float values
            return_values={             # list of return values, these must be known species definitions
                "Hplus": rg.swc.hplus,
                "zsnow": float(abs(kwargs["zsnow"])),
            },
            register=rh                # reservoir_handle to register with
        )

    The dictionary keys of `vr_datafields` will be used to create alias names which c
    an be used to access the respective variables. See the online documentation:
    https://esbmtk.readthedocs.io/

    In the default configuration, ExternalCode instances are computed after all regular connections
    have been established. However, sometimes, a connection may depend on a computed value. In this case
    set the optional parameter ftype to "in_sequence"
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Class."""
        from collections.abc import Callable

        from esbmtk import (
            ConnectionProperties,
            Model,
            Reservoir,
            SinkProperties,
            SourceProperties,
            Species,
            Species2Species,
            SpeciesProperties,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species, SpeciesProperties)],
            "plot_transform_c": ["None", (str, Callable)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "function": ["None", (Callable, str)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceProperties,
                    SinkProperties,
                    Reservoir,
                    Species,
                    Species2Species,
                    ConnectionProperties,
                    GasReservoir,
                    Model,
                    str,
                ),
            ],
            "full_name": ["None", (str)],
            "isotopes": [False, (bool)],
            "volume": ["None", (str, int, float)],
            "vr_datafields": ["None", (dict, str)],
            "function_input_data": ["None", (str, list)],
            "function_params": ["None", (tuple)],
            "fname": ["None", (str)],
            "geometry": ["None", (list, str)],
            "alias_list": ["None", (list, str)],
            "ftype": ["computed", (str)],
            "ref_flux": ["None", (list, str)],
            "return_values": ["None", (list)],
            "arguments": ["None", (str, list)],
            "r_s": ["None", (str, Reservoir)],
            "r_d": ["None", (str, Reservoir)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
            "register",
        ]

        self.__initialize_keyword_variables__(kwargs)
        self._initialize_legacy_attributes(kwargs)

        self.lro: list = []  # list of all return objects.
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx
        self.plt_units = self.mo.c_unit

        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"
        self.mo.lor.append(self)  # add this reservoir to the model
        self.__register_with_parent__()
        self.state = 0
        name = f"{self.full_name}_generic_function".replace(".", "_")
        logging.info(f"creating {name}")

        if self.model.debug:
            print(f"EC: {self.full_name}, isotopes = {self.isotopes}")

        if self.vr_datafields != "None":
            self.alias_list = list(self.vr_datafields.keys())
            # initialize data fields
            self.vr_data = list()
            for e in self.vr_datafields.values():
                if isinstance(e, float | int):
                    self.vr_data.append(np.full(self.mo.steps, e, dtype=float))
                else:
                    self.vr_data.append(e)

        self.mo.lpc_r.append(self)
        # self.mo.lpc_r.append(self.gfh)
        self.mo.lor.remove(self)
        # but lets keep track of  virtual reservoir in lvr.
        self.mo.lvr.append(self)

        if self.alias_list != "None":
            self.create_alialises()

        self.update_parameter_count()

    def create_alialises(self) -> None:
        """Register  alialises for each vr_datafield."""
        for i, a in enumerate(self.alias_list):
            # print(f"{a} = {self.vr_data[i][0]}")
            setattr(self, a, self.vr_data[i])

    def update_parameter_count(self):
        """Update Parameter Count."""
        if len(self.function_params) > 0:
            self.param_start = self.model.vpc
            self.model.vpc = self.param_start + 1
            self.has_p = True
            # upudate global parameter list
            self.model.gpt = (*self.model.gpt, self.function_params)
        else:
            self.has_p = False

    def append(self, **kwargs) -> None:
        """Update GenericFunction parameters.

        After the VirtualSpecies has been initialized. This is most useful
        when parameters have to reference other virtual reservoirs
        which do not yet exist, e.g., when two virtual reservoirs have
        a circular reference.

        Example::

             VR.update(a1=new_parameter, a2=new_parameter)

        """
        allowed_keys: list = ["function_input_data, function_params"]
        # loop over provided kwargs
        for key, value in kwargs.items():
            if key not in allowed_keys:
                raise ESBMTKFunctionError(
                    "you can only change function_input_data, or function_params"
                )
            else:
                getattr(self, key).append(value)

    def __reset_state__(self):
        """Copy the last value to the first position so that we can restart the computation."""
        for i, d in enumerate(self.vr_data):
            d[0] = d[-2]
            setattr(
                self,
                f"vrd_{i}",
                np.append(getattr(self, f"vrd_{i}"), d[0 : -2 : self.mo.reset_stride]),
            )

    def __merge_temp_results__(self) -> None:
        """Replace the data fields with the saved values."""
        # print(f"merging {self.full_name} with whith len of vrd= {len(self.vrd_0)}")
        for i, _d in enumerate(self.vr_data):
            self.vr_data[i] = getattr(self, f"vrd_{i}")

        # update aliases
        self.create_alialises()
        # print(f"new length = {len(self.vr_data[0])}")

    def __read_state__(self, directory: str) -> None:
        """Read virtual reservoir data from csv-file into a dataframe.

        The CSV file must have the following columns

        Model Time     t
        X1
        X2

        """
        from pathlib import Path

        if self.sp.mo.register == "None":
            fn = f"{directory}/state_{self.mo.n}_vr_{self.full_name}.csv"
        elif self.sp.mo.register == "local":
            fn = f"{directory}/state_{self.full_name}.csv"
        else:
            raise ESBMTKFunctionError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        file_path = Path(fn)

        if not file_path.exists():
            raise ESBMTKFunctionError(f"File {fn} not found")
        logging.info(f"reading state for {self.full_name} from {fn}")

        # read csv file into dataframe
        self.df: pd.DataFrame = pd.read_csv(fn)
        self.headers: list = list(self.df.columns.values)
        df = self.df
        headers = self.headers
        # print(f"reading from {fn}")
        for i, _n in enumerate(headers):
            # first column is time
            if i > 0:
                # print(f"i = {i}, header = {n}, data = {df.iloc[-3:, i]}")
                self.vr_data[i - 1][:3] = df.iloc[-3:, i]

    def __sub_sample_data__(self, stride) -> None:
        """Subsample the results before saving, or processing them."""
        # print(f"subsampling {self.fullname}")

        new: list = []
        for d in self.vr_data:
            n = d[2:-2:stride]
            new.append(n)

        self.vr_data = new
        # update aliases
        self.create_alialises()

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """To be called by write_data and save_state."""
        from pathlib import Path

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)

        mo = self.sp.mo  # model handle
        rn = self.full_name  # reservoir name
        mn = self.sp.mo.n  # model name
        mtu = f"{mo.t_unit:~P}"

        if self.sp.mo.register == "None":
            fn = f"{directory}/{prefix}{mn}_vr_{rn}.csv"  # file name
        elif self.sp.mo.register == "local":
            fn = f"{directory}/{prefix}{rn}.csv"  # file name
        else:
            raise ESBMTKFunctionError(
                f"Model register keyword must be 'None'/'local' not {self.sp.mo.register}"
            )

        df: pd.dataframe = DataFrame()

        df[f"{rn} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time

        for i, d in enumerate(self.vr_data):
            h = self.alias_list[i] if self.alias_list != "None" else f"X{i}"
            df[h] = d[start:stop:stride]

        file_path = Path(fn)
        if append and file_path.exists():
            df.to_csv(file_path, header=False, mode="a", index=False)
        else:
            df.to_csv(file_path, header=True, mode="w", index=False)
        return df


class VirtualSpeciesNoSet(ExternalCode):
    """Alias to ensure backwards compatibility."""


class VirtualSpecies(Species):
    """A virtual reservoir.

    Unlike regular reservoirs, the mass of a
    virtual reservoir depends entirely on the return value of a function.

    Example::

        VirtualSpecies(name="foo",
                    volume="10 liter",
                    concentration="1 mmol",
                    species=  ,
                    function=bar,
                    a1 to a3 =  to 3optional function arguments,
                    display_precision = number, optional, inherited from Model,
                    )

    The concentration argument will be used to initialize the reservoir and
    to determine the display units.

    The function definition follows the GenericFunction class.
    which takes a generic function and up to 6 optional
    function arguments, and will replace the mass value(s) of the
    given reservoirs with whatever the function calculates. This is
    particularly useful e.g., to calculate the pH of a given reservoir
    as function of e.g., Alkalinity and DIC.

    The function must return a list of numbers which correspond to the
    data which describe a reservoir i.e., mass, light isotope, heavy
    isotope, delta, and concentration

    In order to use this function we need first declare a function we plan to
    use with the generic function process. This function needs to follow this
    template::

        def my_func(i, a1, a2, a3) -> tuple:
            #
            # i = index of the current max_timestep
            # a1 to a3 =  optional function parameter. These must be present,
            # even if your function will not use it See above for details

            # calc some stuff and return it as

            return [m, l, h, d, c] # where m= mass, and l & h are the respective
                                   # isotopes. d denotes the delta value and
                                   # c the concentration
                                   # Use dummy value as necessary.

    This class provides an update method to resolve cases where e.g., two virtual
    reservoirs have a circular reference. See the documentation of update().
    """

    def __aux_inits__(self) -> None:
        """We us the regular init methods of the Species Class, and extend it in this method."""
        from .processes import GenericFunction

        # if self.register != "None":
        #    self.full_name = f"{self.full_name}.{self.name}"
        name = f"{self.full_name}_generic_function".replace(".", "_")
        logging.info(f"creating {name}")

        self.gfh = GenericFunction(
            name=name,
            function=self.function,
            a1=self.a1,
            a2=self.a2,
            a3=self.a3,
            a4=self.a4,
            act_on=self,
        )

        # we only depend on the above function. so no need
        # to be in the reservoir list
        self.mo.lor.remove(self)
        # but lets keep track of  virtual reservoir in lvr.
        self.mo.lvr.append(self)
        # print(f"added {self.name} to lvr 2")

    def update(self, **kwargs) -> None:
        """Update GenericFunction parameters.

        After the VirtualSpecies has been initialized. This is most useful
        when parameters have to reference other virtual reservoirs
        which do not yet exist, e.g., when two virtual reservoirs have
        a circular reference.

        Example::

            VR.update(a1=new_parameter, a2=new_parameter)

        """
        allowed_keys: list = ["a1", "a2", "a3", "a4", "a5", "a6", "volume"]
        # loop over provided kwargs
        for key, value in kwargs.items():
            if key not in allowed_keys:
                raise ValueError("you can only change a1 to a6")
            setattr(self, key, value)  # update self
            setattr(self.gfh, key, value)  # update function


class GasReservoir(SpeciesBase):
    """reservoir specific information similar to the Species class.

          Example::

                  Species(name = "foo",     # Name of reservoir
                            species = CO2,    # SpeciesProperties handle
                            delta = 20,       # initial delta - optional (defaults  to 0)
                            reservoir_mass = quantity # total mass of all gases
                                             defaults to 1.78E20 mol
                            species_ppm =  number # concentration in ppm
                            plot = "yes"/"no", defaults to yes
                            plot_transform_c = a function reference, optional (see below)
                            legend_left = str, optional, useful for plot transform
                            display_precision = number, optional, inherited from Model
                            register = optional, use to register with Reservoir Group
                            isotopes = True/False otherwise use Model.m_type
                            )



    Accesing Species Data:
    ~~~~~~~~~~~~~~~~~~~~~~~~

    You can access the reservoir data as:

    - Name.m # species mass
    - Name.l # mass of light isotope
    - Name.d # species delta (only avaible after M.get_delta_values()
    - Name.c # partial pressure
    - Name.v # total gas mass

    Useful methods include:

    - Name.write_data() # save data to file
    - Name.info()   # info Species
    """

    def __init__(self, **kwargs) -> None:
        """Initialize a reservoir."""
        from collections.abc import Callable

        from esbmtk import Q_, Model, SpeciesProperties

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, SpeciesProperties)],
            "delta": ["None", (int, float)],
            "reservoir_mass": ["1.7786E20 mol", (str, Q_)],
            "species_ppm": ["None", (str, Q_)],
            "plot_transform_c": ["None", (str, Callable)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "function": ["None", (str, Callable)],
            "display_precision": [0.01, (int, float)],
            "register": ["None", (str, Model)],
            "full_name": ["None", (str)],
            "isotopes": [False, (bool)],
            "geometry": ["None", (str, dict)],
            "rtype": ["regular", (str)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
            "species_ppm",
        ]

        self.__initialize_keyword_variables__(kwargs)
        self._initialize_legacy_attributes(kwargs)

        # we re-use the volume instance variable but instead of a
        # volume, we use it store the mass. Use .m instead?
        self.v_unit = Q_("mole").units

        # setup base data
        if isinstance(self.reservoir_mass, str):
            self.reservoir_mass = Q_(self.reservoir_mass)
        if isinstance(self.species_ppm, str):
            self.species_ppm = Q_(self.species_ppm)

        # not sure this universally true but it works for carbon
        self.species_mass = (self.reservoir_mass * self.species_ppm).to("mol")
        self.display_as = "ppm"
        self.plt_units = "ppm"

        # we use the existing approach to calculate concentration
        # which will divide species_mass/volume.
        self.volume = self.reservoir_mass
        self.model.toc = (*self.model.toc, self.volume.magnitude)
        self.v_index = self.model.gcc
        self.model.gcc = self.model.gcc + 1
        #    Q_(self.species_mass).magnitude / self.species_ppm.to("dimensionless")
        # ).magnitude

        if self.v_unit != self.volume.units:
            raise GasReservoirError(
                f"\n\n{self.full_name} reservoir_mass units must be "
                f"in {self.v_unit} "
                f"not {self.volume.units}"
            )

        # This should probably be species specific?
        self.mu: str = "ppm"  # massunit xxxx

        # save the unit which was provided by the user for display purposes
        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}]"

        # initialize vectors
        self.m: NDArrayFloat = (
            np.zeros(self.species.mo.steps) + self.species_mass.magnitude
        )
        self.l: NDArrayFloat = np.zeros(self.mo.steps)
        # initialize concentration vector
        self.c: NDArrayFloat = self.m / self.volume.to(self.v_unit).magnitude

        if self.delta != "None":
            self.isotopes = True
            self.l = get_l_mass(self.c, self.delta, self.species.r)

        self.v: float = (
            np.zeros(self.mo.steps) + self.volume.to(self.v_unit).magnitude
        )  # mass of atmosphere

        if self.mo.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.lc = np.empty(0)
            self.vc = np.empty(0)

        self.mo.lor.append(self)  # add this reservoir to the model
        self.mo.lic.append(self)  # reservoir type object list
        # register instance name in global name space

        # register to model unless a value is given
        if self.register == "None":
            self.register = self.mo
            self.parent = self.register

        self.__register_with_parent__()

        # decide which setitem functions to use
        if self.isotopes:
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

        # any auxilliary init - normally empty, but we use it here to extend the
        # reservoir class in virtual reservoirs
        self.__aux_inits__()
        self.state = 0

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """Write data by index."""
        self.m[i]: float = value[0]
        self.l[i]: float = value[1]

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """Write data by index."""
        self.m[i]: float = value[0]
        # self.c[i]: float = self.m[i] / self.v[i]  # update concentration
        # self.v[i]: float = self.v[i - 1] + value[0]


class ExternalData(esbmtkBase):
    """Instances of this class hold external X/Y data.

    which can be associated with
    a reservoir.

    Example::

           ExternalData(name       = "Name"
                        filename   = "filename",
                        legend     = "label",
                        offset     = "0 yrs",
                        reservoir  = reservoir_handle,
                        scale      = scaling factor, optional
                        display_precision = number, optional, inherited from Model
                        convert_to = optional, see below
                       )

    The data must exist as CSV file, where the first column contains
    the X-values, and the second column contains the Y-values.

    The x-values must be time and specify the time units in the header between square brackets
    They will be mapped into the model time units.

    The y-values can be any data, but the user must take care that they match the model units
    defined in the model instance. So your data file mujst look like this

    Time [years], Data [units], Data [units]
    1, 12
    2, 13

    By convention, the secon column should contaain the same type of
    data as the reservoir (i.e., a concentration), whereas the third
    column contain isotope delta values. Columns with no data should
    be left empty (and have no header!) The optional scale argument, will
    only affect the Y-col data, not the isotope data

    The column headers are only used for the time or concentration
    data conversion, and are ignored by the default plotting
    methods, but they are available as self.xh,yh

    The file must exist in the local working directory.

    the convert_to keyword can be used to force a specific conversion.
    The default is to convert into the model concentration units.

    Methods
    -------
      - name.plot()

    Data:
      - name.x
      - name.y
      - name.df = dataframe as read from csv file

    """

    def __init__(self, **kwargs: dict[str, str]):
        """Initialize Class."""
        from esbmtk import Q_, DataField, Model, Species

        # dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "filename": ["None", (str)],
            "legend": ["None", (str)],
            "reservoir": ["None", (str, Species, DataField, GasReservoir)],
            "offset": ["0 yrs", (Q_, str)],
            "display_precision": [0.01, (int, float)],
            "scale": [1, (int, float)],
            "disp_units": [True, (bool)],
            "register": [
                "None",
                (str, Model, Species, DataField, GasReservoir, Signal),
            ],
            "plot_transform_c": ["None", (str, callable)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "filename", "legend", ["reservoir", "register"]]
        self.__initialize_keyword_variables__(kwargs)

        # legacy names
        if self.register == "None":
            self.register = self.reservoir

        self.n: str = self.name  # string =  name of this instance
        self.fn: str = self.filename  # string = filename of data

        if isinstance(self.reservoir, Species):
            self.mo: Model = self.reservoir.species.mo
        if isinstance(self.reservoir, Signal):
            self.mo: Model = self.signal.species.mo
        if isinstance(self.register, Model):
            self.mo: Model = self.register
        else:
            self.mo = self.register.mo

        self.model = self.mo
        self.parent = self.register
        self.mo.led.append(self)  # keep track of this instance

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        if not os.path.exists(self.fn):  # check if the file is actually there
            raise ExternalDataError(f"Cannot find file {self.fn}")

        self.df: pd.DataFrame = pd.read_csv(self.fn)  # read file

        ncols = len(self.df.columns)
        if ncols < 2:  # test of we have at elast 2 columns
            raise ExternalDataError("CSV file must have at least 2 columns")
        elif ncols == 2:
            self.isotopes = False
        elif ncols == 3:
            self.isotopes = True
        elif ncols > 3:
            raise ExternalDataError("External data only supports up to 2 Y columns")
        else:
            raise ExternalDataError("ED: This should not happen")

        # print(f"Model = {self.mo.full_name}, t_unit = {self.mo.t_unit}")
        self.offset = self.ensure_q(self.offset)
        self.offset = self.offset.to(self.mo.t_unit).magnitude

        # get unit information from each header
        xh = self.df.columns[0].split("[")[1].split("]")[0]
        yh = self.df.columns[1].split("[")[1].split("]")[0]
        self.zh = (
            self.df.columns[2].split("[")[1].split("]")[0]
            if len(self.df.columns) > 2
            else None
        )

        # create the associated quantities
        self.xq = Q_(xh)
        self.yq = Q_(yh)

        # add these to the data we are are reading
        self.x: NDArrayFloat = self.df.iloc[:, 0].to_numpy() * self.xq
        self.y: NDArrayFloat = self.df.iloc[:, 1].to_numpy() * self.yq

        if self.zh:
            # delta is assumed to be without units
            self.d: NDArrayFloat = self.df.iloc[:, 2].to_numpy()
        else:
            self.zh = False

        # map into model space
        # self.x = self.x - self.x[0] + self.offset
        # map into model units, and strip unit information
        if self.disp_units:
            self.x = self.x.to(self.mo.d_unit).magnitude
        else:
            self.x = self.x.to(self.mo.t_unit).magnitude
        # self.s_data = self.s_data.to(self.mo.f_unit).magnitude * self.scale

        mol_liter = Q_("1 mol/liter").dimensionality
        mol_kg = Q_("1 mol/kg").dimensionality

        if isinstance(self.yq, Q_):
            # test what type of Quantity we have
            if self.yq.is_compatible_with("dimensionless"):  # dimensionless
                self.y = self.y.magnitude
            elif self.yq.is_compatible_with("liter/yr"):  # flux
                self.y = self.y.to(self.mo.r_unit).magnitude
            elif self.yq.is_compatible_with("mol/yr"):  # flux
                self.y = self.y.to(self.mo.f_unit).magnitude
            elif (
                self.yq.dimensionality == mol_liter or self.yq.dimensionality == mol_kg
            ):  # concentration
                self.y = self.y.to(self.mo.c_unit).magnitude
            else:
                SignalError(f"No conversion to model units for {self.scale} specified")

        # test for plt_transform
        if self.plot_transform_c != "None":
            if callable(self.plot_transform_c):
                self.y = self.plot_transform_c(self.y)
            else:
                raise ExternalDataError("Plot transform must be a function")

        # register with reservoir
        self.__register__(self.register)

        # if self.mo.register == "local" and self.register == "None":
        #     self.register = self.mo

        self.__register_with_parent__()

    def __register__(self, obj):
        """Register this dataset with a flux or reservoir.

        This will have the
        effect that the data will be printed together with the model
        results for this reservoir

        Example::

        ExternalData.register(Species)

        """
        self.obj = obj  # reser handle we associate with
        obj.led.append(self)

    def __interpolate__(self) -> None:
        """Interpolate the input data with a resolution of dt across the model domain.

        The first and last data point must coincide with the
        model start and end time. In other words, this method will not
        patch data at the end points.

        This will replace the original values of name.x and name.y. However
        the original data remains accessible as name.df


        """
        xi: NDArrayFloat = self.model.time

        if (self.x[0] > xi[0]) or (self.x[-1] < xi[-1]):
            message = (
                f"\n Interpolation requires that the time domain"
                f"is equal or greater than the model domain"
                f"data t(0) = {self.x[0]}, tmax = {self.x[-1]}"
                f"model t(0) = {xi[0]}, tmax = {xi[-1]}"
            )

            raise ExternalDataError(message)
        else:
            self.y: NDArrayFloat = np.interp(xi, self.x, self.y)
            self.x = xi

    def plot(self) -> None:
        """Plot the data and save a pdf.

        Example::

                ExternalData.plot()

        """
        fig, ax = plt.subplots()  #
        ax.scatter(self.x, self.y)
        ax.set_label(self.legend)
        ax.set_xlabel(self.xh)
        ax.set_ylabel(self.yh)
        plt.show()
        plt.savefig(self.n + ".pdf")
