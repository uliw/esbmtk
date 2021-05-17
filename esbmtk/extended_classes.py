from numbers import Number
from nptyping import NDArray, Float64
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
from pandas import DataFrame
from copy import deepcopy, copy
import time
from time import process_time
from numba.typed import List
import numba
from numba.core import types as nbt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging

import builtins
import os

from .esbmtk import esbmtkBase, Reservoir, Species, Source, Sink, Flux, get_imass


class ReservoirGroup(esbmtkBase):
    """This class allows the creation of a group of reservoirs which share
    a common volume, and potentially connections. E.g., if we have two
    reservoir groups with the same reservoirs, and we connect them
    with a flux, this flux will apply to all reservoirs in this group.

    A typical examples might be ocean water which comprises several
    species.  A reservoir group like ShallowOcean will then contain
    sub-reservoirs like DIC in the form of ShallowOcean.DIC

    Example::

        ReservoirGroup(name = "ShallowOcean",        # Name of reservoir group
                    volume/geometry = "1E5 l",       # see below
                    delta   = {DIC:0, ALK:0, PO4:0]  # dict of delta values
                    mass/concentration = {DIC:"1 unit", ALK: "1 unit"}
                    plot = {DIC:"yes", ALK:"yes"}  defaults to yes
                    isotopes = {DIC: True/False} see Reservoir class for details
               )

    Notes: - The subreservoirs are derived from the keys in the concentration or mass
             dictionary. Toward this end, the keys must be valid species handles and
             -- not species names -- !

    Connecting two reservoir groups requires that the names in both
    group match, or that you specify a dictionary which delineates the
    matching.

    Most parameters are passed on to the Reservoir class. See the reservoir class
    documentation for details

    The geometry keyword specifies the upper depth interval, the lower
    depth interval, and the fraction of the total ocean area inhabited by the reservoir

    If the geometry parameter is supplied, the following instance variables will be
    computed

                 self.volume: in model units (usually liter)
                 self.are:a surface area in m^2 at the upper bounding surface
                 self.area_dz: area of seafloor which is intercepted by this box.
                 self.area_fraction: area of seafloor which is intercepted by this
                                    relative to the total ocean floor area

    """

    def __init__(self, **kwargs) -> None:
        """Initialize a new reservoir group"""

        from . import ureg, Q_
        from .sealevel import get_box_geometry_parameters

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "delta": dict,
            "concentration": dict,
            "mass": dict,
            "volume": (str, Q_),
            "geometry": (str, list),
            "plot": dict,
            "isotopes": dict,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            ["volume", "geometry"],
        ]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "volume": "None",
            "geometry": "None",
        }

        if "concentration" in kwargs:
            self.species: list = list(kwargs["concentration"].keys())
        elif "mass" in kwargs:
            self.species: list = list(kwargs["mass"].keys())
        else:
            raise ValueError("You must provide either mass or concentration")

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.bem.update(
            {
                "mass": "a  string or quantity",
                "concentration": "a string or quantity",
                "volume": "a string or quantity",
                "plot": "yes or no",
                "isotopes": "dict Species: True/False",
                "geometry": "list",
            }
        )

        self.__validateandregister__(kwargs)

        # legacy variable
        self.n = self.name
        self.mo = self.species[0].mo

        # geoemtry information
        if self.volume == "None":
            get_box_geometry_parameters(self)

        # register this group object in the global namespace
        self.__register_name__()

        # dict with all default values
        self.cd: dict = {}
        for s in self.species:
            self.cd[s.name]: dict = {
                "mass": "None",
                "concentration": "None",
                "delta": "None",
                "plot": "yes",
                "isotopes": False,
            }

            # now we loop trough all keys for this reservoir and see
            # if we find a corresponding item in the kwargs
            for kcd, vcd in self.cd[s.name].items():  # kcd  = delta, plot, etc
                if kcd in self.kwargs:  # found entry delta
                    # test if delta relates to any species
                    if s in self.kwargs[kcd]:  # {SO4: xxx}
                        # update the entry with the value provided in kwargs
                        # self.cd['SO4_name']['delta'] = self.kwargs['delta'][SO4]
                        self.cd[s.name][kcd] = self.kwargs[kcd][s]

        self.lor: list = []  # list of reservoirs in this group.
        # loop over all entries in species and create the respective reservoirs
        for s in self.species:
            if not isinstance(s, Species):
                raise ValueError(f"{s.n} needs to be a valid species name")

            # create reservoir without registering it in the global name space
            a = Reservoir(
                name=f"{s.name}",
                register=self,
                species=s,
                delta=self.cd[s.n]["delta"],
                mass=self.cd[s.n]["mass"],
                concentration=self.cd[s.n]["concentration"],
                volume=self.volume,
                geometry=self.geometry,
                plot=self.cd[s.n]["plot"],
                groupname=self.name,
                isotopes=self.cd[s.n]["isotopes"],
            )
            # register as part of this group
            self.lor.append(a)


class SourceSink(esbmtkBase):
    """
    This is a meta class to setup a Source/Sink objects. These are not
    actual reservoirs, but we stil need to have them as objects
    Example::

           Sink(name = "Pyrite",
               species = SO4,
               display_precision = number, optional, inherited from Model
               delta = number or str. optional defaults to "None"
           )

    where the first argument is a string, and the second is a reservoir handle

    """

    def __init__(self, **kwargs) -> None:

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "display_precision": Number,
            "register": (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
            "delta": (Number, str),
            "isotopes": bool,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "display_precision": 0,
            "delta": "None",
            "isotopes": False,
            "register": "None",
        }

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        self.loc: set[Connection] = set()  # set of connection objects

        # legacy names
        # if self.register != "None":
        #    self.full_name = f"{self.name}.{self.register.name}"

        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.bu)
        self.lio: list = []

        if self.register == "None":
            self.pt = self.name
        else:
            self.pt: str = f"{self.register.name}_{self.n}"
            self.groupname = self.register.name

        if self.delta != "None":
            self.isotopes = True
            self.d = np.full(self.mo.steps, self.delta)
        else:
            self.d = np.full(self.mo.steps, 0.0)

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_name__()


class SourceSinkGroup(esbmtkBase):
    """
    This is a meta class to setup  Source/Sink Groups. These are not
    actual reservoirs, but we stil need to have them as objects
    Example::

           Sink(name = "Pyrite",
                species = [SO42, H2S],
                delta = {"SO4": 10}
                )

    where the first argument is a string, and the second is a reservoir handle
    """

    def __init__(self, **kwargs) -> None:

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": list,
            "delta": dict,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        # list of default values if none provided
        self.lod: Dict[any, any] = {"delta": {}}

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy variables
        self.n = self.name

        self.loc: set[Connection] = set()  # set of connection objects

        # register this object in the global namespace
        self.mo = self.species[0].mo  # get model handle
        self.__register_name__()

        self.lor: list = []  # list of sub reservoirs in this group

        # loop over species names and setup sub-objects
        for i, s in enumerate(self.species):
            if not isinstance(s, Species):
                raise ValueError(f"{s.n} needs to be a valid species name")

            if s in self.delta:
                delta = self.delta[s]
            else:
                delta = "None"

            if type(self).__name__ == "SourceGroup":
                a = Source(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=delta,
                )

            elif type(self).__name__ == "SinkGroup":
                a = Sink(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=delta,
                )
            else:
                raise TypeError(f"{type(self).__name__} is not a valid class type")

            # register in local namespace
            self.lor.append(a)


class SinkGroup(SourceSinkGroup):
    """
    This is just a wrapper to setup a Sink object
    Example::

           Sink(name = "Pyrite",species =SO4)

    where the first argument is a string, and the second is a species handle
    """


class SourceGroup(SourceSinkGroup):
    """
    This is just a wrapper to setup a Source object
    Example::

           Sink(name = "SO4_diffusion", species ="SO4")

    where the first argument is a string, and the second is a species handle
    """


class Signal(esbmtkBase):
    """We use a simple generator which will create a signal which is
    described by its startime (relative to the model time), it's
    size (as mass) and duration, or as duration and
    magnitude. Furthermore, we can presribe the signal shape
    (square, pyramid) and whether the signal will repeat. You
    can also specify whether the event will affect the delta value.

    The data in the signal class will simply be added to the data in
    a given flux. So this class cannot be used for scaling (can we
    add this functionality?)

    Example::

          Signal(name = "Name",
                 species = Species handle,
                 start = "0 yrs",     # optional
                 duration = "0 yrs",  #
                 delta = 0,           # optional
                 stype = "addition"   # optional, currently the only type
                 shape = "square"     # square, pyramid
                 mass/magnitude/filename  # give one
                 offset = '0 yrs',     #
                 scale = 1, optional,  #
                 reservoir = r-handle # optional, see below
                 source = s-handle optional, see below
                 display_precision = number, optional, inherited from Model
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
    (i.e., the start time will be mapped to zero). Use the offset
    keyword to shift the external signal data in the time domain.

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

    """

    def __init__(self, **kwargs) -> None:
        """Parse and initialize variables"""

        from . import ureg, Q_

        # provide a list of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": str,
            "duration": str,
            "species": Species,
            "delta": Number,
            "stype": str,
            "shape": str,
            "filename": str,
            "mass": str,
            "magnitude": Number,
            "offset": str,
            "plot": str,
            "scale": Number,
            "display_precision": Number,
            "reservoir": (Reservoir, str),
            "source": (Source, str),
        }

        # provide a list of absolutely required keywords
        self.lrk: List[str] = [
            "name",
            ["duration", "filename"],
            "species",
            ["shape", "filename"],
            ["magnitude", "mass", "filename"],
        ]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "start": "0 yrs",
            "stype": "addition",
            "shape": "external_data",
            "offset": "0 yrs",
            "duration": "0 yrs",
            "plot": "no",
            "delta": 0,
            "scale": 1,
            "display_precision": 0,
            "reservoir": "None",
            "source": "None",
        }

        self.__initerrormessages__()
        self.bem.update(
            {
                "data": "a string",
                "magnitude": "Number",
                "scale": "Number",
            }
        )
        self.__validateandregister__(kwargs)  # initialize keyword values

        # list of signals we are based on
        self.los: List[Signal] = []

        # convert units to model units
        self.st: Number = int(
            Q_(self.start).to(self.species.mo.t_unit).magnitude
        )  # start time

        if "mass" in self.kwargs:
            self.mass = Q_(self.mass).to(self.species.mo.m_unit).magnitude
        elif "magnitude" in self.kwargs:
            self.magnitude = Q_(self.magnitude).to(self.species.mo.f_unit).magnitude

        if "duration" in self.kwargs:
            self.duration = int(Q_(self.duration).to(self.species.mo.t_unit).magnitude)

        self.offset = Q_(self.offset).to(self.species.mo.t_unit).magnitude

        # legacy name definitions
        self.l: int = self.duration
        self.n: str = self.name  # the name of the this signal
        self.sp: Species = self.species  # the species
        self.mo: Model = self.species.mo  # the model handle
        self.ty: str = self.stype  # type of signal
        self.sh: str = self.shape  # shape the event
        self.d: float = self.delta  # delta value offset during the event
        self.kwd: Dict[str, any] = self.kwargs  # list of keywords
        self.led: list = []

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        # initialize signal data
        self.data = self.__init_signal_data__()
        self.data.n: str = self.name + "_data"  # update the name of the signal data
        self.legend_left = self.data.legend_left
        self.legend_right = self.data.legend_right
        # update isotope values
        self.data.li, self.data.hi = get_imass(self.data.m, self.data.d, self.sp.r)
        self.__register_name__()
        self.mo.los.append(self)  # register with model

        if self.reservoir != "None":
            self.__apply_signal__()

    def __apply_signal__(self) -> None:
        """Create a source, and connect signal, source and reservoir"""

        from esbmtk import Source, Connect

        if self.source == "None":
            self.source = Source(name=f"{self.name}_Source", species=self.sp)

        Connect(
            source=self.source,  # source of flux
            sink=self.reservoir,  # target of flux
            rate="0 mol/yr",  # flux rate
            signal=self,  # list of processes
            plot="no",
        )

    def __init_signal_data__(self) -> None:
        """Create an empty flux and apply the shape"""
        # create a dummy flux we can act up
        self.nf: Flux = Flux(
            name=self.n + "_data",
            species=self.sp,
            rate=f"0 {self.sp.mo.f_unit}",
            delta=0,
        )

        # since the flux is zero, the delta value will be undefined. So we set it explicitly
        # this will avoid having additions with Nan values.
        self.nf.d[0:]: float = 0.0

        # find nearest index for start, and end point
        # print(f"Model time units = {self.species.mo.t_unit}")
        # print(f"start_time = {self.st}, dt = {self.mo.dt}")
        # print(f"duration = {self.duration}")

        self.si: int = int(round(self.st / self.mo.dt))  # starting index
        self.ei: int = self.si + int(round(self.duration / self.mo.dt))  # end index
        # print(f"start index = {self.si}")
        # print(f"end index = {self.ei}")

        # create slice of flux vector
        self.s_m: [NDArray, Float[64]] = array(self.nf.m[self.si : self.ei])
        # create slice of delta vector
        self.s_d: [NDArray, Float[64]] = array(self.nf.d[self.si : self.ei])

        if self.sh == "square":
            self.__square__(self.si, self.ei)

        elif self.sh == "pyramid":
            self.__pyramid__(self.si, self.ei)

        elif "filename" in self.kwargs:  # use an external data set
            self.__int_ext_data__(self.si, self.ei)

        else:
            raise ValueError(
                f"argument needs to be either square/pyramid, "
                f"or an ExternalData object. "
                f"shape = {self.sh} is not a valid Value"
            )

        # now add the signal into the flux slice
        self.nf.m[self.si : self.ei] = self.s_m
        self.nf.d[self.si : self.ei] = self.s_d

        return self.nf

    def __square__(self, s, e) -> None:
        """Create Square Signal"""

        if "mass" in self.kwd:
            h = self.mass / self.duration  # get the height of the square

        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError("You must specify mass or magnitude of the signal")

        self.s_m: float = h  # add this to the section
        self.s_d: float = self.d  # add the delta offset

    def __pyramid__(self, s, e) -> None:
        """Create pyramid type Signal

        s = start index
        e = end index
        """

        if "mass" in self.kwd:
            h = 2 * self.mass / self.duration  # get the height of the pyramid

        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError("You must specify mass or magnitude of the signal")

        # create pyramid
        c: int = int(round((e - s) / 2))  # get the center index for the peak
        x: [NDArray, Float[64]] = array([0, c, e - s])  # setup the x coordinates
        y: [NDArray, Float[64]] = array([0, h, 0])  # setup the y coordinates
        d: [NDArray, Float[64]] = array([0, self.d, 0])  # setup the d coordinates
        xi = arange(0, e - s)  # setup the points at which to interpolate
        h: [NDArray, Float[64]] = interp(xi, x, y)  # interpolate flux
        dy: [NDArray, Float[64]] = interp(xi, x, d)  # interpolate delta
        self.s_m: [NDArray, Float[64]] = self.s_m + h  # add this to the section
        self.s_d: [NDArray, Float[64]] = self.s_d + dy  # ditto for delta

    def __int_ext_data__(self, s, e) -> None:
        """Interpolate External data as a signal. Unlike the other signals,
        thiw will replace the values in the flux with those read from the
        external data source. The external data need to be in the following format

        Time [units], Rate [units], delta value [units]
        0,     10,   12

        i.e., the first row needs to be a header line

        """

        from . import ureg, Q_

        if not os.path.exists(self.filename):  # check if the file is actually there
            raise FileNotFoundError(f"Cannot find file {self.filename}")
        # read external dataset
        df = pd.read_csv(self.filename)

        # get unit information from each header
        xh = df.columns[0].split("[")[1].split("]")[0]
        yh = df.columns[1].split("[")[1].split("]")[0]
        # zh = df.iloc[0,2].split("[")[1].split("]")[0]

        # create the associated quantities
        xq = Q_(xh)
        yq = Q_(yh)
        # zq = Q_(zh)

        # add these to the data we are are reading
        x = df.iloc[:, 0].to_numpy() * xq
        y = df.iloc[:, 1].to_numpy() * yq
        d = df.iloc[:, 2].to_numpy()

        # map into model units, and strip unit information
        x = x.to(self.mo.t_unit).magnitude
        y = y.to(self.mo.f_unit).magnitude * self.scale

        # the data can contain 1 to n data points (i.e., index
        # values[0,1,n]) each index value contains a time
        # coordinate. So the duration is x[-1] - X[0]. Duration/dt
        # gives us the steps, so we can setup a vector for
        # interpolation. Insertion off this vector depends on the time
        # offset defined by offset keyword which defines the
        # insertion indexes self.si self.ei

        self.st: float = x[0]  # start time
        self.et: float = x[-1]  # end times
        duration = int(round(self.et - self.st))

        # map the original time coordinate into model space
        x = x - x[0]

        # since everything has been mapped to dt, time equals index
        self.si: int = self.offset  # starting index
        self.ei: int = self.offset + duration  # end index

        # create slice of flux vector
        self.s_m: [NDArray, Float[64]] = array(self.nf.m[self.si : self.ei])

        # create slice of delta vector
        self.s_d: [NDArray, Float[64]] = array(self.nf.d[self.si : self.ei])

        # setup the points at which to interpolate
        xi = arange(0, duration)

        h: [NDArray, Float[64]] = interp(xi, x, y)  # interpolate flux
        dy: [NDArray, Float[64]] = interp(xi, x, d)  # interpolate delta

        # add this to the corresponding section off the flux
        self.s_m: [NDArray, Float[64]] = self.s_m + h
        self.s_d: [NDArray, Float[64]] = self.s_d + dy  # ditto for delta

    def __add__(self, other):
        """ allow the addition of two signals and return a new signal"""

        ns = deepcopy(self)

        # add the data of both fluxes
        ns.data.m: [NDArray, Float[64]] = self.data.m + other.data.m
        ns.data.d: [NDArray, Float[64]] = self.data.d + other.data.d
        ns.data.l: [NDArray, Float[64]]
        ns.data.h: [NDArray, Float[64]]

        [ns.data.l, ns.data.h] = get_imass(ns.data.m, ns.data.d, ns.data.sp.r)

        ns.n: str = self.n + "_and_" + other.n
        print(f"adding {self.n} to {other.n}, returning {ns.n}")
        ns.data.n: str = self.n + "_and_" + other.n + "_data"
        ns.st = min(self.st, other.st)
        ns.l = max(self.l, other.l)
        ns.sh = "compound"
        ns.los.append(self)
        ns.los.append(other)

        return ns

    def repeat(self, start, stop, offset, times) -> None:
        """This method creates a new signal by repeating an existing signal.
        Example::

        new_signal = signal.repeat(start,   # start time of signal slice to be repeated
                                   stop,    # end time of signal slice to be repeated
                                   offset,  # offset between repetitions
                                   times,   # number of time to repeat the slice
                              )

        """

        ns: Signal = deepcopy(self)
        ns.n: str = self.n + f"_repeated_{times}_times"
        ns.data.n: str = self.n + f"_repeated_{times}_times_data"
        start: int = int(start / self.mo.dt)  # convert from time to index
        stop: int = int(stop / self.mo.dt)
        offset: int = int(offset / self.mo.dt)
        ns.start: float = start
        ns.stop: float = stop
        ns.offset: float = stop - start + offset
        ns.times: float = times
        ns.ms: [NDArray, Float[64]] = self.data.m[
            start:stop
        ]  # get the data slice we are using
        ns.ds: [NDArray, Float[64]] = self.data.d[start:stop]

        diff = 0
        for i in range(times):
            start: int = start + ns.offset
            stop: int = stop + ns.offset
            if start > len(self.data.m):
                break
            elif stop > len(self.data.m):  # end index larger than data size
                diff: int = stop - len(self.data.m)  # difference
                stop: int = stop - diff  # new end index
                lds: int = len(ns.ds) - diff
            else:
                lds: int = len(ns.ds)

            ns.data.m[start:stop]: [NDArray, Float[64]] = (
                ns.data.m[start:stop] + ns.ms[0:lds]
            )
            ns.data.d[start:stop]: [NDArray, Float[64]] = (
                ns.data.d[start:stop] + ns.ds[0:lds]
            )

        # and recalculate li and hi
        ns.data.l: [NDArray, Float[64]]
        ns.data.h: [NDArray, Float[64]]
        [ns.data.l, ns.data.h] = get_imass(ns.data.m, ns.data.d, ns.data.sp.r)
        return ns

    def __register_with_flux__(self, flux) -> None:
        """Register this signal with a flux. This should probably be done
        through a process!

        """

        self.fo: Flux = flux  # the flux handle
        self.sp: Species = flux.sp  # the species handle
        model: Model = flux.sp.mo  # the model handle add this process to the
        # list of processes
        flux.lop.append(self)

    def __call__(self) -> NDArray[np.float64]:
        """what to do when called as a function ()"""

        return (array([self.fo.m, self.fo.l, self.fo.h, self.fo.d]), self.fo.n, self)

    def plot(self) -> None:
        """
          Example::

              Signal.plot()

        Plot the signal

        """
        self.data.plot()


class DataField(esbmtkBase):
    """
    DataField: Datafields can be used to plot data which is computed after
    the model finishes in the overview plot windows. Therefore, datafields will
    plot in the same window as the reservoir they are associated with.
    Datafields must share the same x-axis is the model, and can have up to two
    y axis.

    Example::

             DataField(name = "Name"
                       associated_with = reservoir_handle
                       y1_data = np.Ndarray or list of arrays
                       y1_label = Y-Axis label
                       y1_legend = Data legend or list of legends
                       y2_data = np.Ndarray    # optional
                       y2_label = Y-Axis label # optional
                       y2_legend = Data legend # optional
                       common_y_scale = "no",  #optional, default "no"
                       display_precision = number, optional, inherited from Model
                       )

    Note that Datafield data is not mapped to model units. Care must be taken
    that the data units match the model units.

    The instance provides the following data

    Name.x    = X-axis = model X-axis
    Name.y1_data
    Name.y1_label
    Name.y1_legend

    Similarly for y2
    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "associated_with": (Reservoir, ReservoirGroup),
            "y1_data": (NDArray[float], list),
            "y1_label": str,
            "y1_legend": (str, list),
            "y2_data": (str, NDArray[float], list),
            "y2_label": str,
            "y2_legend": (str, list),
            "common_y_scale": str,
            "display_precision": Number,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "associated_with", "y1_data"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "y1_label": "Not Provided",
            "y1_legend": "Not Provided",
            "y2_label": "Not Provided",
            "y2_legend": "Not Provided",
            "y2_data": "None",
            "common_y_scale": "no",
            "display_precision": 0,
        }

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update(
            {
                "associated_with": "a string",
                "y1_data": "a numpy array",
                "y1_label": "a string",
                "y1_legend": "a string",
                "y2_data": "a numpy array",
                "y2_label": "a string",
                "y2_legend": "a string",
                "common_y_scale": "a string",
            }
        )

        self.__validateandregister__(kwargs)  # initialize keyword values

        # set legacy variables
        self.legend_left = self.y1_legend

        self.mo = self.associated_with.mo
        if "self.y2_data" != "None":
            self.d = self.y2_data
            self.legend_right = self.y2_legend
            self.ld = self.y2_label

        self.n = self.name
        self.led = []

        if not isinstance(self.y1_data, list):
            self.y1_data = [self.y1_data]

        if not isinstance(self.y1_legend, list):
            self.y1_legend = [self.y1_legend]

        if not isinstance(self.y2_data, list):
            self.y2_data = [self.y2_data]

        if not isinstance(self.y2_legend, list):
            self.y2_legend = [self.y2_legend]

        # register with reservoir
        self.associated_with.ldf.append(self)
        # register with model. needed for print_reservoirs
        self.mo.ldf.append(self)
        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_name__()
        if self.mo.state == 0:
            print("")
            print(
                "---------------------------------------------------------------------------\n\n"
            )
            print(
                "Warning, you are initializing a datafield before the model results are known\n\n"
            )
            print(
                "---------------------------------------------------------------------------"
            )

    def __write_data__(self, prefix: str, start: int, stop: int, stride: int) -> None:
        """To be called by write_data and save_state"""

        # some short hands
        mo = self.mo  # model handle

        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"

        rn = self.n  # reservoir name
        mn = self.mo.n  # model name
        fn = f"{prefix}{mn}_{rn}.csv"  # file name

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{self.n} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        df[f"{self.n} {self.y1_label}"] = self.y1_data[start:stop:stride]  # y1 data

        if self.y2_data != "None":
            df[f"{self.n} {self.y1_label}"] = self.y2_data[start:stop:stride]  # y2_data

        df.to_csv(fn, index=False)  # Write dataframe to file
        return df


class VirtualReservoir(Reservoir):
    """A virtual reservoir. Unlike regular reservoirs, the mass of a
    virtual reservoir depends entirely on the return value of a function.

    Example::

    VirtualReservoir(name="foo",
                    volume="10 liter",
                    concentration="1 mmol",
                    species=  ,
                    function=bar,
                    a1 to a3 =  to 3optional function arguments,
                    display_precision = number, optional, inherited from Model,
                    )

    the concentration argument will be used to initialize the reservoir and
    to determine the display units.

    The function definition follows the GenericFunction class.
    which takes a generic function and up to 6 optional
    function arguments, and will replace the mass value(s) of the
    given reservoirs with whatever the function calculates. This is
    particularly useful e.g., to calculate the pH of a given reservoir
    as function of e.g., Alkalinity and DIC.
    Parameters:
     - name = name of process,
     - act_on = name of a reservoir this process will act upon
     - function  = a function reference
     - a1 to a3 function arguments

    In order to be compatible with the numba solver, a1 and a2 must be
    an array of 1-D numpy.arrays i.e., [m, l, h, c]. The array can have
    any number of arrays though. a3 must be single array (or list).
    The a3 array must be passed as List([...]), and List must be imported as

    from numba.typed import List


    The function must return a list of numbers which correspond to the
    data which describe a reservoir i.e., mass, light isotope, heavy
    isotope, delta, and concentration

    In order to use this function we need first declare a function we plan to
    use with the generic function process. This function needs to follow this
    template::

        def my_func(i, a1, a2, a3) -> tuple:
            #
            # i = index of the current timestep
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
        """We us the regular init methods of the Reservoir Class, and extend it in this method"""

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

    def update(self, **kwargs) -> None:
        """This method allows to update GenericFunction parameters after the
        VirtualReservoir has been initialized. This is most useful
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
            else:
                setattr(self, key, value)  # update self
                setattr(self.gfh, key, value)  # update function


class ExternalData(esbmtkBase):
    """Instances of this class hold external X/Y data which can be associated with
    a reservoir.

    Example::

           ExternalData(name       = "Name"
                        filename   = "filename",
                        legend     = "label",
                        offset     = "0 yrs",
                        reservoir  = reservoir_handle,
                        scale      = scaling factor, optional
                        display_precision = number, optional, inherited from Model
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
    be left empty (and have no header!) The optional scale argumenty, will
    only affect the Y-col data, not the isotope data

    The column headers are only used for the time or concentration
    data conversion, and are ignored by the default plotting
    methods, but they are available as self.xh,yh

    The file must exist in the local working directory.

    Methods:
      - name.plot()

    Data:
      - name.x
      - name.y
      - name.df = dataframe as read from csv file

    """

    def __init__(self, **kwargs: Dict[str, str]):

        from . import ureg, Q_

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "filename": str,
            "legend": str,
            "reservoir": Reservoir,
            "offset": str,
            "display_precision": Number,
            "scale": Number,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "filename", "legend", "reservoir"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {
            "offset": "0 yrs",
            "display_precision": 0,
            "scale": 1,
        }

        # validate input and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        self.fn: str = self.filename  # string = filename of data
        self.mo: Model = self.reservoir.species.mo

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        if not os.path.exists(self.fn):  # check if the file is actually there
            raise FileNotFoundError(f"Cannot find file {self.fn}")

        self.df: pd.DataFrame = pd.read_csv(self.fn)  # read file

        ncols = len(self.df.columns)
        if ncols != 3:  # test of we have 3 columns
            raise ValueError("CSV file must have 3 columns")

        self.offset = Q_(self.offset).to(self.mo.t_unit).magnitude

        xh = self.df.columns[0]

        # get unit information from each header
        xh = get_string_between_brackets(xh)

        xq = Q_(xh)
        # add these to the data we are are reading
        self.x: [NDArray] = self.df.iloc[:, 0].to_numpy() * xq
        # map into model units
        self.x = self.x.to(self.mo.t_unit).magnitude

        # map into model space
        self.x = self.x - self.x[0] + self.offset

        # check if y-data is present
        yh = self.df.columns[1]
        if not "Unnamed" in yh:
            yh = get_string_between_brackets(yh)
            yq = Q_(yh)
            # add these to the data we are are reading
            self.y: [NDArray] = self.df.iloc[:, 1].to_numpy() * yq
            # map into model units
            self.y = self.y.to(self.mo.t_unit).magnitude * self.scale

        # check if z-data is present
        if ncols == 3:
            zh = self.df.columns[2]
            self.z = self.df.iloc[:, 2].to_numpy()

        # register with reservoir
        self.__register__(self.reservoir)
        self.__register_name__()

    def __register__(self, obj):
        """Register this dataset with a flux or reservoir. This will have the
        effect that the data will be printed together with the model
        results for this reservoir

        Example::

        ExternalData.register(Reservoir)

        """
        self.obj = obj  # reser handle we associate with
        obj.led.append(self)

    def __interpolate__(self) -> None:
        """Interpolate the input data with a resolution of dt across the model
        domain The first and last data point must coincide with the
        model start and end time. In other words, this method will not
        patch data at the end points.

        This will replace the original values of name.x and name.y. However
        the original data remains accessible as name.df


        """

        xi: [NDArray] = self.model.time

        if (self.x[0] > xi[0]) or (self.x[-1] < xi[-1]):
            message = (
                f"\n Interpolation requires that the time domain"
                f"is equal or greater than the model domain"
                f"data t(0) = {self.x[0]}, tmax = {self.x[-1]}"
                f"model t(0) = {xi[0]}, tmax = {xi[-1]}"
            )

            raise ValueError(message)
        else:
            self.y: [NDArray] = interp(xi, self.x, self.y)
            self.x = xi

    def plot(self) -> None:
        """Plot the data and save a pdf

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
