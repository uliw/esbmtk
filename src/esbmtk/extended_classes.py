from __future__ import annotations
from pandas import DataFrame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import math
import copy as cp
import typing as tp
import warnings

if tp.TYPE_CHECKING:
    from esbmtk import Connection, Model

from .esbmtk_base import esbmtkBase
from .esbmtk import ReservoirBase, Reservoir, Species

from .utility_functions import (
    # get_string_between_brackets,
    get_imass,
    get_delta,
    get_l_mass,
)

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class ReservoirGroupError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class SourceSinkGroupError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class SignalError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class DataFieldError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class ESBMTKFunctionError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class ExternalDataError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class GasResrvoirError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class ReservoirGroup(esbmtkBase):
    """This class allows the creation of a group of reservoirs which share
    a common volume, and potentially connections. E.g., if we have twoy
    reservoir groups with the same reservoirs, and we connect them
    with a flux, this flux will apply to all reservoirs in this group.

    A typical examples might be ocean water which comprises several
    species.  A reservoir group like ShallowOcean will then contain
    sub-reservoirs like DIC in the form of ShallowOcean.DIC

    Example::

        ReservoirGroup(name = "ShallowOcean",        # Name of reservoir group
                    volume/geometry = "1E5 l",       # see below
                    delta   = {DIC:0, TA:0, PO4:0]  # dict of delta values
                    mass/concentration = {DIC:"1 unit", TA: "1 unit"}
                    plot = {DIC:"yes", TA:"yes"}  defaults to yes
                    isotopes = {DIC: True/False} see Reservoir class for details
                    seawater_parameters = dict, optional, see below
                    register= model handle, required
               )

    Notes: The subreservoirs are derived from the keys in the concentration or mass
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
    computed:

     - self.volume: in model units (usually liter)
     - self.area: surface area in m^2 at the upper bounding surface
     - self.sed_area: area of seafloor which is intercepted by this box.
     - self.area_fraction: area of seafloor which is intercepted by this relative to the total ocean floor area

    seawater_parameters:
    ~~~~~~~~~~~~~~~~~~~~

    If this optional parameter is specified, a SeaWaterConstants instance will
    be registered for this Reservoir as Reservoir.swc
    See the  SeaWaterConstants class for details how to specify the parameters, e.g.::

        seawater_parameters = {"temperature": 2,
                           "pressure": 240,
                           "salinity" : 35,
                           },

    """

    def __init__(self, **kwargs) -> None:
        """Initialize a new reservoir group"""

        from .utility_functions import dict_alternatives
        from esbmtk import (
            ExternalCode,
            Species,
            SeawaterConstants,
            Model,
            get_box_geometry_parameters,
            Q_,
        )

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
            "register",
            ["volume", "geometry"],
        ]

        if "concentration" in kwargs:
            self.species: list = list(kwargs["concentration"].keys())
        elif "mass" in kwargs:
            self.species: list = list(kwargs["mass"].keys())
        else:
            raise ReservoirGroupError("You must provide either mass or concentration")

        self.__initialize_keyword_variables__(kwargs)

        # legacy variable
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
            raise ReservoirGroupError("Volume must be string or quantity")

        self.__register_name_new__()

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
                if kcd in self.kwargs and s in self.kwargs[kcd]:
                    # update the entry with the value provided in kwargs
                    # self.cd['SO4_name']['delta'] = self.kwargs['delta'][SO4]
                    self.cd[s.name][kcd] = self.kwargs[kcd][s]

        self.lor: list = []  # list of reservoirs in this group.
        # loop over all entries in species and create the respective reservoirs
        for s in self.species:
            if not isinstance(s, Species):
                raise ReservoirGroupError(f"{s.n} needs to be a valid species name")
            if s.flux_only:
                rtype = "flux_only"
            else:
                rtype = "regular"

            a = Reservoir(
                name=f"{s.name}",
                register=self,
                species=s,
                delta=self.cd[s.n]["delta"],
                mass=self.cd[s.n]["mass"],
                concentration=self.cd[s.n]["concentration"],
                volume=self.volume,
                # geometry=self.geometry,
                plot=self.cd[s.n]["plot"],
                groupname=self.name,
                isotopes=self.cd[s.n]["isotopes"],
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
                warnings.warn(
                    f"Using SeawaterConstants without provinding DIC "
                    f"and TA values for {self.name} "
                    f"You need to call swc.update_parameters() "
                    f"once DIC and TA are specified"
                )
        self.register.lrg.append(self)


class SourceSink(esbmtkBase):
    """
    This is a meta class to setup a Source/Sink objects. These are not
    actual reservoirs, but we stil need to have them as objects
    Example::

           Sink(name = "Pyrite",
               species = SO4,
               display_precision = number, optional, inherited from Model
           )

    where the first argument is a string, and the second is a reservoir handle

    """

    def __init__(self, **kwargs) -> None:
        from esbmtk import Species, Model, SourceSinkGroup

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "display_precision": [0.01, (int, float)],
            "register": ["None", (str, Model, SourceSinkGroup)],
            "delta": ["None", (int, float, str)],
            "isotopes": [False, (bool)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species", "register"]

        self.__initialize_keyword_variables__(kwargs)

        self.loc: set[Connection] = set()  # set of connection objects

        # legacy names
        # if self.register != "None":
        #    self.full_name = f"{self.name}.{self.register.name}"

        self.parent = self.register
        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.bu)
        self.lio: list = []

        self.__register_name_new__()


class SourceSinkGroup(esbmtkBase):
    """
    This is a meta class to setup  Source/Sink Groups. These are not
    actual reservoirs, but we stil need to have them as objects
    Example::

           SinkGroup(name = "Pyrite",
                species = [SO42, H2S],
                )

    where the first argument is a string, and the second is a reservoir handle
    """

    def __init__(self, **kwargs) -> None:
        from esbmtk import Model, Species, Source, Sink

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, list)],
            "delta": [{}, dict],
            "isotopes": [False, (dict, bool)],
            "register": ["None", (str, Model)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species", "register"]
        self.__initialize_keyword_variables__(kwargs)

        # legacy variables
        self.n = self.name
        self.parent = self.register
        self.loc: set[Connection] = set()  # set of connection objects

        # register this object in the global namespace
        self.mo = self.species[0].mo  # get model handle
        self.model = self.species[0].mo

        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_name_new__()

        self.lor: list = []  # list of sub reservoirs in this group

        # loop over species names and setup sub-objects
        for i, s in enumerate(self.species):
            if not isinstance(s, Species):
                raise SourceSinkGroupError(f"{s.n} needs to be a valid species name")

            delta = self.delta[s] if s in self.delta else "None"
            if isinstance(self.isotopes, dict):
                isotopes = self.isotopes[s]
            else:
                isotopes = self.isotopes

            if type(self).__name__ == "SourceGroup":
                a = Source(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=delta,
                    isotopes=isotopes,
                )

            elif type(self).__name__ == "SinkGroup":
                a = Sink(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=delta,
                    isotopes=isotopes,
                )
            else:
                raise SourceSinkGroupError(
                    f"{type(self).__name__} is not a valid class type"
                )
            # register in list of reservoirs
            self.lor.append(a)


class SinkGroup(SourceSinkGroup):
    """
    This is just a wrapper to setup a Sink object
    Example::

           SinkGroup(name = "Burial",
                species = [SO42, H2S],
                delta = {"SO4": 10}
                )

    """

    # f __init__(self,super):


class SourceGroup(SourceSinkGroup):
    """
    This is just a wrapper to setup a Source object
    Example::

        SourceGroup(name = "weathering",
                species = [SO42, H2S],
                delta = {"SO4": 10}
                )

    """


class Signal(esbmtkBase):
    """This class will create a signal which is
    described by its startime (relative to the model time), it's
    size (as mass) and duration, or as duration and
    magnitude. Furthermore, we can presribe the signal shape
    (square, pyramid, bell, file )and whether the signal will repeat. You
    can also specify whether the event will affect the delta value.

    The default is to add the signal to a given connection. It is however
    also possible to use the signal data as a scaling factor.

    Example::

          Signal(name = "Name",
                 species = Species handle,
                 start = "0 yrs",     # optional
                 duration = "0 yrs",  #
                 delta = 0,           # optional
                 stype = "addition"   # optional, currently the only type
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

        from esbmtk import Q_, Species, Source, Sink, Reservoir, Model

        # provide a list of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "start": ["0 yrs", (str, Q_)],
            "duration": ["0 yr", (str, Q_)],
            "species": ["None", (Species)],
            "delta": [0, (int, float)],
            "stype": [
                "addition",
                (str),
            ],
            "shape": ["None", (str)],
            "filename": ["None", (str)],
            "mass": ["None", (str, Q_)],
            "magnitude": ["None", (str, Q_)],
            "offset": ["0 yrs", (str, Q_)],
            "plot": ["no", (str)],
            "scale": [1, (int, float)],
            "display_precision": [0.01, (int, float)],
            "reservoir": ["None", (Source, Sink, Reservoir, str)],
            "source": ["None", (Source, Sink, Reservoir, str)],
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
            "register",
        ]

        self.__initialize_keyword_variables__(kwargs)

        # list of signals we are based on
        self.los: list[Signal] = []

        # convert units to model units
        self.st: tp.Union[int, float] = int(
            Q_(self.start).to(self.species.mo.t_unit).magnitude
        )  # start time

        if "mass" in self.kwargs:
            self.mass = Q_(self.mass).to(self.species.mo.m_unit).magnitude
        elif "magnitude" in self.kwargs:
            self.magnitude = Q_(self.magnitude).to(self.species.mo.f_unit).magnitude

        self.duration = int(Q_(self.duration).to(self.species.mo.t_unit).magnitude)

        self.offset = Q_(self.offset).to(self.species.mo.t_unit).magnitude

        if self.duration > 0:
            if self.duration / self.species.mo.max_step < 10:
                warnings.warn(
                    """\n\n Your signal duration is covered by less than 10
                    Intergration steps. This may not be what you want
                    Consider adjusting the max_step parameter of the model object\n"""
                )

        # legacy name definitions
        self.full_name = ""
        self.l: int = self.duration
        self.n: str = self.name  # the name of the this signal
        self.sp: Species = self.species  # the species
        self.mo: Model = self.species.mo  # the model handle
        self.model = self.mo
        self.parent = self.register
        self.ty: str = self.stype  # type of signal
        self.sh: str = self.shape  # shape the event
        self.d: float = self.delta  # delta value offset during the event
        self.kwd: dict[str, any] = self.kwargs  # list of keywords
        self.led: list = []

        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.data = self.__init_signal_data__()
        self.m = self.data.m
        if self.isotopes:
            self.l = self.data.l

        self.data.n: str = self.name + "_data"  # update the name of the signal data
        self.legend_left = self.data.legend_left
        self.legend_right = self.data.legend_right
        # update isotope values
        # self.data.li = get_l_mass(self.data.m, self.data.d, self.sp.r)
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo
        self.__register_name_new__()
        self.mo.los.append(self)  # register with model

        # in case we deal with a sink or source signal
        if self.reservoir != "None":
            self.__apply_signal__()

    def __init_signal_data__(self) -> None:
        """1. Create a vector which contains the signal data. The vector length
           can exceed the modelling domain.
        2. Trim the signal vector in such a way that it fits within the
           modelling domain
        3. Create an empty flux and replace it with the signal vector data.

        Note that this flux will then be added to an existing flux.

        """

        from esbmtk import Flux

        # these are signal times, not model time
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

        # create a dummy flux we can act up
        self.nf: Flux = Flux(
            name=self.n + "_data",
            species=self.sp,
            rate=f"0 {self.sp.mo.f_unit}",
            delta=0,
            isotopes=self.isotopes,
            save_flux_data=True,
            register=self,
        )

        # remove signal fluxes from global flux list
        self.mo.lof.remove(self.nf)

        # map into model space
        insert_start_time = self.st - self.mo.offset
        insert_stop_time = insert_start_time + self.duration

        dt1 = int((self.st - self.mo.offset - self.mo.start))
        dt2 = int((self.st + self.duration - self.mo.stop - self.mo.offset))

        # This fails if actual model steps != dt
        model_start_index = int(max(insert_start_time / self.mo.dt, 0))
        model_stop_index = int(min((self.mo.steps + dt2 / self.mo.dt), self.mo.steps))
        signal_start_index = int(min(dt1, 0) * -1)
        signal_stop_index = int(self.length - max(0, dt2))

        if self.mo.debug:
            print(
                f"dt1 = {dt1}, dt2 = {dt2}, offset = {self.mo.offset}"
                f"insert start time = {insert_start_time} "
                f"insert_stop time = {insert_stop_time} "
                f"duration = {self.duration}\n"
                f"msi = {model_start_index}, msp = {model_stop_index} "
                f"model num_steps = {model_stop_index-model_start_index}\n"
                f"ssi = {signal_start_index}, ssp = {signal_stop_index} "
                f"signal num_steps = {signal_stop_index-signal_start_index}\n"
            )

        if signal_start_index < signal_stop_index:
            self.nf.m[model_start_index:model_stop_index] = self.s_m[
                signal_start_index:signal_stop_index
            ]
            if self.nf.isotopes:
                self.nf.l[model_start_index:model_stop_index] = self.s_l[
                    signal_start_index:signal_stop_index
                ]
        return self.nf

    def __square__(self, s, e) -> None:
        """Create Square Signal"""

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
        """Create pyramid type Signal

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
        """Create a bell curve type signal

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
        """Interpolate External data as a signal. Unlike the other signals,
        this will replace the values in the flux with those read from the
        external data source. The external data need to be in the following format

        Time [units], Rate [units], delta value [units]
        0,     10,   12

        i.e., the first row needs to be a header line

        """
        ed = ExternalData(
            name=f"{self.name}_ed",
            filename=self.filename,
            register=self,
            legend="None",
        )

        self.s_time = ed.x
        self.s_data = ed.y * self.scale

        self.st: float = self.s_time[0]  # start time
        self.et: float = self.s_time[-1]  # end time
        self.duration = int(round((self.et - self.st)))
        num_steps = int(self.duration / self.mo.dt)
        # setup the points at which to interpolate
        xi = np.linspace(self.st, self.et, num_steps)

        self.s_m: NDArrayFloat = np.interp(
            xi, self.s_time, self.s_data
        )  # interpolate flux
        if ed.zh:
            self.s_delta = ed.d
            self.s_d: NDArrayFloat = np.interp(xi, self.s_time, self.s_delta)
            self.s_l = get_l_mass(self.s_m, self.s_d, self.sp.r)
        else:
            self.s_l: NDArrayFloat = np.zeros(num_steps)

        return num_steps

    def __apply_signal__(self) -> None:
        """In case we deal with a source  signal, we need
        to create a source, and connect signal, source and reservoir
        Maybe this logic should be me moved elsewhere?
        """

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

    def __add__(self, other):
        """allow the addition of two signals and return a new signal"""

        ns = cp.deepcopy(self)

        # add the data of both fluxes
        # get delta of self
        sd = get_delta(self.data.l, self.data.m - self.data.l, self.data.sp.r)
        od = get_delta(other.data.l, other.data.m - other.data.l, other.data.sp.r)
        ns.data.m = self.data.m + other.data.m
        ns.data.l = get_l_mass(ns.data.m, sd + od, ns.data.sp.r)

        ns.n: str = self.n + "_and_" + other.n
        # print(f"adding {self.n} to {other.n}, returning {ns.n}")
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

        ns: Signal = cp.deepcopy(self)
        ns.n: str = self.n + f"_repeated_{times}_times"
        ns.data.n: str = self.n + f"_repeated_{times}_times_data"
        start: int = int(start / self.mo.dt)  # convert from time to index
        stop: int = int(stop / self.mo.dt)
        offset: int = int(offset / self.mo.dt)
        ns.start: float = start
        ns.stop: float = stop
        ns.offset: float = stop - start + offset
        ns.times: float = times
        ns.ms: NDArrayFloat = self.data.m[start:stop]  # get the data slice we are using
        ns.ds: NDArrayFloat = self.data.d[start:stop]

        diff = 0
        for _ in range(times):
            start += ns.offset
            stop += ns.offset
            if start > len(self.data.m):
                break
            elif stop > len(self.data.m):  # end index larger than data size
                diff: int = stop - len(self.data.m)  # difference
                stop -= diff
                lds: int = len(ns.ds) - diff
            else:
                lds: int = len(ns.ds)

            ns.data.m[start:stop]: NDArrayFloat = ns.data.m[start:stop] + ns.ms[:lds]
            ns.data.d[start:stop]: NDArrayFloat = ns.data.d[start:stop] + ns.ds[:lds]

        # and recalculate li and hi
        ns.data.l: NDArrayFloat
        ns.data.h: NDArrayFloat
        [ns.data.l, ns.data.h] = get_imass(ns.data.m, ns.data.d, ns.data.sp.r)
        return ns

    def __register_with_flux__(self, flux) -> None:
        """Register this signal with a flux. This should probably be done
        through a process!

        """

        from esbmtk import Flux, Species

        self.fo: Flux = flux  # the flux handle
        self.sp: Species = flux.sp  # the species handle
        # list of processes
        flux.lop.append(self)

    def __call__(self, t) -> list[float, float]:
        """Return Signal value at time t (mass and mass for light
        isotope). This will work as long a t is a multiple of dt, and i = t.
        may extend this by addding linear interpolation but that will
        be costly

        """

        import numpy as np

        m = np.interp(t, self.mo.time, self.data.m)
        if self.isotopes:
            l = np.interp(t, self.mo.time, self.data.l)
        else:
            l = 0

        return [m, l]

    def __plot__(self, M: Model, ax) -> None:
        """Plot instructions.
        M: Model
        ax: matplotlib axes handle
        """

        from esbmtk import set_y_limits

        # convert time and data to display units
        x = (M.time * M.t_unit).to(M.d_unit).magnitude
        y1 = (self.data.m * M.f_unit).to(self.data.plt_units).magnitude
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
            axt.set_ylabel(self.data.ld)
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


class VectorData(esbmtkBase):
    def __init__(self, **kwargs: dict[str, any]) -> None:
        """A simple container for 1-dimensional data. Typically used for
        results obtained by postprocessing.
        """

        from esbmtk import Q_
        from pint import Unit

        self.defaults: dict[str, list(str, tuple)] = {
            "name": ["None", (str,)],
            "register": ["None", (str, ReservoirGroup)],
            "species": ["None", (str, Species)],
            "data": ["None", (str, np.ndarray, float)],
            "isotopes": [False, (bool,)],
            "plt_units": ["None", (str, Unit)],
            "label": ["None", (str, bool)],
        }
        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "register", "species", "data", "label", "plt_units"]
        self.__initialize_keyword_variables__(kwargs)

        self.n = self.name
        self.parent = self.register
        self.sp = self.species
        self.mo = self.species.mo
        self.model = self.species.mo
        self.c = self.data
        self.label = self.name
        self.__register_name_new__()

    def get_plot_format(self):
        """Return concentrat data in plot units"""

        from esbmtk import Q_
        from pint import Unit

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
    """
    DataField: Datafields can be used to plot data which is computed after
    the model finishes in the overview plot windows. Therefore, datafields will
    plot in the same window as the reservoir they are associated with.
    Datafields must share the same x-axis is the model, and can have up to two
    y axis.

    Example::

             DataField(name = "Name"
                       register = Model handle,
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
        """Initialize this instance"""

        from . import Reservoir_no_set, VirtualReservoir, ExternalCode, Model

        # dict of all known keywords and their type
        self.defaults: dict[str, list(str, tuple)] = {
            "name": ["None", (str)],
            "register": [
                "None",
                (
                    Model,
                    Reservoir,
                    ReservoirGroup,
                    Reservoir_no_set,
                    VirtualReservoir,
                    ExternalCode,
                ),
            ],
            "associated_with": [
                "None",
                (
                    Model,
                    Reservoir,
                    ReservoirGroup,
                    Reservoir_no_set,
                    VirtualReservoir,
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
            elif isinstance(self.register, Reservoir):
                self.associated_with = self.register
            else:
                raise DataFieldError(
                    "Set associated_with or register to a reservoir name"
                )

        if isinstance(self.associated_with, Reservoir):
            self.plt_units = self.associated_with.plt_units
        elif isinstance(self.associated_with, ReservoirGroup):
            self.plt_units = self.associated_with.lor[0].plt_units
        else:
            raise ValueError("This needs fixing")

        if self.y1_color == "None":
            self.y1_color = []
            for i, d in enumerate(self.y1_data):
                self.y1_color.append(f"C{i}")

        if self.y1_style == "None":
            self.y1_style = []
            for i, d in enumerate(self.y1_data):
                self.y1_style.append(f"solid")

        if self.y2_color == "None":
            self.y2_color = []
            for i, d in enumerate(self.y2_data):
                self.y2_color.append(f"C{i}")

        if self.y2_style == "None":
            self.y2_style = []
            for i, d in enumerate(self.y2_data):
                self.y2_style.append(f"solid")

        self.n = self.name
        self.led = []

        # register with reservoir
        # self.associated_with.ldf.append(self)
        # register with model. needed for print_reservoirs
        self.register.ldf.append(self)
        if self.display_precision == 0:
            self.display_precision = self.mo.display_precision

        self.__register_name_new__()

    def __write_data__(
        self,
        prefix: str,
        start: int,
        stop: int,
        stride: int,
        append: bool,
        directory: str,
    ) -> None:
        """To be called by write_data and save_state"""

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
        """The input data for the DataField Class can be either missing, be a
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
                for e in range(y_l):
                    x.append(M.time)

        elif isinstance(x, np.ndarray):
            # print(f"np_array, y_l = {y_l}")
            xx = []
            if y_l == 1:  # single y data
                xx.append(x)
            else:  # mutiple y data
                # print(f" no x-data mutiple y-data y_l = {y_l}")
                for e in range(y_l):
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
        elif y_l > 1:
            if not isinstance(label, list):
                ll = []
                for e in y_l:
                    ll.append(label)
                label = ll

        return x, y, label

    def __plot_data__(self, ax, x, y, t, l, i, color, style) -> None:
        """Plot data either as line of scatterplot

        :param ax: axis handle
        :param x: x data
        :param y: y data
        :param t: plot type
        :param l: label str
        :param i: index

        """
        if t == "plot":
            # ax.plot(x, y, color=f"C{i}", label=l)
            ax.plot(x, y, color=color[i], linestyle=style[i], label=l)
        else:
            ax.scatter(x, y, color=color[i], label=l)

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
        for i, d in enumerate(self.y1_data):  # loop over datafield list
            if self.x1_as_time:
                x1 = (self.x1_data[i] * M.t_unit).to(M.d_unit).magnitude
            else:
                x1 = self.x1_data[i]
                # y1 = (self.y * M.c_unit).to(self.plt_units).magnitude
                # 1 = self.y
                # y1_label = f"{self.legend_left} [{self.plt_units:~P}]"

            self.__plot_data__(
                ax,
                x1,
                self.y1_data[i],
                self.y1_type,
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
            for i, d in enumerate(self.y2_data):  # loop over datafield list
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


class Reservoir_no_set(ReservoirBase):
    """This class is similar to a regular reservoir, but we make no
    assumptions about the type of data contained. I.e., all data will be
    left alone

    """

    def __init__(self, **kwargs) -> None:
        """The original class will calculate delta and concentration from mass
        an d and h and l. Since we want to use this class without a
        priory knowledge of how the reservoir arrays are being used we
        overwrite the data generated during initialization with the
        values provided in the keywords

        """

        from esbmtk import (
            ConnectionGroup,
            Species,
            SourceGroup,
            SinkGroup,
            ReservoirGroup,
        )

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "plot_transform_c": ["None", (str, Callable)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "function": ["None", (str, Callable)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
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
        self.__set_legacy_names__(kwargs)

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
        self.__register_name_new__()

        self.__aux_inits__()
        self.state = 0


class ExternalCode(Reservoir_no_set):
    """This class can be used to implement user provided functions. The
    data inside a VR_no_set instance will only change in response to a
    user provided function but will otherwise remain unaffected. That is,
    it is up to the user provided function to manage changes in reponse to
    external fluxes. A VR_no_set is declared in the following way::

        ExternalCode(
                    name="cs",     # instance name
                    species=CO2,   # species, must be given
                    # the vr_data_fields contains any data that is referenced inside the
                    # function, rather than passed as argument, and all data that is
                    # explicitly referenced by the model
                    vr_datafields :dict ={"Hplus": self.swc.hplus,
                                          "Beta": 0.0},
                    function=calc_carbonates, # function reference, see below
                    fname = function name as string
                    function_input_data="DIC TA",
                    # Note that parameters must be individual float values
                    function_params:tuple(float)
                    # list of return values
                    return_values={  # these must be known speces definitions
                                  "Hplus": rg.swc.hplus,
                                   "zsnow": float(abs(kwargs["zsnow"])),
                                   },
                    register=rh # reservoir_handle to register with.
                )

        the dict keys of vr_datafields will be used to create alias
        names which can be used to access the respective variable


    The general template for a user defined function is a follows::

      def calc_carbonates(i: int, input_data: list, vr_data: list, params: list) -> None:
          # i = index of current timestep
          # input_data = list of np.arrays, typically data from other Reservoirs
          # vr_data = list of np.arrays created during instance creation (i.e. the vr data)
          # params = list of float values (at least one!)
          pass
       return

    Note that this function should not return any values, and that all input fields must have
    at least one entry!

    """

    def __init__(self, **kwargs) -> None:
        """The original class will calculate delta and concentration from mass
        an d and h and l. Since we want to use this class without a
        priory knowledge of how the reservoir arrays are being used we
        overwrite the data generated during initialization with the
        values provided in the keywords

        """

        from esbmtk import (
            ConnectionGroup,
            Species,
            SourceGroup,
            SinkGroup,
            ReservoirGroup,
            Reservoir,
            Connection,
            Model,
        )
        from typing import Callable

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "plot_transform_c": ["None", (str, Callable)],
            "legend_left": ["None", (str)],
            "plot": ["yes", (str)],
            "groupname": ["None", (str)],
            "function": ["None", (Callable, str)],
            "display_precision": [0.01, (int, float)],
            "register": [
                "None",
                (
                    SourceGroup,
                    SinkGroup,
                    ReservoirGroup,
                    Reservoir,
                    Connection,
                    ConnectionGroup,
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
            "ftype": ["std", (str)],
            "ref_flux": ["None", (list, str)],
            "return_values": ["None", (list)],
            "arguments": ["None", (str, list)],
            "r_s": ["None", (str, ReservoirGroup)],
            "r_d": ["None", (str, ReservoirGroup)],
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "species",
            "register",
        ]

        self.__initialize_keyword_variables__(kwargs)

        self.__set_legacy_names__(kwargs)
        self.lro: list = []  # list of all return objects.
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx
        self.plt_units = self.mo.c_unit

        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"
        self.mo.lor.append(self)  # add this reservoir to the model
        self.__register_name_new__()
        self.state = 0
        name = f"{self.full_name}_generic_function".replace(".", "_")
        logging.info(f"creating {name}")

        if self.vr_datafields != "None":
            self.alias_list = list(self.vr_datafields.keys())
            # initialize data fields
            self.vr_data = list()
            for e in self.vr_datafields.values():
                if isinstance(e, (float, int)):
                    self.vr_data.append(np.full(self.mo.steps, e, dtype=float))
                else:
                    self.vr_data.append(e)

        self.mo.lpc_r.append(self)
        # self.mo.lpc_r.append(self.gfh)

        self.mo.lor.remove(self)
        # but lets keep track of  virtual reservoir in lvr.
        self.mo.lvr.append(self)

        # print(f"added {self.name} to lvr 2")

        # create temporary memory if we use multiple solver iterations
        if self.mo.number_of_solving_iterations > 0:
            for i, d in enumerate(self.vr_data):
                setattr(self, f"vrd_{i}", np.empty(0))

        if self.alias_list != "None":
            self.create_alialises()

        self.update_parameter_count()

    def create_alialises(self) -> None:
        """Register  alialises for each vr_datafield"""

        for i, a in enumerate(self.alias_list):
            # print(f"{a} = {self.vr_data[i][0]}")
            setattr(self, a, self.vr_data[i])

    def update_parameter_count(self):
        if len(self.function_params) > 0:
            self.param_start = self.model.vpc
            self.model.vpc = self.param_start + 1
            self.has_p = True
            # upudate global parameter list
            self.model.gpt = (*self.model.gpt, self.function_params)
        else:
            self.has_p = False

    def append(self, **kwargs) -> None:
        """This method allows to update GenericFunction parameters after the
        VirtualReservoir has been initialized. This is most useful
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
        """Copy the last value to the first position so that we can restart the computation"""

        for i, d in enumerate(self.vr_data):
            d[0] = d[-2]
            setattr(
                self,
                f"vrd_{i}",
                np.append(getattr(self, f"vrd_{i}"), d[0 : -2 : self.mo.reset_stride]),
            )

    def __merge_temp_results__(self) -> None:
        """Once all iterations are done, replace the data fields
        with the saved values

        """

        # print(f"merging {self.full_name} with whith len of vrd= {len(self.vrd_0)}")
        for i, d in enumerate(self.vr_data):
            self.vr_data[i] = getattr(self, f"vrd_{i}")

        # update aliases
        self.create_alialises()
        # print(f"new length = {len(self.vr_data[0])}")

    def __read_state__(self, directory: str) -> None:
        """read virtual reservoir data from csv-file into a dataframe

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
        for i, n in enumerate(headers):
            # first column is time
            if i > 0:
                # print(f"i = {i}, header = {n}, data = {df.iloc[-3:, i]}")
                self.vr_data[i - 1][:3] = df.iloc[-3:, i]

    def __sub_sample_data__(self, stride) -> None:
        """There is usually no need to keep more than a thousand data points
        so we subsample the results before saving, or processing them

        """

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
        """To be called by write_data and save_state"""

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


class VirtualReservoir_no_set(ExternalCode):
    """Alias to ensure backwards compatibility"""


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
        # print(f"added {self.name} to lvr 2")

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
            setattr(self, key, value)  # update self
            setattr(self.gfh, key, value)  # update function


class GasReservoir(ReservoirBase):
    """This object holds reservoir specific information similar to the Reservoir class

          Example::

                  Reservoir(name = "foo",     # Name of reservoir
                            species = CO2,    # Species handle
                            delta = 20,       # initial delta - optional (defaults  to 0)
                            reservoir_mass = quantity # total mass of all gases
                                             defaults to 1.833E20 mol
                            species_ppm =  number # concentration in ppm
                            plot = "yes"/"no", defaults to yes
                            plot_transform_c = a function reference, optional (see below)
                            legend_left = str, optional, useful for plot transform
                            display_precision = number, optional, inherited from Model
                            register = optional, use to register with Reservoir Group
                            isotopes = True/False otherwise use Model.m_type
                            )



    Accesing Reservoir Data:
    ~~~~~~~~~~~~~~~~~~~~~~~~

    You can access the reservoir data as:

    - Name.m # species mass
    - Name.l # mass of light isotope
    - Name.d # species delta (only avaible after M.get_delta_values()
    - Name.c # partial pressure
    - Name.v # total gas mass

    Useful methods include:

    - Name.write_data() # save data to file
    - Name.info()   # info Reservoir
    """

    def __init__(self, **kwargs) -> None:
        """Initialize a reservoir."""

        from esbmtk import Q_, Species, Model
        from typing import Callable

        # provide a dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "species": ["None", (str, Species)],
            "delta": [0, (int, float)],
            "reservoir_mass": ["1.833E20 mol", (str, Q_)],
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
            "register",
        ]

        self.__initialize_keyword_variables__(kwargs)

        self.__set_legacy_names__(kwargs)

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
        # isotope mass
        # self.l = get_l_mass(self.m, self.delta, self.species.r)
        self.l = get_l_mass(self.c, self.delta, self.species.r)
        # delta of reservoir
        self.v: float = (
            np.zeros(self.mo.steps) + self.volume.to(self.v_unit).magnitude
        )  # mass of atmosphere

        if self.mo.number_of_solving_iterations > 0:
            self.mc = np.empty(0)
            self.cc = np.empty(0)
            self.lc = np.empty(0)
            self.vc = np.empty(0)

        self.mo.lor.append(self)  # add fthis reservoir to the model
        self.mo.lic.append(self)  # reservoir type object list
        # register instance name in global name space

        # register this group object in the global namespace
        if self.mo.register == "local" and self.register == "None":
            self.register = self.mo

        self.__register_name_new__()

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
        """write data by index"""

        self.m[i]: float = value[0]
        self.l[i]: float = value[1]
        # self.c[i]: float = self.m[i] / self.v[i]  # update concentration
        # self.v[i]: float = self.v[i - 1] + value[0]
        # self.c[i]: float = self.m[i] / self.v[i]  # update concentration
        # self.h[i]: float = value[2]
        # self.d[i]: float = get_delta(self.l[i], self.h[i], self.sp.r)

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """write data by index"""
        self.m[i]: float = value[0]
        # self.c[i]: float = self.m[i] / self.v[i]  # update concentration
        # self.v[i]: float = self.v[i - 1] + value[0]


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

    Methods:
      - name.plot()

    Data:
      - name.x
      - name.y
      - name.df = dataframe as read from csv file

    """

    def __init__(self, **kwargs: dict[str, str]):
        from esbmtk import Q_, Model, Reservoir, DataField

        # dict of all known keywords and their type
        self.defaults: dict[str, list[any, tuple]] = {
            "name": ["None", (str)],
            "filename": ["None", (str)],
            "legend": ["None", (str)],
            "reservoir": ["None", (str, Reservoir, DataField, GasReservoir)],
            "offset": ["0 yrs", (Q_, str)],
            "display_precision": [0.01, (int, float)],
            "scale": [1, (int, float)],
            "register": [
                "None",
                (str, Model, Reservoir, DataField, GasReservoir, Signal),
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
        if isinstance(self.reservoir, Reservoir):
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
        xq = Q_(xh)
        yq = Q_(yh)

        # add these to the data we are are reading
        self.x: NDArrayFloat = self.df.iloc[:, 0].to_numpy() * xq
        self.y: NDArrayFloat = self.df.iloc[:, 1].to_numpy() * yq

        if self.zh:
            # delta is assumed to be without units
            self.d: NDArrayFloat = self.df.iloc[:, 2].to_numpy()
        else:
            self.zh = False

        # map into model space
        # self.x = self.x - self.x[0] + self.offset
        # map into model units, and strip unit information
        self.x = self.x.to(self.mo.t_unit).magnitude
        # self.s_data = self.s_data.to(self.mo.f_unit).magnitude * self.scale

        mol_liter = Q_("1 mol/liter").dimensionality
        mol_kg = Q_("1 mol/kg").dimensionality

        if isinstance(yq, Q_):
            # test what type of Quantity we have
            if yq.is_compatible_with("dimensionless"):  # dimensionless
                self.y = self.y.magnitude
            elif yq.is_compatible_with("liter/yr"):  # flux
                self.y = self.y.to(self.mo.r_unit).magnitude
            elif yq.is_compatible_with("mol/yr"):  # flux
                self.y = self.y.to(self.mo.f_unit).magnitude
            elif yq.dimensionality == mol_liter:  # concentration
                self.y = self.y.to(self.mo.c_unit).magnitude
            elif yq.dimensionality == mol_kg:  # concentration
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

        self.__register_name_new__()

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
