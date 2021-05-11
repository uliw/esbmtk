from numbers import Number
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
from pandas import DataFrame
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

set_printoptions(precision=4)
# from .utility_functions import *
from .esbmtk import esbmtkBase, Reservoir, Flux, Signal, Source, Sink
from .utility_functions import sort_by_type
from .solver import get_imass, get_frac, get_delta, get_flux_data

# from .connections import ConnnectionGroup


class Process(esbmtkBase):
    """This class defines template for process which acts on one or more
    reservoir flux combinations. To use it, you need to create an
    subclass which defines the actual process implementation in their
    call method. See 'PassiveFlux as example'

    """

    __slots__ = ("reservoir", "r", "flux", "r", "mo", "direction")

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """

        self.__defaultnames__()  # default kwargs names
        self.__initerrormessages__()  # default error messages
        self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

    def __postinit__(self) -> None:
        """Do some housekeeping for the process class"""

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.r: Reservoir = self.reservoir
        self.f: Flux = self.flux
        self.m: Model = self.r.sp.mo  # the model handle
        self.mo: Model = self.m

        # self.rm0: float = self.r.m[0]  # the initial reservoir mass
        if isinstance(self.r, Reservoir):
            self.direction: Dict[Flux, int] = self.r.lio[self.f]

        self.__misc_init__()

    def __delayed_init__(self) -> None:
        """
        Initialize stuff which is only known after the entire model has been defined.
        This will be executed just before running the model. You need to add the following
        two lines somewhere in the init procedure (preferably by redefining __misc_init__)
        and redefine __delayed_init__ to actually execute any code below

        # this process requires delayed init
        self.mo.lto.append(self)

        """

        pass

    def __misc_init__(self) -> None:
        """This is just a place holder method which will be called by default
        in __post_init__() This can be overloaded to add additional
        code to the init procedure without the need to redefine
        init. This useful for processes which only define a call method.

        """

        pass

    def __defaultnames__(self) -> None:
        """Set up the default names and dicts for the process class. This
        allows us to extend these values without modifying the entire init process

        """

        from .connections import ConnectionGroup
        from esbmtk import Reservoir, ReservoirGroup, Flux

        # provide a dict of known keywords and types
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": (Reservoir, Source, Sink),
            "flux": Flux,
            "rate": Number,
            "delta": Number,
            "lt": Flux,
            "alpha": Number,
            "scale": Number,
            "ref_reservoirs": (Flux, Reservoir, list, str),
            "register": (str, ConnectionGroup, Reservoir, ReservoirGroup, Flux),
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # default type hints
        self.scale: Number
        self.delta: Number
        self.alpha: Number

    def __register__(self, reservoir: Reservoir, flux: Flux) -> None:
        """Register the flux/reservoir pair we are acting upon, and register
        the process with the reservoir

        """

        # register the reservoir flux combination we are acting on
        self.f: Flux = flux
        self.r: Reservoir = reservoir
        # add this process to the list of processes acting on this reservoir
        reservoir.lop.append(self)
        flux.lop.append(self)

    def show_figure(self, x, y) -> None:
        """Apply the current process to the vector x, and show the result as y.
        The resulting figure will be automatically saved.

        Example::
             process_name.show_figure(x,y)

        """
        pass


class GenericFunction(Process):
    """This Process class takes a generic function and up to 6 optional
     function arguments, and will replace the mass value(s) of the
     given reservoirs with whatever the function calculates. This is
     particularly useful e.g., to calculate the pH of a given reservoir
     as function of e.g., Alkalinity and DIC.

     Parameters:
      - name = name of process,
      - act_on = name of a reservoir this process will act upon
      - function  = a function reference
      - a1 to a3,

     in order to use this function we need first declare a function we plan to
     use with the generic function process. This function needs to follow the
     below template.

    In order to be compatible with the numba solver, a1 and a2 must be
    an array of 1-D numpy.arrays i.e., [m, l, h, c]. The array can have
    any number of arrays though. a3 must be single array (or list)
    containing an arbitrary number of entries. All numbers in a1 to a3
    _must_ be of type float64.

    The function must return a list of numbers which correspond to the
    data which describe a reservoir i.e., mass, light isotope, heavy
    isotope, delta, and concentration

    In order to use this function we need first declare a function we plan to
    use with the generic function process. This function needs to follow this
    template::

        def my_func(i, a1, a2, a3) -> tuple:
            #
            # i = index of the current timestep
            # a1 to a2 =  optional function parameter. These must be present,
            # even if your function will not use it
            # will be applied to this reservoir.

            # calc some stuff and return it as

            return [m, l, h, d, c] # where m= mass, and l & h are the respective
                                   # isotopes. d denotes the delta value and
                                   # c the concentration
                                   # Use dummy value as necessary.


         This function can then be used as::

         GenericFunction(name="foo",
                 function=my_func,
                 a1 = some argument,
                 a2 = some argument,
                 a3 = some argument
                 a4 = some argument
                 acton_on = reservoir reference
                 )

     see calc_H in the carbonate chemistry module as example

    """

    __slots__ = (
        "function",
        "act_on",
        "a1",
        "a2",
        "a3",
        "a4",
        "i",
        "act_on",
    )

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options

        """

        self.__defaultnames__()  # default kwargs names

        # list of allowed keywords
        self.lkk: Dict[str, any] = {
            "name": str,
            "act_on": (Flux, Reservoir, any),
            "function": any,
            "a1": any,
            "a2": any,
            "a3": any,
            "a4": any,
            "act_on": any,
        }

        # required arguments
        self.lrk: list = ["name", "act_on", "function"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "a1": List(),
            "a2": List(),
            "a3": List(),
            "a4": List(np.zeros(3)),
            "act_on": List(),
        }

        self.__initerrormessages__()  # default error messages
        self.bem.update(
            {
                "act_on": "a reservoir or flux",
                "function": "a function",
                "a1": "must be a numpy array",
                "a2": "must be a numpy array",
                "a3": "must be a numba List",
                "a4": "must be a numpy array",
            }
        )
        self.__validateandregister__(kwargs)  # initialize keyword values

        if not callable(self.function):
            raise ValueError("function must be defined before it can be used here")

        self.__postinit__()  # do some housekeeping
        self.__register_name__()  #

        # register with reservoir
        if isinstance(self.act_on, Reservoir):
            self.act_on.lpc.append(self)  # register with Reservoir
            self.act_on.mo.lpc_r.append(self)  # Register with Model
        elif isinstance(self.act_on, Flux):
            self.act_on.lpc.append(self)  # register with Flux
            self.act_on.mo.lpc_f.append(self)  # Register with Model
        else:
            raise ValueError("functions can only act upon reservoirs or fluxes")

    def __call__(self, i: int) -> None:
        """Here we execute the user supplied function and assign the
        return value to the flux or reservoir

        Where i = index of the current timestep
              acting_on = reservoir or flux we are acting on.

        """

        r = self.function(
            i,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
        )

        return r

    # redefine post init
    def __postinit__(self) -> None:
        """Do some housekeeping for the process class"""

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.m: Model = self.act_on.mo  # the model handle
        self.mo: Model = self.act_on.mo

        self.__misc_init__()

    def get_process_args(self) -> tuple:
        """return the data associated with this object"""

        func_name: function = self.function

        return (
            func_name,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            # List(
            [
                self.act_on.m,
                self.act_on.l,
                self.act_on.h,
                self.act_on.d,
                self.act_on.c,
            ]
            # ),
        )


class LookupTable(Process):
    """This process replaces the flux-values with values from a static
    lookup table

         Example::

         LookupTable("name", upstream_reservoir_handle, lt=flux-object)

         where the flux-object contains the mass, li, hi, and delta values
         which will replace the current flux values.

    """

    def __call__(self, r: Reservoir, i: int) -> None:
        """Here we replace the flux value with the value from the flux object
        which we use as a lookup-table

        """
        self.m[i]: float = self.lt.m[i]
        self.d[i]: float = self.lt.d[i]
        self.l[i]: float = self.lt.l[i]
        self.h[i]: float = self.lt.h[i]


class AddSignal(Process):
    """This process adds values to the current flux based on the values provided by the sifnal object.
    This class is typically invoked through the connector object

     Example::

     AddSignal(name = "name",
               reservoir = upstream_reservoir_handle,
               flux = flux_to_act_upon,
               lt = flux with lookup values)

     where - the upstream reservoir is the reservoir the process belongs too
             the flux is the flux to act upon
             lt= contains the flux object we lookup from

    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
        Create a new process object with a given process type and options
        """

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["lt", "flux", "reservoir"])  # new required keywords

        self.__initerrormessages__()
        # self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

        # decide whichh call function to use
        # if self.mo.m_type == "both":
        if self.reservoir.isotopes:
            self.__execute__ = self.__with_isotopes__
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function
    def __call__(self, r: Reservoir, i: int):
        return self.__execute__(r, i)

    # use this when we do isotopes
    def __with_isotopes__(self, r, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
        the flux value with the value from the signal object which
        we use as a lookup-table (self.lt)

        """
        # add signal mass to flux mass

        self.f.m[i] = self.f.m[i] + self.lt.m[i]
        self.f.d[i] = self.f.d[i] + self.lt.d[i]
        if self.f.m[i] != 0:
            # self.f[i] = self.f[i] + self.lt[i]
            self.f.l[i], self.f.h[i] = get_imass(self.f.m[i], self.f.d[i], r.rvalue)
        # signals may have zero mass, but may have a delta offset. Thus, we do not know
        # the masses for the light and heavy isotope. As such we have to calculate the masses
        # after we add the signal to a flux

    # use this when we do isotopes
    def __without_isotopes__(self, r, i) -> None:
        """Each process is associated with a flux (self.f). Here we replace
        the flux value with the value from the signal object which
        we use as a lookup-table (self.lt)

        """
        # add signal mass to flux mass
        self.f.m[i] = self.f.m[i] + self.lt.m[i]


class PassiveFlux(Process):
    """This process sets the output flux from a reservoir to be equal to
    the sum of input fluxes, so that the reservoir concentration does
    not change. Furthermore, the isotopic ratio of the output flux
    will be set equal to the isotopic ratio of the reservoir The init
    and register methods are inherited from the process class. The
    overall result can be scaled, i.e., in order to create a split flow etc.
    Example::

    PassiveFlux(name = "name",
                reservoir = upstream_reservoir_handle
                flux = flux handle)

    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux"])  # new required keywords
        self.__initerrormessages__()
        # self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values
        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

    def __misc_init__(self) -> None:
        """This is just a place holder method which will be called by default
        in __post_init__() This can be overloaded to add additional
        code to the init procedure without the need to redefine
        init. This useful for processes which only define a call method.

        """

        # this process requires delayed init.
        self.mo.lto.append(self)

    def __delayed_init__(self) -> None:
        """
        Initialize stuff which is only known after the entire model has been defined.
        This will be executed just before running the model.

        """

        # Create a list of fluxes wich excludes the flux this process
        # will be acting upon

        print(f"delayed init for {self.name}")
        self.fws: List[Flux] = self.r.lof.copy()
        self.fws.remove(self.f)  # remove this handle

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. That is, we calculate the sum of all fluxes
        excluding this flux. This sum will be equal to this flux. This will likely only
        work for outfluxes though.

        Should this be done for output fluxes as well?

        """

        new: float = 0.0

        # calc sum of fluxes in fws. Note that at this point, not all fluxes
        # will be known so we need to use the flux values from the previous times-step
        for j, f in enumerate(self.fws):
            # print(f"{f.n} = {f.m[i-1] * reservoir.lio[f]}")
            new += f.m[i - 1] * reservoir.lio[f]

        # print(f"sum = {new:.0f}\n")
        self.f[i] = get_flux_data(new, reservoir.d[i - 1], reservoir.rvalue)

        # m = new
        # r = reservoir.l[i - 1] / reservoir.m[i - 1]
        # l = m * r
        # h = m - l
        # self.f.m[i] = m
        # self.f.l[i] = l
        # self.f.h[i] = h

    # self.f.d[i] = reservoir.d[i - 1]


class PassiveFlux_fixed_delta(Process):
    """This process sets the output flux from a reservoir to be equal to
    the sum of input fluxes, so that the reservoir concentration does
    not change. However, the isotopic ratio of the output flux is set
    at a fixed value. The init and register methods are inherited
    from the process class. The overall result can be scaled, i.e.,
    in order to create a split flow etc.  Example::

    PassiveFlux_fixed_delta(name = "name",
                            reservoir = upstream_reservoir_handle,
                            flux handle,
                            delta = delta offset)

    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """

        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "delta", "flux"])  # new required keywords

        self.__initerrormessages__()
        # self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        # legacy names
        self.f: Flux = self.flux
        # legacy variables
        self.mo = self.reservoir.mo

        print("\nn *** Warning, you selected the PassiveFlux_fixed_delta method ***\n ")
        print(
            " This is not a particularly phyiscal process is this really what you want?\n"
        )
        print(self.__doc__)
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
        apply_flux_modifier method of a reservoir which itself is
        called by the model execute method

        """

        r: float = reservoir.rvalue  # the isotope reference value

        varflux: Flux = self.f
        flux_list: List[Flux] = reservoir.lof.copy()
        flux_list.remove(varflux)  # remove this handle

        # sum up the remaining fluxes
        newflux: float = 0
        for f in flux_list:
            newflux = newflux + f.m[i - 1] * reservoir.lio[f]

        # set isotope mass according to keyword value
        self.f[i] = array(get_flux_data(newflux, self.delta, r))


class VarDeltaOut(Process):
    """Unlike a passive flux, this process sets the flux istope ratio
    equal to the isotopic ratio of the reservoir. The
    init and register methods are inherited from the process
    class.

    VarDeltaOut(name = "name",
                reservoir = upstream_reservoir_handle,
                flux = flux handle,
                rate = rate,)

    """

    __slots__ = ("rate", "flux", "reservoir")

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """Initialize this Process"""

        from . import ureg, Q_
        from .connections import ConnectionGroup
        from esbmtk import Flux, Reservoir, ReservoirGroup

        # get default names and update list for this Process
        self.__defaultnames__()
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": (Reservoir, Source, Sink),
            "flux": Flux,
            "rate": (str, Q_),
            "register": (ConnectionGroup, ReservoirGroup, Reservoir, Flux, str),
        }
        self.lrk.extend(["reservoir", "flux"])  # new required keywords
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

        # decide which call function to use
        # if self.mo.m_type == "both":
        if self.reservoir.isotopes:
            # print(
            #    f"vardeltaout with isotopes for {self.reservoir.register.name}.{self.reservoir.name}"
            # )
            if isinstance(self.reservoir, Reservoir):
                # print("Using reservoir")
                self.__execute__ = self.__with_isotopes_reservoir__
            elif isinstance(self.reservoir, Source):
                # print("Using Source")
                self.__execute__ = self.__with_isotopes_source__
            else:
                raise ValueError(
                    f"{self.name}, reservoir must be of type Source or Reservoir, not {type(self.reservoir)}"
                )
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function
    def __call__(self, reservoir: Reservoir, i: int):
        return self.__execute__(reservoir, i)

    def __with_isotopes_reservoir__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
        apply_flux_modifier method of a reservoir which itself is
        called by the model execute method

        """

        m: float = self.flux.m[i]
        if m != 0:
            # if reservoir.register.name == "db":
            #    print(f"{reservoir.name} d={reservoir.d[i-1]}")
            r: float = reservoir.species.element.r
            d: float = reservoir.d[i - 1]
            l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
            h: float = m - l

            self.flux[i] = [m, l, h, d]

    def get_process_args(self, reservoir: Reservoir):

        # if upstream is a source, we only have a single delta value
        # so we need to patch this. Maybe this should move to source?
        if isinstance(self.reservoir, Source):
            delta = self.reservoir.d
        else:
            delta = reservoir.d

        func_name: function = self.p_vardeltaout

        res_data = List([self.flux.m, delta, reservoir.c])
        flux_data = List([self.flux.m, self.flux.l, self.flux.h, self.flux.d])
        proc_const = List([float(reservoir.species.element.r), float(0)])

        # print(f"rd = {res_data[2][0]}, fd = {flux_data[0][0]}")
        return func_name, res_data, flux_data, proc_const

    @staticmethod
    @njit()
    def p_vardeltaout(res_data, flux_data, proc_const, i) -> None:
        # concentration times scale factor
        # res_data[0] is actually flux data. See get_process args for details

        m: float = res_data[0][i - 1]  # r mass
        d: float = res_data[1][i - 1]  # delta
        l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)

        flux_data[0][i] = m
        flux_data[1][i] = l
        flux_data[2][i] = m - l
        flux_data[3][i] = d

    def __with_isotopes_source__(self, reservoir: Reservoir, i: int) -> None:
        """If the source of the flux is a source, there is only a single delta value.
        Changes to the flux delta are applied through the Signal class.

        """

        m: float = self.flux.m[i]
        if m != 0:
            d: float = self.reservoir.delta
            r: float = reservoir.species.element.r
            l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
            h: float = m - l

            self.flux[i] = [m, l, h, d]

    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
        apply_flux_modifier method of a reservoir which itself is
        called by the model execute method

        """

        pass


class ScaleFlux(Process):
    """This process scales the mass of a flux (m,l,h) relative to another
    flux but does not affect delta. The scale factor "scale" and flux
    reference must be present when the object is being initalized

    Example::
         ScaleFlux(name = "Name",
                   reservoir = reservoir_handle (upstream or downstream)
                   scale = 1
                   ref_reservoirs = flux we use for scale)

    """

    __slots__ = ("rate", "scale", "ref_reservoirs", "reservoir", "flux")

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """Initialize this Process"""
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(
            ["reservoir", "flux", "scale", "ref_reservoirs"]
        )  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

        # decide which call function to use
        # if self.mo.m_type == "both":
        if self.reservoir.isotopes:
            self.__execute__ = self.__with_isotopes__
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function
    def __call__(self, reservoir: Reservoir, i: int):
        return self.__execute__(reservoir, i)

    def __with_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the reference object, but that we will set the
        delta according to the reservoir

        """

        m: float = self.ref_reservoirs.m[i - 1] * self.scale
        r: float = reservoir.species.element.r
        d: float = reservoir.d[i - 1]
        l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)

        self.flux[i]: np.array = [m, l, m - l, d]

    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the reference object, but that we will set the
        delta according to the reservoir (or the flux?)

        """
        self.f[i] = self.ref_reservoirs[i - 1] * self.scale

    def get_process_args(self, reservoir: Reservoir):
        """"""

        func_name: function = self.p_scale_flux

        res_data = List([self.ref_reservoirs.m, reservoir.d, self.ref_reservoirs.m])
        flux_data = List([self.flux.m, self.flux.l, self.flux.h, self.flux.d])
        proc_const = List([float(reservoir.species.element.r), float(self.scale)])

        return func_name, res_data, flux_data, proc_const

    @staticmethod
    @njit()
    def p_scale_flux(res_data, flux_data, proc_const, i) -> None:

        m: float = res_data[0][i - 1] * proc_const[1]
        d: float = res_data[1][i - 1]
        l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)
        # print(f"ScaleFlux m = {m:.2e}, scale = {proc_const[1]}")

        flux_data[0][i] = m
        flux_data[1][i] = l
        flux_data[2][i] = m - l
        flux_data[3][i] = d


class Reaction(ScaleFlux):
    """This process approximates the effect of a chemical reaction between
    two fluxes which belong to a differents species (e.g., S, and O).
    The flux belonging to the upstream reservoir will simply be
    scaled relative to the flux it reacts with. The scaling is given
    by the ratio argument. So this function is equivalent to the
    ScaleFlux class.

    Example::

       Connect(source=IW_H2S,
               sink=S0,
               ctype = "react_with",
               scale=1,
               ref_reservoirs = O2_diff_to_S0,
               scale =1,
       )
    """


class FluxDiff(Process):
    """The new flux will be the difference of two fluxes"""

    """This process scales the mass of a flux (m,l,h) relative to another
     flux but does not affect delta. The scale factor "scale" and flux
     reference must be present when the object is being initalized

     Example::
          ScaleFlux(name = "Name",
                    reservoir = upstream_reservoir_handle,
                    scale = 1
                    ref_reservoirs = flux we use for scale)

     """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """Initialize this Process"""
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(
            ["reservoir", "flux", "scale", "ref_reservoirs"]
        )  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        # legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the reference object, but that we will set the
        delta according to the reservoir (or the flux?)

        """
        self.f[i] = (self.ref_reservoirs[0][i] - self.ref_reservoirs[1][i]) * self.scale


class Fractionation(Process):
    """This process offsets the isotopic ratio of the flux by a given
       delta value. In other words, we add a fractionation factor

    Example::
         Fractionation(name = "Name",
                       reservoir = upstream_reservoir_handle,
                       flux = flux handle
                       alpha = 12 in permil (e.f)

    """

    __slots__ = ("flux", "reservoir")

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "alpha"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        # alpha is given in permil, but the fractionation routine expects
        # it as 1 + permil, i.e., 70 permil would 1.007
        # legacy variables
        self.alp = 1 + self.alpha / 1000
        self.mo = self.reservoir.mo
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
        Set flux isotope masses based on fractionation factor

        """

        if self.f.m[i] != 0:
            # print(f"delta before = {get_delta(self.f.l[i], self.f.h[i], self.f.rvalue)}")
            self.f.l[i], self.f.h[i] = get_frac(self.f.m[i], self.f.l[i], self.alp)
            # update delta
            # self.f.d[i] = get_delta(self.f.l[i], self.f.h[i], self.f.rvalue)
            self.f.d[i] = self.f.d[i] + self.alpha
            # print(f"delta after = {get_delta(self.f.l[i], self.f.h[i], self.f.rvalue)}\n")

        return

    def get_process_args(self, reservoir: Reservoir):

        func_name: function = self.p_fractionation

        res_data = List([reservoir.m, reservoir.d, reservoir.c])
        flux_data = List([self.flux.m, self.flux.l, self.flux.h, self.flux.d])
        proc_const = List([float(reservoir.species.element.r), float(self.alpha)])

        return func_name, res_data, flux_data, proc_const

    @staticmethod
    @njit()
    def p_fractionation(res_data, flux_data, proc_const, i) -> None:
        #
        d: float = flux_data[3][i] + proc_const[1]
        m: float = flux_data[0][i]
        l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)

        flux_data[0][i] = m
        flux_data[1][i] = l
        flux_data[2][i] = m - l
        flux_data[3][i] = d


class RateConstant(Process):
    """This is a wrapper for a variety of processes which depend on rate constants
    Please see the below class definitions for details on how to call them
    At present, the following processes are defined

    ScaleRelativeToNormalizedConcentration
    ScaleRelativeToConcentration

    """

    __slots__ = ("scale", "ref_value", "k_value", "flux", "reservoir")

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """Initialize this Process"""

        from . import ureg, Q_
        from .connections import SourceGroup, SinkGroup, ReservoirGroup
        from .connections import ConnectionGroup
        from esbmtk import Flux

        # Note that self.lkk values also need to be added to the lkk
        # list of the connector object.

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names

        # update the allowed keywords
        self.lkk: dict = {
            "scale": Number,
            "k_value": Number,
            "name": str,
            "reservoir": (Reservoir, Source, Sink),
            "flux": Flux,
            "ref_reservoirs": list,
            "left": (list, Reservoir, Number),
            "right": (list, Reservoir, Number),
            "register": (
                SourceGroup,
                SinkGroup,
                ReservoirGroup,
                ConnectionGroup,
                Flux,
                str,
            ),
        }

        # new required keywords
        self.lrk.extend(["reservoir", ["scale", "k_value"]])

        # dict with default values if none provided
        # self.lod = {r

        self.__initerrormessages__()

        # add these terms to the known error messages
        self.bem.update(
            {
                "scale": "a number",
                "reservoir": "Reservoir handle",
                "ref_reservoirs": "List of Reservoir handle(s)",
                "ref_value": "a number or flux quantity",
                "name": "a string value",
                "flux": "a flux handle",
                "left": "list, reservoir or number",
                "right": "list, reservoir or number",
            }
        )

        # initialize keyword values
        self.__validateandregister__(kwargs)

        # xxx if self.register != "None":
        #    self.name = f"{self.register}.{self.name}"

        self.__postinit__()  # do some housekeeping
        # legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()

        # decide which call function to use
        # if self.mo.m_type == "both":
        if self.reservoir.isotopes:
            self.__execute__ = self.__with_isotopes__
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function
    def __call__(self, reservoir: Reservoir, i: int):
        return self.__execute__(reservoir, i)


# class ScaleRelativeToNormalizedConcentration(RateConstant):
#     """This process scales the flux as a function of the upstream
#      reservoir concentration C and a constant which describes the
#      strength of relation between the reservoir concentration and
#      the flux scaling

#      F = (C/C0 -1) * k

#      where C denotes the concentration in the ustream reservoir, C0
#      denotes the baseline concentration and k is a constant
#      This process is typically called by the connector
#      instance. However you can instantiate it manually as


#      ScaleRelativeToNormalizedConcentration(
#                        name = "Name",
#                        reservoir= upstream_reservoir_handle,
#                        flux = flux handle,
#                        Scale =  1000,
#                        ref_value = 2 # reference_concentration
#     )

#     """

#     def __call__(self, reservoir: Reservoir, i: int) -> None:
#         """
#         this will be called by the Model.run() method
#         """

#         scale: float = (reservoir.c[i - 1] / self.ref_value - 1) * self.scale
#         # scale = scale * (scale >= 0)  # prevent negative fluxes.
#         self.f[i] = self.f[i] + self.f[i] * array([scale, scale, scale, 1])

# from numba import typed, typeof, types
# from numba.experimental import jitclass
# import numba

# lmo = typed.List()
# ldo = typed.List()
# reg_time = types.float64

# spec = [
#     ("reservoir.m", numba.float64[:]),
#     ("i", numba.int32),
#     ("reservoir.volume", numba.float64),
#     ("scale", numba.float64),
#     ("flux.m", numba.float64[:]),
#     ("reservoir.species.element.r", numba.float64),
#     ("reservoir.d", numba.float64[:]),
# ]


# @jitclass(spec)
class ScaleRelativeToConcentration(RateConstant):
    """This process calculates the flux as a function of the upstream
     reservoir concentration C and a constant which describes thet
     strength of relation between the reservoir concentration and
     the flux scaling

     F = C * k

     where C denotes the concentration in the ustream reservoir, k is a
     constant. This process is typically called by the connector
     instance. However you can instantiate it manually as

     ScaleRelativeToConcentration(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       Scale =  1000,
    )

    """

    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        m: float = self.reservoir.m[i - 1]
        if m > 0:  # otherwise there is no flux
            # convert to concentration
            c = m / self.reservoir.volume
            m = c * self.scale
            self.flux.m[i] = m

    def __with_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """
        C = M/V so we express this as relative to mass which allows us to
        use the isotope data.

        The below calculates the flux as function of reservoir concentration,
        rather than scaling the flux.
        """

        c: float = self.reservoir.c[i - 1]
        if c > 0:  # otherwise there is no flux
            m = c * self.scale
            r: float = reservoir.species.element.r
            d: float = reservoir.d[i - 1]
            l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
            self.flux[i]: np.array = [m, l, m - l, d]

    def get_process_args(self, reservoir: Reservoir):

        func_name: function = self.p_scale_relative_to_concentration

        res_data = List([reservoir.m, reservoir.d, reservoir.c])
        flux_data = List([self.flux.m, self.flux.l, self.flux.h, self.flux.d])
        proc_const = List([float(reservoir.species.element.r), float(self.scale)])

        return func_name, res_data, flux_data, proc_const

    @staticmethod
    @njit()
    def p_scale_relative_to_concentration(res_data, flux_data, proc_const, i) -> None:
        # concentration times scale factor
        m: float = res_data[2][i - 1] * proc_const[1]
        d: float = res_data[1][i - 1]  # delta
        l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)
        flux_data[0][i] = m
        flux_data[1][i] = l
        flux_data[2][i] = m - l
        flux_data[3][i] = d


class ScaleRelativeToMass(RateConstant):
    """This process scales the flux as a function of the upstream
     reservoir Mass M and a constant which describes the
     strength of relation between the reservoir mass and
     the flux scaling

     F = F0 *  M * k

     where M denotes the mass in the ustream reservoir, k is a
     constant and F0 is the initial unscaled flux. This process is
     typically called by the connector instance. However you can
     instantiate it manually as

     Note that we scale the flux, rather than compute the flux!

     This is faster than setting a new flux, computing the isotope
     ratio and setting delta. So you either have to set the initial
     flux F0 to 1, or calculate the scale accordingly

     ScaleRelativeToMass(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       Scale =  1000,
    )

    """

    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        m: float = self.reservoir.m[i - 1] * self.scale
        self.flux.m[i]: float = m

    def __with_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """
        this will be called by the Model.run() method

        """
        m: float = self.reservoir.m[i - 1] * self.scale
        r: float = reservoir.species.element.r
        d: float = reservoir.d[i - 1]
        l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
        self.flux[i]: np.array = [m, l, m - l, d]

    def get_process_args(self, reservoir: Reservoir):
        """return the data associated with this object"""

        func_name: function = self.p_scale_relative_to_mass

        res_data = List([self.reservoir.m, reservoir.d, reservoir.c])
        flux_data = List([self.flux.m, self.flux.l, self.flux.h, self.flux.d])
        proc_const = List([float(reservoir.species.element.r), float(self.scale)])

        return func_name, res_data, flux_data, proc_const

    @staticmethod
    @njit()
    def p_scale_relative_to_mass(res_data, flux_data, proc_const, i) -> None:
        # concentration times scale factor
        m: float = res_data[0][i - 1] * proc_const[1]
        d: float = res_data[1][i - 1]  # delta
        l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)
        flux_data[0][i] = m
        flux_data[1][i] = l
        flux_data[2][i] = m - l
        flux_data[3][i] = d

    # def p_scale_relative_to_mass(res_data, flux_data, proc_const, i) -> tuple:

    #     m: float = res_data[0][i - 1] * proc_const[1]
    #     d: float = res_data[1][i - 1]
    #     l: float = (1000.0 * m) / ((d + 1000.0) * proc_const[0] + 1000.0)
    #     return [m, l, m - l, d]


# class ScaleRelativeToNormalizedMass(RateConstant):
#     """This process scales the flux as a function of the upstream
#      reservoir mass M and a constant which describes the
#      strength of relation between the reservoir concentration and
#      the flux scaling

#      F = (M/M0 -1) * k

#      where M denotes the mass in the ustream reservoir, M0
#      denotes the reference mass, and k is a constant
#      This process is typically called by the connector
#      instance. However you can instantiate it manually as


#      ScaleRelativeToNormalizedConcentration(
#                        name = "Name",
#                        reservoir= upstream_reservoir_handle,
#                        flux = flux handle,
#                        Scale =  1,
#                        ref_value = 1e5 # reference_mass
#     )

#     """

#     def __call__(self, reservoir: Reservoir, i: int) -> None:
#         """
#         this will be called by the Model.run() method
#         """
#         scale: float = (reservoir.m[i - 1] / self.ref_value - 1) * self.scale
#         # scale = scale * (scale >= 0)  # prevent negative fluxes.
#         self.f[i] = self.f[i] + self.f[i] * array([scale, scale, scale, 1])


class ScaleRelative2otherReservoir(RateConstant):
    """This process scales the flux as a function one or more reservoirs
    constant which describes the
    strength of relation between the reservoir concentration and
    the flux scaling

    F = C1 * C1 * k

    where Mi denotes the concentration in one  or more reservoirs, k is one
    or more constant(s). This process is typically called by the connector
    instance when you specify the connection as

    Connect(source =  upstream reservoir,
              sink = downstream reservoir,
              ctype = "scale_relative_to_multiple_reservoirs"
              ref_reservoirs = [r1, r2, k etc] # you must provide at least one
                                               # reservoir or constant
              scale = a overall scaling factor
           )
    """

    def __misc_init__(self) -> None:
        """Test that self.reservoir only contains numbers and reservoirs"""

        self.rs: list = []
        self.constant: Number = 1

        for r in self.ref_reservoirs:
            if isinstance(r, (Reservoir)):
                self.rs.append(r)
            elif isinstance(r, (Number)):
                self.constant = self.constant * r
            else:
                raise ValueError(f"{r} must be reservoir or number, not {type(r)}")

    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        c: float = 1
        for r in self.rs:
            c = c * r.c[i - 1]

        scale: float = c * self.scale * self.constant

        # scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] = [scale, scale, scale, 1]

    def __with_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """
        not sure that this correct WRT isotopes

        """

        raise NotImplementedError(
            "Scale relative to multiple reservoirs is undefined for isotope calculations"
        )

        # c: float = 1
        # for r in self.rs:
        #     c = c * r.c[i - 1]

        # scale: float = c * self.scale * self.constant

        # # scale = scale * (scale >= 0)  # prevent negative fluxes.
        # self.f[i] = [scale, scale, scale, 1]


class Flux_Balance(RateConstant):
    """This process calculates a flux between two reservoirs as a function
    of multiple reservoir concentrations and constants.

    Note that could result in negative fluxes. which might cause
    issues with isotope ratios (untested)

    This will work with equilibrium reactions between two reservoirs where the
    reaction can be described as

    K * [R1] = R[2] * [R3]

    you can have more than two terms on each side as long as they are
    constants or reservoirs

    Equilibrium(
                name = "Name",
                reservoir = reservoir handle,
                left = [] # list with reservoir names or constants
                right = [] # list with reservoir names or constants
                flux = flux handle,
                k_value = a constant, defaults to 1
    )

    """

    # redefine misc_init which is being called by post-init
    def __misc_init__(self):
        """Sort out input variables"""

        Rl: List[Reservoir] = []
        Rr: List[Reservoir] = []
        Cl: List[float] = []
        Cr: List[float] = []
        # parse the left hand side

        em = "left/right values must be constants or reservoirs"
        [self.Rl, self.Cl] = sort_by_type(self.left, [Reservoir, Number], em)
        [self.Rr, self.Cr] = sort_by_type(self.right, [Reservoir, Number], em)

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
        this will be called by the Model.run() method

        """

        kl: NDArray = np.array([1.0, 1.0, 1.0, 1.0])
        kr: NDArray = np.array([1.0, 1.0, 1.0, 1.0])
        scale: NDArray = np.array([1.0, 1.0, 1.0, 1.0])

        # calculate the product of reservoir concentrations for left side
        for r in self.Rl:
            kl *= r[i - 1]
        # multiply with any any constants on the right
        for c in self.Cl:
            kl *= c

        # calculate the product of reservoir concentrations for right side
        for r in self.Rr:
            kr *= r[i - 1]
        # multiply with any any constants on the right
        for c in self.Cr:
            kr *= c

        # set flux
        self.f[i] = (kl - kr) * self.k_value


class Monod(Process):
    """This process scales the flux as a function of the upstream
    reservoir concentration using a Michaelis Menten type
    relationship

    F = F * a * F0 x C/(b+C)

    where F0 denotes the unscaled flux (i.e., at t=0), C denotes
    the concentration in the ustream reservoir, and a and b are
    constants.

    Example::
         Monod(name = "Name",
               reservoir =  upstream_reservoir_handle,
               flux = flux handle ,
               ref_value = reference concentration
               a_value = constant,
               b_value = constant )

    """

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """"""

        from . import ureg, Q_

        """ Initialize this Process """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names

        # update the allowed keywords
        self.lkk: dict = {
            "a_value": Number,
            "b_value": Number,
            "ref_value": (Number, str, Q_),
            "name": str,
            "reservoir": (Reservoir, Source, Sink),
            "flux": Flux,
            "register": (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
        }

        self.lrk.extend(
            ["reservoir", "a_value", "b_value", "ref_value"]
        )  # new required keywords

        self.__initerrormessages__()
        self.bem.update(
            {
                "a_value": "a number",
                "b_value": "a number",
                "reservoir": "Reservoir handle",
                "ref_value": "a number",
                "name": "a string value",
                "flux": "a flux handle",
            }
        )

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping
        # legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
        this willbe called by Model.execute apply_processes
        """

        scale: float = (
            self.a_value
            * (self.ref_value * reservoir.c[i - 1])
            / (self.b_value + reservoir.c[i - 1])
        )

        scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] + self.f[i] * scale

    def __plot__(
        self, start: int, stop: int, ref_reservoirs: float, a: float, b: float
    ) -> None:
        """Test the implementation"""

        y = []
        x = range(start, stop)

        for e in x:
            y.append(a * ref_reservoirs * e / (b + e))

        fig, ax = plt.subplots()  #
        ax.plot(x, y)
        # Create a scatter plot for ax
        plt.show()
