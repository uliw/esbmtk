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

class Process(esbmtkBase):
    """This class defines template for process which acts on one or more
     reservoir flux combinations. To use it, you need to create an
     subclass which defines the actual process implementation in their
     call method. See 'PassiveFlux as example'
    
    """
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
        """ Do some housekeeping for the process class
        
        """

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.r: Reservoir = self.reservoir
        self.f: Flux = self.flux
        self.m: Model = self.r.sp.mo  # the model handle
        self.mo: Model = self.m

        # Create a list of fluxes wich excludes the flux this process
        # will be acting upon
        self.fws: List[Flux] = self.r.lof.copy()
        self.fws.remove(self.f)  # remove this handle

        self.rm0: float = self.r.m[0]  # the initial reservoir mass
        self.direction: Dict[Flux, int] = self.r.lio[self.f]

        self.__misc_init__()

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
        
        # provide a dict of known keywords and types
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "rate": Number,
            "delta": Number,
            "lt": Flux,
            "alpha": Number,
            "scale": Number,
            "ref": (Flux, list),
            'register': (str,ConnectionGroup),
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # default type hints
        self.scale: t
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
        """ Apply the current process to the vector x, and show the result as y.
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
     - a1 to a6, up to 6 optional function arguments
    
    in order to use this function we need first declare a function we plan to
    use with the generic function process. This function needs to follow this
    template::

        def my_func(i, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0) -> tuple:
            # 
            # i = index of the current timestep
            # a1 to a2 =  optional function parameter. These must be present, 
            # even if your function will not use it
            
            # calc some stuff and return it as

            return [m, l, h] # where m= mass, and l & h are the respective 
                             # isotopes. If there are none, dummmy values
                             # instead

    
    This function can then be used as::
    
        GenericFunction(name="foo",
                function=my_func,
                a1 = some argument,
                a2 = some argument,
                act_on = reservoir name)
    
    """

    __slots__ = ('function', 'act_on', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'i')

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
          Create a new process object with a given process type and options
          """

        self.__defaultnames__()  # default kwargs names
        self.lkk: Dict[str, any] = {
            "name": str,
            "act_on": (Flux, Reservoir),
            "function": any,
            "a1": any,
            "a2": any,
            "a3": any,
            "a4": any,
            "a5": any,
            "a6": any,
        }

        # required arguments
        self.lrk: list = (["name", "act_on", "function"])

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            'a1': 0,
            'a2': 0,
            'a3': 0,
            'a4': 0,
            'a5': 0,
            'a6': 0,
        }

        self.__initerrormessages__()  # default error messages
        self.bem.update({
            "act_on": "a reservoir or flux",
            "function": "a function",
            "a1": "a number etc",
            "a2": "a number etc",
            "a3": "a number etc",
            "a4": "a number etc",
            "a5": "a number etc",
            "a6": "a number etc",
        })
        self.__validateandregister__(kwargs)  # initialize keyword values

        if not callable(self.function):
            raise ValueError(
                "function must be defined before it can be used here")

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
            raise ValueError(
                "functions can only act upon reservoirs or fluxes")

    def __call__(self, i: int) -> None:
        """Here we execute the user supplied function and assign the 
        return value to the flux or reservoir

        Where i = index of the current timestep
              acting_on = reservoir or flux we are acting on.

        """

        self.act_on[i] = self.function(
            i,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.a5,
            self.a6,
        )

    # redefine post init
    def __postinit__(self) -> None:
        """ Do some housekeeping for the process class

          """

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.m: Model = self.act_on.sp.mo  # the model handle
        self.mo: Model = self.m

        self.__misc_init__()

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
          self.m[i] :float  = self.lt.m[i]
          self.d[i] :float  = self.lt.d[i]
          self.l[i] :float = self.lt.l[i]
          self.h[i] :float = self.lt.h[i]

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
        #self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values

        #legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

        # decide whichh call function to use
        if self.mo.m_type == "both":
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
        # add signal delta to flux delta
        self.f.d[i] = self.f.d[i] + self.lt.d[i]

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
        #self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values
        #legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. That is, we calculate the sum of all fluxes
          excluding this flux. This sum will be equl to this flux. This will likely only
          work for outfluxes though.

        Should this be done for output fluxes as well?
          
          """

        new: float = 0
        for j, f in enumerate(self.fws):
            new += f.m[i] * reservoir.lio[f]

        # self.f[i] = get_flux_data(new,reservoir.d[i-1],reservoir.rvalue)

        m = new
        r = reservoir.l[i - 1] / reservoir.m[i - 1]
        l = m * r
        h = m - l
        self.f.m[i] = m
        self.f.l[i] = l
        self.f.h[i] = h
        self.f.d[i] = reservoir.d[i - 1]

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

     def __init__(self, **kwargs :Dict[str, any]) -> None:
          """ Initialize this Process """


          self.__defaultnames__()  # default kwargs names
          self.lrk.extend(["reservoir","delta", "flux"]) # new required keywords

          self.__initerrormessages__()
          #self.bem.update({"rate": "a string"})
          self.__validateandregister__(kwargs)  # initialize keyword values
          self.__postinit__()  # do some housekeeping

          # legacy names
          self.f :Flux = self.flux
          #legacy variables
          self.mo = self.reservoir.mo

          print("\nn *** Warning, you selected the PassiveFlux_fixed_delta method ***\n ")
          print(" This is not a particularly phyiscal process is this really what you want?\n")
          print(self.__doc__)
          self.__register_name__()
     
     def __call__(self, reservoir :Reservoir, i :int) -> None:
          """Here we re-balance the flux. This code will be called by the
          apply_flux_modifier method of a reservoir which itself is
          called by the model execute method

          """

          r :float = reservoir.rvalue # the isotope reference value

          varflux :Flux = self.f 
          flux_list :List[Flux] = reservoir.lof.copy()
          flux_list.remove(varflux)  # remove this handle

          # sum up the remaining fluxes
          newflux :float = 0
          for f in flux_list:
               newflux = newflux + f.m[i-1] * reservoir.lio[f]

          # set isotope mass according to keyword value
          self.f[i] = array(get_flux_data(newflux, self.delta, r))

class VarDeltaOut(Process):
    """Unlike a passive flux, this process sets the output flux from a
     reservoir to a fixed value, but the isotopic ratio of the output
     flux will be set equal to the isotopic ratio of the reservoir The
     init and register methods are inherited from the process
     class. The overall result can be scaled, i.e., in order to create
     a split flow etc.  Example::

     VarDeltaOut(name = "name",
                 reservoir = upstream_reservoir_handle,
                 flux = flux handle,
                 rate = rate,)

     """

    __slots__ = ('rate')

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process
          
          """

        from . import ureg, Q_
        from .connections import ConnectionGroup

        # get default names and update list for this Process
        self.__defaultnames__()
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "rate": (str, Q_),
            "register": (ConnectionGroup, str),
        }
        self.lrk.extend(["reservoir", "rate"])  # new required keywords
        self.__initerrormessages__()
        self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values

        # parse rate term, and map to legacy name
        self.rateq = Q_(self.rate)
        self.rate = Q_(self.rate).to(self.reservoir.mo.f_unit).magnitude
        #legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Here we re-balance the flux. This code will be called by the
          apply_flux_modifier method of a reservoir which itself is
          called by the model execute method

        """

        # set flux according to keyword value

        # this explicit expression is siginificantly faster than the below
        # function call
        m = self.rate
        r = reservoir.l[i - 1] / reservoir.m[i - 1]
        l = m * r
        h = m - l
        self.f.m[i] = m
        self.f.l[i] = l
        self.f.h[i] = h
        self.f.d[i] = reservoir.d[i-1]
       
        # self.f[i] = get_flux_data(self.rate,reservoir.d[i-1], reservoir.rvalue)

class ScaleFlux(Process):
    """This process scales the mass of a flux (m,l,h) relative to another
     flux but does not affect delta. The scale factor "scale" and flux
     reference must be present when the object is being initalized

     Example::
          ScaleFlux(name = "Name",
                    reservoir = upstream_reservoir_handle,
                    scale = 1
                    ref = flux we use for scale)

    """

    __slots__ = ('rate', 'scale')
    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process 

        """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "scale",
                         "ref"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values

        #legacy variables
        self.mo = self.reservoir.mo
        self.__postinit__()  # do some housekeeping
        self.__register_name__()

        

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
          model execute method.
          Note that this will use the mass of the reference object, but that we will set the 
          delta according to the reservoir (or the flux?)

          """
        self.f[i] = self.ref[i] * self.scale
        self.f[i] = get_flux_data(self.f.m[i], reservoir.d[i - 1],
                                  reservoir.rvalue)


class FluxDiff(Process):
    """ The new flux will be the difference of two fluxes

    """
    """This process scales the mass of a flux (m,l,h) relative to another
     flux but does not affect delta. The scale factor "scale" and flux
     reference must be present when the object is being initalized

     Example::
          ScaleFlux(name = "Name",
                    reservoir = upstream_reservoir_handle,
                    scale = 1
                    ref = flux we use for scale)

     """
    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process 

        """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux", "scale",
                         "ref"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        #legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()


    
    def __call__(self, reservoir :Reservoir, i: int) -> None:
        """Apply the scale factor. This is typically done through the the
        model execute method.
        Note that this will use the mass of the reference object, but that we will set the 
        delta according to the reservoir (or the flux?)

        """

        self.f[i] = (self.ref[0][i] - self.ref[1][i]) * self.scale

class Fractionation(Process):
    """This process offsets the isotopic ratio of the flux by a given
        delta value. In other words, we add a fractionation factor

     Example::
          Fractionation(name = "Name",
                        reservoir = upstream_reservoir_handle,
                        flux = flux handle
                        alpha = 12 in permil (e.f)

     """
    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        self.lrk.extend(["reservoir", "flux",
                         "alpha"])  # new required keywords

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping

        # alpha is given in permil, but the fractionation routine expects
        # it as 1 + permil, i.e., 70 permil would 1.007
        #legacy variables
        self.alpha = 1 + self.alpha / 1000
        self.mo = self.reservoir.mo
        self.__register_name__()

        # decide which call function to use
        if self.mo.m_type == "both":
            self.__execute__ = self.__with_isotopes__
        else:
            self.__execute__ = self.__without_isotopes__

    # setup a placeholder call function        
    def __call__(self, reservoir: Reservoir, i: int):
        return self.__execute__(reservoir, i)

    # use this when we do isotopes
    def __with_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """
        Set flux isotope masses based on fractionation factor

        """

        self.f.l[i], self.f.h[i] = get_frac(self.f.m[i], self.f.l[i],
                                            self.alpha)

        #update delta
        self.f.d[i] = get_delta(self.f.l[i], self.f.h[i], self.f.rvalue)
        return

    # use this when we don't do isotopes
    def __without_isotopes__(self, reservoir: Reservoir, i: int) -> None:
        """
        Set flux isotope masses based on fractionation factor

        """

        return

class RateConstant(Process):
    """This is a wrapper for a variety of processes which depend on rate constants
    Please see the below class definitions for details on how to call them
    At present, the following processes are defined

    ScaleRelativeToNormalizedConcentration
    ScaleRelativeToConcentration

    """
    __slots__ = ('k_value', 'ref_value')
    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this Process

        """

        from . import ureg, Q_

        # Note that self.lkk values also need to be added to the lkk
        # list of the connector object.

        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names

        # update the allowed keywords
        self.lkk :dict = {
            "k_value": Number,
            "ref_value": Number,
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "ref_reservoir": list,
            "left": (list, Reservoir, Number),
            "right": (list, Reservoir, Number),
            "register":
            (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
        }

        # new required keywords
        self.lrk.extend(["reservoir", "k_value"])

        # dict with default values if none provided
        # self.lod = {r

        self.__initerrormessages__()

        # add these terms to the known error messages
        self.bem.update({
            "k_value": "a number",
            "reservoir": "Reservoir handle",
            "ref_reservoirs": "List of Reservoir handle(s)",
            "ref_value": "a number or flux quantity",
            "name": "a string value",
            "flux": "a flux handle",
            "left": "list, reservoir or number",
            "right": "list, reservoir or number",
        })

        # initialize keyword values
        self.__validateandregister__(kwargs)
        self.__postinit__()  # do some housekeeping
        # legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()


class ScaleRelativeToNormalizedConcentration(RateConstant):
    """This process scales the flux as a function of the upstream
     reservoir concentration C and a constant which describes the
     strength of relation between the reservoir concentration and
     the flux scaling

     F = (C/C0 -1) * k

     where C denotes the concentration in the ustream reservoir, C0
     denotes the baseline concentration and k is a constant
     This process is typically called by the connector
     instance. However you can instantiate it manually as


     ScaleRelativeToNormalizedConcentration(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       K_value =  1000,
                       ref_value = 2 # reference_concentration
    )

    """

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this will be called by the Model.run() method
          """

        scale: float = (reservoir.c[i - 1] / self.ref_value - 1) * self.k_value
        # scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] = self.f[i] + self.f[i] * array([scale, scale, scale, 1])


class ScaleRelativeToConcentration(RateConstant):
    """This process scales the flux as a function of the upstream
     reservoir concentration C and a constant which describes the
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
                       K_value =  1000,
    )

    """

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this will be called by the Model.run() method
          """
        #print(f"k= {self.k_value}")
        scale: float = reservoir.c[i - 1] * self.k_value
        # scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] = self.f[i] * array([scale, scale, scale, 1])


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
     flux F0 to 1, or calculate the k_value accordingly

     ScaleRelativeToMass(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       K_value =  1000,
    )

    """

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this will be called by the Model.run() method
          """
        scale: float = reservoir.m[i - 1] * self.k_value
        self.f[i] = self.f[i] * array([scale, scale, scale, 1])


class ScaleRelativeToNormalizedMass(RateConstant):
    """This process scales the flux as a function of the upstream
     reservoir mass M and a constant which describes the
     strength of relation between the reservoir concentration and
     the flux scaling

     F = (M/M0 -1) * k

     where M denotes the mass in the ustream reservoir, M0
     denotes the reference mass, and k is a constant
     This process is typically called by the connector
     instance. However you can instantiate it manually as


     ScaleRelativeToNormalizedConcentration(
                       name = "Name",
                       reservoir= upstream_reservoir_handle,
                       flux = flux handle,
                       K_value =  1,
                       ref_value = 1e5 # reference_mass
    )

    """

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this will be called by the Model.run() method
          """
        scale: float = (reservoir.m[i - 1] / self.ref_value - 1) * self.k_value
        # scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] = self.f[i] + self.f[i] * array([scale, scale, scale, 1])


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
               k_value = a overall scaling factor
            )
    """

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
        this will be called by the Model.run() method

        """

        c: float = 1
        for r in self.ref_reservoir:
            c = c * r.c[i - 1]

        scale: float = c * self.k_value
        # scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] = self.f[i] * array([scale, scale, scale, 1])

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
        """ Sort out input variables

        """

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

        kl: NDArray    = np.array([1.0, 1.0, 1.0, 1.0])
        kr: NDArray    = np.array([1.0, 1.0, 1.0, 1.0])
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
        self.f[i] = (kl - kr) *  self.k_value

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
        """

        """

        from . import ureg, Q_

        """ Initialize this Process """
        # get default names and update list for this Process
        self.__defaultnames__()  # default kwargs names
        
        # update the allowed keywords
        self.lkk :dict = {
            "a_value": Number,
            "b_value": Number,
            "ref_value": (Number,str, Q_),
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "register":
            (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
        }

        self.lrk.extend(["reservoir", "a_value", "b_value",
                         "ref_value"])  # new required keywords

        self.__initerrormessages__()
        self.bem.update({
            "a_value": "a number",
            "b_value": "a number",
            "reservoir": "Reservoir handle",
            "ref_value": "a number",
            "name": "a string value",
            "flux": "a flux handle",
        })

        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()  # do some housekeeping
        #legacy variables
        self.mo = self.reservoir.mo
        self.__register_name__()

    def __call__(self, reservoir: Reservoir, i: int) -> None:
        """
          this willbe called by Model.execute apply_processes
          """

        scale: float = self.a_value * (self.ref_value * reservoir.c[i - 1]) / (
            self.b_value + reservoir.c[i - 1])

        scale = scale * (scale >= 0)  # prevent negative fluxes.
        self.f[i] + self.f[i] * scale

    def __plot__(self, start: int, stop: int, ref: float, a: float,
                 b: float) -> None:
        """ Test the implementation

          """

        y = []
        x = range(start, stop)

        for e in x:
            y.append(a * ref * e / (b + e))

        fig, ax = plt.subplots()  #
        ax.plot(x, y)
        # Create a scatter plot for ax
        plt.show()
