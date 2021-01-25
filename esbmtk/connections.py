"""

     esbmtk.connections
     ~~~~~~~~~~~~~~~~~~

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
from .processes import *
from .esbmtk import *

class Connect(esbmtkBase):
    """Two reservoirs connect to each other via at least one flux. This
    module creates the connecting flux and creates a connector object
    which stores all connection properties.

    For simple connections, the type flux type is derived implcitly from the specified parameters.
    For complex connections, the flux type must be set explicitly. See the examples below:

    Parameters:
        - name: A string which determines the name of this object. Optional, if not provided
          the connection name will be derived as "Source_2_Sink_Connection"
        - source: An object handle for a Source or Reservoir
        - sink: An object handle for a Sink or Reservoir
        - rate: A quantity (e.g., "1 mol/s"), optional
        - delta: The isotope ratio, optional
        - alpha: A fractionation factor, optional
        - id: A string wich will become part of the object name, optional
        - plot: "yes" or "no", defaults to "yes"
        - signal: An object handle of signal, optional
        - pl: A list of process objects, optional
        - ctype: connection type, optional, this allows to scale a flux in response to other
          reservoirs and fluxes

    Connection Types:
    -----------------
    Basic Connections (the advanced ones are below):

    - If both =rate= and =delta= are given, the flux is treated as a
       fixed flux with a given isotope ratio. This is usually the case for
       most source objects (they can still be affected by a signal, see
       above), but makes little sense for reservoirs and sinks.

    - If both the =rate= and =alpha= are given, the flux rate is fixed
      (subject to any signals), but the isotopic ratio of the output
      flux depends on the isotopic ratio of the upstream reservoir
      plus any isotopic fractionation specified by =alpha=. This is
      typically the case for fluxes which include an isotopic
      fractionation (i.e., pyrite burial). This combination is not
      particularly useful for source objects.

    - If the connection specifies only =delta= the flux is treated as a
      variable flux which is computed in such a way that the reservoir
      maintains steady state with respect to it's mass.

    - If the connection specifies only =rate= the flux is treated as a
      fixed flux which is computed in such a way that the reservoir
      maintains steady state with respect to it's isotope ratio.

    Examples of Basic Connections
    -----------------------------

    Connecting a Source to a Reservoir
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Unless you use a Signal, a source typically provides a steady stream with a given isotope ratio (if used)
    
    Example::

       Connect(source =  Source,
               sink = downstrean reservoir,
               rate = "1 mol/s",
               delta = optional,
               signal = optional, see the signal documentation)
    
    Connecting a Reservoir to Sink or another Reservoir
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Here we can distinguish between cases where we use fixed flux, or a flux which reacts to in some way to the 
    upstream reservoir (see the Reservoir to Reservoir section for a more complete treatment):

    Fixed outflux, with no isotope fractionation

    Example::
    
         Connect(source =  upstream reservoir,
               sink = Sink,
               rate = "1 mol/s",)

    Fixed outflux, with isotope fractionation

    Example::
    
         Connect(source =  upstream reservoir,
               sink = Sink,
               alpha = -28,
               rate = "1 mol/s",)

    Advanced Connections
    --------------------

    You can aditionally define connection properties via the ctype
    keyword. This requires additional keyword parameters. The following values are
    recognized

    ctype = "scale_with_flux"
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This will scale a flux relative to another flux:

    Example::
     
        Connect(source =  upstream reservoir,
               sink = downstream reservoir,
               ctype = "scale_with_flux",
               ref = flux handle, 
               k_value = a scaling factor)

    ctype = "scale_with_mass" and "scale_with_concentration"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    This will scale a flux relative to the mass or concentration of a reservoir

    Example::
     
        Connect(source =  upstream reservoir,
               sink = downstream reservoir,
               ctype = "scale_with_mass",
               ref = reservoir handle, 
               k_value = a scaling factor)

    ctype = "scale_relative_to_multiple_reservoirs"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    This process scales the flux as a function one or more reservoirs
    or constants which describes the
    strength of relation between the reservoir concentrations and
    the flux scaling

    F = C1 * C2 * k

    where Ci denotes the concentrartion in one  or more reservoirs, k is one
    or more constants. 

    Example:: 

        Connect(source =  upstream reservoir,
               sink = downstream reservoir,
               ctype = "scale_relative_to_multiple_reservoirs"
               ref_reservoirs = [r1, r2, k etc] # you must provide at least one
               k_value = a scaling factor)
    
    K_value is an overall scaling factor.


    ctype = "flux_balance"
    ~~~~~~~~~~~~~~~~~~~~~~

   This type can be used to express equilibration fluxes 
   between two reservoirs. This connection type, takes three parameters:

   - =left= is a list which can contain constants and/or reservoirs. The
     list must contain at least one valid element. All elements in this
     list will be multiplied with each other. E.g. if we have a list
     with one constant and one reservoir, the reservoir concentration
     will be multiplied with the constant. If we have two reservoirs,
     the respective reservoir concentrations will be multiplied with
     each other.
   - =right= similar to =left= The final flux rate will be computed as
     the difference between =left= and =right=
   - =k_value= a constant which will be multiplied with the difference
     between =left=and =right=

    Example::

        Connect(source=R_CO2,         # target of flux
                sink=R_HCO3,          # source of flux
                rate="1 mol/s",       # flux rate
                ctype="flux_balance", # connection type 
                k_value=1,            # global scaling factor
                left=[K1, R_CO2],     # where K1 is a constant
                right=[R_HCO3, R_Hplus])

    
    Useful methods in this class
    ----------------------------
    The following methods might prove useful
    
     - describe() will provide a short description of the connection objects.
     - list_processes() which will list all the processes which are associated with this connection.
     - update() which allows you to update connection properties after the connection has been created

    """
    
    def __init__(self, **kwargs):
        """ The init method of the connector obbjects performs sanity checks e.g.:
               - whether the reservoirs exist
               - correct flux properties (this will be handled by the process object)
               - whether the processes do exist (hmmh, that implies that the optional processes do get registered with the model)
               - creates the correct default processes
               - and connects the reservoirs

        see the class documentation for details and examples

        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name":
            str,
            "id":
            str,
            "source": (Source, Reservoir),
            "sink": (Sink, Reservoir),
            "delta": (Number, str),
            "rate": (str, Number, Q_),
            "pl":
            list,
            "alpha": (Number, str),
            "species":
            Species,
            "ctype":
            str,
            "ref": (Flux, list),
            "react_with":
            Flux,
            "ratio":
            Number,
            "scale":
            Number,
            "ref_value": (str, Number, Q_),
            "ref_reservoir": (list, Reservoir),
            "k_value": (Number, str, Q_),
            "a_value":
            Number,
            "b_value":
            Number,
            "left": (list, Number, Reservoir),
            "right": (list, Number, Reservoir),
            "plot":
            str,
            "register":
            (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
            "signal": (Signal, str),
        }

        if "name" not in kwargs:
            n = kwargs["source"].name + "_2_" + kwargs[
                "sink"].n + "_Connector"  # set the name
            kwargs.update({"name": n})  # and add it to the kwargs

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "source", "sink"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "id": "None",
            "plot": "yes",
            "ctype": "None",
            "delta": "None",
            "alpha": "None",
            "rate": "None",
            "k_value": 1,
            "signal": "None",
        }

        # validate and initialize instance variables
        self.__initerrormessages__()

        self.bem.update({
            "k_concentration": "a number",
            "k_mass": "a number",
            "k_value": "a number",
            "a_value": "a number",
            "ref_value": "a number, string, or quantity",
            "b_value": "a number",
            "name": "a string",
            "id": "a string",
            "plot": "a string",
            "left": "Number, list or Reservoir",
            "right": "Number, list or Reservoir",
            "signal": "Signal Handle",
        })

        self.drn = {
            "alpha": "_alpha",
            "rate": "_rate",
            "delta": "_delta",
        }

        self.__validateandregister__(kwargs)

        if kwargs["id"] != "None":
            self.name = self.name + f"_{self.id}"
        if 'pl' in kwargs:
            self.lop: list[Process] = self.pl
        else:
            self.lop: list[Process] = []

        if self.signal != "None":
            self.lop.append(self.signal)

        # if no reference reservoir is specified, default to the upstream
        # reservoir
        if 'ref_reservoir' not in kwargs:
            self.ref_reservoir = kwargs["source"]

        # legacy names
        self.influx: int = 1
        self.outflux: int = -1
        self.n = self.name
        self.mo = self.source.sp.mo
        self.p = 0  # the default process handle
        self.r1: (Process, Reservoir) = self.source
        self.r2: (Process, Reservoir) = self.sink

        self.get_species(self.r1, self.r2)  #
        self.mo: Model = self.sp.mo  # the current model handle
        self.lof: list[Flux] = []  # list of fluxes in this connection
        # get a list of all reservoirs registered for this species
        self.lor: list[Reservoir] = self.mo.lor

        self.source.loc.add(self)  # register connector with reservoir
        self.sink.loc.add(self)  # register connector with reservoir
        self.mo.loc.add(self)  # register connector with model

        self.__create_flux__()  # Source/Sink/Regular

        self.__set_process_type__()  # derive flux type and create flux(es)

        self.__register_name__()  # register connection in namespace

        # This should probably move to register fluxes
        self.__register_process__()

        if self.register == "yes":
            print(f"Created connection {self.name}")
        else:
            print(f"Created group connection {self.register.name}.{self.name}")

    def update(self, **kwargs):
        """Update connection properties. This will delete existing processes
        and fluxes, replace existing key-value pairs in the
        self.kwargs dict, and then re-initialize the connection.

        """
        self.__delete_process__()
        self.__delete_flux__()
        self.kwargs.update(kwargs)
        self.__init_connection__(self.kwargs)
        print(f"Updated {self.n}")

    def get_species(self, r1, r2) -> None:
        """In most cases the species is set by r2. However, if we have
        backward fluxes the species depends on the r2

        """
        #print(f"r1 = {r1.n}, r2 = {r2.n}")
        if isinstance(self.r1, Source):
            self.r = r1
        else:  # in this case we do have an upstream reservoir
            self.r = r2

        # test if species was explicitly given
        if "species" in self.kwargs:  # this is a quick fix only
            self.sp = self.kwargs["species"]
        else:
            self.sp = self.r.sp  # get the parent species

    def __create_flux__(self) -> None:
        """Create flux object, and register with reservoir and global namespace

        """

        # test if default arguments present
        if self.delta == "None":
            d = 0
        else:
            d = self.delta

        if self.rate == "None":
            r = f"1 {self.sp.mo.f_unit}"
            #self._rate = r
        else:
            r = self.rate

        # flux name
        if self.id == "None":
            n = self.r1.n + '_2_' + self.r2.n + "_Flux"
        else:
            n = self.r1.n + '_2_' + self.r2.n + "_" + \
                self.id + "_Flux"  # flux name r1_2_r2

        # derive flux unit from species obbject
        funit = self.sp.mu + "/" + str(self.sp.mo.bu)  # xxx

        self.fh = Flux(
            name=n,  # flux name
            species=self.sp,  # Species handle
            delta=d,  # delta value of flux
            rate=r,  # flux value
            plot=self.plot,  # display this flux?
            register=self.register,  # is this part of a group?
        )

        # register flux with its reservoirs
        if isinstance(self.r1, Source):
            # add the flux name direction/pair
            self.r2.lio[self.fh] = self.influx
            # add the handle to the list of fluxes
            self.r2.lof.append(self.fh)
            # register flux and element in the reservoir.
            self.__register_species__(self.r2, self.r1.sp)

        elif isinstance(self.r2, Sink):
            # add the flux name direction/pair
            self.r1.lio[self.fh] = self.outflux
            # add flux to the upstream reservoir
            self.r1.lof.append(self.fh)
            # register flux and element in the reservoir.
            self.__register_species__(self.r1, self.r2.sp)

        elif isinstance(self.r1, Sink):
            raise NameError(
                "The Sink must be specified as a destination (i.e., as second argument"
            )

        elif isinstance(self.r2, Source):
            raise NameError("The Source must be specified as first argument")

        else:  # this is a regular connection
            # add the flux name direction/pair
            self.r1.lio[self.fh] = self.outflux
            # add the flux name direction/pair
            self.r2.lio[self.fh] = self.influx
            # add flux to the upstream reservoir
            self.r1.lof.append(self.fh)
            # add flux to the downstream reservoir
            self.r2.lof.append(self.fh)
            self.__register_species__(self.r1, self.r1.sp)
            self.__register_species__(self.r2, self.r2.sp)

        self.lof.append(self.fh)

    def __register_species__(self, r, sp) -> None:
        """ Add flux to the correct element dictionary"""
        # test if element key is present in reservoir
        if sp.eh in r.doe:
            # add flux handle to dictionary list
            r.doe[sp.eh].append(self.fh)
        else:  # add key and first list value
            r.doe[sp.eh] = [self.fh]

    def __register_process__(self) -> None:
        """ Register all flux related processes"""

        # first test if we have a signal in the list. If so,
        # remove signal and replace with process

        p_copy = copy(self.lop)
        for p in p_copy:  # loop over process list if provided during init
            if isinstance(p, Signal):
                self.lop.remove(p)
                if p.ty == "addition":
                    # create AddSignal Process object
                    n = AddSignal(name=p.n + "_addition_process",
                                  reservoir=self.r,
                                  flux=self.fh,
                                  lt=p.data)
                    self.lop.append(n)
                else:
                    raise ValueError(f"Signal type {p.ty} is not defined")

        # nwo we can register everythig on lop
        for p in self.lop:
            p.__register__(self.r, self.fh)

    def __set_process_type__(self) -> None:
        """ Deduce flux type based on the provided flux properties. The method calls the
        appropriate method init routine
        """

        if isinstance(self.r1, Source):
            self.r = self.r2
        else:
            self.r = self.r1

        # set process name
        #if len(self.kwargs["id"]) > 0:
        if self.id == "None":
            self.pn = self.r1.n + "_2_" + self.r2.n
        else:
            self.pn = self.r1.n + "_2_" + self.r2.n + f"_{self.id}"

        # if connection type is not set explicitly
        if self.ctype == "None":
            # set the fundamental flux type based on the flux arguments given
            if self.delta != "None" and self.rate != "None":
                pass  # do nothing but exit this conditional
            # variable flux with fixed delta
            elif self.delta != "None":  # rate must None
                self.__passivefluxfixeddelta__()
            elif self.rate != "None":  # delta must be None
                self.__vardeltaout__()  # variable delta with fixed flux
            else:  # if neither are given -> default varflux type
                self._delta = 0
                self.__passiveflux__()
                print("passive flux")

        elif self.ctype == "flux_diff":
            self.__vardeltaout__()
            self.__flux_diff__()
        elif self.ctype == "scale_with_flux":
            self.__vardeltaout__()
            self.__scaleflux__()
        elif self.ctype == "copy_flux":
            self.__vardeltaout__()
            self.__scaleflux__()
        elif self.ctype == "scale_with_mass":
            self.__rateconstant__()
        elif self.ctype == "scale_with_concentration":
            self.__rateconstant__()
        elif self.ctype == "scale_with_concentration_normalized":
            self.__rateconstant__()
        elif self.ctype == "scale_with_mass_normalized":
            self.__rateconstant__()
        elif self.ctype == "scale_relative_to_multiple_reservoirs":
            self.__rateconstant__()
        elif self.ctype == "flux_balance":
            self.__rateconstant__()
        elif self.ctype == "monod_type_limit":
            self.__vardeltaout__()
            self.__rateconstant__()
        else:
            print(f"Connection Type {self.type} is unknown")
            raise ValueError(f"Unknown connection type {self.ctype}")

        # Set optional flux processes
        if self.alpha != "None":
            self.__alpha__()  # Set optional flux processes

    def __passivefluxfixeddelta__(self) -> None:
        """ Just a wrapper to keep the if statement manageable

        """

        ph = PassiveFlux_fixed_delta(
            name=self.pn + "_Pfd",
            reservoir=self.r,
            flux=self.fh,
            register=self.register,
            delta=self.delta)  # initialize a passive flux process object
        self.lop.append(ph)

    def __vardeltaout__(self) -> None:
        """Unlike a passive flux, this process sets the output flux from a
        reservoir to a fixed value, but the isotopic ratio of the
        output flux will be set equal to the isotopic ratio of the
        upstream reservoir.

        """

        ph = VarDeltaOut(name=self.pn + "_Pvdo",
                         reservoir=self.r,
                         flux=self.fh,
                         register=self.register,
                         rate=self.rate)
        self.lop.append(ph)

    def __scaleflux__(self) -> None:
        """ Scale a flux relative to another flux

        """

        if not isinstance(self.kwargs["ref"], Flux):
            raise ValueError("Scale reference must be a flux")

        ph = ScaleFlux(name=self.pn + "_PSF",
                       reservoir=self.r,
                       flux=self.fh,
                       register=self.register,
                       scale=self.kwargs["k_value"],
                       ref=self.kwargs["ref"])
        self.lop.append(ph)

    def __flux_diff__(self) -> None:
        """ Scale a flux relative to the difference between
        two fluxes

        """

        if not isinstance(self.kwargs["ref"], list):
            raise ValueError("ref must be a list")

        ph = FluxDiff(name=self.pn + "_PSF",
                      reservoir=self.r,
                      flux=self.fh,
                      register=self.register,
                      scale=self.kwargs["k_value"],
                      ref=self.kwargs["ref"])
        self.lop.append(ph)

    def __reaction__(self) -> None:
        """ Just a wrapper to keep the if statement manageable

        """

        if not isinstance(self.kwargs["react_with"], Flux):
            raise ValueError("Scale reference must be a flux")
        ph = Reaction(name=self.pn + "_RF",
                      reservoir=self.r,
                      flux=self.fh,
                      register=self.register,
                      scale=self.kwargs["ratio"],
                      ref=self.kwargs["react_with"])
        # we need to make sure to remove the flux referenced by
        # react_with is removed from the list of fluxes in this
        # reservoir.
        self.r2.lof.remove(self.kwargs["react_with"])
        self.lop.append(ph)

    def __passiveflux__(self) -> None:
        """ Just a wrapper to keep the if statement manageable

        """

        ph = PassiveFlux(
            name=self.pn + "_PF",
            reservoir=self.r,
            register=self.register,
            flux=self.fh)  # initialize a passive flux process object
        self.lop.append(ph)  # add this process to the process list

    def __alpha__(self) -> None:
        """ Just a wrapper to keep the if statement manageable

        """

        ph = Fractionation(name=self.pn + "_Pa",
                           reservoir=self.r,
                           flux=self.fh,
                           register=self.register,
                           alpha=self.kwargs["alpha"])
        self.lop.append(ph)  #

    def __rateconstant__(self) -> None:
        """ Add rate constant type process

        """

        from . import ureg, Q_

        # this process requires that we use the vardeltaout process
        if self.mo.m_type != "mass_only":
            self.__vardeltaout__()

        if self.ctype == "scale_with_mass":
            self.k_value = map_units(self.k_value, self.mo.m_unit)
            ph = ScaleRelativeToMass(name=self.pn + "_PkM",
                                     reservoir=self.ref_reservoir,
                                     flux=self.fh,
                                     register=self.register,
                                     k_value=self.k_value)

        elif self.ctype == "scale_with_mass_normalized":
            self.k_value = map_units(self.k_value, self.mo.m_unit)
            self.ref_value = map_units(self.ref_value, self.mo.m_unit)
            ph = ScaleRelativeToNormalizedMass(name=self.pn + "_PknM",
                                               reservoir=self.ref_reservoir,
                                               flux=self.fh,
                                               register=self.register,
                                               ref_value=self.ref_value,
                                               k_value=self.k_value)

        elif self.ctype == "scale_with_concentration":
            self.k_value = map_units(self.k_value, self.mo.c_unit,
                                     self.mo.f_unit, self.mo.r_unit)
            ph = ScaleRelativeToConcentration(name=self.pn + "_PkC",
                                              reservoir=self.ref_reservoir,
                                              flux=self.fh,
                                              register=self.register,
                                              k_value=self.k_value)

        elif self.ctype == "scale_relative_to_multiple_reservoirs":
            self.k_value = map_units(self.k_value, self.mo.c_unit,
                                     self.mo.f_unit, self.mo.r_unit)
            ph = ScaleRelative2otherReservoir(name=self.pn + "_PkC",
                                              reservoir=self.source,
                                              ref_reservoir=self.ref_reservoir,
                                              flux=self.fh,
                                              register=self.register,
                                              k_value=self.k_value)

        elif self.ctype == "flux_balance":
            self.k_value = map_units(self.k_value, self.mo.c_unit,
                                     self.mo.f_unit, self.mo.r_unit)
            ph = Flux_Balance(name=self.pn + "_Pfb",
                              reservoir=self.source,
                              left=self.left,
                              right=self.right,
                              flux=self.fh,
                              register=self.register,
                              k_value=self.k_value)

        elif self.ctype == "scale_with_concentration_normalized":
            self.k_value = map_units(self.k_value, self.mo.c_unit,
                                     self.mo.f_unit, self.mo.r_unit)
            self.ref_value = map_units(self.ref_value, self.mo.c_unit)
            ph = ScaleRelativeToNormalizedConcentration(
                name=self.pn + "_PknC",
                reservoir=self.ref_reservoir,
                flux=self.fh,
                register=self.register,
                ref_value=self.ref_value,
                k_value=self.k_value)

        elif self.ctype == "monod_ctype_limit":
            self.ref_value = map_units(self.ref_value, self.mo.c_unit)
            ph = Monod(name=self.pn + "_PMonod",
                       reservoir=self.ref_reservoir,
                       flux=self.fh,
                       register=self.register,
                       ref_value=self.ref_value,
                       a_value=self.a_value,
                       b_value=self.b_value)

        else:
            raise ValueError(
                f"This should not happen,and points to a keywords problem in {self.name}"
            )

        self.lop.append(ph)

    def describe(self, **kwargs) -> None:
        """ Show an overview of the object properties.
        Optional arguments are
        index  :int = 0 this will show data at the given index
        indent :int = 0 indentation

        """
        off: str = "  "
        if "index" not in kwargs:
            index = 0
        else:
            index = kwargs["index"]

        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = ' ' * indent

        # print basic data bout this Connection
        print(f"{ind}{self.__str__(indent=indent)}")

        print(f"{ind}Fluxes:")
        for f in sorted(self.lof):
            f.describe(indent=indent, index=index)

    def __delete_process__(self) -> None:
        """ Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old processes, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lop = copy(self.lop)
        for p in lop:
            self.r1.lop.remove(p)
            self.fh.lop.remove(p)
            self.lop.remove(p)
            self.r1.mo.lmo.remove(p.n)
            del p

    def __delete_flux__(self) -> None:
        """ Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old flux, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lof = copy(self.lof)
        for f in lof:
            self.r1.lof.remove(f)
            self.lof.remove(f)
            self.r1.mo.lmo.remove(f.n)
            del f

    # ---- Property definitions to allow for connection updates --------
    """ Changing the below properties requires that we delete all
    associated objects (processes), and determines the new flux type,
    and initialize/register these with the connection and model.
    We also have to update the keyword arguments as these are used
    for the log entry

    """

    # ---- alpha ----
    @property
    def alpha(self) -> Number:
        return self._alpha

    @alpha.setter
    def alpha(self, a: Number) -> None:
        self.__delete_process__()
        self.__delete_flux__()
        self._alpha = a
        self.kwargs["alpha"] = a
        self.__set_process_type__()  # derive flux type and create flux(es)
        self.__register_process__()

    # ---- rate  ----
    @property
    def rate(self) -> Number:
        return self._rate

    @rate.setter
    def rate(self, r: str) -> None:
        from . import ureg, Q_
        self.__delete_process__()
        self.__delete_flux__()
        self._rate = Q_(r).to(self.mo.f_unit)
        self.kwargs["rate"] = r
        self.__create_flux__()  # Source/Sink/Regular
        self.__set_process_type__()  # derive flux type and create flux(es)
        self.__register_process__()

    # ---- delta  ----
    @property
    def delta(self) -> Number:
        return self._delta

    @delta.setter
    def delta(self, d: Number) -> None:
        self.__delete_process__()
        self.__delete_flux__()
        self._delta = d
        self.kwargs["delta"] = d
        self.__create_flux__()  # Source/Sink/Regular
        self.__set_process_type__()  # derive flux type and create flux(es)
        self.__register_process__()

class Connection(Connect):
    """ Alias for the Connect class

    """

class ConnectionGroup(esbmtkBase):
    """Name:

        ConnectionGroup

        Connect reservoir/sink/source groups when at least one of the
        arguments is a reservoirs_group object. This method will
        create regular connections for each matching species. 

        Use the connection.update() method to fine tune connections 
        after creation

    Example::

        ConnectionGroup(source =  upstream reservoir / upstream reservoir group
           sink = downstrean reservoir / downstream reservoirs_group
           delta = defaults to zero and has to be set manually
           alpha =  defaults to zero and has to be set manually
           rate = shared between all connections
           ref = shared between all connections
           species = list, optional, if present, only these species will be connected
           ctype = if set it will be shared between all connections. To
           pl = [list]) process list. optional, shared between all connections
           id = optional identifier, shared between all connections
           plot = "yes/no" # defaults to yes, shared between all connections
        )

        Notes: if species is given as a list, shared arguments like, delta, alpha, rate, ref,
        ctype pl, and plot can also be provided as list. As long as there is a one to one mapping
        the species list and the list of a shared property, the shared property will be mapped
        to each species, e.g.:

        species = [CO, Hplus]
        alpha = [1.02, 1.03]

        will create two connections, the first one with an alpha of 1.02, and the second with an alpha of 1.03

    """
    def __init__(self, **kwargs) -> None:

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "id": dict,
            "name": str,
            "source": (SourceGroup, ReservoirGroup),
            "sink": (SinkGroup, ReservoirGroup),
            "delta": dict,
            "rate": dict,
            "pl": dict,
            "signal": Signal,
            "alpha": dict,
            "species": dict,
            "ctype": str,
            "ref": list,
            "plot": dict,
        }

        self.base_name = kwargs["source"].name + "_2_" + kwargs["sink"].name

        n = kwargs["source"].name + "_2_" + kwargs[
            "sink"].name + "_ConnectionGroup"  # set the name

        # set connection group name
        kwargs.update({"name": n})  # and add it to the kwargs

        # provide a list of absolutely required keywords
        self.lrk: list = ["source", "sink"]

        # get the number of sub reservoirs in the source and sink
        nor_sink = len(kwargs["sink"].species)
        nor_source = len(kwargs["source"].species)

        #if nor_source != nor_sink:
        #    raise ValueError(
        #        "Number of sub reservoirs does not match. Specify match explicitly"
        #    )

        cid: dict = {}
        plot: dict = {}
        delta: dict = {}
        alpha: dict = {}
        rate: dict = {}
        # loop over names and create dicts
        for n in kwargs['sink'].species:
            cid[n] = 'None'
            plot[n] = 'no'
            delta[n] = 'None'
            alpha[n] = 'None'
            rate[n] = 'None'

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "id": cid,
            "plot": plot,
            "delta": delta,
            "alpha": alpha,
            "rate": rate
        }

        # turn kwargs into instance variables
        self.__validateandregister__(kwargs)

        self.loc: list = []  # list of connections in this group

        # self.source.lor is a  list with the object names in the group
        self.mo = self.sink.lor[0].mo

        # loop over sub-reservoirs and create connections
        for i, r in enumerate(self.source.lor):
            if not isinstance(r, (Reservoir, Source, Sink)):
                raise ValueError(
                    f"{r} must be of type reservoir, source or sink")
                # take the species of this sub reservoir
                # in the source, and find matching
                # species in the sink

            # loop over sink list until a match is found
            for j, s in enumerate(self.sink.lor):
                if not isinstance(s, (Reservoir, Source, Sink)):
                    raise ValueError(
                        f"{r} must be of type reservoir, source or sink")

                if r.species == s.species:  # match found
                    # name = f"{self.source.name}_{r.species.name}_2_{self.sink.name}_{s.species.name}"
                    name = f"{r.species.name}_2_{s.species.name}_Connector"
                    a = Connect(
                        name=name,
                        source=r,
                        sink=s,
                        rate=self.rate[s.species],
                        delta=self.delta[s.species],
                        alpha=self.alpha[s.species],
                        plot=self.plot[s.species],
                        id=self.id[s.species],
                        register=self,
                    )
                elif j == nor_sink:  # no match was found
                    raise ValueError("{r.species} has no match")

            # register connection with connection group
            #this should happen automatically
            #setattr(self, a.name, a)
            self.loc.append(a)

        # register connection group in global namespace
        self.__register_name__()

    def describe(self) -> None:
        """ List all connections in this group
        
        """

        print(f"Group Connection from {self.source.name} to {self.sink.name}\n")
        print("    The following Connections are part of this group\n")

        for c in self.loc:
            print(c.name)

        print("")

        print(f"        You can query the details of each connection like this:\n")
        print(f"          {self.name}.{self.loc[0].name}.describe()")
