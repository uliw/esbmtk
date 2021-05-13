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
from .esbmtk import *


class Connect(esbmtkBase):
    """Two reservoirs connect to each other via at least one flux. This
     module creates the connecting flux and creates a connector object
     which stores all connection properties.

     For simple connections, the type flux type is derived implcitly from the specified parameters.
     For complex connections, the flux type must be set explicitly. See the examples below:

     Parameters:
         - name: A string which determines the name of this object. Optional, if not provided
           the connection name will be derived as "C_Source_2_Sink"
         - source: An object handle for a Source or Reservoir
         - sink: An object handle for a Sink or Reservoir
         - rate: A quantity (e.g., "1 mol/s"), optional
         - delta: The isotope ratio, optional
         - ref_reservoirs: Reservoir or flux reference
         - alpha: A fractionation factor, optional
         - id: A string wich will become part of the object name, optional
         - plot: "yes" or "no", defaults to "yes"
         - signal: An object handle of signal, optional
         - pl: A list of process objects, optional
         - ctype: connection type, optional, this allows to scale a flux in response to other
           reservoirs and fluxes
         - bypass :str optional defaults to "None" see scale with flux

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
                ref_reservoirs = flux handle,
                scale = a scaling factor, optional, defaults to 1
                bypass :str = "source"/"sink"
                )

    if bypass is set the flux will not affect the upstream or
    downstream reservoir. This is a more generalized approach than the
    virtual flux type. This is useful to describe reactions which
    split into two elements (i.e., deprotonation), or combine fluxes
    from two reservoirs (say S, and O) into a reservoir which records
    SO4.

    ctype = "virtual_flux"
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This will scale a flux relative to another flux, similar to
    scaleflux. However, in this case, the flux will only affect the
    sink, and not the source. This is useful to describe reactions which
    split into two elements (i.e., deprotonation).

    Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_flux",
                ref_reservoirs = flux handle,
                scale = a scaling factor, optional, defaults to 1
                )

     ctype = "scale_with_mass" and "scale_with_concentration"
     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     This will scale a flux relative to the mass or concentration of a reservoir

     Example::

         Connect(source =  upstream reservoir,
                sink = downstream reservoir,
                ctype = "scale_with_mass",
                ref_reservoirs = reservoir handle,
                scale = a scaling factor, optional, defaults to 1
    )

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
                ref_reservoirs_reservoirs = [r1, r2, k etc] # you must provide at least one
                scale = a scaling factor, optional, defaults to 1
    )

     scale is an overall scaling factor.


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
                 scale=1,            # global scaling factor, optional
                 left=[K1, R_CO2],     # where K1 is a constant
                 right=[R_HCO3, R_Hplus])


     Useful methods in this class
     ----------------------------
     The following methods might prove useful

      - info() will provide a short description of the connection objects.
      - list_processes() which will list all the processes which are associated with this connection.
      - update() which allows you to update connection properties after the connection has been created

    """

    def __init__(self, **kwargs):
        """The init method of the connector obbjects performs sanity checks e.g.:
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
            "name": str,
            "id": str,
            "source": (Source, Reservoir),
            "sink": (Sink, Reservoir),
            "delta": (Number, str),
            "rate": (str, Number, Q_),
            "pl": list,
            "alpha": (Number, str),
            "species": Species,
            "ctype": str,
            "ref_reservoirs": (Flux, Reservoir, str, list),
            "ratio": Number,
            "scale": (Number, Q_, str),
            "ref_value": (str, Number, Q_),
            "k_value": (Number, str, Q_),
            "a_value": Number,
            "b_value": Number,
            "left": (list, Number, Reservoir),
            "right": (list, Number, Reservoir),
            "plot": str,
            "groupname": bool,
            "register": (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
            "signal": (Signal, str),
            "bypass": str,
            "isotopes": bool,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["source", "sink"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "id": "",
            "plot": "yes",
            "ctype": "None",
            "delta": "None",
            "alpha": "None",
            "rate": "None",
            "k_value": "None",
            "scale": 1,
            "signal": "None",
            "groupname": False,
            "bypass": "None",
            "name": "None",
            "isotopes": False,
            "ref_reservoirs": "None",
        }

        # validate and initialize instance variables
        self.__initerrormessages__()

        self.bem.update(
            {
                "k_concentration": "a number",
                "k_mass": "a number",
                "k_value": "a number",
                "scale": "a number or Quantity",
                "a_value": "a number",
                "ref_value": "a number, string, or quantity",
                "b_value": "a number",
                "name": "a string",
                "id": "a string",
                "plot": "a string",
                "left": "Number, list or Reservoir",
                "right": "Number, list or Reservoir",
                "signal": "Signal Handle",
                "groupname": "True or False",
                "bypass": "source/sink",
            }
        )

        self.drn = {
            "alpha": "_alpha",
            "rate": "_rate",
            "delta": "_delta",
        }

        self.__validateandregister__(kwargs)

        # if kwargs["id"] != "None":
        #    self.name = self.name + f"_{self.id}"
        if "pl" in kwargs:
            self.lop: list[Process] = self.pl
        else:
            self.lop: list[Process] = []

        if self.signal != "None":
            self.lop.append(self.signal)

        # if no reference reservoir is specified, default to the upstream
        # reservoir
        if self.ref_reservoirs == "None":
            self.ref_reservoirs = kwargs["source"]

        # decide if this connection needs isotope calculations
        if self.source.isotopes or self.sink.isotopes:
            self.isotopes = True

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

        # make sure scale is a number in model units

        if self.scale == "None":
            self.scale = 1.0

        if isinstance(self.scale, str):
            self.scale = Q_(self.scale)

        if isinstance(self.scale, Q_):
            # test what type of Quantity we have
            if self.scale.check(["volume]/[time"]):  # flux
                self.scale = self.scale.to(self.mo.r_unit)
            elif self.scale.check(["mass] / [time"]):  # flux
                self.scale = self.scale.to(self.mo.f_unit)
            elif self.scale.check("[mass]/[volume]"):  # concentration
                self.scale = self.scale.to(self.mo.c_unit)
            else:
                ValueError(f"No conversion to model units for {self.scale} specified")

        # if sink and source a regular, the name will be simply C_S_2_S
        # if we deal with ReservoirGroups we need to reflect this in the
        # connection name

        if self.name == "None":
            self.name = f"{self.source.name}_2_{self.sink.name}"
            # this yields something like PO42PO4
        # if we are part of connection group we need to add this
        # to the connectiongroup

        if self.id == "None" or self.id == "":
            pass
        else:
            self.name = f"{self.name}_{self.id}"

        if self.register == "None":
            self.full_name = self.name
        else:
            self.full_name = f"{self.register.name}.{self.name}"

        self.__create_flux__()  # Source/Sink/Regular

        self.__set_process_type__()  # derive flux type and create flux(es)

        self.__register_name__()  # register connection in namespace

        self.source.loc.add(self)  # register connector with reservoir
        self.sink.loc.add(self)  # register connector with reservoir
        self.mo.loc.add(self)  # register connector with model

        # This should probably move to register fluxes
        self.__register_process__()

        # if self.register == "None":
        #     print(f"Created connection {self.name}")
        # else:
        #     print(f"Created connection {self.register.name}.{self.name}")

        logging.info(f"Created {self.full_name}")

    def update(self, **kwargs):
        """Update connection properties. This will delete existing processes
        and fluxes, replace existing key-value pairs in the
        self.kwargs dict, and then re-initialize the connection.

        """
        self.__delete_process__()
        self.__delete_flux__()
        self.kwargs.update(kwargs)
        self.__init_connection__(self.kwargs)

    def get_species(self, r1, r2) -> None:
        """In most cases the species is set by r2. However, if we have
        backward fluxes the species depends on the r2

        """
        # print(f"r1 = {r1.n}, r2 = {r2.n}")
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
        """Create flux object, and register with reservoir and global namespace"""

        # test if default arguments present
        if self.delta == "None":
            d = 0
        else:
            d = self.delta

        if self.rate == "None":
            r = f"0 {self.sp.mo.f_unit}"
            # self._rate = r
        else:
            r = self.rate

        # flux name
        # if self.groupname == False:
        if self.id == "None" or self.id == "":
            n = f"{self.r1.n}_2_{self.r2.n}_F"
        else:
            n = f"{self.r1.n}_2_{self.r2.n}_{self.id}_F"
        # else:
        #    n = "F_" + self.r1.full_name + "_2_" + self.r2.full_name

        # derive flux unit from species obbject
        funit = self.sp.mu + "/" + str(self.sp.mo.bu)  # xxx

        self.fh = Flux(
            name=n,  # flux name
            species=self.sp,  # Species handle
            delta=d,  # delta value of flux
            rate=r,  # flux value
            plot=self.plot,  # display this flux?
            register=self.register,  # is this part of a group?
            isotopes=self.isotopes,
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
                    n = AddSignal(
                        name=p.n + "_addition_process",
                        reservoir=self.r,
                        flux=self.fh,
                        lt=p.data,
                    )
                    self.lop.insert(0, n)  # signals must come first
                    logging.debug(f"Inserting {n.n} in {self.name} for {self.r.n}")
                else:
                    raise ValueError(f"Signal type {p.ty} is not defined")

        # ensure that vardeltaout is first in list
        self.__move_process_to_top_of_queue__(self.lop, VarDeltaOut)
        # nwo we can register everythig on lop
        for p in self.lop:
            p.__register__(self.r, self.fh)

    def __move_process_to_top_of_queue__(self, lop: list, ptype: any) -> None:
        """Return a copy of lop where ptype has been moved to the top of lop"""
        p_copy = copy(lop)
        for p in p_copy:  # loop over process list if provided during init
            if isinstance(p, ptype):
                lop.remove(p)
                lop.insert(0, p)  # insert at top

    def __set_process_type__(self) -> None:
        """Deduce flux type based on the provided flux properties. The method calls the
        appropriate method init routine
        """

        if isinstance(self.r1, Source):
            self.r = self.r2
        else:
            self.r = self.r1

        # if signal is provided but rate is omitted
        if self.signal != "None" and self.rate == "None":
            self._rate = "0 mmol/y"

        # if connection type is not set explicitly
        if self.ctype == "None" or self.ctype == "Regular":
            # set the fundamental flux type based on the flux arguments given
            if self.delta != "None" and self.rate != "None":
                # self.__vardeltaout__()
                pass  # if delta and rate are specified, do nothing
            # variable flux with fixed delta
            elif self.delta != "None":  # if delta is set
                self.__passivefluxfixeddelta__()
            elif self.rate != "None":  # if rate is set
                self.__vardeltaout__()  # variable delta with fixed flux
            else:  # if neither are given -> default varflux type
                self._delta = 0
                self.__passiveflux__()

        elif self.ctype == "flux_diff":
            self.__vardeltaout__()
            self.__flux_diff__()
        elif self.ctype == "scale_with_flux":
            self.__scaleflux__()
        elif self.ctype == "virtual_flux":
            self.__virtual_flux__()
            # self.__vardeltaout__()
        elif self.ctype == "copy_flux":
            self.__scaleflux__()
            # self.__vardeltaout__()
        elif self.ctype == "scale_with_mass":
            self.__rateconstant__()
        elif self.ctype == "scale_with_concentration":
            self.__rateconstant__()
            # self.__vardeltaout__()
        # elif self.ctype == "scale_with_concentration_normalized":
        #    self.__rateconstant__()
        # elif self.ctype == "scale_with_mass_normalized":
        #     self.__rateconstant__()
        elif self.ctype == "scale_relative_to_multiple_reservoirs":
            self.__rateconstant__()
        elif self.ctype == "flux_balance":
            self.__rateconstant__()
        elif self.ctype == "react_with":
            self.__reaction__()
        elif self.ctype == "monod_type_limit":
            # self.__vardeltaout__()
            self.__rateconstant__()
        elif self.ctype == "manual":
            pass
        else:
            print(f"Connection Type {self.ctype} is unknown")
            raise ValueError(f"Unknown connection type {self.ctype}")

        # Set optional flux processes
        if self.alpha != "None":
            self.__alpha__()  # Set optional flux processes
            # self.__vardeltaout__()

    def __passivefluxfixeddelta__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = PassiveFlux_fixed_delta(
            name="Pfd",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            delta=self.delta,
        )  # initialize a passive flux process object
        self.lop.append(ph)

    def __vardeltaout__(self) -> None:
        """Unlike a passive flux, this process sets the output flux from a
        reservoir to a fixed value, but the isotopic ratio of the
        output flux will be set equal to the isotopic ratio of the
        upstream reservoir.

        """

        ph = VarDeltaOut(
            name="Pvdo",
            reservoir=self.source,
            flux=self.fh,
            register=self.fh,
            rate=self.rate,
        )
        self.lop.append(ph)

    def __scaleflux__(self) -> None:
        """Scale a flux relative to another flux"""

        if not isinstance(self.ref_reservoirs, Flux):
            raise ValueError("Scale reference must be a flux")

        if self.k_value != "None":
            self.scale = self.k_value
            print(f"\n Warning: use scale instead of k_value for scaleflux type\n")

        # if self.bypass == "source":
        #     target = self.sink
        # elif self.bypass == "sinks":
        #     target = self.source
        # elif self.bypass == "None":
        #     target = self.r
        # else:
        #     raise ValueError(f"bypass must be None/source/sink but not {self.bypass}")

        ph = ScaleFlux(
            name="PSF",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )
        self.lop.append(ph)

        # this flux must not affect the source reservoir
        # self.r.lof.remove(self.fh)

    def __virtual_flux__(self) -> None:
        """Create a virtual flux. This is similar to __scaleflux__, however the new flux
        will only affect the sink, and not the source.

        """

        if self.k_value != "None":
            self.scale = self.k_value
            print(
                f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
            )

        if not isinstance(self.kwargs["ref_reservoirs"], Flux):
            raise ValueError("Scale reference must be a flux")

        ph = ScaleFlux(
            name="PSFV",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )
        self.lop.append(ph)

        # this flux must not affect the source reservoir
        self.r.lof.remove(self.fh)

    def __flux_diff__(self) -> None:
        """Scale a flux relative to the difference between
        two fluxes

        """

        if self.k_value != "None":
            self.scale = self.k_value
            print(
                f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
            )

        if not isinstance(self.kwargs["ref_reservoirs"], list):
            raise ValueError("ref must be a list")

        ph = FluxDiff(
            name="PSF",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )
        self.lop.append(ph)

    def __reaction__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        if not isinstance(self.ref_reservoirs, (Flux)):
            raise ValueError("Scale reference must be a flux")

        if self.k_value != "None":
            self.scale = self.k_value
            print(f"\n Warning: use scale instead of k_value for reaction type\n")

        ph = Reaction(
            name="RF",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            scale=self.scale,
            ref_reservoirs=self.ref_reservoirs,
        )

        # we need to make sure to remove the flux referenced by
        # react_with is removed from the list of fluxes in this
        # reservoir. This should probably move in to the React Class
        self.r2.lof.remove(self.ref_reservoirs)
        self.lop.append(ph)

    def __passiveflux__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = PassiveFlux(
            name="_PF", reservoir=self.r, register=self.fh, flux=self.fh
        )  # initialize a passive flux process object
        self.lop.append(ph)  # add this process to the process list

    def __alpha__(self) -> None:
        """Just a wrapper to keep the if statement manageable"""

        ph = Fractionation(
            name="_Pa",
            reservoir=self.r,
            flux=self.fh,
            register=self.fh,
            alpha=self.kwargs["alpha"],
        )
        self.lop.append(ph)  #

    def __rateconstant__(self) -> None:
        """Add rate constant type process"""

        from . import ureg, Q_

        # this process requires that we use the vardeltaout process
        if self.mo.m_type != "mass_only":
            # self.__vardeltaout__()
            pass

        if self.ctype == "scale_with_mass":
            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale with mass type\n"
                )

            self.scale = map_units(self.scale, self.mo.m_unit)
            ph = ScaleRelativeToMass(
                name="_PkM",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )

        elif self.ctype == "scale_with_concentration":

            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale with concentration type\n"
                )

            self.scale = map_units(
                self.scale, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            # print(
            #    f"Registering PKC with ref = {self.ref.full_name}, scale = {self.scale}"
            # )

            ph = ScaleRelativeToConcentration(
                name="PkC",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )
            # print(f"Process Name {ph.full_name}")

        elif self.ctype == "scale_relative_to_multiple_reservoirs":

            if self.k_value != "None":
                self.scale = self.k_value
                print(
                    f"\n Warning: use scale instead of k_value for scale relative to multiple reservoirs\n"
                )

            self.scale = map_units(
                self.scale, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            ph = ScaleRelative2otherReservoir(
                name="PkC",
                reservoir=self.source,
                ref_reservoirs=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                scale=self.scale,
            )

        elif self.ctype == "flux_balance":
            self.k_value = map_units(
                self.k_value, self.mo.c_unit, self.mo.f_unit, self.mo.r_unit
            )
            ph = Flux_Balance(
                name="_Pfb",
                reservoir=self.source,
                left=self.left,
                right=self.right,
                flux=self.fh,
                register=self.fh,
                k_value=self.k_value,
            )

        elif self.ctype == "monod_ctype_limit":
            self.ref_value = map_units(self.ref_value, self.mo.c_unit)
            ph = Monod(
                name="_PMonod",
                reservoir=self.ref_reservoirs,
                flux=self.fh,
                register=self.fh,
                ref_value=self.ref_value,
                a_value=self.a_value,
                b_value=self.b_value,
            )

        else:
            raise ValueError(
                f"This should not happen,and points to a keywords problem in {self.name}"
            )

        self.lop.append(ph)

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.
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
            ind = " " * indent

        # print basic data bout this Connection
        print(f"{ind}{self.__str__(indent=indent)}")

        print(f"{ind}Fluxes:")
        for f in sorted(self.lof):
            f.info(indent=indent, index=index)

    def __delete_process__(self) -> None:
        """Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old processes, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lop = copy(self.lop)

        for p in lop:
            for f in self.lof:
                if isinstance(f.register, ConnectionGroup):
                    # remove from Connection group list of model objects
                    self.register.lmo.remove(f)
                else:
                    self.r1.lop.remove(p)
                    self.fh.lop.remove(p)
                    self.lop.remove(p)
                    self.r1.mo.lmo.remove(p.n)
                    del p

    def __delete_flux__(self) -> None:
        """Updates to the connection properties may change the connection type and thus
        the processes which are associated with this connection. We thus have to
        first delete the old flux, before we re-initialize the connection

        """

        # identify which processes we need to delete
        # unregister process from connection.lop, reservoir.lop, flux.lop, model.lmo
        # delete process from global name space if present

        lof = copy(self.lof)
        for f in lof:
            if isinstance(f.register, ConnectionGroup):
                # remove from Connection group list of model objects
                self.register.lmo.remove(f)
            else:
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
    """Alias for the Connect class"""


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
           ref_reservoirs = shared between all connections
           species = list, optional, if present, only these species will be connected
           ctype = needs to be set for all connections. Use "Regular"
                   unless you require a specific connection type
           pl = [list]) process list. optional, shared between all connections
           id = optional identifier, passed on to individual connection
           plot = "yes/no" # defaults to yes, shared between all connections
        )

        ConnectionGroup(
                  source=OM_Weathering,
                  sink=Ocean,
                  rate={DIC: f"{OM_w} Tmol/yr" ,
                        ALK: f"{0} Tmol/yr"},
                  ctype = {DIC: "Regular",
                           ALK: "Regular"},
    )


    """

    def __init__(self, **kwargs) -> None:

        self.__parse_kwargs__(kwargs)

        # # self.source.lor is a  list with the object names in the group
        self.mo = self.sink.lor[0].mo
        self.loc: list = []  # list of connection objects

        self.__create_connections__()

        # register connection group in global namespace
        self.__register_name__()
        logging.info(f"Created {self.name}")

    def update(self, **kwargs) -> None:
        """Add a connection to the connection group
        This will overwrite the original kwargs though

        """

        self.__parse_kwargs__(kwargs)
        self.__create_connections__()

    def __parse_kwargs__(self, kwargs) -> None:
        """
        Parse and register the keyword arguments

        """
        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "id": str,
            "name": str,
            "source": (SourceGroup, ReservoirGroup),
            "sink": (SinkGroup, ReservoirGroup),
            "delta": dict,
            "rate": dict,
            "pl": dict,
            "signal": Signal,
            "alpha": dict,
            "species": dict,
            "ctype": dict,
            "ref_reservoirs": dict,
            "plot": dict,
            "scale": dict,
        }

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            "name": "None",
            "id": "",
        }

        if "name" in kwargs:
            self.base_name = kwargs["name"]
        else:
            name = f"{kwargs['source'].name}2{kwargs['sink'].name}"
            self.base_name = kwargs["source"].name + "2" + kwargs["sink"].name
            n = f"C_{name}"
            # set connection group name
            kwargs.update({"name": n})  # and add it to the kwargs

        # provide a list of absolutely required keywords
        self.lrk: list = ["source", "sink"]
        self.__validateandregister__(kwargs)

    def __create_connections__(self) -> None:
        """Create Connections"""
        # find all sub reservoirs which have been specified by the ctype keyword
        self.connections: list = []
        for r in self.source.lor:
            if self.ctype == "None":
                raise ValueError(
                    f"Connectiongroup {self.name} must specify 'ctype'. See help(Connectiongroup)"
                )
            if r.sp in self.ctype:
                self.connections.append(r)
                logging.debug(f"found species {r.sp.n}")

        # now we need to create defaults for all connections
        # we setup a dict like this  {SO4:{cid:xxx,plot:xxx}}
        self.cd: dict = {}  # connection dictionary
        for r in self.connections:  # ["SO4", "H2S"]
            self.cd[r.n] = {
                "cid": self.id,
                "plot": "yes",
                "delta": "None",
                "alpha": "None",
                "rate": "None",
                "scale": "None",
                "ctype": "None",
                "ref_reservoirs": "None",
            }
            # now we loop trough all keys for this connection and see
            # if we find a corresponding item in the kwargs
            for kcd, vcd in self.cd[r.n].items():
                if kcd in self.kwargs:  # found entry like ctype
                    if r.sp in self.kwargs[kcd]:  # {SO4: xxx}
                        # update the entry
                        self.cd[r.n][kcd] = self.kwargs[kcd][r.sp]
            # now we can create the connection

            name = f"{r.n}"

            a = Connect(
                # name=name,
                source=getattr(self.source, r.n),
                sink=getattr(self.sink, r.n),
                rate=self.cd[r.n]["rate"],
                delta=self.cd[r.n]["delta"],
                alpha=self.cd[r.n]["alpha"],
                plot=self.cd[r.n]["plot"],
                ctype=self.cd[r.n]["ctype"],
                scale=self.cd[r.n]["scale"],
                ref_reservoirs=self.cd[r.n]["ref_reservoirs"],
                groupname=True,
                id=self.id,
                register=self,
            )

            ## add connection to list of connections
            self.loc.append(a)

    def info(self) -> None:
        """List all connections in this group"""

        print(f"Group Connection from {self.source.name} to {self.sink.name}\n")
        print("The following Connections are part of this group\n")
        print(f"You can query the details of each connection like this:\n")
        for c in self.loc:
            print(f"{c.name}: {self.name}.{c.name}.info()")

        print("")
