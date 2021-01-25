"""
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
import os

class esbmtkBase(object):
    """The esbmtk base class template. This class handles keyword
    arguments, name registration and other common tasks

    """

    __slots__ = ('__dict__')

    from typing import Dict

    def __init__(self) -> None:
        raise NotImplementedError

    def __global_defaults__(self) -> None:
        """ Initial variables which should be present in every object

        """
        self.lmo: list = []

        if 'register' not in self.kwargs:
            self.register = "yes"

    def __validateandregister__(self, kwargs: Dict[str, any]) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        and register the instance variables and the instance in teh global name space
        
        """

        # validate input
        self.__validateinput__(kwargs)

        # add global key-value pairs which should be present in each object
        self.__global_defaults__()

        # register all key/value pairs as instance variables
        self.__registerkeys__()

    def __register_name__(self) -> None:
        """
        Register object name in global name space, and test if
        name is unique

        There are two possible cases: This is a regular object, which will be registered
        in the global namespace (self.register is not set).

        Case B) This object should be registered in the local namespace of a group. In which case
        self.register should be set to the group object.
        
        """

        # we use this to suppress the echo during object creation
        self.reg_time = time.monotonic()

        # if self register is set, it points to the group object which contains
        # this sub object.
        if self.register == "yes":  # This is part of a group
            if isinstance(self, Model):  # Model only exist in the global NS
                setattr(builtins, self.name, self)
            else:
                if self.name in self.mo.lmo:
                    raise NameError(f"{self.name} is a duplicate. Please fix")
                else:
                    self.mo.lmo.append(self.name)
                    setattr(builtins, self.name, self)

        else:
            if self in self.register.lmo:
                raise NameError(
                    f"{self.name} is a duplicate in {self.register}. Please fix"
                )
            setattr(self.register, self.name, self)
            self.register.lmo.append(self)

        logging.info(self.__repr__(1))

    def __validateinput__(self, kwargs: Dict[str, any]) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        """

        self.kwargs = kwargs  # store the kwargs
        self.provided_kwargs = kwargs.copy()  # preserve a copy

        if not hasattr(self, 'lkk'):
            self.lkk: Dict[str, any] = {}
        if not hasattr(self, 'lrk'):
            self.lrk: List[str] = []
        if not hasattr(self, 'lod'):
            self.lod: Dict[str, any] = []
        if not hasattr(self, 'drn'):
            self.drn: Dict[str, any] = []

        # check that mandatory keys are present
        # and that all keys are allowed
        self.__checkkeys__()

        # initialize missing parameters

        self.kwargs = self.__addmissingdefaults__(self.lod, kwargs)

        # check if key values are of correct type
        self.__checktypes__(self.lkk, self.kwargs)

    def __checktypes__(self, av: Dict[any, any], pv: Dict[any, any]) -> None:
        """ this method will use the the dict key in the user provided
        key value data (pv) to look up the allowed data type for this key in av

        av = dictinory with the allowed input keys and their type
        pv = dictionary with the user provided key-value data
        """

        k: any
        v: any

        # provide more meaningful error messages

        # loop over provided keywords
        for k, v in pv.items():
            # check av if provided value v is of correct type
            if av[k] != any:
                #print(f"key = {k}, value  = {v}")
                if not isinstance(v, av[k]):
                   
                    raise TypeError(
                        f"{type(v)} is the wrong type for '{k}', should be '{av[k]}'"
                    )

    def __initerrormessages__(self):
        """ Init the list of known error messages"""
        self.bem: Dict[str, str] = {
            "Number": "a number",
            "Model": "a model handle (i.e. the name without quotation marks)",
            "Element":
            "an element handle (i.e. the name without quotation marks)",
            "Species":
            "a species handle (i.e. the name without quotation marks)",
            "Flux": "a flux handle (i.e. the name without quotation marks)",
            "Reservoir":
            "a reservoir handle (i.e. the name without quotation marks)",
            "Signal":
            "a signal handle (i.e. the name without quotation marks)",
            "Process":
            "a process handle (i.e. the name without quotation marks)",
            "Unit": "a string",
            "File": "a filename inb the local directory",
            "Legend": " a string",
            "Source": " a string",
            "Sink": " a string",
            "Ref": " a Flux reference",
            "Alpha": " a Number",
            "Delta": " a Number",
            "Scale": " a Number",
            "Ratio": " a Number",
            "number": "a number",
            "model": "a model handle (i.e. the name without quotation marks)",
            "element":
            "an element handle (i.e. the name without quotation marks)",
            "species":
            "a species handle (i.e. the name without quotation marks)",
            "flux": "a flux handle (i.e. the name without quotation marks)",
            "reservoir":
            "a reservoir handle (i.e. the name without quotation marks)",
            "signal":
            "a signal handle (i.e. the name without quotation marks)",
            "Process":
            "a process handle (i.e. the name without quotation marks)",
            "unit": "a string",
            "file": "a filename inb the local directory",
            "legend": " a string",
            "source": " a string",
            "sink": " a string",
            "ref": " a Flux reference",
            "alpha": " a Number",
            "delta": " a Number",
            "scale": "a Number",
            "ratio": "a Number",
            "concentration": "a Number",
            "pl": " a list with one or more process handles",
            "react_with": "a Flux handle",
            "data": "External Data Object",
            "register": "esbmtk object",
            str: "a string with quotation marks",
        }

    def __registerkeys__(self) -> None:
        """ register the kwargs key/value pairs as instance variables
        and complain about unknown keywords"""
        k: any  # dict keys
        v: any  # dict values

        # need list of replacement values
        # "alpha" : _alpha

        for k, v in self.kwargs.items():
            # check wheather the variable name needs to be replaced
            if k in self.drn:
                k = self.drn[k]
            setattr(self, k, v)

    def __checkkeys__(self) -> None:
        """ check if the mandatory keys are present"""

        k: str
        v: any
        # test if the required keywords are given
        for k in self.lrk:  # loop over required keywords
            if isinstance(k, list):  # If keyword is a list
                s: int = 0  # loop over allowed substitutions
                for e in k:  # test how many matches are in this list
                    s = s + int(e in self.kwargs)
                if s != 1:  # if none, or more than one match, throw error
                    raise ValueError(
                        f"You need to specify exactly one from this list: {k}")

            else:  # keyword is not a list
                if k not in self.kwargs:
                    raise ValueError(f"You need to specify a value for {k}")

        tl: List[str] = []
        # get a list of all known keywords
        for k, v in self.lkk.items():
            tl.append(k)

        # test if we know all keys
        for k, v in self.kwargs.items():
            if k not in self.lkk:
                raise ValueError(
                    f"{k} is not a valid keyword. \n Try any of \n {tl}\n")

    def __addmissingdefaults__(self, lod: dict, kwargs: dict) -> dict:
        """
        test if the keys in lod exist in kwargs, otherwise add them with the default values
        in lod
        """
        new: dict = {}
        if len(self.lod) > 0:
            for k, v in lod.items():
                if k not in kwargs:
                    new.update({k: v})

        kwargs.update(new)
        return kwargs

    def __repr__(self, log=0) -> str:
        """ Print the basic parameters for this class when called via the print method

        """
        from esbmtk import Q_

        m: str = ""

        # suppress output during object initialization
        tdiff = time.monotonic() - self.reg_time

        # do not echo input unless explicitly requestted

        m = f"{self.__class__.__name__}(\n"
        for k, v in self.provided_kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    m = m + f"    {k} = {v.name},\n"
                # if this is a string
                elif isinstance(v, str):
                    m = m + f"    {k} = '{v}',\n"
                # if this is a quantity
                elif isinstance(v, Q_):
                    m = m + f"    {k} = '{v}',\n"
                # if this is a list
                elif isinstance(v, (list, np.ndarray)):
                    m = m + f"    {k} = '{v[0:3]}',\n"
                # all other cases
                else:
                    m = m + f"    {k} = {v},\n"

        m = m + ")"

        if log == 0 and tdiff < 1:
            m = ""

        return m

    def __str__(self, **kwargs):
        """ Print the basic parameters for this class when called via the print method
        Optional arguments

        indent :int = 0 printing offset

        """
        from esbmtk import Q_

        m: str = ""
        off: str = "  "

        if "indent" in kwargs:
            ind: str = kwargs["indent"] * " "
        else:
            ind: str = ""

        m = f"{ind}{self.name} ({self.__class__.__name__})\n"
        for k, v in self.provided_kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    pass
                elif isinstance(v, str) and not (k == "name"):
                    m = m + f"{ind}{off}{k} = {v}\n"
                elif isinstance(v, Q_):
                    m = m + f"{ind}{off}{k} = {v}\n"
                elif k != "name":
                    m = m + f"{ind}{off}{k} = {v}\n"

        return m

    def __lt__(self, other) -> None:
        """ This is needed for sorting with sorted()

        """

        return self.n < other.n

    def __gt__(self, other) -> None:
        """ This is needed for sorting with sorted()

        """

        return self.n > other.n

    def describe(self, **kwargs) -> None:
        """ Show an overview of the object properties.
        Optional arguments are

        indent :int = 0 indentation

        """

        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = ' ' * indent

        # print basic data bout this object
        print(f"{ind}{self.__str__(indent=indent)}")

# class ClassName(object):
#     """Documentation for ClassName

#     """
#     def __init__(self, args):
#         super(ClassName, self).__init__()
#         self.args = args


class Model(esbmtkBase):
    """This class is used to specify a new model

    Example:

          esbmtkModel(name   =  "Test_Model",
                      start    = "0 yrs",    # optional: start time
                      stop     = "1000 yrs", # end time
                      timestep = "2 yrs",    # as a string "2 yrs"
                      offset = "0 yrs",    # optional: time offset for plot
                      mass_unit = "mol/l",   #required
                      volume_unit = "mol/l", #required
                      time_label = optional, defaults to "Time"
                      display_precision = optional, defaults to 0.01,
                      m_type = "mass_only", defaults to both (mass & isotope)
                      plot_style = 'default', optional defaults to 'default'
                      )

    The 'ref_time' keyword will offset the time axis by the specified
    amount, when plotting the data, .i.e., the model time runs from to
    100, but you want to plot data as if where from 2000 to 2100, you would
    specify a value of 2000. This is for display purposes only, and does not affect
    the model. Care must be taken that any external data references the model
    time domain, and not the display time.

    The display precision affects the on-screen display of data. It is
    also cutoff for the graphicak output. I.e., the interval f the y-axis will not be
    smaller than the display_precision.

    All of the above keyword values are available as variables with
    Model_Name.keyword

    The user facing methods of the model class are
       - Model_Name.describe()
       - Model_Name.save_data()
       - Model_Name.plot_data()
       - Model_Name.save_state() Save the model state
       - Model_name.read_state() Initialize with a previous model state
       - Model_Name.run()
       - Model_Name.list_species()

    User facing variable are Model_Name.time which contains the time
    axis.

    Optional, you can provide the element keyword which will setup a
    default set of Species for Carbon and Sulfur. In this case, there
    is no need to define elements or species. The argument to this
    keyword are either "Carbon", or "Sulfur" or both as a list
    ["Carbon", "Sulfur"].

    """

    __slots__ = ('lor')

    def __init__(self, **kwargs: Dict[any, any]) -> None:
        """ Init Sequence

        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "start": str,
            "stop": str,
            "timestep": str,
            "offset": str,
            "element": (str, list),
            "mass_unit": str,
            "volume_unit": str,
            "time_label": str,
            "display_precision": float,
            "m_type": str,
            "plot_style": str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = [
            "name", "stop", "timestep", "mass_unit", "volume_unit"
        ]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': "0 years",
            'offset': "0 years",
            'time_label': "Time",
            'display_precision': 0.01,
            'm_type': "mass_only",
            'plot_style': "default",
        }

        self.__initerrormessages__()
        self.bem.update({
            "offset": "a string",
            "timesetp": "a string",
            "element": "element name or list of names",
            "mass_unit": "a string",
            "volume_unit": "a string",
            "time_label": "a string",
            "display_precision": "a number",
            "m_type": "a string",
            "plot_style": "a string",
        })

        self.__validateandregister__(kwargs)  # initialize keyword values

        # empty list which will hold all reservoir references
        self.lor: list = []
        # empty list which will hold all connector references
        self.loc: set = set()  # set with connection handles
        self.lel: list = []  # list which will hold all element references
        self.lsp: list = []  # list which will hold all species references
        self.lop: list = []  # list flux processe
        self.lpc_f: list = []  # list of external functions affecting fluxes
        # list of external functions affecting reservoirs
        self.lpc_r: list = []
        # optional keywords for use in the connector class
        self.olkk: list = []

        # Parse the strings which contain unit information and convert
        # into model base units For this we setup 3 variables which define
        self.l_unit = ureg.meter  # the length unit
        self.t_unit = Q_(self.timestep).units  # the time unit
        self.d_unit = Q_(self.stop).units  # display time units
        self.m_unit = Q_(self.mass_unit).units  # the mass unit
        self.v_unit = Q_(self.volume_unit).units  # the volume unit
        # the concentration unit (mass/volume)
        self.c_unit = self.m_unit / self.v_unit
        self.f_unit = self.m_unit / self.t_unit  # the flux unit (mass/time)
        self.r_unit = self.v_unit / self.t_unit  # flux as volume/time
        # this is now defined in __init__.py
        #ureg.define('Sverdrup = 1e6 * meter **3 / second = Sv = Sverdrups')

        # legacy variable names
        self.start = Q_(self.start).to(self.t_unit).magnitude
        self.stop = Q_(self.stop).to(self.t_unit).magnitude
        self.offset = Q_(self.offset).to(self.t_unit).magnitude

        self.bu = self.t_unit
        self.base_unit = self.t_unit
        self.dt = Q_(self.timestep).magnitude
        self.tu = str(self.bu)  # needs to be a string
        self.n = self.name
        self.mo = self.name
        self.plot_style: list = [self.plot_style]

        self.xl = f"Time [{self.bu}]"  # time axis label
        self.length = int(abs(self.stop - self.start))
        self.steps = int(abs(round(self.length / self.dt)))
        self.time = ((arange(self.steps) * self.dt) + self.start)

        # set_printoptions(precision=self.display_precision)

        if "element" in self.kwargs:
            if isinstance(self.kwargs["element"], list):
                element_list = self.kwargs["element"]
            else:
                element_list = [self.kwargs["element"]]

            for e in element_list:

                if e == "Carbon":
                    carbon(self)
                elif e == "Sulfur":
                    sulfur(self)
                elif e == "Hydrogen":
                    hydrogen(self)
                elif e == "Phosphor":
                    phosphor(self)
                else:
                    raise ValueError(f"{e} not implemented yet")
                warranty = (
                    f"\n"
                    f"ESBMTK  Copyright (C) 2020  Ulrich G.Wortmann\n"
                    f"This program comes with ABSOLUTELY NO WARRANTY\n"
                    f"For details see the LICENSE file\n"
                    f"This is free software, and you are welcome to redistribute it\n"
                    f"under certain conditions; See the LICENSE file for details.\n"
                )
                print(warranty)

        # start a log file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        fn: str = f"{kwargs['name']}.log"
        logging.basicConfig(filename=fn, filemode='w', level=logging.INFO)
        self.__register_name__()

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

        # print basic data bout this object
        print(self)

        # list elements
        print("Currently defined elements and their species:")
        for e in self.lel:
            print(f"{ind}{e}")
            print(f"{off} Defined Species:")
            for s in e.lsp:
                print(f"{off}{off}{ind}{s.n}")

    def save_state(self) -> None:
        """ Save model state. Similar to save data, but only saves the last 10
        time-steps

        """

        start: int = -10
        stop: int = -1
        stride: int = 1
        prefix: str = "state_"

        for r in self.lor:
            r.__write_data__(prefix, start, stop, stride)

    def save_data(self, **kwargs) -> None:
        """Save the model results to a CSV file. Each reservoir will have
        their own CSV file

        Optional arguments:
        stride = int  # every nth element
        start = int   # start index
        stop = int    # end index


        """

        for k, v in kwargs.items():
            if not isinstance(v, int):
                print(f"{k} must be an integer number")
                raise ValueError(f"{k} must be an integer number")

        if "stride" in kwargs:
            stride = kwargs["stride"]
        else:
            stride = 1

        if "start" in kwargs:
            start = kwargs["start"]
        else:
            start = 0

        if "stop" in kwargs:
            stop = kwargs["stop"]
        else:
            stop = None

        prefix = ""
        for r in self.lor:
            r.__write_data__(prefix, start, stop, stride)

    def read_state(self):
        """This will initialize the model with the result of a previous model
        run.  For this to work, you will need issue a
        Model.save_state() command at then end of a model run. This
        will create the necessary data files to initialize a
        subsequent model run.

        """
        for r in self.lor:
            r.__read_state__()

    def plot_data(self, **kwargs: dict) -> None:
        """
        Loop over all reservoirs and either plot the data into a
        window, or save it to a pdf

        This method has the optional keyword ptype which can be

        both = plot both, concentraqqtion and isotope data
        iso  = plot isotope data alone
        concentration = plot only concentration data.

        """

        ptype: int = get_ptype(self, kwargs)

        i = 0
        for r in self.lor:
            r.__plot__(i, ptype)
            i = i + 1

        plt.show()  # create the plot windows

    def plot_reservoirs(self, **kwargs: dict) -> None:
        """Loop over all reservoirs and either plot the data into a window,
            or save it to a pdf

        This method has the optional keyword ptype which can be

        both = plot both, concentration and isotope data
        iso  = plot isotope data alone
        concentration = plot only concentration data.
        """

        ptype: int = get_ptype(self, kwargs)

        i: int = 0
        for r in self.lor:
            r.__plot_reservoirs__(i, ptype)
            i = i + 1

        plt.show()  # create the plot windows

    def run(self) -> None:
        """Loop over the time vector, and for each time step, calculate the
        fluxes for each reservoir
        """

        # this has nothing todo with self.time below!
        start: float = process_time()
        new: [NDArray, Float] = zeros(4)

        # put direction dictionary into a list
        for r in self.lor:  # loop over reservoirs
            r.lodir = []
            for f in r.lof:  # loop over fluxes
                a = r.lio[f]
                r.lodir.append(a)

        i = self.execute(new, self.time, self.lor, self.lpc_f, self.lpc_r)

        duration: float = process_time() - start
        print(f"\n Execution took {duration} seconds \n")

    @staticmethod
    def execute(new: [NDArray, Float], time: [NDArray, Float], lor: list,
                f_lpc: list, r_lpc: list) -> None:
        """ Moved this code into a separate function to enable numba optimization

        """

        i = 1  # processes refer to the previous time step -> start at 1
        dt = lor[0].mo.dt

        for t in time[0:-1]:  # loop over the time vector except the first
            # we first need to calculate all fluxes
            for r in lor:  # loop over all reservoirs
                for p in r.lop:  # loop over reservoir processes
                    p(r, i)  # update fluxes

            # update all process based fluxes. This can be done in a global lpc list
            for p in f_lpc:
                p(i)

            # and then update all reservoirs
            for r in lor:  # loop over all reservoirs
                flux_list: List[str] = r.lof
                direction_list: List[int] = r.lodir
                new[0] = new[1] = new[2] = new[3] = 0.0

                # sum fluxes
                for j, f in enumerate(flux_list):
                    new += f[i] * direction_list[j]

                # add to data from last time step
                r[i] = r[i - 1] + new * dt

            # update reservoirs which are calculated
            # lrp # list calculated reservoir
            # update all process based fluxes. This can be done in a global lpc list
            for p in r_lpc:
                p(i)

            i = i + 1  # next time step

    def __step_process__(self, r, i) -> None:
        """ For debugging. Provide reservoir and step number,
        """
        for p in r.lop:  # loop over reservoir processes
            print(f"{p.n}")
            p(r, i)  # update fluxes

    def __step_update_reservoir__(self, r, i) -> None:
        """ For debugging. Provide reservoir and step number,
        """
        flux_list = r.lof
        # new = sum_fluxes(flux_list,r,i) # integrate all fluxes in self.lof

        ms = ls = hs = 0
        for f in flux_list:  # do sum of fluxes in this reservoir
            direction = r.lio[f]
            ms = ms + f.m[i] * direction  # current flux and direction
            ls = ls + f.l[i] * direction  # current flux and direction
            hs = hs + f.h[i] * direction  # current flux and direction

        new = array([ms, ls, hs])
        new = new * r.mo.dt  # get flux / timestep
        new = new + r[i - 1]  # add to data from last time step
        # new = new * (new > 0)  # set negative values to zero
        r[i] = new  # update reservoir data

    def list_species(self):
        """ List all  defined species.

        """
        for e in self.lel:
            print(f"{e.n}")
            e.list_species()

class Element(esbmtkBase):
    """Each model, can have one or more elements.  This class sets
    element specific properties

    Example::

            Element(name      = "S "           # the element name
                    model     = Test_model     # the model handle
                    mass_unit =  "mol",        # base mass unit
                    li_label  =  "$^{32$S",    # Label of light isotope
                    hi_label  =  "$^{34}S",    # Label of heavy isotope
                    d_label   =  r"$\delta^{34}$S",  # Label for delta value
                    d_scale   =  "VCDT",       # Isotope scale
                    r         = 0.044162589,   # isotopic abundance ratio for element
                  )

    """

    # set element properties
    def __init__(self, **kwargs) -> any:
        """ Initialize all instance variables

        """

        # provide a dict of known keywords and types
        self.lkk = {
            "name": str,
            "model": Model,
            "mass_unit": str,
            "li_label": str,
            "hi_label": str,
            "d_label": str,
            "d_scale": str,
            "r": Number
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "model", "mass_unit"]
        # list of default values if none provided
        self.lod = {
            'li_label': "NONE",
            'hi_label': "NONE",
            'd_label': "NONE",
            'd_scale': "NONE",
            'r': 1,
        }

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

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
        self.__register_name__()

    def list_species(self) -> None:
        """ List all species which are predefined for this element

        """

        for e in self.lsp:
            print(e.n)

class Species(esbmtkBase):
    """Each model, can have one or more species.  This class sets species
specific properties
      
      Example::
        
            Species(name = "SO4",
                    element = S,
)

    """

    __slots__ = ('r')

    # set species properties
    def __init__(self, **kwargs) -> None:
        """ Initialize all instance variables
            """

        # provide a list of all known keywords
        self.lkk: Dict[any, any] = {
            "name": str,
            "element": Element,
            'display_as': str,
            'm_weight': Number
        }

        # provide a list of absolutely required keywords
        self.lrk = ["name", "element"]

        # list of default values if none provided
        self.lod = {"display_as": kwargs["name"], 'm_weight': 0}

        self.__initerrormessages__()

        self.__validateandregister__(kwargs)  # initialize keyword values

        if not "display_as" in kwargs:
            self.display_as = self.name

        # legacy names
        self.n = self.name  # display name of species
        self.mu = self.element.mu  # display name of mass unit
        self.ln = self.element.ln  # display name of light isotope
        self.hn = self.element.hn  # display name of heavy isotope
        self.dn = self.element.dn  # display string for delta
        self.ds = self.element.ds  # display string for delta scale
        self.r = self.element.r  # ratio of isotope standard
        self.mo = self.element.mo  # model handle
        self.eh = self.element.n  # element name
        self.e = self.element  # element handle
        self.dsa = self.display_as  # the display string.

        #self.mo.lsp.append(self)   # register self on the list of model objects
        self.e.lsp.append(self)  # register this species with the element
        self.__register_name__()

class Reservoir(esbmtkBase):
    """Tis object holds reservoir specific information.

      Example::

              Reservoir(name = "foo",      # Name of reservoir
                        species = S,          # Species handle
                        delta = 20,           # initial delta - optional (defaults  to 0)
                        mass/concentration = "1 unit"  # species concentration or mass
                        volume = "1E5 l",      # reservoir volume (m^3)
                        plot = "yes"/"no", defaults to yes
                        transform_m = a function reference, optional (see below)
                        )

      you must either give mass or concentration. The result will always be displayed as concentration

      Using a transform function
      ~~~~~~~~~~~~~~~~~~~~~~~~~~

      In some cases, it is useful to transform the reservoir
      concentration data before plotting it.  A good example is the H+
      concentration in water which is better displayed as pH.  We can
      do this by specifying a function to convert the reservoir
      concentration into pH units::

          def phc(m):
              # convert m into the negative log space
              pH = -np.log10(m)
              return m

      this function can then be added to a reservoir as::
    
          hplus.transform_m = phc
    
      Note, at present the transform_m function will only take one
      argument, which always defaults to the reservoir
      mass. The function must return a single argument which
      will be interpreted as the transformed reservoir concentration.

      You can access the reservoir data as:

      - Name.m # mass
      - Name.d # delta
      - Name.c # concentration

    Useful methods include:

      - Name.write_data() # save data to file
      - Name.describe()   # describe Reservoir

    """

    __slots__ = ('m', 'l', 'h', 'd', 'c', 'lio', 'rvalue', 'lodir', 'lof', 'lpc')

    def __init__(self, **kwargs) -> None:
        """ Initialize a reservoir.

        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name":
            str,
            "species":
            Species,
            "delta": (Number, str),
            "concentration": (str, Q_),
            "mass": (str, Q_),
            "volume": (str, Q_),
            "transform_m":
            any,
            "plot":
            str,
            "register":
            (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup, str),
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name", "species", "volume", ["mass", "concentration"]
        ]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            'delta': "None",
            'plot': "yes",
            'transform_m': "None",
        }

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.bem.update({
            "mass": "a  string or quantity",
            "concentration": "a string or quantity",
            "volume": "a string or quantity",
            "plot": "yes or no",
            'register': 'Group Object',
        })
        self.__validateandregister__(kwargs)

        if self.delta == "None":
            self.delta = 0

        # legacy names
        self.n: str = self.name  # name of reservoir
        self.sp: Species = self.species  # species handle
        self.mo: Model = self.species.mo  # model handle
        self.rvalue = self.sp.r

        # convert units
        self.volume: Number = Q_(self.volume).to(self.mo.v_unit).magnitude
        self.v: Number = self.volume  # reservoir volume

        # This should probably be species specific?
        self.mu: str = self.sp.e.mass_unit  # massunit xxxx

        if "concentration" in kwargs:
            c = Q_(self.concentration)
            self.plt_units = c.units
            self.concentration: Number = c.to(self.mo.c_unit).magnitude
            self.mass: Number = self.concentration * self.volume  # caculate mass
            self.display_as = "concentration"
        elif "mass" in kwargs:
            m = Q_(self.mass)
            self.plt_units = self.mo.m_unit
            self.mass: Number = m.to(self.mo.m_unit).magnitude
            self.concentration = self.mass / self.volume
            self.display_as = "mass"
        else:
            raise ValueError("You need to specify mass or concentration")

        # save the unit which was provided by the user for display purposes

        self.lof: list[Flux] = []  # flux references
        self.led: list[ExternalData] = []  # all external data references
        self.lio: dict[str, int] = {}  # flux name:direction pairs
        self.lop: list[Process] = []  # list holding all processe references
        self.loe: list[Element] = []  # list of elements in thiis reservoir
        self.doe: Dict[Species, Flux] = {}  # species flux pairs
        self.loc: set[Connection] = set()  # set of connection objects
        self.ldf: list[DataField] = []  # list of datafield objects
        self.lpc: list[Process] = [] # list of processes which calculate reservoirs

        # initialize mass vector
        self.m: [NDArray, Float[64]] = zeros(self.species.mo.steps) + self.mass
        self.l: [NDArray, Float[64]] = zeros(self.mo.steps)
        self.h: [NDArray, Float[64]] = zeros(self.mo.steps)

        if self.mass == 0:
            self.c: [NDArray, Float[64]] = zeros(self.species.mo.steps)
            self.d: [NDArray, Float[64]] = zeros(self.species.mo.steps)
        else:
            # initialize concentration vector
            self.c: [NDArray, Float[64]] = self.m / self.v
            # isotope mass
            [self.l, self.h] = get_imass(self.m, self.delta, self.species.r)
            # delta of reservoir
            self.d: [NDArray, Float[64]] = get_delta(self.l, self.h,
                                                     self.species.r)

        # left y-axis label
        self.lm: str = f"{self.species.n} [{self.mu}/l]"
        # right y-axis label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"
        self.xl: str = self.mo.xl  # set x-axis lable to model time

        self.legend_left = self.species.dsa
        self.legend_right = f"{self.species.dn} [{self.species.ds}]"
        self.mo.lor.append(self)  # add this reservoir to the model
        # register instance name in global name space
        self.__register_name__()

        # decide which setitem functions to use
        if self.mo.m_type == "both":
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

    # setup a placeholder setitem function
    def __setitem__(self, i: int, value: float):
        return self.__set_data__(i, value)

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return self

    def __getitem__(self, i: int) -> NDArray[np.float64]:
        """ Get flux data by index

        """

        return np.array([self.m[i], self.l[i], self.h[i], self.d[i]])

    def __set_with_isotopes__(self, i: int, value: float) -> None:
        """ write data by index

        """

        self.m[i]: float = value[0]
        self.l[i]: float = value[1]
        self.h[i]: float = value[2]
        # update concentration and delta next. This is computationally inefficient
        # but the next time step may depend on on both variables.
        self.d[i]: float = get_delta(self.l[i], self.h[i], self.sp.r)
        self.c[i]: float = self.m[i] / self.v  # update concentration

    def __set_without_isotopes__(self, i: int, value: float) -> None:
        """ write data by index

        """

        self.m[i]: float = value[0]
        self.c[i]: float = self.m[i] / self.v  # update concentration

    def __write_data__(self, prefix: str, start: int, stop: int,
                       stride: int) -> None:
        """ To be called by write_data and save_state
        
        """

        # some short hands
        sn = self.sp.n  # species name
        sp = self.sp  # species handle
        mo = self.sp.mo  # model handle

        smu = f"{mo.m_unit:~P}"
        mtu = f"{mo.t_unit:~P}"
        fmu = f"{mo.f_unit:~P}"
        cmu = f"{mo.c_unit:~P}"

        sdn = self.sp.dn  # delta name
        sds = f"[{self.sp.ds}]"  # delta scale
        rn = self.n  # reservoir name
        mn = self.sp.mo.n  # model name
        fn = f"{prefix}{mn}_{rn}.csv"  # file name

        # build the dataframe
        df: pd.dataframe = DataFrame()

        df[f"{self.n} Time [{mtu}]"] = self.mo.time[start:stop:stride]  # time
        df[f"{self.n} {sn} [{smu}]"] = self.m[start:stop:stride]  # mass
        df[f"{self.n} {sp.ln} [{smu}]"] = self.l[start:stop:
                                                 stride]  # light isotope
        df[f"{self.n} {sp.hn} [{smu}]"] = self.h[start:stop:
                                                 stride]  # heavy isotope
        df[f"{self.n} {sdn} [{sds}]"] = self.d[start:stop:
                                               stride]  # delta value
        df[f"{self.n} {sn} [{cmu}]"] = self.c[start:stop:
                                              stride]  # concentration

        for f in self.lof:  # Assemble the headers and data for the reservoir fluxes
            df[f"{f.n} {sn} [{fmu}]"] = f.m[start:stop:stride]  # mass
            df[f"{f.n} {sn} [{sp.ln}]"] = f.l[start:stop:
                                              stride]  # light isotope
            df[f"{f.n} {sn} [{sp.hn}]"] = f.h[start:stop:
                                              stride]  # heavy isotope
            df[f"{f.n} {sn} {sdn} [{sds}]"] = f.d[start:stop:stride]  # delta

        df.to_csv(fn, index=False)  # Write dataframe to file
        return df

    def __read_state__(self) -> None:
        """ read data from csv-file into a dataframe

        The CSV file must have the following columns

        Model Time     t
        Reservoir_Name m
        Reservoir_Name l
        Reservoir_Name h
        Reservoir_Name d
        Reservoir_Name c
        Flux_name m
        Flux_name l etc etc.

        """

        read: set = set()
        curr: set = set()

        # get a set of all current fluxes
        for e in self.lof:
            curr.add(e.name)

        fn = "state_" + self.mo.n + "_" + self.n + ".csv"

        if not os.path.exists(fn):
            print(f"Cannot find {fn}\n")
            raise FileNotFoundError(f"{fn} does not exist")

        df: pd.DataFrame = pd.read_csv(fn)
        headers = list(df.columns.values)
        self.df = df

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
        i: int = 0
        for n in header_list:
            name = n.split(" ")[0]
            # this finds the reservoir name
            if name == self.name:
                col = self.__assign__data__(self, df, col, True)
            # this loops over all fluxes in a reservoir
            elif is_name_in_list(name, self.lof):
                obj = get_object_from_list(name, self.lof)
                read.add(obj.name)
                col = self.__assign__data__(obj, df, col, False)
                i += 1
            else:
                print(f"No '{name}' in {self.n}\n")
                raise ValueError("Unable to find Reservoir of Flux Name")

        # test if we missed any fluxes
        for e in list(curr.difference(read)):
            print(f"\n Warning: Did not find values for '{e}'\n")

    def __assign__data__(self, obj: any, df: pd.DataFrame, col: int,
                         res: bool) -> int:
        """
        Assign the third last entry data to all values in flux or reservoir

        parameters: df = dataframe
                    col = column number
                    res = true if reservoir

        """

        ovars: list = ["m", "l", "h", "d"]

        obj.m[:] = df.iloc[-3, col]
        obj.l[:] = df.iloc[-3, col + 1]
        obj.h[:] = df.iloc[-3, col + 2]
        obj.d[:] = df.iloc[-3, col + 3]
        col = col + 4

        if res:  # if type is reservoir
            obj.c[:] = df.iloc[-3, col]
            col += 1

        return col

    def __plot__(self, i: int, ptype: int) -> None:
        """ Plot data from reservoirs and fluxes into a multiplot window

        """

        model = self.sp.mo
        species = self.sp
        obj = self
        # time = model.time + model.offset  # get the model time
        #xl = f"Time [{model.bu}]"

        size, geo = get_plot_layout(self)  # adjust layout
        filename = f"{model.n}_{self.n}.pdf"
        fn = 1  # counter for the figure number

        plt.style.use(model.plot_style)
        fig = plt.figure(i)  # Initialize a plot window
        fig.canvas.set_window_title(f"Reservoir Name: {self.n}")
        fig.set_size_inches(size)

        # plot reservoir data
        if self.plot == "yes":
            plot_object_data(geo, fn, self, ptype)

            # plot the fluxes assoiated with this reservoir
            for f in sorted(self.lof):  # plot flux data
                if f.plot == "yes":
                    fn = fn + 1
                    plot_object_data(geo, fn, f, ptype)

            for d in sorted(self.ldf):  # plot data fields
                fn = fn + 1
                plot_object_data(geo, fn, d, ptype)

            if geo != [1, 1]:
                fig.suptitle(f"Model: {model.n}, Reservoir: {self.n}\n",
                             size=16)

            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            fig.savefig(filename)

    def __plot_reservoirs__(self, i: int, ptype: int) -> None:
        """ Plot only the  reservoirs data, and ignore the fluxes

        """

        model = self.sp.mo
        species = self.sp
        obj = self
        time = model.time + model.offset  # get the model time
        xl = f"Time [{model.bu}]"

        size: list = [5, 3]
        geo: list = [1, 1]
        filename = f"{model.n}_{self.n}.pdf"
        fn: int = 1  # counter for the figure number

        plt.style.use(model.plot_style)
        fig = plt.figure(i)  # Initialize a plot window
        fig.set_size_inches(size)

        # plt.legend()ot reservoir data
        plot_object_data(geo, fn, self, ptype)

        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(filename)

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

        # print basic data bout this reservoir
        print(f"{ind}{self.__str__(indent=indent)}")
        print(f"{ind}Data sample:")
        show_data(self, index=index, indent=indent)

        print(f"\n{ind}Connnections:")
        for p in sorted(self.loc):
            print(f"{off}{ind}{p.n}")

        print()
        print("Use the describe method on any of the above connections")
        print("to see information on fluxes and processes")

class ReservoirGroup(esbmtkBase):
    """This class allows the creation of a group of reservoirs which share
    a common volume, and potentially connections. E.g., if we have two
    reservoir groups with the same reservoirs, and we connect them
    with a flux, this flux will apply to all reservoirs in this group. 

    A typical examples might be ocean water which comprises several
    species.  A reservoir group like ShallowOcean will then contain
    sub-reservoirs like DIC in the form of ShallowOcean.DIC

    Example::

        ReservoirGroup(name = "ShallowOcean",    # Name of reservoir group
                    volume = "1E5 l",            # reservoir volume (m^3)
                    delta   = {DIC:0, ALK:0, PO4:0]            # dict of delta values
                    mass/concentration = {DIC:"1 unit", ALK: "1 unit", PO$: "1 unit"] # 
                    plot = {DIC:"yes", ALK:"yes", PO4: "no"] defaults to yes
               )

    Notes: - The subreservoirs are derived from the keys in the concentration or mass
             dictionary. Toward this end, the keys must be valid species handles and
             -- not species names -- !
    
    Connecting two reservoir groups requires that the names in both
    group match, or that you specify a dictionary which delineates the
    matching.

    """
    def __init__(self, **kwargs) -> None:
        """ Initialize a new reservoir group

        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "delta": dict,
            "concentration": dict,
            "mass": dict,
            "volume": (str, Q_),
            "plot": dict,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = [
            "name",
            "volume",
            ["mass", "concentration"],
        ]

        # Create a list of default values if none provided
        plot: dict = {}
        delta: dict = {}
        concentration: dict = {}
        mass: dict = {}

        if 'concentration' in kwargs:
            self.species: list = list(kwargs['concentration'].keys())
        elif 'mass' in kwargs:
            self.species: list = list(kwargs['mass'].keys())
        else:
            raise ValueError("You must provide either mass or concentration")

        # loop over names and create dicts
        for n in self.species:
            delta[n] = 'None'
            plot[n] = 'yes'
            concentration[n] = 'None'
            mass[n] = 'None'

        self.lod: Dict[str, any] = {
            'delta': delta,
            'concentration': concentration,
            'mass': concentration,
            'plot': plot,
        }

        # validate and initialize instance variables
        self.__initerrormessages__()
        self.bem.update({
            "mass": "a  string or quantity",
            "concentration": "a string or quantity",
            "volume": "a string or quantity",
            "plot": "yes or no",
        })

        self.__validateandregister__(kwargs)

        # legacy variable
        self.n = self.name
        
        # get model handle
        self.mo = self.species[0].mo

        # register this group object in the global namespace
        self.__register_name__()

        self.lor: list = []  # list of reservoirs in this group.
        # loop over all entries in species and create the respective reservoirs
        for i, s in enumerate(self.species):
            if not isinstance(s, Species):
                raise ValueError(f"{s} needs to be a valid species name")

            if self.concentration[s] == "None":
                # create reservoir without registering it in the global name space
                a = Reservoir(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=self.delta[s],
                    mass=self.mass[s],
                    volume=self.volume,
                    plot=self.plot[s],
                )
            elif self.mass[s] == "None":
                # create reservoir without registering it in the global name space
                a = Reservoir(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                    delta=self.delta[s],
                    concentration=self.concentration[s],
                    volume=self.volume,
                    plot=self.plot[s],
                )
            else:
                raise ValueError("You must specify mass or concentration")

            # register as part of this group
            self.lor.append(a)

class Flux(esbmtkBase):
    """A class which defines a flux object. Flux objects contain
      information which links them to an species, describe things like
      the mass and time unit, and store data of the total flux rate at
      any given time step. Similarly, they store the flux of the light
      and heavy isotope flux, as well as the delta of the flux. This
      is typically handled through the Connect object. If you set it up manually
      
      Flux = (name = "Name"
              species = species_handle,
              delta = any number,
              rate  = "12 mol/s" # must be a string
      )

       You can access the flux data as
      - Name.m # mass
      - Name.d # delta
      - Name.c # concentration
      
    """

    __slots__ = ('m', 'l', 'h', 'd', 'rvalue', 'lpc')

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """
        Initialize a flux. Arguments are the species name the flux rate
        (mol/year), the delta value and unit
        
        """

        from . import ureg, Q_

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "delta": Number,
            "rate": (str, Q_),
            "plot": str,
            "register": (SourceGroup,SinkGroup,ReservoirGroup,ConnectionGroup,str),
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "species", "rate"]

        # list of default values if none provided
        self.lod: Dict[any, any] = {
            'delta': 0,
            "plot": "yes",
        }

        # initialize instance
        self.__initerrormessages__()
        self.bem.update({"rate": "a string", "plot": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # name of flux
        self.sp: Species = self.species  # species name
        self.mo: Model = self.species.mo  # model name
        self.model: Model = self.species.mo  # model handle
        self.rvalue = self.sp.r

        # model units
        self.plt_units = Q_(self.rate).units
        self.mu: str = f"{self.species.mu}/{self.mo.tu}"

        # and convert flux into model units
        fluxrate: float = Q_(self.rate).to(self.mo.f_unit).magnitude

        self.m: [NDArray, Float[64]
                 ] = zeros(self.model.steps) + fluxrate  # add the flux
        self.l: [NDArray, Float[64]] = zeros(self.model.steps)
        self.h: [NDArray, Float[64]] = zeros(self.model.steps)
        self.d: [NDArray, Float[64]] = zeros(self.model.steps) + self.delta

        if self.rate != 0:
            [self.l, self.h] = get_imass(self.m, self.delta, self.species.r)

        # if self.delta == 0:
        #     self.d: [NDArray, Float[64]] = zeros(self.model.steps)
        # else:  # update delta
        #     self.d: [NDArray, Float[64]] = get_delta(self.l, self.h, self.sp.r)

        self.lm: str = f"{self.species.n} [{self.mu}]"  # left y-axis a label
        self.ld: str = f"{self.species.dn} [{self.species.ds}]"  # right y-axis a label
        self.legend_left: str = self.species.dsa
        self.legend_right: str = f"{self.species.dn} [{self.species.ds}]"

        self.xl: str = self.model.xl  # se x-axis label equal to model time
        self.lop: list[Process] = []  # list of processes
        self.lpc: list = [] # list of external functions
        self.led: list[ExternalData] = []  # list of ext data
        self.source: str = ""  # Name of reservoir which acts as flux source
        self.sink: str = ""  # Name of reservoir which acts as flux sink
        self.__register_name__()

        # decide which setitem functions to use
        if self.mo.m_type == "both":
            self.__set_data__ = self.__set_with_isotopes__
        else:
            self.__set_data__ = self.__set_without_isotopes__

    # setup a placeholder setitem function
    def __setitem__(self, i: int, value: [NDArray, float]):
        return self.__set_data__(i, value)

    def __getitem__(self, i: int) -> NDArray[np.float64]:
        """ Get data by index
        
        """

        return array([self.m[i], self.l[i], self.h[i], self.d[i]])

    def __set_with_isotopes__(self, i: int, value: [NDArray, float]) -> None:
        """ Write data by index
        
        """

        self.m[i] = value[0]
        self.l[i] = value[1]
        self.h[i] = value[2]
        self.d[i] = get_delta(self.l[i], self.h[i], self.sp.r)  # update delta

    def __set_without_isotopes__(self, i: int, value: [NDArray,
                                                       float]) -> None:
        """ Write data by index
        
        """

        self.m[i] = value[0]

    def __call__(self) -> None:  # what to do when called as a function ()
        pass
        return

    def __add__(self, other):
        """ adding two fluxes works for the masses, but not for delta

        """

        self.m = self.m + other.m
        self.l = self.l + other.l
        self.h = self.h + other.h
        self.d = get_delta(self.l, self.h, self.sp.r)

    def __sub__(self, other):
        """ adding two fluxes works for the masses, but not for delta

        """

        self.m = self.m - other.m
        self.l = self.l - other.l
        self.h = self.h - other.h
        self.d = get_delta(self.l, self.h, self.sp.r)

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

        # print basic data bout this object
        print(f"{ind}{self.__str__(indent=indent)}")
        print(f"{ind}Data sample:")
        show_data(self, index=index, indent=indent)

        if len(self.lop) > 0:
            print(f"\n{ind}Process(es) acting on this flux:")
            for p in self.lop:
                print(f"{off}{ind}{p.__repr__()}")

            print("")
            print(
                "Use help on the process name to get an explanation what this process does"
            )
            if self.register == "yes":
                print(f"e.g., help({self.lop[0].n})")
            else:
                print(f"e.g., help({self.register.name}.{self.lop[0].n})")
        else:
            print("There are no processes for this flux")

    def plot(self, **kwargs: dict) -> None:
        """Plot the flux data:
        This method has the optional keyword ptype which can be

        both = plot both, concentration and isotope data
        iso  = plot isotope data alone
        concentration = plot only concentration data.

        """

        ptype: int = get_ptype(kwargs)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(5, 4)  # Set figure size in inches
        fig.set_dpi(100)  # Set resolution in dots per inch

        ax1.plot(self.mo.time, self.m, c="C0")
        ax2 = ax1.twinx()  # get second y-axis
        ax2.plot(self.mo.time, self.d, c="C1", label=self.n)

        ax1.set_title(self.n)
        ax1.set_xlabel(f"Time [{self.mo.tu}]")  #
        ax1.set_ylabel(f"{self.sp.n} [{self.sp.mu}]")
        ax2.set_ylabel(f"{self.sp.dn} [{self.sp.ds}]")
        ax1.spines['top'].set_visible(False)  # remove unnecessary frame
        ax2.spines['top'].set_visible(False)  # remove unnecessary frame

        fig.tight_layout()
        plt.show()
        plt.savefig(self.n + ".pdf")

class SourceSink(esbmtkBase):
    """
    This is a meta class to setup a Source/Sink objects. These are not 
    actual reservoirs, but we stil need to have them as objects
    Example::
    
           Sink(name = "Pyrite",species = SO4)

    where the first argument is a string, and the second is a reservoir handle
    
    """

    def __init__(self, **kwargs) -> None:


        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": Species,
            "register": (SourceGroup, SinkGroup, ReservoirGroup, ConnectionGroup,str),
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        self.loc: set[Connection]  = set()  # set of connection objects

        # legacy names
        self.n = self.name
        self.sp = self.species
        self.mo = self.species.mo
        self.u = self.species.mu + "/" + str(self.species.mo.bu)

        self.__register_name__()


class Sink(SourceSink):
    """
    This is just a wrapper to setup a Sink object
    Example::
    
           Sink(name = "Pyrite",species =SO4)

    where the first argument is a string, and the second is a species handle
    """


class Source(SourceSink):
    """
    This is just a wrapper to setup a Source object
    Example::
    
           Sink(name = "SO4_diffusion", species ="SO4")

    where the first argument is a string, and the second is a species handle
    """

class SourceSinkGroup(esbmtkBase):
    """
    This is a meta class to setup  Source/Sink Groups. These are not 
    actual reservoirs, but we stil need to have them as objects
    Example::
    
           Sink(name = "Pyrite",species = SO4)

    where the first argument is a string, and the second is a reservoir handle
    """

    
    def __init__(self, **kwargs) -> None:

        # provide a dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "species": list,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name", "species"]
        # list of default values if none provided

        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        self.loc: set[Connection] = set()  # set of connection objects

        # register this object in the global namespace
        self.mo = self.species[0].mo  # get model handle
        self.__register_name__()

        self.lor: list = []  # list of sub reservoirs in this group
        # loop over names and setup sub-objects
        for i, s in enumerate(self.species):
            if not isinstance(s, Species):
                raise ValueError(f"{s} needs to be a valid species name")

            if type(self).__name__ == "SourceGroup":
                a = Source(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                )
            elif type(self).__name__ == "SinkGroup":
                a = Sink(
                    name=f"{s.name}",
                    register=self,
                    species=s,
                )
            else:
                raise TypeError(
                    f"{type(self).__name__} is not a valid class type")

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
                   scale = 1, optional
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
      keyword to shift the external signal data in teh time domain.


      This class has the following methods

        Signal.repeat()
        Signal.plot()
        Signal.describe()
    
    """
    def __init__(self, **kwargs) -> None:
        """ Parse and initialize variables
        
        """

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
            "scale": Number
        }

        # provide a list of absolutely required keywords
        self.lrk: List[str] = [
            "name", ["duration", "filename"], "species", ["shape", "filename"],
            ["magnitude", "mass", "filename"]
        ]

        # list of default values if none provided
        self.lod: Dict[str, any] = {
            'start': "0 yrs",
            'stype': "addition",
            'shape': "external_data",
            'offset': "0 yrs",
            'duration': "0 yrs",
            'delta': 0,
            'scale': 1,
        }

        self.__initerrormessages__()
        self.bem.update({
            "data": "a string",
            "magnitude": "Number",
            "scale": "Number",
        })
        self.__validateandregister__(kwargs)  # initialize keyword values

        # list of signals we are based on.
        self.los: List[Signal] = []

        # convert units to model units
        self.st: Number = Q_(self.start).to(
            self.species.mo.t_unit).magnitude  # start time

        if "mass" in self.kwargs:
            self.mass = Q_(self.mass).to(self.species.mo.m_unit).magnitude
        elif "magnitude" in self.kwargs:
            self.magnitude = Q_(self.magnitude).to(
                self.species.mo.f_unit).magnitude

        if "duration" in self.kwargs:
            self.duration = Q_(self.duration).to(
                self.species.mo.t_unit).magnitude

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

        # initialize signal data
        self.data = self.__init_signal_data__()
        self.data.n: str = self.name + "_data"  # update the name of the signal data
        # update isotope values
        self.data.li, self.data.hi = get_imass(self.data.m, self.data.d,
                                               self.sp.r)
        self.__register_name__()

    def __init_signal_data__(self) -> None:
        """ Create an empty flux and apply the shape
            """
        # create a dummy flux we can act up
        self.nf: Flux = Flux(name=self.n + "_data",
                             species=self.sp,
                             rate=f"0 {self.sp.mo.f_unit}",
                             delta=0)

        # since the flux is zero, the delta value will be undefined. So we set it explicitly
        # this will avoid having additions with Nan values.
        self.nf.d[0:]: float = 0.0

        # find nearest index for start, and end point
        self.si: int = int(round(self.st / self.mo.dt))  # starting index
        self.ei: int = self.si + int(round(self.duration / self.mo.dt))  # end index

        # create slice of flux vector
        self.s_m: [NDArray, Float[64]] = array(self.nf.m[self.si:self.ei])
        # create slice of delta vector
        self.s_d: [NDArray, Float[64]] = array(self.nf.d[self.si:self.ei])

        if self.sh == "square":
            self.__square__(self.si, self.ei)

        elif self.sh == "pyramid":
            self.__pyramid__(self.si, self.ei)

        elif "filename" in self.kwargs:  # use an external data set
            self.__int_ext_data__(self.si, self.ei)

        else:
            raise ValueError(f"argument needs to be either square/pyramid, "
                             f"or an ExternalData object. "
                             f"shape = {self.sh} is not a valid Value")

        # now add the signal into the flux slice
        self.nf.m[self.si:self.ei] = self.s_m
        self.nf.d[self.si:self.ei] = self.s_d

        return self.nf

    def __square__(self, s, e) -> None:
        """ Create Square Signal

        """

        if "mass" in self.kwd:
            h = self.mass / self.duration  # get the height of the square
           
        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError(
                "You must specify mass or magnitude of the signal")

        self.s_m: float = h  # add this to the section
        self.s_d: float = self.d  # add the delta offset

    def __pyramid__(self, s, e) -> None:
        """ Create pyramid type Signal

        s = start index
        e = end index
        """

        if "mass" in self.kwd:
            h = 2 * self.mass / self.duration  # get the height of the pyramid
            
        elif "magnitude" in self.kwd:
            h = self.magnitude
        else:
            raise ValueError(
                "You must specify mass or magnitude of the signal")

        # create pyramid
        c: int = int(round((e - s) / 2))  # get the center index for the peak
        x: [NDArray, Float[64]] = array([0, c,
                                         e - s])  # setup the x coordinates
        y: [NDArray, Float[64]] = array([0, h, 0])  # setup the y coordinates
        d: [NDArray, Float[64]] = array([0, self.d,
                                         0])  # setup the d coordinates
        xi = arange(0, e - s)  # setup the points at which to interpolate
        h: [NDArray, Float[64]] = interp(xi, x, y)  # interpolate flux
        dy: [NDArray, Float[64]] = interp(xi, x, d)  # interpolate delta
        self.s_m: [NDArray,
                   Float[64]] = self.s_m + h  # add this to the section
        self.s_d: [NDArray, Float[64]] = self.s_d + dy  # ditto for delta

    def __int_ext_data__(self, s, e) -> None:
        """ Interpolate External data as a signal. Unlike the other signals,
        thiw will replace the values in the flux with those read from the
        external data source. The external data need to be in the following format

        Time [units], Rate [units], delta value [units]
        0,     10,   12

        i.e., the first row needs to be a header line
        
        """

        from . import ureg, Q_

        if not os.path.exists(
                self.filename):  # check if the file is actually there
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
        self.s_m: [NDArray, Float[64]] = array(self.nf.m[self.si:self.ei])

        # create slice of delta vector
        self.s_d: [NDArray, Float[64]] = array(self.nf.d[self.si:self.ei])

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
        """ This method creates a new signal by repeating an existing signal.
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
        ns.ms: [NDArray, Float[64]
                ] = self.data.m[start:stop]  # get the data slice we are using
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

            ns.data.m[start:stop]: [NDArray, Float[64]
                                    ] = ns.data.m[start:stop] + ns.ms[0:lds]
            ns.data.d[start:stop]: [NDArray, Float[64]
                                    ] = ns.data.d[start:stop] + ns.ds[0:lds]

        # and recalculate li and hi
        ns.data.l: [NDArray, Float[64]]
        ns.data.h: [NDArray, Float[64]]
        [ns.data.l, ns.data.h] = get_imass(ns.data.m, ns.data.d, ns.data.sp.r)
        return ns

    def __register__(self, flux) -> None:
        """ Register this signal with a flux. This should probably be done
            through a process!
        
        """

        self.fo: Flux = flux  # the flux handle
        self.sp: Species = flux.sp  # the species handle
        model: Model = flux.sp.mo  # the model handle add this process to the
        # list of processes
        flux.lop.append(self)

    def __call__(self) -> NDArray[np.float64]:
        """ what to do when called as a function ()

        """

        return (array([self.fo.m, self.fo.l, self.fo.h,
                       self.fo.d]), self.fo.n, self)

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
                       y1_data = np.Ndarray
                       y1_label = Y-Axis label
                       y1_legend = Data legend
                       y2_data = np.Ndarray    # optional
                       y2_label = Y-Axis label # optional
                       y2_legend = Data legend # optional

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
            "associated_with": (Reservoir,ReservoirGroup),
            "y1_data": NDArray[float],
            "y1_label": str,
            "y1_legend": str,
            "y2_data": (str,NDArray[float]),
            "y2_label": str,
            "y2_legend": str,
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
        }

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({
            "associated_with": "a string",
            "y1_data": "a numpy array",
            "y1_label": "a string",
            "y1_legend": "a string",
            "y2_data": "a numpy array",
            "y2_label": "a string",
            "y2_legend": "a string"
        })

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
        # register with reservoir
        self.associated_with.ldf.append(self)
        self.__register_name__()

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
            "scale": Number,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "filename", "legend", "reservoir"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {"offset": "0 yrs", "scale": 1}

        # validate input and initialize instance variables
        self.__initerrormessages__()
        self.__validateandregister__(kwargs)  # initialize keyword values

        # legacy names
        self.n: str = self.name  # string =  name of this instance
        self.fn: str = self.filename  # string = filename of data
        self.mo: Model = self.reservoir.species.mo

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

        if ((self.x[0] > xi[0]) or (self.x[-1] < xi[-1])):
            message = (f"\n Interpolation requires that the time domain"
                       f"is equal or greater than the model domain"
                       f"data t(0) = {self.x[0]}, tmax = {self.x[-1]}"
                       f"model t(0) = {xi[0]}, tmax = {xi[-1]}")

            raise ValueError(message)
        else:
            self.y: [NDArray] = interp(xi, self.x, self.y)
            self.x = xi

    def plot(self) -> None:
        """ Plot the data and save a pdf

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

from .connections import *
from .processes import *
from .species_definitions import *
