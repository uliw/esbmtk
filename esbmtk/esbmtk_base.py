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

     This module defines some shared methods

"""
from __future__ import annotations
import time
from numba.typed import List
import numpy as np
import logging
import builtins
import typing as tp

if tp.TYPE_CHECKING:
    from .esbmtk import Model


class input_parsing(object):
    """Provides various routines to parse and process keyword
    arguments.  All derived classes need to declare the allowed
    keyword arguments, their defualt values and the type in the
    following format:

    defaults = {"key": [value, (allowed instances)]

    the recommended sequence is to first set default values via
    __register_variable_names__()

    __update_dict_entries__(defaults,kwargs) will  compare the provided kwargs against
    this data, and upon succesful parsing update the default dict
    with the new values
    """

    def __init__(self):
        raise NotImplementedError("input parsing has no instance!")

    def __initialize_keyword_variables__(self, kwargs) -> None:
        """check, register and update keyword variables"""

        self.update = False
        self.__check_mandatory_keywords__(self.lrk, kwargs)
        self.__register_variable_names__(self.defaults, kwargs)
        self.__update_dict_entries__(self.defaults, kwargs)
        self.update = True

    def __check_mandatory_keywords__(self, lrk: list, kwargs: dict) -> None:
        """Verify that all elements of lrk have a corresponding key in
        kwargs.  If not, print error message"""

        for key in lrk:
            if isinstance(key, list):
                has_key = 0
                for k in key:
                    if k in kwargs and kwargs[k] != "None":
                        has_key += 1
                if has_key != 1:
                    raise ValueError(f"give only one of {key}")
            else:
                if key not in kwargs:
                    raise ValueError(f"{key} is a mandatory keyword")

    def __register_variable_names__(
        self,
        defaults: dict[str, list[any, tuple]],
        kwargs: dict,
    ) -> None:
        """Register the key value[0] pairs as local instance variables.
        We register them with their actual variable name and as _variable_name
        in case we use setter and getter methods.
        to avoid name conflicts.
        """
        for key, value in defaults.items():
            setattr(self, "_" + key, value[0])
            setattr(self, key, value[0])

        # save kwargs dict
        self.kwargs: dict = kwargs

    def __update_dict_entries__(
        self,
        defaults: dict[str, list[any, tuple]],
        kwargs: dict[str, list],
    ) -> None:
        """This function compares the kwargs dictionary with the defaults
        dictionary. If the kwargs key cannot be found, raise an
        error. Otherwise test that the value is of the correct type. If
        yes, update the defaults dictionary with the new value.

        defaults = {"key": [value, (allowed instances)]
        kwargs = {"key": value

        Note that this function assumes that all defaults have been registered
        with the instance via __register_variable_names__()
        """
        for key, value in kwargs.items():
            if key not in defaults:
                raise ValueError(f"{key} is not a valid key")

            if not isinstance(value, defaults[key][1]):
                raise TypeError(
                    f"{value} for {key} must be of type {defaults[key][1]}, not {type(value)}"
                )

            defaults[key][0] = value  # update defaults dictionary
            setattr(self, key, value)  # update instance variables
            setattr(self, "_" + key, value)  # and their property shadows

    def __register_name_new__(self) -> None:
        """if self.parent is set, register self as attribute of self.parent,
        and set full name to parent.full-name + self.name
        if self.parent == "None", full_name = name
        """

        if self.parent == "None":
            self.full_name = self.name
            reg = self
        else:
            self.full_name = self.parent.full_name + "." + self.name
            reg = self.parent.model
            # check for naming conflicts
            if self.full_name in reg.lmo:
                raise NameError(f"{self.full_name} is a duplicate name in reg.lmo")
            else:
                # register with model
                reg.lmo.append(self.full_name)
                reg.lmo2.append(self)
                reg.dmo.update({self.full_name: self})
                setattr(self.parent, self.name, self)
                self.kwargs["full_name"] = self.full_name
        self.reg_time = time.monotonic()


class esbmtkBase(input_parsing):
    """The esbmtk base class template. This class handles keyword
    arguments, name registration and other common tasks

    Useful methods in this class:

    define required keywords in lrk dict:
       self.lrk: list = ["name"]

    define allowed type per keyword in lkk dict:
       self.lkk: dict[str, any] = {
                                  "name": str,
                                  "model": Model,
                                  "salinity": (int, float), # int or float
                                  }

    define default values if none provided in lod dict
       self.lod: dict[str, any] = {"salinity": 35.0}

    validate keyword input:
        self.__validateinput__(kwargs)

    add global defaults each esbmtk object should have, even if they are not set or used
        self.__global_defaults__()

    register all key/value pairs as instance variables
        self.__registerkeys__()

    register name in global name space. This is only necessary if you
    want to reference the instance by name from the console, otherwise
    use the normal python way (i.e.  instance = class(keywords)
    self.__register_name__ ()

    """

    #  __slots__ = "__dict__"

    # from typing import dict

    def __init__(self) -> None:
        raise NotImplementedError

    def __global_defaults__(self) -> None:
        """Initial variables which should be present in every object
        Note that this is executed before we register the kwargs as instance
        variables

        """

        self.lmo: list = []
        self.lmo2: list = []
        self.ldo: list = ["full_name", "register", "groupname", "ctype"]

        for n in self.ldo:
            if n not in self.kwargs:
                self.kwargs[n] = "None"
                # logging.debug(
                #    f"set {self.kwargs['name']} self.kwargs[{n}] to {self.kwargs[n]}"
                # )

    def __validateandregister__(self, kwargs: dict) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        and register the instance variables and the instance in the global name space

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

        Case B) This object should be registered in the local namespace of a group.
        In that case self.register should be set to the group object.

        """

        from esbmtk import Model

        # we use this to suppress the echo during object creation
        self.reg_time = time.monotonic()

        # if self register is set, it points to the group object which contains
        # this sub object.

        logging.debug(f"self.register = {self.register}")

        # models are either in the global namespace, or do not get registered at all
        if isinstance(self, Model):
            if self.register == "None":
                setattr(builtins, self.name, self)
            else:
                pass

        # all other objects can be either part of another esbmtk object
        # register=object reference
        # or be registered globally
        else:
            # get model registry
            if isinstance(self.register, Model):
                reg = self.register
            elif isinstance(self.register, str):
                reg = self.mo
            else:
                reg = self.register.mo

            # checl model default
            if self.register == "None":  # global registration
                setattr(builtins, self.name, self)
                self.full_name = self.name

                # check for naming conflicts
                if self.full_name in reg.lmo:
                    raise NameError(f"{self.full_name} is a duplicate name. Please fix")
                else:
                    # register with model
                    reg.lmo.append(self.full_name)
                    reg.lmo2.append(self)
                    reg.dmo.update({self.full_name: self})

            else:  # local registration
                # get full_name of parent object
                if self.register.full_name != "None":
                    fn: str = f"{self.register.full_name}.{self.name}"
                else:
                    fn: str = f"{self.register.name}.{self.name}"

                self.full_name = fn
                setattr(self.register, self.name, self)
                # register with model
                reg.lmo.append(self.full_name)
                reg.dmo.update({self.full_name: self})

        # add fullname to kwargs so it shows up in __repr__
        # its a dirty hack though
        self.kwargs["full_name"] = self.full_name
        logging.info(self.__repr__(1))

    def __validateinput__(self, kwargs: dict) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        """

        self.kwargs = kwargs  # store the kwargs
        self.provided_kwargs = kwargs.copy()  # preserve a copy

        if not hasattr(self, "lkk"):
            self.lkk: dict = {}
        if not hasattr(self, "lrk"):
            self.lrk: list = []
        if not hasattr(self, "lod"):
            self.lod: dict = []
        if not hasattr(self, "drn"):
            self.drn: dict = []

        # check that mandatory keys are present
        # and that all keys are allowed
        self.__checkkeys__()

        # initialize missing parameters

        self.kwargs = self.__addmissingdefaults__(self.lod, kwargs)

        # check if key values are of correct type
        self.__checktypes__(self.lkk, self.kwargs)

    def __checktypes__(self, av: dict[any, any], pv: dict[any, any]) -> None:
        """this method will use the the dict key in the user provided
        key value data (pv) to look up the allowed data type for this key in av

        av = dictinory with the allowed input keys and their type
        pv = dictionary with the user provided key-value data
        """

        k: any
        v: any

        # loop over known keywords in av
        for k, v in av.items():
            if k in pv:  # check type of k
                # check if entry matches required type
                if v != any:
                    # print()
                    if not isinstance(pv[k], v):
                        raise TypeError(
                            f"{type(pv[k])} is the wrong type for '{k}', should be '{av[k]}'"
                        )

    def __initerrormessages__(self):
        """Init the list of known error messages"""
        self.bem: dict = {
            "Number": "a number",
            "Model": "a model handle (i.e. the name without quotation marks)",
            "Element": "an element handle (i.e. the name without quotation marks)",
            "Species": "a species handle (i.e. the name without quotation marks)",
            "Flux": "a flux handle (i.e. the name without quotation marks)",
            "Reservoir": "a reservoir handle (i.e. the name without quotation marks)",
            "Signal": "a signal handle (i.e. the name without quotation marks)",
            "Process": "a process handle (i.e. the name without quotation marks)",
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
            "element": "an element handle (i.e. the name without quotation marks)",
            "species": "a species handle (i.e. the name without quotation marks)",
            "flux": "a flux handle (i.e. the name without quotation marks)",
            "reservoir": "a reservoir handle (i.e. the name without quotation marks)",
            "signal": "a signal handle (i.e. the name without quotation marks)",
            "Process": "a process handle (i.e. the name without quotation marks)",
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
        """register the kwargs key/value pairs as instance variables
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
        """check if the mandatory keys are present"""

        k: str
        v: any
        # test if the required keywords are given
        for k in self.lrk:  # loop over required keywords
            if isinstance(k, list):  # If keyword is a list
                s: int = 0  # loop over allowed substitutions
                for e in k:  # test how many matches are in this list
                    if e in self.kwargs:
                        # print(self.kwargs[e])
                        if not isinstance(e, (np.ndarray, np.float64, list)):
                            # print (type(self.kwargs[e]))
                            if self.kwargs[e] != "None":
                                s = s + 1
                if s > 1:  # if more than one match
                    raise ValueError(
                        f"You need to specify exactly one from this list: {k}"
                    )

            else:  # keyword is not a list
                if k not in self.kwargs:
                    raise ValueError(f"You need to specify a value for {k}")

        tl: List[str] = []
        # get a list of all known keywords
        for k, v in self.lkk.items():
            tl.append(k)

        # test if we know all keys
        # for k, v in self.kwargs.items():
        #     if k not in self.lkk:
        #         raise ValueError(f"{k} is not a valid keyword. \n Try any of \n {tl}\n")

    def __addmissingdefaults__(self, lod: dict, kwargs: dict) -> dict:
        """
        test if the keys in lod exist in kwargs, otherwise add them
        with the default values from lod

        """
        new: dict = {}
        if len(self.lod) > 0:
            for k, v in lod.items():
                if k not in kwargs:
                    new.update({k: v})

        kwargs.update(new)
        return kwargs

    def __repr__(self, log=0) -> str:
        """Print the basic parameters for this class when called via the print method"""
        from esbmtk import Q_

        m: str = ""

        # suppress output during object initialization
        tdiff = time.monotonic() - self.reg_time

        # do not echo input unless explicitly requestted

        m = f"{self.__class__.__name__}(\n"
        for k, v in self.kwargs.items():
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

    def __str__(self, kwargs={}):
        """Print the basic parameters for this class when called via the print method
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

        if "index" in kwargs:
            index = int(kwargs["index"])
        else:
            index = -2

        m = f"{ind}{self.name} ({self.__class__.__name__})\n"
        for k, v in self.kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    pass
                elif isinstance(v, str) and not (k == "name"):
                    m = m + f"{ind}{off}{k} = {v}\n"
                elif isinstance(v, Q_):
                    m = m + f"{ind}{off}{k} = {v}\n"
                elif isinstance(v, np.ndarray):
                    m = m + f"{ind}{off}{k}[{index}] = {v[index]:.2e}\n"
                elif k != "name":
                    m = m + f"{ind}{off}{k} = {v}\n"

        return m

    def __lt__(self, other) -> None:
        """This is needed for sorting with sorted()"""

        return self.n < other.n

    def __gt__(self, other) -> None:
        """This is needed for sorting with sorted()"""

        return self.n > other.n

    def info(self, **kwargs) -> None:
        """Show an overview of the object properties.
        Optional arguments are

        indent :int = 0 indentation

        """

        if "indent" not in kwargs:
            indent = 0
            ind = ""
        else:
            indent = kwargs["indent"]
            ind = " " * indent

        # print basic data bout this object
        print(f"{ind}{self.__str__(kwargs)}")

    def __aux_inits__(self) -> None:
        """Aux initialization code. Not normally used"""

        pass
