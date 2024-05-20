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
import numpy as np
import numpy.typing as npt
import typing as tp

if tp.TYPE_CHECKING:
    from .esbmtk import SpeciesProperties

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


class KeywordError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class MissingKeywordError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class InputError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class FluxSpecificationError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


class SpeciesPropertiesMolweightError(Exception):
    def __init__(self, message):
        message = f"\n\n{message}\n"
        super().__init__(message)


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

        # import copy

        self.update = False
        # self.defaults_copy = copy.deepcopy(self.defaults)
        # self.defaults_copy = {**self.defaults} # probably the wrong place.
        self.__check_mandatory_keywords__(self.lrk, kwargs)
        self.__register_variable_names__(self.defaults, kwargs)
        self.__update_dict_entries__(self.defaults, kwargs)
        self.update = True

    def __check_mandatory_keywords__(self, lrk: tp.List, kwargs: dict) -> None:
        """Verify that all elements of lrk have a corresponding key in
        kwargs.  If not, print error message"""

        for key in lrk:
            if isinstance(key, list):
                has_key = sum(k in kwargs and kwargs[k] != "None" for k in key)
                if has_key != 1:
                    raise ValueError(f"give only one of {key}")
            elif key not in kwargs:
                raise MissingKeywordError(f"{key} is a mandatory keyword")

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
            setattr(self, f"_{key}", value[0])
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
                raise KeywordError(f"{key} is not a valid keyword")

            if not isinstance(value, defaults[key][1]):
                raise InputError(
                    f"{value} for {key} must be of type {defaults[key][1]}, not {type(value)}"
                )

            defaults[key][0] = value  # update defaults dictionary
            setattr(self, key, value)  # update instance variables
            setattr(self, f"_{key}", value)

    def __register_name_new__(self) -> None:
        """if self.parent is set, register self as attribute of self.parent,
        and set full name to parent.full-name + self.name
        if self.parent == "None", full_name = name
        """

        if self.parent == "None":
            self.full_name = self.name
            reg = self
        else:
            self.full_name = f"{self.parent.full_name}.{self.name}"
            reg = self.parent.model
            # self.full_name, reg.lmo = self.__test_and_resolve_duplicates__(self.full_name, reg.lmo)
            if self.full_name in reg.lmo:
                print("\n -------------- Warning ------------- \n")
                print(f"\t {self.full_name} is a duplicate name in reg.lmo")
                print("\n ---------------------- ------------- \n")
                # raise NameError(f"{self.full_name} is a duplicate name in reg.lmo")
            # register with model
            reg.lmo.append(self.full_name)
            reg.lmo2.append(self)
            reg.dmo.update({self.full_name: self})
            setattr(self.parent, self.name, self)
            self.kwargs["full_name"] = self.full_name
        self.reg_time = time.monotonic()

    def __test_and_resolve_duplicates__(self, name, lmo):
        if name in lmo:
            print(f"\n Warning, {name} is a duplicate, trying with {name}_1\n")
            name = name + "_1"
            name, lmo = self.__test_and_resolve_duplicates__(name, lmo)
        else:
            lmo.append(name)

        return name, lmo


class esbmtkBase(input_parsing):
    """The esbmtk base class template. This class handles keyword
    arguments, name registration and other common tasks

    Useful methods in this class:

    define required keywords in lrk dict:
       self.lrk: tp.List = ["name"]

    define allowed type per keyword in lkk dict:
       self.defaults: dict[str, list[any, tuple]] = {
                                  "name": ["None", (str)],
                                  "model": ["None",(str, Model)],
                                  "salinity": [35, (int, float)], # int or float
                                  }

    parse and register all keywords with the instance
    self.__initialize_keyword_variables__(kwargs)

    register the instance
    self.__register_name_new__ ()

    """

    def __init__(self) -> None:
        raise NotImplementedError

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
                    m = f"{m}    {k} = {v.name},\n"
                elif isinstance(v, str):
                    m = f"{m}    {k} = '{v}',\n"
                elif isinstance(v, Q_):
                    m = f"{m}    {k} = '{v}',\n"
                elif isinstance(v, (list, np.ndarray)):
                    m = f"{m}    {k} = '{v[:3]}',\n"
                else:
                    m = f"{m}    {k} = {v},\n"

        m = "" if log == 0 and tdiff < 1 else f"{m})"
        return m

    def __str__(self, kwargs=None):
        """Print the basic parameters for this class when called via the print method
        Optional arguments

        indent :int = 0 printing offset

        """
        if kwargs is None:
            kwargs = {}
        from esbmtk import Q_

        m: str = ""
        off: str = "  "

        ind: str = kwargs["indent"] * " " if "indent" in kwargs else ""
        index = int(kwargs["index"]) if "index" in kwargs else -2
        m = f"{ind}{self.name} ({self.__class__.__name__})\n"
        for k, v in self.kwargs.items():
            if not isinstance({k}, esbmtkBase):
                # check if this is not another esbmtk object
                if "esbmtk" in str(type(v)):
                    pass
                elif isinstance(v, str) and k != "name":
                    m = f"{m}{ind}{off}{k} = {v}\n"
                elif isinstance(v, Q_):
                    m = f"{m}{ind}{off}{k} = {v}\n"
                elif isinstance(v, np.ndarray):
                    m = f"{m}{ind}{off}{k}[{index}] = {v[index]:.2e}\n"
                elif k != "name":
                    m = f"{m}{ind}{off}{k} = {v}\n"

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

    def ensure_q(self, arg):
        """Test that a given input argument is a quantity. If not convert
        into quantity
        """
        from esbmtk import Q_

        if isinstance(arg, Q_):
            pass
        elif isinstance(arg, str):
            arg = Q_(arg)
        else:
            raise InputError(f"{arg} must be string or Quantity, not {type(arg)}")

        return arg

    def help(self) -> None:
        """Show all keywords, their fdefault values and allowed types."""
        print(f"\n{self.full_name} has the following keywords:\n")
        for k, v in self.defaults_copy.items():
            print(f"{k} defaults to {v[0]}, allowed types = {v[1]}")
        print()
        print("The following keywords are mandatory:")
        for kw in self.lrk:
            print(f"{kw}")

    def set_flux(self, mass: str, time: str, substance: SpeciesProperties):
        """
        set_flux converts() a flux rate that is specified as rate, time, substance
        so that it matches the correct model units (i.e., kg/s or mol/s)

        Example:

        .. code-block:: python

           M.set_flux("12 Tmol", "year", M.C)

        if model mass units are in mol, no change will be made 
        if model mass units are in kg, the above will return kg C/a (and vice verso)

        :param mass: e.g., "12 Tmol"
        :param time: e.g., "year"
        :param substance: e.g., SpeciesProperties Instance e.g., M.PO4

        :returns: mol/year or g/year

        :raises: FluxSpecificationError
        :raises: SpeciesPropertiesMolweightError
    """
        from esbmtk import Q_, ureg

        if substance.m_weight > 0:
            mass = Q_(mass)
            g_per_mol = ureg("g/mol")
            if mass.is_compatible_with("g") or mass.is_compatible_with("mol"):
                r = mass.to(
                    substance.mo.m_unit,  # mol
                    "chemistry",  # context
                    mw=substance.m_weight * g_per_mol,  # g/mol
                )
            else:
                message = f"no known conversion for {mass}, and {substance}"
                raise FluxSpecificationError(message)
        else:
            message = f"no mol weight defintion for {substance.full_name}"
            raise SpeciesPropertiesMolweightError(message)

        return r / ureg(time)

    def __update_ode_constants__(self, value) -> None:
        """Place the value of self.name onto the global parameter list
        store the index position and advance the index pointer

        :param name: value for the parameter list
        """
        if value != "None":
            self.model.toc = (*self.model.toc, value)
            index = self.model.gcc
            self.model.gcc = self.model.gcc + 1
        else:
            index = 0

        return index
