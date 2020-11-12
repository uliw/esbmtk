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
from .esbmtk import *

class esbmtkBase():
    """The esbmtk base class template. This class handles keyword
    arguments, name registration and other common tasks

    """

    from typing import Dict
    
    def __init__(self) -> None:
        raise NotImplementedError

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

        # register all key/value pairs as instance variables
        self.__registerkeys__()

        # register instance name in global name space
        self.reg_time = time.monotonic()
        setattr(builtins, self.name, self)

    def __validateinput__(self, kwargs: Dict[str, any]) -> None:
        """Validate the user provided input key-value pairs. For this we need
        kwargs = dictionary with the user provided key-value pairs
        self.lkk = dictionary with allowed keys and type
        self.lrk = list of mandatory keywords
        self.lod = dictionary of default values for keys

        """

        self.kwargs = kwargs  # store the kwargs
        self.provided_kwargs = kwargs.copy()  # preserve a copy

        if not self.lkk:  #dictionary with allowed keys and type
            self.lkk: Dict[str, any] = {}
        if not self.lrk:  # list with mandatory keywords
            self.lrk: List[str] = []
        if not self.lod:  # dictionary of default values for keys
            self.lod: Dict[str, any] = []

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
            if not isinstance(v, av[k]):
                raise TypeError(f"{type(v)} is the wrong type for '{k}', should be '{av[k]}'")

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
            str: "a string with quotation marks",
        }

    def __registerkeys__(self) -> None:
        """ register the kwargs key/value pairs as instance variables
        and complain about uknown keywords"""
        k: any
        v: any

        for k, v in self.kwargs.items():
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

    def __repr__(self):
        """ Print the basic parameters for this class when called via the print method
        
        """
        
        m: str = ""

        # suppress output during object initialization 
        tdiff = time.monotonic() -self.reg_time
        if tdiff > 1:
            for k, v in self.provided_kwargs.items():
                if not isinstance({k}, esbmtkBase):
                    m = m + f"{k} = {v}\n"
                    
        return m
