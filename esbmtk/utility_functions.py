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
from nptyping import NDArray
from typing import Dict, Union
from numpy import array, set_printoptions, arange, zeros, interp, mean
from pandas import DataFrame
from copy import deepcopy, copy
import time
from time import process_time
import numba
from numba.core import types
from numba import njit, prange
from numba.typed import List
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import mpmath

import logging
import time

# import builtins
# import math

set_printoptions(precision=4)


def add_to(l, e):
    """
    add element e to list l, but check if the entry already exist. If so, throw
    exception. Otherwise add
    """

    if not (e in l):  # if not present, append element
        l.append(e)


def get_plot_layout(obj):
    """Simple function which selects a row, column layout based on the number of
    objects to display.  The expected argument is a reservoir object which
    contains the list of fluxes in the reservoir

    """

    noo = 1  # the reservoir is the fisrt object
    for f in obj.lof:  # count numbert of fluxes
        if f.plot == "yes":
            noo += 1

    for d in obj.ldf:  # count number of data fields
        noo += 1

    # noo = len(obj.lof) + 1  # number of objects in this reservoir
    logging.debug(f"{noo} subplots for {obj.n} required")

    size, geo = plot_geometry(noo)

    return size, geo


def plot_geometry(noo: int) -> tuple():
    """Define plot geometry based on number of objects to plot"""

    if noo < 2:
        geo = [1, 1]  # one row, one column
        size = [5, 3]  # size in inches
    elif 1 < noo < 3:
        geo = [2, 1]  # two rows, one column
        size = [5, 6]  # size in inches
    elif 2 < noo < 5:
        geo = [2, 2]  # two rows, two columns
        size = [10, 6]  # size in inches
    elif 4 < noo < 7:
        geo = [2, 3]  # two rows, three columns
        size = [15, 6]  # size in inches
    elif 6 < noo < 10:
        geo = [3, 3]  # two rows, three columns
        size = [15, 9]  # size in inches
    elif 9 < noo < 13:
        geo = [4, 3]  # two rows, three columns
        size = [15, 12]  # size in inches
    elif 12 < noo < 16:
        geo = [5, 3]  # two rows, three columns
        size = [15, 15]  # size in inches
    else:
        m = (
            "plot geometry for more than 15 fluxes is not yet defined"
            "Consider calling flux.plot individually on each flux in the reservoir"
        )
        raise ValueError(m)

    return size, geo


def list_fluxes(self, name, i) -> None:
    """
    Echo all fluxes in the reservoir to the screen
    """
    print(f"\nList of fluxes in {self.n}:")

    for f in self.lof:  # show the processes
        direction = self.lio[f.n]
        if direction == -1:
            t1 = "From:"
            t2 = "Outflux from"
        else:
            t1 = "To  :"
            t2 = "Influx to"

        print(f"\t {t2} {self.n} via {f.n}")

        for p in f.lop:
            p.describe()

    print(" ")
    for f in self.lof:
        f.describe(i)  # print out the flux data


def show_data(self, **kwargs) -> None:
    """Print the 3 lines of the data starting with index

    Optional arguments:

    index :int = 0 starting index
    indent :int = 0 indentation
    """

    off: str = "  "

    if "index" not in kwargs:
        index = 0
    else:
        index = kwargs["index"]

    if "indent" in kwargs:
        ind: str = kwargs["indent"] * " "
    else:
        ind: str = ""

    # show the first 4 entries
    for i in range(index, index + 3):
        print(f"{off}{ind}i = {i}, Mass = {self.m[i]:.2e}, delta = {self.d[i]:.2f}")


def set_y_limits(ax: plt.Axes, obj: any) -> None:
    """Prevent the display or arbitrarily small differences"""
    lower: float
    upper: float

    bottom, top = ax.get_ylim()
    if (top - bottom) < obj.display_precision:
        top = bottom + obj.display_precision
        ax.set_ylim(bottom, top)


def get_ptype(obj, **kwargs: dict) -> int:
    """
    Set plot type variable based on ptype or isotope keyword

    """

    from esbmtk import Flux, Reservoir, Signal, DataField, Source, Sink

    ptype: int = 0

    if isinstance(obj, (Reservoir, Source, Sink, Flux)):
        if obj.isotopes:
            ptype = 0
        else:
            ptype = 2
    elif "ptype" in kwargs:
        if kwargs["ptype"] == "both":
            ptype = 0
        elif kwargs["ptype"] == "iso":
            ptype = 1
        elif kwargs["ptype"] == "concentration":
            ptype = 2
        elif kwargs["ptype"] == "mass_only":
            ptype = 2

    return ptype


def plot_object_data(geo: list, fn: int, obj: any) -> None:
    """collection of commands which will plotqand annotate a reservoir or flux
    object into an existing plot window.

    geo: geometry info
    fn: figure number in plot
    obj: the object to plot

    """

    from . import ureg, Q_
    from esbmtk import Flux, Reservoir, Signal, DataField, Source

    # geo = list with rows and cols
    # fn  = figure number
    # yl  = array with y values for the left side
    # yr  = array with y values for the right side
    # obj = object handle, i.e., reservoir or flux

    first_axis: bool = False
    second_axis: bool = False

    rows = geo[0]
    cols = geo[1]
    # species = obj.sp
    model = obj.mo
    time = model.time + model.offset

    # convert data from model units to display units (i.e. the same
    # units the input data was defined).
    # time units are the same regardless of object
    time = (time * model.t_unit).to(model.d_unit).magnitude

    # we do not map isotope values
    yr = obj.d

    # get plot type
    ptype: int = get_ptype(obj)

    # remap concentration & flux values
    if isinstance(obj, Flux):
        yl = (obj.m * model.f_unit).to(obj.plt_units).magnitude
        y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

    elif isinstance(obj, (Reservoir)):
        if obj.display_as == "mass":
            yl = (obj.m * model.m_unit).to(obj.plt_units).magnitude
            y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

        elif obj.plot_transform_c != "None":
            if callable(obj.plot_transform_c):
                # yl = (obj.m * model.m_unit).to(obj.plt_units).magnitude
                yl = obj.plot_transform_c(obj.c)
                y_label = f"{obj.legend_left}"
            else:
                raise ValueError("plot_transform_c must be function")

        else:
            yl = (obj.c * model.c_unit).to(obj.plt_units).magnitude
            y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

    elif isinstance(obj, Signal):
        # use the same units as the associated flux
        yl = (obj.data.m * model.f_unit).to(obj.data.plt_units).magnitude
        y_label = f"{obj.n} [{obj.data.plt_units:~P}]"

    elif isinstance(obj, DataField):
        time = (time * model.t_unit).to(model.d_unit).magnitude
        yl = obj.y1_data
        y_label = obj.y1_label
        if type(obj.y2_data) == str:
            ptype = 2
        else:
            ptype = 0

    else:  # sources, sinks, external data should not show up here
        raise ValueError(f"{obj.n} = {type(obj)}")

    # decide what to plot
    if ptype == 0:
        first_axis = True
        second_axis = True
    elif ptype == 1:
        first_axis = False
        second_axis = True
    elif ptype == 2:
        first_axis = True
        second_axis = False

    # start subplot
    ax1 = plt.subplot(rows, cols, fn)

    # set color index
    cn = 0
    col = f"C{cn}"

    if first_axis:
        if isinstance(obj, DataField):
            if not isinstance(obj.y1_data[0], str):
                for i, d in enumerate(obj.y1_data):  # loop over datafield list
                    yl = d
                    label = obj.y1_legend[i]
                    # print(f"label = {label}")
                    ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=label)
                    cn = cn + 1
                    col = f"C{cn}"

                ax1.set_xlabel(f"{model.time_label} [{model.d_unit:~P}]")
                ax1.set_ylabel(y_label)
                # remove unnecessary frame species
                ax1.spines["top"].set_visible(False)
                set_y_limits(ax1, obj)
                plt.legend()

        else:
            ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=y_label)
            cn = cn + 1
            col = f"C{cn}"

            ax1.set_xlabel(f"{model.time_label} [{model.d_unit:~P}]")
            ax1.set_ylabel(y_label)
            # remove unnecessary frame species
            ax1.spines["top"].set_visible(False)
            set_y_limits(ax1, obj)

    if second_axis:
        if isinstance(obj, DataField):
            if not isinstance(obj.y2_data[0], str):
                if obj.common_y_scale == "yes":
                    for i, d in enumerate(obj.y2_data):  # loop over datafield list
                        yl = d
                        label = obj.y2_legend[i]
                        ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=label)
                        cn = cn + 1
                        col = f"C{cn}"
                        set_y_limits(ax1, model)
                        ax1.legend()
                        second_axis = False
                else:
                    ax2 = ax1.twinx()  # create a second y-axis
                    for i, d in enumerate(obj.y2_data):  # loop over datafield list
                        yl = d
                        label = obj.y2_legend[i]
                        ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=label)
                        cn = cn + 1
                        col = f"C{cn}"

                    ax2.set_ylabel(obj.ld)  # species object delta label
                    set_y_limits(ax2, model)
                    # remove unneeded frame
                    ax2.spines["top"].set_visible(False)
            else:
                second_axis = False

        elif isinstance(obj, Signal):
            # use the same units as the associated flux
            ax2 = ax1.twinx()  # create a second y-axis
            # plof right y-scale data
            ln2 = ax2.plot(
                time[1:-2], obj.data.d[1:-2], color=col, label=obj.legend_right
            )
            ax2.set_ylabel(obj.data.ld)  # species object delta label
            set_y_limits(ax2, model)
            ax2.spines["top"].set_visible(False)  # remove unnecessary frame speciess
        else:
            ax2 = ax1.twinx()  # create a second y-axis
            # plof right y-scale data
            ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)
            ax2.set_ylabel(obj.ld)  # species object delta label
            set_y_limits(ax2, model)
            ax2.spines["top"].set_visible(False)  # remove unnecessary frame speciess

    # adjust display properties for title and legend

    if isinstance(obj, (Reservoir)):
        # ax1.set_title(obj.pt)
        ax1.set_title(obj.full_name)
    else:
        ax1.set_title(obj.full_name)

    plt.rcParams["axes.titlepad"] = 14  # offset title upwards
    plt.rcParams["legend.facecolor"] = "0.8"  # show a gray background
    plt.rcParams["legend.edgecolor"] = "0.8"  # make frame the same color
    plt.rcParams["legend.framealpha"] = 0.4  # set transparency

    for d in obj.led:  # loop over external data objects if present

        if isinstance(d.x[0], str):  # if string, something is off
            raise ValueError("No time axis in external data object {d.name}")
        if "y" in dir(d):  # mass or concentration data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{obj.lm} {d.legend}"
            ln3 = ax1.scatter(d.x[1:-2], d.y[1:-2], color=col, label=leg)
        if "z" in dir(d) and second_axis:  # isotope data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{d.legend}"
            ln3 = ax2.scatter(d.x, d.z, color=col, label=leg)

    # collect all labels and print them in one legend
    if first_axis:
        handler1, label1 = ax1.get_legend_handles_labels()
        plt.gca().spines["right"].set_visible(False)

    if second_axis:
        handler2, label2 = ax2.get_legend_handles_labels()

    if first_axis and second_axis:
        legend = ax2.legend(handler1 + handler2, label1 + label2, loc=0).set_zorder(6)
    # elif first_axis:
    #    legend = ax1.legend(handler1 + label1, loc=0).set_zorder(6)
    # elif second_axis:
    #   legend = ax2.legend(handler2 + label2, loc=0).set_zorder(6)

    # Matplotlib will show arbitrarily small differences which can be confusing
    # yl_min = min(yl)
    # yl_max = max(yl)
    # if (yl_max - yl_min) < 0.1:


def is_name_in_list(n: str, l: list) -> bool:
    """Test if an object name is part of the object list"""

    r: bool = False
    for e in l:
        if e.full_name == n:
            r = True
    return r


def get_object_from_list(name: str, l: list) -> any:
    """Match a name to a list of objects. Return the object"""

    match: bool = False
    for o in l:
        if o.full_name == name:
            r = o
            match = True

    if match:
        return r
    else:
        raise ValueError(f"Object = {o.full_name} has no matching flux {name}")


def sort_by_type(l: list, t: list, m: str) -> list:
    """divide a list by type into new lists. This function will return a
    list and it is up to the calling code to unpack the list

    l is list with various object types
    t is a list which contains the object types used for sorting
    m is a string for the error function
    """

    # from numbers import Number

    lc = l.copy()
    rl = []

    for ot in t:  # loop over object types
        a = []
        for e in l:  # loop over list elements
            if isinstance(e, ot):
                a.append(e)  # add to temporary list
                lc.remove(e)  # remove this element

        rl.append(a)  # save the temporary list to rl

    # at this point, all elements of lc should have been processed
    # if not, lc contains element which are of a different type
    if len(lc) > 0:
        raise TypeError(m)

    return rl


def split_key(k: str, M: any) -> Union[any, any, str]:
    """split the string k with letter 2, and test if optional
    id string is present

    """

    if "2" in k:
        source = k.split("2")[0]
        sinkandid = k.split("2")[1]
    else:
        raise ValueError("Name must follow 'Source2Sink' format")

    if "@" in sinkandid:
        sink = sinkandid.split("@")[0]
        cid = sinkandid.split("@")[1]
    else:
        sink = sinkandid
        cid = ""

    sink = M.dmo[sink]
    source = M.dmo[source]
    return (source, sink, cid)


def make_dict(keys: list, values: list) -> dict:
    """Create a dictionary from a list and value, or from
    two lists

    """
    if isinstance(values, list):
        if len(values) == len(keys):
            d: dict = dict(zip(keys, values))
        else:
            print(f"len values ={len(values)}, len keys ={len(keys)}")
            print(f"values = {values}")
            for k in keys:
                print(f"key = {k}")
            raise ValueError(f"key and value list must be of equal length")
    else:
        values: list = [values] * len(keys)
        d: dict = dict(zip(keys, values))

    return d


def get_typed_list(data: list) -> list:

    tl = List()
    for x in data:
        tl.append(x)
    return tl


def create_reservoirs(bn: dict, ic: dict, M: any, cs: bool = False) -> dict:
    """boxes are defined by area and depth interval here we use an ordered
    dictionary to define the box geometries. The next column is temperature
    in deg C, followed by pressure in bar
    the geometry is [upper depth datum, lower depth datum, area percentage]

    bn = dictionary with box parameters
    bn: dict = {  # name: [[geometry], T, P]
                 "sb": {"g": [0, 200, 0.9], "T": 20, "P": 5},
                 "ib": {"g": [200, 1200, 1], "T": 10, "P": 100},
                }

    ic = dictionary with species default values. This is used to et up
         initial conditions. Here we use shortcut and use the same conditions
         in each box. If you need box specific initial conditions
         use the output of build_concentration_dicts as starting point


    ic: dict = { # species: concentration, Isotopes
                   PO4: [Q_("2.1 * umol/liter"), False],
                   DIC: [Q_("2.1 mmol/liter"), False],
                   ALK: [Q_("2.43 mmol/liter"), False],
               }

    M: Model object handle

    cs: add virtual reservoir for the carbonate system. Defaults to False

    """

    from esbmtk import SeawaterConstants, ReservoirGroup, build_concentration_dicts
    from esbmtk import SourceGroup, SinkGroup, carbonate_system, Q_

    # parse for sources and sinks, create these and remove them from the list

    # setup the remaining boxes
    # icd: dict = build_concentration_dicts(ic, bn)

    # loop over reservoir names
    for k, v in bn.items():
        if "ty" in v:  # type is given
            if v["ty"] == "Source":
                SourceGroup(name=k, species=v["sp"])
            elif v["ty"] == "Sink":
                SinkGroup(name=k, species=v["sp"])
            else:
                raise ValueError("'ty' must be either Source or Sink")

        else:  # create reservoirs
            icd: dict = build_concentration_dicts(ic, k)
            swc = SeawaterConstants(
                name=f"SW_{k}",
                model=M,
                temperature=v["T"],
                pressure=v["P"],
            )

            rg = ReservoirGroup(
                name=k,
                geometry=v["g"],
                concentration=icd[k][0],
                isotopes=icd[k][1],
                delta=icd[k][2],
            )

            if cs:
                volume = Q_(f"{rg.lor[0].volume} l")
                carbonate_system(
                    Q_(f"{swc.ca} mol/l"),
                    Q_(f"{swc.hplus} mol/l"),
                    volume,
                    swc,
                    rg,
                )

    return icd


def build_concentration_dicts(cd: dict, bg: dict) -> dict:
    """Build a dict which can be used by create_reservoirs

    bg : dict where the box_names are dict keys.
    cd: dictionary with the following format:
        cd = {
             # species: [concentration, isotopes]
             PO4: [Q_("2.1 * umol/liter"), False],
             DIC: [Q_("2.1 mmol/liter"), False],
            }

    This function returns a new dict in the following format

    #  box_names: [concentrations, isotopes]
    d= {"bn": [{PO4: .., DIC: ..},{PO4:False, DIC:False}]}

    """

    if isinstance(bg, dict):
        box_names: list = bg.keys()
    elif isinstance(bg, list):
        box_names: list = bg
    elif isinstance(bg, str):
        box_names: list = [bg]
    else:
        raise ValueError("This should never happen")

    icd: dict = OrderedDict()
    td1: dict = {}  # temp dictionary
    td2: dict = {}  # temp dictionary
    td3: dict = {}  # temp dictionary

    # create the dicts for concentration and isotopes
    for k, v in cd.items():
        td1.update({k: v[0]})
        td2.update({k: v[1]})
        td3.update({k: v[2]})

    # box_names: list = bg.keys()
    for bn in box_names:  # loop over box names
        icd.update({bn: [td1, td2, td3]})

    return icd


def calc_volumes(bg: dict, M: any, h: any) -> list:
    """Calculate volume contained in a given depth interval
    bg is an ordered dictionary in the following format

    bg=  {
          "hb": (0.1, 0, 200),
          "sb": (0.9, 0, 200),
         }

    where the key must be a valid box name, the first entry of the list denoted
    the areal extent in percent, the second number is upper depth limit, and last
    number is the lower depth limit.

    M must be a model handle
    h is the hypsometry handle

    The function returns a list with the corresponding volumes

    """

    # from esbmtk import hypsometry

    v: list = []  # list of volumes

    for k, v in bg.items():
        a = v[0]
        u = v[1]
        l = v[2]

        v.append(h.volume(u, l) * a)

    return v


def get_longest_dict_entry(d: dict) -> int:
    """Get length of each item in the connection dict"""
    l_length = 0  # length of  longest list
    p_length = 0  # length of single parameter
    nl = 0  # number of lists
    ll = []

    # we need to cover the case where we have two lists of different length
    # this happens if we have a long list of tuples with matched references,
    # as well as a list of species
    for k, v in d.items():
        if isinstance(v, list):
            nl = nl + 1
            if len(v) > l_length:
                l_length = len(v)
                ll.append(l_length)

        else:
            p_length = 1

    if nl > 1:
        # if lists have different lengths
        if ll.count(l_length) != len(ll):
            raise ValueError("Mapping for multiple lists is not supported")

    if l_length > 0 and p_length == 0:
        case = 0  # Only lists present
    if l_length == 0 and p_length == 1:
        case = 1  #  Only parameters present
    if l_length > 0 and p_length == 1:
        case = 2  #  Lists and parameters present

    return case, l_length


def convert_to_lists(d: dict, l: int) -> dict:
    """expand mixed dict entries (i.e. list and single value) such
    that they are all lists of equal length

    """
    cd = d.copy()

    for k, v in cd.items():
        if not isinstance(v, list):
            p = []
            for i in range(l):
                p.append(v)
            d[k] = p

    return d


def get_sub_key(d: dict, i: int) -> dict:
    """take a dict which has where the value is a list, and return the
    key with the n-th value of that list

    """

    rd: dict = {}
    for k, v in d.items():
        rd[k] = v[i]

    return rd


def expand_dict(d: dict, mt: str = "1:1") -> int:

    """Determine dict structure

    in case we have mutiple connections with mutiple species, the
    default action is to map connections to species (t = '1:1'). If
    you rather want to create mutiple connections (one for each
    species) in each connection set t = '1:N'

    """
    # loop over dict entries
    # ck = connection key
    # cd = connection dict

    r: dict = {}  # the dict we will return

    for ck, cd in d.items():  # loop over connections

        # print(f"ck = {ck}")
        # print(f"cd = {cd}")

        rd: dict = {}  # temp dict
        nd: dict = {}  # temp dict
        case, length = get_longest_dict_entry(cd)

        # print(f"length = {length}")

        if isinstance(ck, tuple):
            # assume 1:1 mapping between tuple and connection parameters
            if mt == "1:1":

                # prep dictionaries
                if case == 0:
                    nd = cd
                    # print("case 0")
                elif case == 1:  # only parameters present. Expand for each tuple entry
                    length = len(ck)
                    nd = convert_to_lists(cd, length)
                    # print("case 1")
                elif case == 2:  # mixed list present, Expand list
                    # print(f"case 2, length = {length}")
                    nd = convert_to_lists(cd, length)
                    # print(nd)

                # for each connection group in the tuple
                if length != len(ck):
                    message = (
                        f"The number of connection properties ({length})\n"
                        f"does not match the number of connection groups ({len(ck)})\n"
                        f"did you intend to do a 1:N mapping?"
                    )
                    raise ValueError(message)

                # map property dicts to connection group names
                i = 0
                for t in ck:
                    rd[t] = get_sub_key(nd, i)
                    i = i + 1

            elif mt == "1:N":  # apply each species to each connection
                if case == 0:
                    nd = cd
                elif case == 1:  # only parameters present. Expand for each tuple entry
                    length = len(ck)
                    nd = convert_to_lists(cd, length)
                elif case == 2:  # mixed list present, Expand list
                    nd = convert_to_lists(cd, length)

                for t in ck:  # apply the entire nd dict to all connections
                    rd[t] = nd
            else:
                raise ValueError(f"{mt} is not defined. must be '1:1' or '1:N'")

        else:
            if case == 0:  # only lists present, case 3
                nd = cd
            elif case == 1:  # only parameters present
                nd = cd
            elif case == 2:  # list and parameters present case 4
                nd = convert_to_lists(cd, length)
            rd[ck] = nd

        # update the overall dict and move to the next entry
        r.update(rd)

    return r


def create_bulk_connections(ct: dict, M: any, mt: int = "1:1") -> None:
    """Create connections from a dictionary. The dict can have the following keys
    following format:

    mt = mapping type. See below for explanation

    # na: names, tuple or str. If lists, all list elements share the same properties
    # sp: species list or species
    # ty: type, str
    # ra: rate, Quantity
    # sc: scale, Number
    # re: reference, optional
    # al: alpha, optional
    # de: delta, optional
    # mx: True, optional defaults to False. If set, it will create forward
          and backward fluxes (i.e. mixing)

    There are 6 different cases how to specify connections

    Case 1 One connection, one set of parameters
           ct1 = {"sb2hb": {"ty": "scale", 'ra'....}}

    Case 2 One connection, one set of instructions, one subset with mutiple parameters
           This will be expanded to create connections for each species
           ct2 = {"sb2hb": {"ty": "scale", "sp": ["a", "b"]}}

    Case 3 One connection complete set of mutiple characters. Similar to case 2, but now
           all parameters are given explicitly
           ct3 = {"sb2hb": {"ty": ["scale", "scale"], "sp": ["a", "b"]}}

    Case 4 Mutiple connections, one set of parameters. This will create
           identical connection for "sb2hb" and  "ib2db"
           ct4 = {("sb2hb", "ib2db"): {"ty": "scale", 'ra': ...}}

    Case 5 Mutiple connections, one subset of mutiple set of parameters. This wil
          create a connection for species 'a' in sb2hb and with species 'b' in ib2db
           ct5 = {("sb2hb", "ib2db"): {"ty": "scale", "sp": ["a", "b"]}}

    Case 6 Mutiple connections, complete set of parameters of mutiple parameters
           Same as case 5, but now all parameters are specified explicitly
           ct6 = {("sb2hb", "ib2db"): {"ty": ["scale", "scale"], "sp": ["a", "b"]}}


    The default interpretation for cases 5 and 6 is that each list
    entry corresponds to connection. However, sometimes we want to
    create mutiple connections for multiple entries. In this case
    provide the mt='1:N' parameter which will create a connection for
    each species in each connection group. See the below example.

    It is easy to shoot yourself in the foot. It is best to try the above first with
    some simple examples, e.g.,

    from esbmtk import expand_dict
    ct2 = {"sb2hb": {"ty": "scale", "sp": ["a", "b"]}}

    It is best to use the show_dict function to verify that your input
    dictionary produces the corrrect results!

    """

    from esbmtk import create_connection, expand_dict

    # expand dictionary into a well formed dict where each connection
    # has a fully formed entry
    c_ct = expand_dict(ct, mt=mt)

    # loop over dict entries and create the respective connections
    for k, v in c_ct.items():
        if isinstance(k, tuple):
            # loop over names in tuple
            for c in k:
                create_connection(c, v, M)
        elif isinstance(k, str):
            create_connection(k, v, M)
        else:
            raise ValueError(f"{connection} must be string or tuple")

    return ct


def create_connection(n: str, p: dict, M: any) -> None:

    """called by create_bulk_connections in order to create a connection
    group It is assumed that all rates are in liter/year or mol per
    year. This may not be what you want or need.

    You need to provide a connection key e.g., sb2db@mix which will be
    interpreted as mixing a connection between sb and db and thus
    create connections in both directions

    """

    from esbmtk import ConnectionGroup, Q_

    # get the reservoir handles by splitting the key
    source, sink, cid = split_key(n, M)

    # create default connections parameters and replace with values in
    # the parameter dict if present.
    los = list(p["sp"]) if isinstance(p["sp"], list) else [p["sp"]]
    typ = "None" if not "ty" in p else p["ty"]
    scale = 1 if not "sc" in p else p["sc"]
    rate = Q_("0 mol/a") if not "ra" in p else p["ra"]
    ref_reservoirs = "None" if not "re" in p else p["re"]
    alpha = "None" if not "al" in p else p["al"]
    delta = "None" if not "de" in p else p["de"]
    mix = False if not "mx" in p else p["mx"]
    cid = f"{cid}_f" if mix else f"{cid}"

    if isinstance(scale, Q_):
        scale = scale.to("l/a").magnitude

    # if name in M.dmo: # group already exist in this case we nee to update the
    # connection group

    # Case one, no mixing, test if exist, otherwise create
    # Case 2, mixing: test if forward connection exists, if not create
    #                  test if backwards connection exists, or create

    if not mix:
        name = f"C_{source.name}2{sink.name}"
        update_or_create(
            name,
            source,
            sink,
            los,
            typ,
            scale,
            rate,
            ref_reservoirs,
            alpha,
            delta,
            cid,
            M,
        )

    else:  # this is a connection with mixing
        # create forward connection
        name = f"C_{source.name}2{sink.name}"
        update_or_create(
            name,
            source,
            sink,
            los,
            typ,
            scale,
            rate,
            ref_reservoirs,
            alpha,
            delta,
            cid,
            M,
        )

        # create backwards connection
        name = f"C_{sink.name}2{source.name}"
        cid = cid.replace("_f", "_b")
        update_or_create(
            name,
            sink,
            source,
            los,
            typ,
            scale,
            rate,
            ref_reservoirs,
            alpha,
            delta,
            cid,
            M,
        )


def update_or_create(
    name, source, sink, los, typ, scale, rate, ref_reservoirs, alpha, delta, cid, M
):
    """Create or update connection"""

    from esbmtk import ConnectionGroup

    if name in M.dmo:  # update connection
        cg = M.dmo[name]
        cg.update(
            name=name,
            source=source,
            sink=sink,
            ctype=make_dict(los, typ),
            scale=make_dict(los, scale),  # get rate from dictionary
            rate=make_dict(los, rate),
            ref_reservoirs=make_dict(los, ref_reservoirs),
            alpha=make_dict(los, alpha),
            delta=make_dict(los, delta),
            id=cid,  # get id from dictionary
        )
    else:  # create connection
        cg = ConnectionGroup(
            name=name,
            source=source,
            sink=sink,
            ctype=make_dict(los, typ),
            scale=make_dict(los, scale),  # get rate from dictionary
            rate=make_dict(los, rate),
            ref_reservoirs=make_dict(los, ref_reservoirs),
            alpha=make_dict(los, alpha),
            delta=make_dict(los, delta),
            id=cid,  # get id from dictionary
        )


def get_name_only(o: any) -> any:
    """Test if item is an esbmtk type. If yes, extract the name"""

    from esbmtk import Flux, Reservoir, ReservoirGroup, Species
    from esbmtk import Sink, Source, SourceGroup, SinkGroup
    from esbmtk import Process, DataField, VirtualReservoir

    if isinstance(o, (Flux, Reservoir, ReservoirGroup, Species)):
        r = o.full_name
    else:
        r = o

    return r


def get_simple_list(l: list) -> list:
    """return a list which only has the full name
    rather than all the object properties

    """

    from esbmtk import Flux, Reservoir, Species

    r: list = []
    for e in l:
        r.append(get_name_only(e))

    return r


def show_dict(d: dict, mt: str = "1:1") -> None:
    """show dict entries in an organized manner"""

    from esbmtk import expand_dict, get_simple_list, get_name_only

    ct = expand_dict(d, mt)
    for ck, cv in ct.items():
        print(f"{ck}")

        for pk, pv in cv.items():
            if isinstance(pv, list):
                x = get_simple_list(pv)
            else:
                x = get_name_only(pv)
            print(f"     {pk} : {x}")


def find_matching_fluxes(M: any, filter_by: str) -> list:
    """Loop over all reservoir, and extract the names of all fluxes
    which match the filter string. Return the list of names (not objects!)

    """

    lof: set = set()

    for r in M.loc:
        for f in r.lof:
            if filter_by in f.full_name:
                lof.add(f)

    return list(lof)


def reverse_key(key: str) -> str:
    """ reverse a connection key e.g., sb2db@POM becomes db2sb@POM """

    # print(f"key = {key}")
    l = key.split("@")
    left = l[0]
    # right = l[1]
    rs = left.split("2")
    r1 = rs[0]
    r2 = rs[1]

    return f"{r2}2{r1}"


def get_connection_keys(s: set, fstr: str, nstr: str, inverse: bool) -> list:
    """extract connection keys from set of flux names, replace fstr with
    nstr so that the key can be used in create_bulk_connnections()

    The optional inverse parameter, can be used where in cases where the
    flux direction needs to be reversed, i.e., the returned key will not read
    sb2db@POM, but db2s@POM

    """

    cl: list = []

    for n in s:
        # get connection and flux name
        l = n.full_name.split(".")
        cn = l[0][2:]  # get key without leadinf C_
        if inverse:
            cn = reverse_key(cn)
        cn.replace(fstr, nstr)
        cn = f"{cn}@{nstr}"
        cl.append(cn)

    return cl


def gen_dict_entries(M: any, **kwargs) -> tuple:
    """find all fluxes which contain the reference string, and create
    matching keys which contain the target string. The function will
    return two lists, which can be used to create a dict for the
    create_bulk_connnection function.

    E.g., to create a dict which will create new fluxes based on
    existing fluxes

    dk:list, fl:list = gen_dict_entries(ref_id = 'POP', target_id = 'POM')

    this will find all fluxes with the POP-id and put these into
    fl. It will also generate a list with suitable connections keys
    (dk) which contain the 'POM' id.

    The optional inverse parameter, can be used where in cases where the
    flux direction needs to be reversed, i.e., the returned key will not read
    sb2db@POM, but db2s@POM

    """

    reference = kwargs["ref_id"]
    target = kwargs["target_id"]
    if "inverse" in kwargs:
        inverse = kwargs["inverse"]
    else:
        inverse = False

    flist: list = find_matching_fluxes(M, filter_by=reference)
    klist: list = get_connection_keys(flist, reference, target, inverse)

    return tuple(klist), flist


def build_ct_dict(d: dict, p: dict) -> dict:
    """build a connection dictionary from a dict containing connection
    keys, and a dict containing connection properties. This is most
    useful for connections which a characterized by a fixed rate but
    apply to many species. E.g., mixing fluxes in a complex model etc.

    """

    ct: dict = {}

    for k, v in d.items():
        td: dict = {}  # temp dict for scale
        td["sc"] = v
        td.update(p)
        ct[k] = td

    return ct


def get_string_between_brackets(s: str) -> str:
    """Parse string and extract substring between square brackets"""

    s = s.split("[")
    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    s = s[1]

    s = s.split("]")

    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    return s[0]


def map_units(v: any, *args) -> float:
    """parse v to see if it is a string. if yes, map to quantity.
    parse v to see if it is a quantity, if yes, map to model units
    and extract magnitude, assign mangitude to return value
    if not, assign value to return value

    v : a keyword value number/string/quantity
    args: one or more quantities (units) see the Model class (e.g., f_unit)

    """

    from . import Q_

    m: float = 0
    match: bool = False

    # test if string, map to quantity if yes
    if isinstance(v, str):
        v = Q_(v)

    # test if we find a matching dimension, map if true
    if isinstance(v, Q_):
        for q in args:
            if v.dimensionality == q.dimensionality:
                m = v.to(q).magnitude
                match = True

        if not match:
            message = f"{v} is none of {print(*args)}"
            raise ValueError(message)

    else:  # no quantity, so it should be a number
        m = v

    if not isinstance(m, Number):
        raise ValueError(f"m is {type(m)}, must be float, v={v}. Something is fishy")

    return m
