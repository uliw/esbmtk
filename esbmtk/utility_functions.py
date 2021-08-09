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


from esbmtk import Q_


# import builtins
# import math

set_printoptions(precision=4)


def insert_into_namespace(name, value, name_space=globals()):
    name_space[name] = value


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
        # time = (time * model.t_unit).to(model.d_unit).magnitude
        # yl = obj.y1_data
        # y_label = obj.y1_label
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
            y_label = obj.y1_label
            if not isinstance(obj.y1_data[0], str):
                for i, d in enumerate(obj.y1_data):  # loop over datafield list
                    y1_legend = obj.y1_legend[i]
                    # print(f"label = {y1_legend}")
                    ln1 = ax1.plot(
                        obj.x1_data[i], obj.y1_data[i], color=col, label=y1_legend
                    )
                    cn = cn + 1
                    col = f"C{cn}"

                ax1.set_xlabel(f"{model.time_label} [{model.d_unit:~P}]")
                ax1.set_ylabel(y_label)
                # remove unnecessary frame species
                ax1.spines["top"].set_visible(False)
                # set_y_limits(ax1, obj)
                # plt.legend()

        else:
            ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=y_label)
            cn = cn + 1
            col = f"C{cn}"

            ax1.set_xlabel(f"{model.time_label} [{model.d_unit:~P}]")
            ax1.set_ylabel(y_label)
            # remove unnecessary frame species
            ax1.spines["top"].set_visible(False)
            # set_y_limits(ax1, obj)

    if second_axis:
        if isinstance(obj, DataField):
            y_label = obj.y2_label
            if not isinstance(obj.y2_data[0], str):
                if obj.common_y_scale == "yes":
                    for i, d in enumerate(obj.y2_data):  # loop over datafield list
                        y2_legend = obj.y2_legend[i]
                        ln1 = ax1.plot(
                            obj.x2_data[i], obj.y2_data[i], color=col, label=y2_legend
                        )
                        cn = cn + 1
                        col = f"C{cn}"
                        # set_y_limits(ax1, model)
                        # ax1.legend()
                        second_axis = False
                else:
                    ax2 = ax1.twinx()  # create a second y-axis
                    for i, d in enumerate(obj.y2_data):  # loop over datafield list
                        y2_legend = obj.y2_legend[i]
                        ln1 = ax2.plot(
                            obj.x2_data[i], obj.y2_data[i], color=col, label=y2_legend
                        )
                        cn = cn + 1
                        col = f"C{cn}"

                    ax2.set_ylabel(obj.y2_label)  # species object delta label
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
        elif isinstance(obj, Reservoir):
            ax2 = ax1.twinx()  # create a second y-axis
            # plof right y-scale data
            ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)
            ax2.set_ylabel(obj.ld)  # species object delta label
            set_y_limits(ax2, model)
            ax2.spines["top"].set_visible(False)  # remove unnecessary frame species

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
    elif first_axis:
        legend = ax1.legend(handler1, label1, loc=0).set_zorder(6)
    elif second_axis:
        legend = ax2.legend(handler2, label2, loc=0).set_zorder(6)
    else:
        raise TypeError("This should never happen!")


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


def get_object_handle(res, M):
    """Test if we the key is a global reservoir handle
    or exists in the model namespace

    res: list, str, or reservoir handle
    M: Model handle
    """

    rlist: list = []

    if not isinstance(res, list):
        res = [res]

    for o in res:
        if o in M.dmo:  # is object known in global namespace
            rlist.append(M.dmo[o])
        elif o in M.__dict__:  # or does it exist in Model namespace
            rlist.append(getattr(M, o))
        else:
            raise ValueError(f"{o} is not known for model {M.name}")

    if len(rlist) == 1:
        rlist = rlist[0]

    return rlist


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

    sink = get_object_handle(sink, M)
    source = get_object_handle(source, M)

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


def create_reservoirs(bn: dict, ic: dict, M: any, register: any = "None") -> dict:
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

    register reservoir groups in global name space (default), or with the
    provided object reference

    """

    from esbmtk import SeawaterConstants, ReservoirGroup, build_concentration_dicts
    from esbmtk import SourceGroup, SinkGroup, Q_

    # parse for sources and sinks, create these and remove them from the list

    # loop over reservoir names
    for k, v in bn.items():
        # test key format
        if M.name in k:
            k = k.split(".")[1]

        if "ty" in v:  # type is given
            if v["ty"] == "Source":
                SourceGroup(name=k, species=v["sp"], register=register)
            elif v["ty"] == "Sink":
                SinkGroup(name=k, species=v["sp"], register=register)
            else:
                raise ValueError("'ty' must be either Source or Sink")

        else:  # create reservoirs
            icd: dict = build_concentration_dicts(ic, k)
            rg = ReservoirGroup(
                name=k,
                geometry=v["g"],
                concentration=icd[k][0],
                isotopes=icd[k][1],
                delta=icd[k][2],
                seawater_parameters={"temperature": v["T"], "pressure": v["P"]},
                register=register,
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
    # bp: bypass, see scale_with_flux
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
        if isinstance(k, tuple):  # loop over names in tuple
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
    bypass = "None" if not "bp" in p else p["bp"]

    if isinstance(scale, Q_):
        scale = scale.to("l/a").magnitude

    # if name in M.dmo: # group already exist in this case we nee to update the
    # connection group

    # Case one, no mixing, test if exist, otherwise create
    # Case 2, mixing: test if forward connection exists, if not create
    #                  test if backwards connection exists, or create

    if not mix:
        if M.register == "local":
            name = f"CG_{source.name}2{sink.name}"
        else:
            name = f"CG_{source.name}2{sink.name}"

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
            bypass,
        )

    else:  # this is a connection with mixing
        # create forward connection
        if M.register == "local":
            name = f"CG_{source.name}2{sink.name}"
        else:
            name = f"CG_{source.name}2{sink.name}"

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
            bypass,
        )

        # create backwards connection
        if M.register == "local":
            name = f"CG_{sink.name}2{source.name}"
        else:
            name = f"CG_{sink.name}2{source.name}"

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
            bypass,
        )


def update_or_create(
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
    bypass,
):
    """Create or update connection"""

    from esbmtk import ConnectionGroup

    if M.register == "local":
        register = M
    else:
        register = "None"

    # update connection if already known
    if f"{name}" in M.lmo or f"{M.name}.{name}" in M.lmo:
        if M.register == "local":
            cg = getattr(M, name)
        else:
            cg = __builtins__[name]

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
            bypass=make_dict(los, bypass),
            register=register,
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
            bypass=make_dict(los, bypass),
            register=register,
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


def find_matching_fluxes(l: list, filter_by: str, exclude: str) -> list:
    """Loop over all reservoir in l, and extract the names of all fluxes
    which match the filter string. Return the list of names (not objects!)

    """

    lof: set = set()

    for r in l:
        for f in r.lof:
            if filter_by in f.full_name and exclude not in f.full_name:
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


def get_connection_keys(
    s: set, fstr: str, nstr: str, inverse: bool, exclude: str
) -> list:
    """extract connection keys from set of flux names, replace fstr with
    nstr so that the key can be used in create_bulk_connnections()

    The optional inverse parameter, can be used where in cases where the
    flux direction needs to be reversed, i.e., the returned key will not read
    sb2db@POM, but db2s@POM

    E.g., if

    s = ( M4.CG_P_sb2P_ib.PO4.POP_F)
    fstr = "POP"
    nstr = "POM_DIC"

    M4.CG_P_sb2P_ib.PO4.POP_F will become

    P_sb2P_ib@POM_DIC

    """

    cl: list = []

    for n in s:
        # get connection and flux name
        l = n.full_name.split(".")
        cn = l[1][3:]  # get key without leadinf C_
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

    from esbmtk import Model

    reference = kwargs["ref_id"]
    target = kwargs["target_id"]
    if "inverse" in kwargs:
        inverse = kwargs["inverse"]
    else:
        inverse = False

    if "exclude" in kwargs:
        exclude_str = kwargs["exclude"]
    else:
        exclude_str = "None"

    if isinstance(M, Model):
        flist: list = find_matching_fluxes(
            M.loc, filter_by=reference, exclude=exclude_str
        )
    elif isinstance(M, list):
        flist: list = find_matching_fluxes(
            M,
            filter_by=reference,
            exclude=exclude_str,
        )
    else:
        raise ValueError(f"gen_dict_entries: M must be list or Model, not {type(M)}")

    klist: list = get_connection_keys(flist, reference, target, inverse, exclude_str)

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


def check_for_quantity(kw) -> Q_:
    """check if keyword is quantity or string an convert as necessary

    kw = str or Q_

    """

    from esbmtk import Q_

    if isinstance(kw, str):
        kw = Q_(kw)
    elif isinstance(kw, Q_):
        pass
    else:
        raise ValueError(f"kw must be string or Quantity")

    return kw


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


def add_carbonate_system_1(rgs: list):
    """Creates a new carbonate system virtual reservoir for each
    reservoir in rgs. Note that rgs must be a list of reservoir groups.

    Required keywords:
        rgs: list = []  of Reservoir Group objects

    These new virtual reservoirs are registered to their respective Reservoir
    as 'cs'.

    The respective data fields are available as rgs.r.cs.xxx where xxx stands
    for a given key key in the  vr_datafields dictionary (i.e., H, CA, etc.)

    """

    from esbmtk import ExternalCode, calc_carbonates_1

    # get object handle even if it defined in model namespace
    # rgs = get_object_handle(rgs)

    for rg in rgs:

        if rg.mo.register == "local":
            species = rg.mo.CO2
        else:
            species = __builtins__["CO2"]

        if hasattr(rg, "DIC") and hasattr(rg, "TA"):
            ExternalCode(
                name="cs",
                species=species,
                function=calc_carbonates_1,
                vr_datafields={
                    "H": rg.swc.hplus,
                    "CA": rg.swc.ca,
                    "HCO3": rg.swc.hco3,
                    "CO3": rg.swc.co3,
                    "CO2aq": rg.swc.co2,
                    "Omega": 0.0,
                },
                function_input_data=List([rg.DIC.c, rg.TA.c]),
                function_params=List(
                    [
                        rg.swc.K1,  # 1
                        rg.swc.K2,  # 2
                        rg.swc.KW,  # 3
                        rg.swc.KB,  # 4
                        rg.swc.boron,  # 5
                        rg.swc.hplus,  # 5
                        rg.swc.ca2,  # 6
                        rg.swc.Ksp,  # 7
                        rg.swc.Ksp0,  # 8
                    ]
                ),
                register=rg,
            )
        else:
            raise AttributeError(f"{rg.full_name} must have a TA and DIC reservoir")


def add_carbonate_system_2(**kwargs) -> None:
    """Creates a new carbonate system virtual reservoir
    which will compute carbon species, saturation, compensation,
    and snowline depth, and compute the associated carbonate burial fluxes

    Required keywords:
        rgs: list of ReservoirGroup objects
        carbonate_export_fluxes: list of flux objects which mus match the
                                 list of ReservoirGroup objects.
        zsat_min = depth of the upper boundary of the deep box
        z0 = upper depth limit for carbonate burial calculations
             typically the lower boundary of the surface water box

    Optional Parameters:

        zsat = initial saturation depth (m)
        zcc = initial carbon compensation depth (m)
        zsnow = initial snowline depth (m)
        zsat0 = characteristic depth (m)
        Ksp0 = solubility product of calcite at air-water interface (mol^2/kg^2)
        kc = heterogeneous rate constant/mass transfer coefficient for calcite dissolution (kg m^-2 yr^-1)
        Ca2 = calcium ion concentration (mol/kg)
        pc = characteristic pressure (atm)
        pg = seawater density multiplied by gravity due to acceleration (atm/m)
        I = dissolvable CaCO3 inventory
        co3 = CO3 concentration (mol/kg)
        Ksp = solubility product of calcite at in situ sea water conditions (mol^2/kg^2)

    """

    from esbmtk import carbonate_chemistry
    from esbmtk import ExternalCode, calc_carbonates_2

    # list of known keywords
    lkk: dict = {
        "rgs": list,
        "carbonate_export_fluxes": list,
        "AD": float,
        "zsat": int,
        "zsat_min": int,
        "zcc": int,
        "zsnow": int,
        "zsat0": int,
        "Ksp0": float,
        "kc": float,
        "Ca2": float,
        "pc": (float, int),
        "pg": (float, int),
        "I_caco3": (float, int),
        "alpha": float,
        "zmax": (float, int),
        "z0": (float, int),
        "Ksp": (float, int),
    }
    # provide a list of absolutely required keywords
    lrk: list[str] = ["rgs", "carbonate_export_fluxes", "zsat_min", "z0"]

    # we need the reference to the Model in order to set some
    # default values.

    reservoir = kwargs["rgs"][0]
    model = reservoir.mo
    # list of default values if none provided
    lod: dict = {
        "zsat": -3715,  # m
        "zcc": -4750,  # m
        "zsnow": -4750,  # m
        "zsat0": -5078,  # m
        "Ksp0": reservoir.swc.Ksp0,  # mol^2/kg^2
        "kc": 8.84 * 1000,  # m/yr converted to kg/(m^2 yr)
        "AD": model.hyp.area_dz(-200, -6000),
        "alpha": 0.6,  # 0.928771302395292, #0.75,
        "pg": 0.103,  # pressure in atm/m
        "pc": 511,  # characteristic pressure after Boudreau 2010
        "I_caco3": 529,  #  dissolveable CaCO3 in mol/m^2
        "zmax": -6000,  # max model depth
        "Ksp": reservoir.swc.Ksp,  # mol^2/kg^2
    }

    # make sure all mandatory keywords are present
    __checkkeys__(lrk, lkk, kwargs)

    # add default values for keys which were not specified
    kwargs = __addmissingdefaults__(lod, kwargs)

    # test that all keyword values are of the correct type
    __checktypes__(lkk, kwargs)

    # establish some shared parameters
    # depths_table = np.arange(0, 6001, 1)
    depths: NDArray = np.arange(0, 6002, 1, dtype=float)
    rgs = kwargs["rgs"]
    Ksp0 = kwargs["Ksp0"]
    ca2 = rgs[0].swc.ca2
    pg = kwargs["pg"]
    pc = kwargs["pc"]
    z0 = kwargs["z0"]
    Ksp = kwargs["Ksp"]

    # C saturation(z) after Boudreau 2010
    Csat_table: NDArray = (Ksp0 / ca2) * np.exp((depths * pg) / pc)
    area_table = model.hyp.get_lookup_table(0, -6002)  # area in m^2(z)
    area_dz_table = model.hyp.get_lookup_table_area_dz(0, -6002) * -1  # area'
    sa = model.hyp.sa  # Total earth area
    AD = model.hyp.area_dz(z0, -6000)  # Total Ocean Area
    dt = model.dt

    for i, rg in enumerate(rgs):  # Setup the virtual reservoirs

        if rg.mo.register == "local":
            species = rg.mo.CO2
        else:
            species = __builtins__["CO2"]

        ExternalCode(
            name="cs",
            species=species,
            function=calc_carbonates_2,
            # datafield hold the results of the VR_no_set function
            # provide a default values which will be use to initialize
            # the respective datafield/
            vr_datafields={
                "H": rg.swc.hplus,  # 0 H+
                "CA": rg.swc.ca,  # 1 carbonate alkalinity
                "HCO3": rg.swc.hco3,  # 2 HCO3
                "CO3": rg.swc.co3,  # 3 CO3
                "CO2aq": rg.swc.co2,  # 4 CO2aq
                "zsat": kwargs["zsat"],  # 5 zsat
                "zcc": kwargs["zcc"],  # 6 zcc
                "zsnow": kwargs["zsnow"],  # 7 zsnow
                "Fburial": 0.0,  # 8 carbonate burial
                "B": 0.0,  # 9 carbonate export productivity
                # temp fields, delete eventually
                "BNS": 0.0,  # 10 BNS
                "BDS_under": 0.0,  # 11 BDS_under
                "BDS_resp": 0.0,  # 12 BDS_resp
                "BDS": 0.0,  # 13 BDS
                "BCC": 0.0,  # 14 BCC
                "BPDC": 0.0,  # 15 BPDC
                "BD": 0.0,  # 16 BD
                "bds_area": 0.0,  # 17 bds_area
                "zsnow_dt": 0.0,  # 18 zsnow_dt
                "Omega": 0.0,  # 19 omega
            },
            function_input_data=List(
                [
                    rg.DIC.m,  # 0 DIC mass
                    rg.DIC.l,  # 1 DIC light isotope mass
                    rg.DIC.h,  # 2 DIC heavy isotope mass
                    rg.DIC.c,  # 3 DIC concentration
                    rg.TA.m,  # 4 TA mass
                    rg.TA.c,  # 5 TA concentration
                    kwargs["carbonate_export_fluxes"][i].m,  # 6
                    area_table,  # 7
                    area_dz_table,  # 8
                    Csat_table,  # 9
                ]
            ),
            function_params=List(
                [
                    rg.swc.K1,  # 0
                    rg.swc.K2,  # 1
                    rg.swc.KW,  # 2
                    rg.swc.KB,  # 3
                    rg.swc.boron,  # 4
                    Ksp0,  # 5
                    float(kwargs["kc"]),  # 6
                    float(sa),  # 7
                    float(rg.volume.to("liter").magnitude),  # 8
                    float(AD),  # 9
                    float(abs(kwargs["zsat0"])),  # 10
                    float(rg.swc.ca2),  # 11
                    rg.mo.dt,  # 12
                    float(kwargs["pc"]),  # 13
                    float(kwargs["pg"]),  # 14
                    float(kwargs["I_caco3"]),  # 15
                    float(kwargs["alpha"]),  # 16
                    float(abs(kwargs["zsat_min"])),  # 17
                    float(abs(kwargs["zmax"])),  # 18
                    float(abs(kwargs["z0"])),  # 19
                    Ksp,  # 20
                ]
            ),
            register=rg,
        )


def __find_flux__(reservoirs: list, full_name: str):
    """Helper function to find a Flux object based on its full_name in the reservoirs
    in the list of provided reservoirs.

    PRECONDITIONS: full_name must contain the full_name of the Flux

    Parameters:
        reservoirs: List containing all reservoirs
        full_name: str specifying the full name of the flux (boxes.flux_name)
    """
    needed_flux = None
    for res in reservoirs:
        for flux in res.lof:
            if flux.full_name == full_name:
                needed_flux = flux
                break
        if needed_flux != None:
            break
    if needed_flux == None:
        raise NameError(
            f"add_carbonate_system: Flux {full_name} cannot be found in any of the reservoirs in the Model!"
        )

    return needed_flux


def __checktypes__(av: Dict[any, any], pv: Dict[any, any]) -> None:
    """this method will use the the dict key in the user provided
    key value data (pv) to look up the allowed data type for this key in av

    av = dictinory with the allowed input keys and their type
    pv = dictionary with the user provided key-value data
    """

    k: any
    v: any

    # loop over provided keywords
    for k, v in pv.items():
        # check av if provided value v is of correct type
        if av[k] != any:
            # print(f"key = {k}, value  = {v}")
            if not isinstance(v, av[k]):
                # print(f"k={k}, v= {v}, av[k] = {av[k]}")
                raise TypeError(
                    f"{type(v)} is the wrong type for '{k}', should be '{av[k]}'"
                )


def __checkkeys__(lrk: list, lkk: list, kwargs: dict) -> None:
    """check if the mandatory keys are present

    lrk = list of required keywords
    lkk = list of all known keywords
    kwargs = dictionary with key-value pairs

    """

    k: str
    v: any
    # test if the required keywords are given
    for k in lrk:  # loop over required keywords
        if isinstance(k, list):  # If keyword is a list
            s: int = 0  # loop over allowed substitutions
            for e in k:  # test how many matches are in this list
                if e in kwargs:
                    # print(self.kwargs[e])
                    if not isinstance(e, (np.ndarray, np.float64, list)):
                        # print (type(self.kwargs[e]))
                        if kwargs[e] != "None":
                            s = s + 1
            if s > 1:  # if more than one match
                raise ValueError(f"You need to specify exactly one from this list: {k}")

        else:  # keyword is not in list
            if k not in kwargs:
                raise ValueError(f"You need to specify a value for {k}")

    tl: List[str] = []
    # create a list of known keywords
    for k, v in lkk.items():
        tl.append(k)

    # test if we know all keys
    for k, v in kwargs.items():
        if k not in lkk:
            raise ValueError(f"{k} is not a valid keyword. \n Try any of \n {tl}\n")


def __addmissingdefaults__(lod: dict, kwargs: dict) -> dict:
    """
    test if the keys in lod exist in kwargs, otherwise add them with the default values
    from lod

    """

    new: dict = {}
    if len(lod) > 0:
        for k, v in lod.items():
            if k not in kwargs:
                new.update({k: v})

    kwargs.update(new)
    return kwargs
