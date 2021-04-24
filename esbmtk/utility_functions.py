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
import time
from time import process_time
import numba
from numba.core import types
from numba import njit, prange
from numba.typed import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpmath

import logging
import time
import builtins
import math
set_printoptions(precision=4)

# @njit()
def get_imass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species
    
    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l
    return [l, h]

# @njit()
def get_frac(m: float, l: float, a: float) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha on
    the ratio between the light and heavy isotope.

    """

    li: float = -l * m / (a * l - a * m - l)
    hi: float = m - li  # get the new heavy isotope value
    return li, hi

# @njit()
def get_flux_data(m: float, d: float, r: float) -> [NDArray, float]:
    """ 
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species. Unlike get_mass, this function returns the full array
    
    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l

    return np.array([m, l, h, d])

# @njit()
def get_delta(l: [NDArray, [Float64]], h: [NDArray, [Float64]],
              r: float) -> [NDArray, [Float64]]:
    """Calculate the delta from the mass of light and heavy isotope
     Arguments are l and h which are the masses of the light and
     heavy isotopes respectively, r = abundance ratio of the
     respective element. Note that this equation can result in a
     siginificant loss of precision (on the order of 1E-13). I
     therefore round the results to numbers largers 1E12 (otherwise a
     delta of zero may no longer be zero)

   """
    # d = 1000 * (h / l - r) / r
    d: float = 1E3 * (np.abs(h) / np.abs(l) - r) / r
    return d

def add_to (l, e):
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

def list_fluxes(self,name,i) -> None:
            """
            Echo all fluxes in the reservoir to the screen
            """
            print(f"\nList of fluxes in {self.n}:")
            
            for f in self.lof: # show the processes
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
                  f.describe(i) # print out the flux data

def show_data(self, **kwargs) -> None:
    """ Print the 3 lines of the data starting with index

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
        print(
            f"{off}{ind}i = {i}, Mass = {self.m[i]:.2e}, delta = {self.d[i]:.2f}"
        )

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
        # plot left y-scale data
        ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=obj.legend_left)
        # set labels
        ax1.set_xlabel(f"{model.time_label} [{model.d_unit:~P}]")
        ax1.set_ylabel(y_label)
        # remove unnecessary frame species
        ax1.spines["top"].set_visible(False)
        set_y_limits(ax1, obj)

    # set color index
    cn = cn + 1
    col = f"C{cn}"

    if second_axis:
        if isinstance(obj, DataField):
            if obj.common_y_scale == "yes":
                ln2 = ax1.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)
                set_y_limits(ax1, model)
                ax1.legend()
                second_axis = False
            else:
                ax2 = ax1.twinx()  # create a second y-axis
                # plof right y-scale data
                ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)
                ax2.set_ylabel(obj.ld)  # species object delta label
                set_y_limits(ax2, model)
                ax2.spines["top"].set_visible(
                    False
                )  # remove unnecessary frame speciess

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

    #from numbers import Number

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
        cid = "None"

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

def create_bulk_connections(ct: dict, M: any) -> None:
    """Create connections from a dictionary. The dict shoudl have the
    following format:

    # Setup the dict which describes all fluxes
    # na: names, tuple or str. If lists, all list elements share the same properties
    # sp: species list or species
    # ty: type, str
    # ra: rate, Quantity
    # sc: scale, Number
    # re: reference, optional
    # al: alpha, optional
    # de: delta, optional
    # mx: True, optional defaults to False
    sl: list = list(ic.keys())  # get species list
    ct = {  # thermohaline circulation
            # Apply to all boxes in the tuple
         ("hb2db@thc", "db2ib@thc", "ib2hb@thc"): {
          "ty": "scale_with_concentration",
          "sp": sl,  # species list
          "ra": Q_('20*Sv'),
         },
        # mixing fluxes
        "sb2ib@mix": {
           "ty": "scale_with_concentration",
           "ra": Q_('63 Sv'),
           "sp": "sl",
           "mx": True,
       },
      },
    # particulate fluxes due to biological production
    "sb2ib@POP": {"ty": "scale_with_mass", "sc": 0.8, "re": sb.PO4, "sp": PO4},
    }

    """

    from esbmtk import create_connection

    # loop over values in ct dict
    for k, v in ct.items():
        if isinstance(k, tuple):
            # loop over names in tuple
            for c in k:
                create_connection(c, v, M)
        elif isinstance(k, str):
            create_connection(k, v, M)
        else:
            raise ValueError(f"{connection} must be string or tuple")


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
    ref = "None" if not "re" in p else p["re"]
    alpha = "None" if not "al" in p else p["al"]
    delta = "None" if not "de" in p else p["de"]
    mix = False if not "mx" in p else p["mx"]
    cid = f"{cid}_f" if mix else f"{cid}"

    if isinstance(scale, Q_):
        scale = scale.to("l/a").magnitude

    cg = ConnectionGroup(
        source=source,
        sink=sink,
        ctype=make_dict(los, typ),
        scale=make_dict(los, scale),  # get rate from dictionary
        rate=make_dict(los, rate),
        ref=make_dict(los, ref),
        alpha=make_dict(los, alpha),
        delta=make_dict(los, delta),
        id=cid,  # get id from dictionary
    )

    # if mixing is set to True create reverse connection
    if mix:
        cid = cid.replace("_f", "_b")
        cg2 = ConnectionGroup(
            source=sink,
            sink=source,
            ctype=make_dict(los, typ),
            scale=make_dict(los, scale),  # get rate from dictionary
            rate=make_dict(los, rate),
            ref=make_dict(los, ref),
            alpha=make_dict(los, alpha),
            delta=make_dict(los, delta),
            id=cid,  # get id from module import symbol
        )

def execute(
    new: [NDArray, Float64],
    time: [NDArray, Float64],
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:

    """This is the original object oriented solver"""

    i = 1  # some processes refer to the previous time step
    for t in time[1:-1]:  # loop over the time vector except the first
        # we first need to calculate all fluxes
        for r in lor:  # loop over all reservoirs
            for p in r.lop:  # loop over reservoir processes
                p(r, i)  # update fluxes

        # update all process based fluxes. This can be done in a global lpc list
        for p in lpc_f:
            p(i)

        # and then update all reservoirs
        for r in lor:  # loop over all reservoirs
            flux_list = r.lof

            new[0] = new[1] = new[2] = 0
            for f in flux_list:  # do sum of fluxes in this reservoir
                direction = r.lio[f]
                new[0] = new[0] + f.m[i] * direction  # current flux and direction
                new[1] = new[1] + f.l[i] * direction  # current flux and direction
                new[2] = new[2] + f.h[i] * direction  # current flux and direction

            # print(f"fsum = {new[0]:.2e}")
            # new = array([ms, ls, hs])
            new = new * r.mo.dt  # get flux / timestep
            new = new + r[i - 1]  # add to data from last time step
            new = new * (new > 0)  # set negative values to zero
            # print(f"updating {r.full_name} from {r.m[i]:.2e}")
            r[i] = new  # update reservoir data
            # print(f"to  {r.m[i]:.2e}\n")

        # update reservoirs which do not depend on fluxes but on
        # functions
        for p in lpc_r:
            p.act_on[i] = p(i)

        i = i + 1


def execute_h(
    new: [NDArray, Float64],
    time: [NDArray, Float64],
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:

    """Moved this code into a separate function to enable numba optimization"""

    i: int = 1  # processes refer to the previous time step -> start at 1
    dt: float = lor[0].mo.dt
    ratio: float = lor[0].sp.r
    ratio = 1

    a, b, c, d, e = build_flux_lists_all(lor)
    for t in time[1:-1]:  # loop over the time vector except the first
        # we first need to calculate all fluxes
        for r in lor:  # loop over all reservoirs
            for p in r.lop:  # loop over reservoir processes
                p(r, i)  # update fluxes

        # update all process based fluxes. This can be done in a global lpc list
        for p in lpc_f:
            p(i)

        summarize_fluxes(a, b, c, d, e, i, dt)

        # update reservoirs which do not depend on fluxes but on
        # functions
        for p in lpc_r:
            p(i)

        i = i + 1  # next time step


def execute_n(
    new: [NDArray, Float64],
    time: [NDArray, Float64],
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:

    """Moved this code into a separate function to enable numba optimization"""
    # config.THREADING_LAYER = "threadsafe"
    # numba.set_num_threads(2)

    i: int = 1  # processes refer to the previous time step -> start at 1
    dt: float = lor[0].mo.dt
    ratio: float = lor[0].sp.r
    ratio = 1

    fn_vr, a1, a2, a3, a4, a5, a6, a7 = build_vr_list(lpc_r)
    fn, rd, fd, pc = build_process_list(lor)
    a, b, c, d, e = build_flux_lists_all(lor)
    for t in time[1:-1]:  # loop over the time vector except the first
        # update_fluxes for each reservoir
        update_fluxes(fn, rd, fd, pc, i)

        # update all process based fluxes. This can be done in a global lpc list
        # for p in lpc_f:
        #    p(i)

        # calculate the resulting reservoir concentrations
        summarize_fluxes(a, b, c, d, e, i, dt)

        # update reservoirs which do not depend on fluxes but on
        # functions
        # calc_v_reservoir_data()
        update_virtual_reservoirs(fn_vr, a1, a2, a3, a4, a5, a6, a7, i)

        i = i + 1  # next time step


def execute_e(
    new: [NDArray, Float64],
    time_array: [NDArray, Float64],
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:

    """Moved this code into a separate function to enable numba optimization"""
    # numba.config.THREADING_LAYER = "threadsafe"
    #numba.set_num_threads(2)

    # this has nothing todo with self.time below!
    start: float = process_time()
    dt: float = lor[0].mo.dt
    fn_vr, a1, a2, a3, a7 = build_vr_list(lpc_r)
    fn, rd, fd, pc = build_process_list(lor)
    a, b, c, d, e = build_flux_lists_all(lor)

    duration: float = process_time() - start
    print(f"\n Setup time {duration} cpu seconds\n")

    wts = time.time()
    start: float = process_time()
    foo(fn_vr, a1, a2, a3, a7, fn, rd, fd, pc, a, b, c, d, e, time_array[:-1], dt)

    duration: float = process_time() - start
    wcd = time.time() - wts
    print(f"\n Total solver time {duration} cpu seconds, wt = {wcd}\n")


@njit(parallel=False)
def foo(fn_vr, a1, a2, a3, a7, fn, rd, fd, pc, a, b, c, d, e, maxt, dt):

    i = 1
    for t in maxt:
        for j, f_list in enumerate(fn):
            for u, function in enumerate(f_list):
                fn[j][u](rd[j][u], fd[j][u], pc[j][u], i)

        # calculate the resulting reservoir concentrations
        # summarize_fluxes(a, b, c, d, e, i, dt)
        r_steps: int = len(b)
        # loop over reservoirs
        for j in range(r_steps):
            # for j, r in enumerate(b):  # this will catch the list for each reservoir

            # sum fluxes in each reservoir
            mass = li = 0.0
            f_steps = len(b[j])
            for u in range(f_steps):
                direction = c[j][u]
                # for u, f in enumerate(r):  # this should catch each flux per reservoir
                mass += b[j][u][0][i] * direction  # mass
                li += b[j][u][1][i] * direction  # li

            # update masses
            a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
            a[j][1][i] = a[j][1][i - 1] + li * dt  # li
            a[j][2][i] = a[j][0][i] - a[j][1][i]  # hi
            # update delta
            a[j][3][i] = 1e3 * (a[j][2][i] / a[j][1][i] - e[j]) / e[j]
            # update concentrations
            a[j][4][i] = a[j][0][i] / d[j]

        # # update reservoirs which do not depend on fluxes but on
        # # functions
        # # calc_v_reservoir_data()
        for j, f in enumerate(fn_vr):
            a7[j][0][i], a7[j][1][i], a7[j][2][i], a7[j][3][i], a7[j][4][i] = fn_vr[j](
                i, a1[j], a2[j], a3[j]
            )
        i = i + 1  # next time step


@njit
def update_fluxes(fn, rd, fd, pc, i):
    """Loop over all processes and update fluxes"""

    for j, f_list in enumerate(fn):
        for u, function in enumerate(f_list):
            fn[j][u](rd[j][u], fd[j][u], pc[j][u], i)


@njit()
def sum_p(r_list, f_list, dir_list, v_list, r0_list, i, dt):

    j = 0
    for e in range(2):
        sum_lists(r_list[j], f_list[j], dir_list[j], v_list[j], r0_list[j], i, dt)

@njit()
def summarize_fluxes(a, b, c, d, e, i, dt):
    """Sum fluxes in reservoirs with isostopes"""
    
    r_steps: int = len(b)
    # loop over reservoirs
    for j in range(r_steps):
        # for j, r in enumerate(b):  # this will catch the list for each reservoir

        # sum fluxes in each reservoir
        mass = li = 0.0
        f_steps = len(b[j])
        for u in range(f_steps):
            direction = c[j][u]
            # for u, f in enumerate(r):  # this should catch each flux per reservoir
            mass += b[j][u][0][i] * direction  # mass
            li += b[j][u][1][i] * direction  # li

        # update masses
        a[j][0][i] = a[j][0][i - 1] + mass * dt  # mass
        a[j][1][i] = a[j][1][i - 1] + li * dt  # li
        a[j][2][i] = a[j][0][i] - a[j][1][i]  # hi
        # update delta
        a[j][3][i] = (
            1e3 * (a[j][2][i] / a[j][1][i] - e[j]) / e[j]
        )
        # update concentrations
        a[j][4][i] = a[j][0][i] / d[j]

# GenericFunction(
#      name = 'db_VH_generic_function',
#      function = <function calc_H at 0x7f8bc474c160>,
#      a1 = VCA,
#      a2 = DIC,
#      a3 = SW_db,
#      a4 = 0,
#      a5 = 0,
#      a6 = 0,
#      act_on = VH,
#      full_name = 'db_VH_generic_function',
#  )]

# a1 to a6 should probably be lists so they can contain any type of data


def build_vr_list(lor: list) -> tuple:
    """Build lists which contain all function references for
    virtual reservoirs as well aas their input values

    """

    fn = List()  # List() # list of functions
    a1 = List()  # reservoir data
    a2 = List()  # flux data  flux.m flux.l, flux.h, flux.d
    a3 = List()  # list of constants
    a7 = List()

    fn = numba.typed.List.empty_list(
        types.UniTuple(types.float64, 5)(
            types.int64,  # i
            types.float64[::1],  # a1
            types.float64[::1],  # a2
            types.ListType(types.float64), # a3
        ).as_type()
    )

    for p in lor:  # loop over reservoir processes

        func_name, a1d, a2d, a3d, a7d = p.get_process_args()
        print(f"fname = {func_name}")
        fn.append(func_name)
        a1.append(a1d)
        a2.append(a2d)
        a3.append(a3d)
        a7.append(List(a7d))

    return fn, a1, a2, a3, a7

def build_flux_lists(lor, iso: bool = False) -> tuple:
    """flux_list :list [] contains all fluxes as
    [f.m, f.l, f.h, f.d], where each sublist relates to one reservoir

    i.e. per reservoir we have list [f1, f2, f3], where f1 = [m, l, h, d]
    and m = np.array()

    iso = False/True

    """

    r_list: list = List()
    v_list: list = List()
    r0_list: list = List()

    f_list: list = List()
    dir_list: list = List()
    rd_list: list = List()
    fd_list: list = List()

    for r in lor:  # loop over all reservoirs
        if r.isotopes == iso:
            rd_list = List([r.m, r.l, r.h, r.d, r.c])

            r_list.append(rd_list)
            v_list.append(float(r.volume))
            r0_list.append(float(r.sp.r))

            i = 0
            # add fluxes for each reservoir entry
            tf: list = List()  # temp list for flux data
            td: list = List()  # temp list for direction data

            # loop over all fluxes
            for f in r.lof:
                fd_list = List([f.m, f.l, f.h, f.d])
                tf.append(fd_list)
                td.append(float(r.lodir[i]))
                i = i + 1

            f_list.append(tf)
            dir_list.append(td)

    # v_list = tuple(v_list)
    # r0_list = tuple(r0_list)
    # dir_list = tuple(dir_list)

    return r_list, f_list, dir_list, v_list, r0_list


def build_flux_lists_all(lor, iso: bool = False) -> tuple:
    """flux_list :list [] contains all fluxes as
    [f.m, f.l, f.h, f.d], where each sublist relates to one reservoir

    i.e. per reservoir we have list [f1, f2, f3], where f1 = [m, l, h, d]
    and m = np.array()

    iso = False/True

    """

    r_list: list = List()
    v_list: list = List()
    r0_list: list = List()

    f_list: list = List()
    dir_list: list = List()
    rd_list: list = List()
    fd_list: list = List()

    for r in lor:  # loop over all reservoirs
        rd_list = List([r.m, r.l, r.h, r.d, r.c])

        r_list.append(rd_list)
        v_list.append(float(r.volume))
        r0_list.append(float(r.sp.r))

        i = 0
        # add fluxes for each reservoir entry
        tf: list = List()  # temp list for flux data
        td: list = List()  # temp list for direction data

        # loop over all fluxes
        for f in r.lof:
            fd_list = List([f.m, f.l, f.h, f.d])
            tf.append(fd_list)
            td.append(float(r.lodir[i]))
            i = i + 1

        f_list.append(tf)
        dir_list.append(td)

    return r_list, f_list, dir_list, v_list, r0_list

def build_process_list(lor: list) -> tuple:
    from numba.typed import List
    import numba
    from numba.core import types

    fn = List()  # List() # list of functions
    rd = List()  # reservoir data
    fd = List()  # flux data  flux.m flux.l, flux.h, flux.d
    pc = List()  # list of constants

    # func_name : function reference
    # res_data :list = reservoir data (m,l,d,c)
    # flux_data :list = flux data (m,l,h,d)
    # proc_const : list = any constants, must be float

    f_time = 0
    d_time = 0
    
    for r in lor:  # loop over reservoirs

        # note that types.List is differenfr from Types.ListType. Also
        # note that [::1]  declares C-style arrays see
        # https://numba.discourse.group/t/list-mistaken-as-list-when-creating-list-of-function-references/677/3
        tfn = numba.typed.List.empty_list(
            types.ListType(types.void)(  # return value
                types.ListType(types.float64[::1]),
                types.ListType(types.float64[::1]),
                types.ListType(types.float64), 
                types.int64,  # parameter 4
            ).as_type()
        )

        trd = List()
        tfd = List()
        tpc = List()
        for p in r.lop:  # loop over reservoir processes

            print(f"adding function {p.full_name}  with {r.full_name}")
            start: float = process_time()
            func_name, res_data, flux_data, proc_const = p.get_process_args(r)
            duration =  process_time() - start
            f_time = f_time +  duration
            
            start: float = process_time()
            tfn.append(func_name)
            trd.append(res_data)
            tfd.append(flux_data)
            tpc.append(proc_const)
            duration =  process_time() - start
            d_time = d_time +  duration

        fn.append(tfn)
        rd.append(trd)
        fd.append(tfd)
        pc.append(tpc)

    print(f"f_time = {f_time}")
    print(f"d_time = {d_time}")
    
    return fn, rd, fd, pc

def get_string_between_brackets(s :str) -> str:
    """ Parse string and extract substring between square brackets

    """
    
    s =  s.split("[")
    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    s = s[1]

    s = s.split("]")

    if len(s) < 2:
        raise ValueError(f"Column header {s} must include units in square brackets")

    return s[0]

def map_units(v: any, *args) -> float:
    """ parse v to see if it is a string. if yes, map to quantity. 
        parse v to see if it is a quantity, if yes, map to model units
        and extract magnitude, assign mangitude to return value
        if not, assign value to return value
        
        v : a keyword value number/string/quantity
        args: one or more quantities (units) see the Model class (e.g., f_unit)

    """

    from . import Q_

    m: float = 0
    match :bool = False

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
