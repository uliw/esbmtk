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
#from numba import jit, njit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpmath

import logging
import time
import builtins
import math
set_printoptions(precision=4)

def get_imass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species
    
    """

    li: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    hi: float = m - li
    return [li, hi]


def get_frac(m: float, l: float, a: float) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha on
    the ratio between the light and heavy isotope.

    """

    li: float = -l * m / (a * l - a * m - l)
    hi: float = m - li  # get the new heavy isotope value
    return li, hi


def get_flux_data(m: float, d: float, r: float) -> [NDArray, float]:
    """ 
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species. Unlike get_mass, this function returns the full array
    
    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l

    return np.array([m, l, h, d])

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

    d: float = 1E3 * (abs(h) / abs(l) - r) / r
    return d

def add_to (l, e):
    """
      add element e to list l, but check if the entry already exist. If so, throw
      exception. Otherwise add
    """

    if not (e in l):  # if not present, append element
        l.append(e)

def get_plot_layout(obj):
      """ Simple function which selects a row, column layout based on the number of
      objects to display.  The expected argument is a reservoir object which
      contains the list of fluxes in the reservoir

      """

      noo = 1 # the reservoir is the fisrt object
      for f in obj.lof: # count numbert of fluxes
            if f.plot == "yes":
                  noo += 1
                  
      for d in obj.ldf: # count number of data fields
            noo +=1
            
      # noo = len(obj.lof) + 1  # number of objects in this reservoir
      logging.debug(f"{noo} subplots for {obj.n} required")

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
      else:
          print("plot geometry for more than 8 fluxes is not yet defined")
          print("Consider calling flux.plot individually on each flux in the reservoir")
          # print(f"Selected Geometry: rows = {geo[0]}, cols = {geo[1]}")

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

def set_y_limits(ax: plt.Axes, model: any) -> None:
    """ Prevent the display or arbitrarily small differences
    """
    lower: float
    upper: float

    bottom, top = ax.get_ylim()
    if (top - bottom) < model.display_precision:
        top = bottom + model.display_precision
        ax.set_ylim(bottom, top)


def get_ptype(obj, kwargs: dict) -> int:
    """
    Set plot type variable
    
    """

    ptype: int = 0
    if "ptype" in kwargs:
        if kwargs["ptype"] == "both":
            ptype = 0
        elif kwargs["ptype"] == "iso":
            ptype = 1
        elif kwargs["ptype"] == "concentration":
            ptype = 2
        elif kwargs["ptype"] == "mass_only":
            ptype = 2
    else:
        if obj.m_type == "mass_only":
            ptype = 2
        elif obj.m_type == "both":
            ptype = 0
        else:
            raise ValueError(
                "ptype must be one of 'both/iso/concentration/mass_only'")

    return ptype


def plot_object_data(geo: list, fn: int, obj, ptype: int) -> None:
    """collection of commands which will plot and annotate a reservoir or flux
      object into an existing plot window. 
      """

    from . import ureg, Q_
    from esbmtk import Flux, Reservoir, Signal, DataField

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

    # remap concentration & flux values
    if isinstance(obj, Flux):
        yl = (obj.m * model.f_unit).to(obj.plt_units).magnitude
        y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

    elif isinstance(obj, Reservoir):
        if obj.display_as == "mass":
            yl = (obj.m * model.m_unit).to(obj.plt_units).magnitude
            y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

        elif obj.transform_m != "None":
            if callable(obj.transform_m):
                #yl = (obj.m * model.m_unit).to(obj.plt_units).magnitude
                yl = obj.transform_m(obj.m)
                y_label = f"{obj.legend_left}"
            else:
                raise ValueError("transform_m must be function")

        else:
            yl = (obj.c * model.c_unit).to(obj.plt_units).magnitude
            y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"

    elif isinstance(obj, Signal):
        # use the same units as the associated flux
        yl = (obj.c * model.c_unit).to(obj.fo.plt_units).magnitude
        y_label = f"{obj.n} [{obj.fo.plt_units:~P}]"

    elif isinstance(obj, DataField):
        time = (time * model.t_unit).to(model.d_unit).magnitude
        yl = obj.y1_data
        y_label = obj.y1_label
        if obj.y2_data == "None":
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
    ax1 = plt.subplot(rows, cols, fn, title=obj.n)

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
        ax1.spines['top'].set_visible(False)
        set_y_limits(ax1, model)

    # set color index
    cn = cn + 1
    col = f"C{cn}"

    if second_axis:
        ax2 = ax1.twinx()  # create a second y-axis

        # plof right y-scale data
        ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)

        ax2.set_ylabel(obj.ld)  # species object delta label
        ax2.spines['top'].set_visible(
            False)  # remove unnecessary frame speciess
        set_y_limits(ax2, model)

    # adjust display properties for title and legend
    ax1.set_title(obj.n)
    plt.rcParams['axes.titlepad'] = 14  # offset title upwards
    plt.rcParams["legend.facecolor"] = '0.8'  # show a gray background
    plt.rcParams["legend.edgecolor"] = '0.8'  # make frame the same color
    plt.rcParams["legend.framealpha"] = 0.4  # set transparency

    for d in obj.led:  # loop over external data objects if present

        if isinstance(d.x[0], str):  # if string, something is off
            raise ValueError("No time axis in external data object {d.name}")
        if "y" in dir(d):  # mass or concentration data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{obj.lm} {d.legend}"
            ln3 = ax1.scatter(d.x, d.y, color=col, label=leg)
        if "z" in dir(d) and second_axis:  # isotope data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{d.legend}"
            ln3 = ax2.scatter(d.x, d.z, color=col, label=leg)

    # collect all labels and print them in one legend
    if first_axis:
        handler1, label1 = ax1.get_legend_handles_labels()

    if second_axis:
        handler2, label2 = ax2.get_legend_handles_labels()

    if first_axis and second_axis:
        legend = ax2.legend(handler1 + handler2, label1 + label2,
                            loc=0).set_zorder(6)
    #elif first_axis:
    #    legend = ax1.legend(handler1 + label1, loc=0).set_zorder(6)
    #elif second_axis:
    #   legend = ax2.legend(handler2 + label2, loc=0).set_zorder(6)

    # Matplotlib will show arbitrarily small differences which can be confusing
    #yl_min = min(yl)
    #yl_max = max(yl)
    #if (yl_max - yl_min) < 0.1:

def is_name_in_list(n: str, l: list) -> bool:
    """ Test if an object name is part of the object list
    
    """

    r: bool = False
    for e in l:
        if e.n == n:
            r = True
    return r


def get_object_from_list(n: str, l: list) -> any:
    """ Match a name to a list of objects. Return the object
    
    """

    for o in l:
        if o.n == n:
            r = o
    return r

def get_hplus(dic :float, ta :float)->float:
    """
    Calculate H+ concentration based on DIC concentration and Alkalinity
    according to eq 11 in Follows et al 2006
    
    """

    pk1 = 5.81  # at this ph value CO2 and HCO3 have the same concentration
    pk2 = 8.92
    K1 = 10**-pk1
    K2 = 10**-pk2
    
    g = dic / ta
    hplus = 0.5 * ((g - 1) * K1 + ((1 - g)**2 * K1**2 - 4 * K1 * K2 *
                                   (1 - 2 * g))**0.5)

    return hplus

def get_pco2(dic :float, ta :float) -> float:
    """Calculate pCO2 in uatm at 25C and a Salinity of 35

    DIC has to be in mmol/l!

    """
    pk1 = 5.81  # at this ph value CO2 and HCO3 have the same concentration
    pk2 = 8.92
    K1 = 10**-pk1
    K2 = 10**-pk2
    K0 = 36

    hplus = get_hplus(dic,ta)

    # get [CO2] in water
    co2 = dic / (1 + K1/hplus + K1*K2/hplus**2)

    # get pco2 as a function of co2 fugacity
    pco2 = co2/K0 *1E6

    # this cam also be expressed in teh following way
    #pco2a = (ta/K0 * ( K1/hplus + 2*K1*K2/hplus**2)**-1) * 1.e6
    #pco2b = (dic/K0 * (1 + K1/hplus + (K1*K2)/hplus**2)**-1) * 1.e6
   
    return pco2

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
