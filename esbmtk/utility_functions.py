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

def get_mass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species
    
    """
    
    li: float = (1000 * m) / ((d + 1000) * r + 1000)
    hi: float = ((d * m + 1000 * m) * r) / ((d + 1000) * r + 1000)
    return [li, hi]


def set_mass(m: float, d: float, r: float) -> [float, float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio 
    species. Unlike get_mass, this function returns the full array
    
    """
    
    l: float = (1000 * m) / ((d + 1000) * r + 1000)
    h: float = ((d * m + 1000 * m) * r) / ((d + 1000) * r + 1000)

    return array([m, l, h])

def get_delta(l, h, r):
    """
      Calculate the delta from the mass of light and heavy isotope
      Arguments are l and h which are the masses of the light and
      heavy isotopes respectively, r = abundance ratio of the respective
      element
    """
   
    d = 1000 * (h / l - r) / r
    return d

def add_to (l, e):
    """
      add element e to list l, but check if the entry already exist. If so, throw
      exception. Otherwise add
    """

    if not (e in l):  # if not present, append element
        l.append(e)

def get_mag(unit, base_unit):
    """
      Compare the unit associated with the object obj (i.e., a flux, etc)
      with the base unit set for the species or model (base_unit)
      ms = magnitude string, s = scaling factor
    """

    # E.g., unit = mmol, and base_unit = mol -> ms = m, and thus s = 1E-3
    if len(base_unit) > len(unit):
        ms = base_unit.replace(unit, "")
    else:
        ms = unit.replace(base_unit, "")  # get the magnitude string of the species
        
        if ms == "":  # species unit and reservoir units are the same
            s = 1  # -> no scaling
        elif ms == "G":  # value is provided in mega
            s = 1E9  # thus they need to be scaled by 1e9
        elif ms == "M":  # value is provided in mega
            s = 1E6  # thus they need to be scaled by 1e6
        elif ms == "k":  # value is provided in kilo
            s = 1E3  # thus they need to be scaled by 1e3
        elif ms == "m":  # value is provided in milli
            s = 1E-3  # thus they need to be scaled by 1e-3
        elif ms == "u":  # value is provided in micro
            s = 1E-6  # thus they need to be scaled by 1e-6
        elif ms == "n":  # value is provided in nano
            s = 1E-9  # thus they need to be scaled by 1e-9
        elif ms == "p":  # value is print(replace. Tab to end.)ovided in pico
            s = 1E-12  # thus they need to be scaled by 1e-12
        elif ms == "f":  # value is print(replace. Tab to end.)ovided in femto
            s = 1E-15  # thus they need to be scaled by 1e-15
        else:  # unknown conversion
            s = 1  # -> no scaling
            raise ValueError(
                (f"magnitude = {ms}, unit = {unit} "
                 f"base_unit = {base_unit} ."
                 f"This case is not defined (yet?)")
            )
        
    if len(base_unit) > len(unit):
        s = 1 / s
        
    return s

def get_plot_layout(obj):
      """ Simple function which selects a row, column layout based on the number of
      objects to display.  The expected argument is a reservoir object which
      contains the list of fluxes in the reservoir

      """
      
      noo = len(obj.lof) + 1  # number of objects in this reservoir
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

def sum_fluxes(flux_list,reservoir,i):
    """This function takes a list of fluxes and calculates the sum for each
    flux element.  It will return a an array with [mass, li, hi] The
    summation is a bit more involved than I like, but I issues with summing
    small values, so here we doing is using fsum. For this step, we need to
    look up the reservoir specific flux direction which is stored as
    key-value pair in the reservoir.lio dictionary (flux name: direction)
    """
    
    ms  = 0
    ls  = 0
    hs  = 0

    for f in flux_list:  # do sum of fluxes in this reservoir
        direction = reservoir.lio[f.n]
        ms  = ms + f.m[i] * direction # current flux and direction
        ls  = ls + f.l[i] * direction # current flux and direction
        hs  = hs + f.h[i] * direction # current flux and direction

    # sum up the each array component individually
    new = array([ms, ls, hs])
    return new

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

def show_data(self,name,i) -> None:
    """ Print the first 4, and last 3 lines of the data for a given flux or reservoir object
    """
    
    # show the first 4 entries
    print(f"{name}:")
    for i in range(i,i+3):
        print(f"\t i = {i}, Mass = {self.m[i]:.2f}, LI = {self.l[i]:.2f}, HI = {self.h[i]:.2f}, delta = {self.d[i]:.2f}")
    
    print(".......................")

def plot_object_data(geo, fn, obj) -> None:
    """collection of commands which will plot and annotate a reservoir or flux
      object into an existing plot window. 
      """

    from . import ureg, Q_
    from esbmtk import Flux, Reservoir, Signal

    # geo = list with rows and cols
    # fn  = figure number
    # yl  = array with y values for the left side
    # yr  = array with y values for the right side
    # obj = object handle, i.e., reservoir or flux

    rows = geo[0]
    cols = geo[1]
    species = obj.sp
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
        yl = (obj.c * model.c_unit).to(obj.plt_units).magnitude
        y_label = f"{obj.legend_left} [{obj.plt_units:~P}]"
    elif isinstance(obj, Signal):
        # use the same units as the associated flux
        yl = (obj.c * model.c_unit).to(obj.fo.plt_units).magnitude
        y_label = f"{obj.n} [{obj.fo.plt_units:~P}]"
    else:  # sources, sinks, external data should not show up here
        raise ValueError(f"{obj.n} = {type(obj)}")

    # start subplot
    ax1 = plt.subplot(rows, cols, fn, title=obj.n)

    # set color index
    cn = 0
    col = f"C{cn}"
    # plot left y-scale data
    ln1 = ax1.plot(time[1:-2], yl[1:-2], color=col, label=obj.legend_left)

    # set labels
    ax1.set_xlabel(f"[{model.d_unit:~P}]")  # set the x-axis label
    ax1.set_ylabel(y_label)  # the y label
    ax1.spines['top'].set_visible(False)  # remove unnecessary frame species

    # set color index
    cn = cn + 1
    col = f"C{cn}"

    ax2 = ax1.twinx()  # create a second y-axis

    # plof right y-scale data
    ln2 = ax2.plot(time[1:-2], yr[1:-2], color=col, label=obj.legend_right)

    ax2.set_ylabel(obj.ld)  # species object delta label
    ax2.spines['top'].set_visible(False)  # remove unnecessary frame speciess

    # adjust display properties for title and legend
    ax1.set_title(obj.n)
    plt.rcParams['axes.titlepad'] = 14 # offset title upwards
    plt.rcParams["legend.facecolor"] = '0.8' # show a gray background
    plt.rcParams["legend.edgecolor"] = '0.8' # make frame the same color
    plt.rcParams["legend.framealpha"] = 0.4  # set transparency

    for d in obj.led:  # loop over external data objects if present
        if isinstance(d.x[0], str):  # if string, something is off
            raise ValueError("No time axis in external data object {d.name}")
        if isinstance(d.y[0],
                      str) is False:  # mass or concentration data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{obj.lm} {d.legend}"
            ln3 = ax1.scatter(d.x, d.y, color=col, label=leg)
        if isinstance(d.d[0], str) is False:  # isotope data is present
            cn = cn + 1
            col = f"C{cn}"
            leg = f"{obj.ld} {d.legend}"
            ln3 = ax2.scatter(d.x, d.d, color=col, label=leg)

    # collect all labels and print them in one legend
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(handler1 + handler2, label1 + label2,
                        loc=0).set_zorder(6)
    
    # Matplotlib will show arbitrarily small differences which can be confusing
    #yl_min = min(yl)
    #yl_max = max(yl)
    #if (yl_max - yl_min) < 0.1: