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

# import pandas as pd
# import mpmath

import logging
import time
import builtins
import math


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
def get_delta(
    l: [NDArray, [Float64]], h: [NDArray, [Float64]], r: float
) -> [NDArray, [Float64]]:
    """Calculate the delta from the mass of light and heavy isotope
    Arguments are l and h which are the masses of the light and
    heavy isotopes respectively, r = abundance ratio of the
    respective element. Note that this equation can result in a
    siginificant loss of precision (on the order of 1E-13). I
    therefore round the results to numbers largers 1E12 (otherwise a
    delta of zero may no longer be zero)

    """
    # d = 1000 * (h / l - r) / r
    d: float = 1e3 * (np.abs(h) / np.abs(l) - r) / r
    return d


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

            new[0] = new[1] = new[2] = new[3] = 0
            for f in flux_list:  # do sum of fluxes in this reservoir
                direction = r.lio[f]
                new[0] = new[0] + f.m[i] * direction  # current flux and direction
                new[1] = new[1] + f.l[i] * direction  # current flux and direction
                # new[2] = new[2] + f.h[i] * direction  # current flux and direction

            # print(f"fsum = {new[0]:.2e}")
            # new = array([ms, ls, hs])
            new[2] = new[0] - new[1]
            # new[3] = delta will be set by the setitem method in the reserevoir
            # ditto for the concentration
            new = new * r.mo.dt  # get flux / timestep
            # print(f"{i} new = {new}, dt = {r.mo.dt}")
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
    # numba.set_num_threads(2)

    # this has nothing todo with self.time below!
    start: float = process_time()
    dt: float = lor[0].mo.dt
    fn_vr, a1, a2, a3, a4, a7 = build_vr_list(lpc_r)
    fn, rd, fd, pc = build_process_list(lor)
    a, b, c, d, e = build_flux_lists_all(lor)

    duration: float = process_time() - start
    print(f"\n Setup time {duration} cpu seconds\n")
    print("Starting solver")

    wts = time.time()
    start: float = process_time()
    foo(fn_vr, a1, a2, a3, a4, a7, fn, rd, fd, pc, a, b, c, d, e, time_array[:-1], dt)

    duration: float = process_time() - start
    wcd = time.time() - wts
    print(f"\n Total solver time {duration} cpu seconds, wt = {wcd}\n")


@njit(parallel=False, fastmath=True)
def foo(fn_vr, a1, a2, a3, a4, a7, fn, rd, fd, pc, a, b, c, d, e, maxt, dt):

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
                i, a1[j], a2[j], a3[j], a4[j]
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
        a[j][3][i] = 1e3 * (a[j][2][i] / a[j][1][i] - e[j]) / e[j]
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
    a4 = List()
    a7 = List()

    fn = numba.typed.List.empty_list(
        types.UniTuple(types.float64, 5)(
            types.int64,  # i
            types.float64[::1],  # a1
            types.float64[::1],  # a2
            types.ListType(types.float64),  # a3
            types.float64[::1],  # a4
        ).as_type()
    )

    for p in lor:  # loop over reservoir processes

        func_name, a1d, a2d, a3d, a4d, a7d = p.get_process_args()
        # print(f"fname = {func_name}")
        fn.append(func_name)
        a1.append(a1d)
        a2.append(a2d)
        a3.append(a3d)
        a4.append(a4d)
        a7.append(List(a7d))

    return fn, a1, a2, a3, a4, a7


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
    print(f"Building Process List")

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

            start: float = process_time()
            func_name, res_data, flux_data, proc_const = p.get_process_args(r)
            duration = process_time() - start
            f_time = f_time + duration

            start: float = process_time()
            tfn.append(func_name)
            trd.append(res_data)
            tfd.append(flux_data)
            tpc.append(proc_const)
            duration = process_time() - start
            d_time = d_time + duration

        fn.append(tfn)
        rd.append(trd)
        fd.append(tfd)
        pc.append(tpc)

    print(f"f_time = {f_time}")
    print(f"d_time = {d_time}")

    return fn, rd, fd, pc
