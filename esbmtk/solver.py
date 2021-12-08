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
from nptyping import *
from typing import *
from numpy import array, set_printoptions, arange, zeros, interp, mean
import time
from time import process_time
import numba
from numba.core import types
from numba import njit, prange
from numba.typed import List

import numpy as np

# import pandas as pd
# import mpmath

import time
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

    start: float = process_time()
    i = 1  # some processes refer to the previous time step
    for t in time[1:-1]:  # loop over the time vector except the first
        # we first need to calculate all fluxes
        for r in lor:  # loop over all reservoirs
            for p in r.lop:  # loop over reservoir processes
                p(r, i)  # update fluxes

        for p in lpc_f:  # update all process based fluxes.
            p(i)

        for r in lor:  # loop over all reservoirs
            flux_list = r.lof

            new[0] = new[1] = new[2] = new[3] = 0
            for f in flux_list:  # do sum of fluxes in this reservoir
                direction = r.lio[f]
                new[0] = new[0] + f.m[i] * direction  # current flux and direction
                new[1] = new[1] + f.l[i] * direction  # current flux and direction

            new[2] = new[0] - new[1]
            # new[3] = delta will be set by the setitem method in the reserevoir
            # ditto for the concentration
            new = new * r.mo.dt  # get flux / timestep
            new = new + r[i - 1]  # add to data from last time step
            new = new * (new > 0)  # set negative values to zero
            r[i] = new  # update reservoir data

        # update reservoirs which do not depend on fluxes but on
        # functions
        for p in lpc_r:
            p(i)

        i = i + 1

    duration: float = process_time() - start
    print(f"\n Execution time {duration:.2e} cpu seconds\n")


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


def execute_e(model, new, lor, lpc_f, lpc_r):

    """ """

    # numba.set_num_threads(2)

    # this has nothing todo with self.time below!

    dt: float = lor[0].mo.dt

    if model.first_start:
        start: float = process_time()
        (
            model.fn_vr,
            model.input_data,
            model.vr_data,
            model.vr_params,
            model.count,
        ) = build_vr_list(lpc_r)

        model.fn, model.da, model.pc = build_process_list(lor)
        model.a, model.b, model.c, model.d, model.e = build_flux_lists_all(lor)
        model.first_start = False

        duration: float = process_time() - start
        print(f"\n Setup time {duration} cpu seconds\n")

    print("Starting solver")

    wts = time.time()
    start: float = process_time()

    if model.count > 0:
        foo(
            model.fn_vr,
            model.input_data,
            model.vr_data,
            model.vr_params,
            model.fn,
            model.da,
            model.pc,
            model.a,
            model.b,
            model.c,
            model.d,
            model.e,
            model.time[:-1],
            model.dt,
        )
    else:
        foo_no_vr(
            model.fn,
            model.da,
            model.pc,
            model.a,
            model.b,
            model.c,
            model.d,
            model.e,
            model.time[:-1],
            model.dt,
        )

    duration: float = process_time() - start
    wcd = time.time() - wts
    print(f"\n Total solver time {duration} cpu seconds, wt = {wcd}\n")


@njit(parallel=False, fastmath=True, error_model="numpy")
def foo(fn_vr, input_data, vr_data, vr_params, fn, da, pc, a, b, c, d, e, maxt, dt):

    i = 1
    for t in maxt:
        for j, f_list in enumerate(fn):
            for u, function in enumerate(f_list):
                # print(i)
                fn[j][u](da[j][u], pc[j][u], i)

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
            fn_vr[j](i, input_data[j], vr_data[j], vr_params[j])

        i = i + 1  # next time step


@njit(parallel=False, fastmath=True, error_model="numpy")
def foo_no_vr(fn, da, pc, a, b, c, d, e, maxt, dt):
    """Same as foo but no virtual reservoirs present."""
    i = 1
    for t in maxt:
        for j, f_list in enumerate(fn):
            for u, function in enumerate(f_list):
                fn[j][u](da[j][u], pc[j][u], i)

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

        i = i + 1


def build_vr_list(lvr: list) -> tuple:
    """Build lists which contain all function references for
    virtual reservoirs as well aas their input values

    """

    fn = List()  # List() # list of functions
    input_data = List()  # reservoir data
    vr_data = List()  # flux data  flux.m flux.l, flux.h, flux.d
    vr_params = List()  # list of constants
    fn = numba.typed.List.empty_list(
        types.UniTuple(types.float64, 4)(
            types.int64,  # i
            types.ListType(types.float64[::1]),
            types.ListType(types.float64[::1]),
            types.ListType(types.float64),  # a3
        ).as_type()
    )

    count = 0
    for r in lvr:  # loop over reservoir processes

        func_name, in_d, vr_d, params = r.get_process_args()
        fn.append(func_name)
        input_data.append(in_d)
        vr_data.append(vr_d)
        vr_params.append(params)
        count = count + 1

    return fn, input_data, vr_data, vr_params, count


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
        if len(r.lof) > 0:
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
    da = List()  # data
    pc = List()  # list of constants

    print(f"Building Process List")

    for r in lor:  # loop over reservoirs
        # print(f"for {r.full_name}")
        # note that types.List is differenfr from Types.ListType. Also
        # note that [::1]  declares C-style arrays see
        # https://numba.discourse.group/t/list-mistaken-as-list-when-creating-list-of-function-references/677/3
        tfn = numba.typed.List.empty_list(
            types.ListType(types.void)(  # return value
                types.ListType(types.float64[::1]),
                types.ListType(types.float64),
                types.int64,  # parameter 4
            ).as_type()
        )

        tda = List()  # temp list for data
        tpc = List()  # temp list for constants
        have_data = False
        for p in r.lop:  # loop over reservoir processes
            # print(f"working on {p.name}")
            func_name, data, proc_const = p.get_process_args(r)
            tfn.append(func_name)
            tda.append(data)
            tpc.append(proc_const)
            have_data = True

        if have_data:
            fn.append(tfn)
            da.append(tda)
            pc.append(tpc)

    return fn, da, pc
