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
from __future__ import annotations

# from pint import UnitRegistry
# from nptyping import *
from time import process_time
import numpy as np
import numpy.typing as npt
import typing as tp
from numba import njit

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from esbmtk import Reservoir, GasReservoir, Flux

@njit(fastmath=True)
def get_l_mass(m: float, d: float, r: float) -> float:
    """
    :param m: mass or concentration
    :param d: delta value
    :param r: isotopic reference ratio

    return mass or concentration of the light isotopeb
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def get_imass(m: float, d: float, r: float) -> [float, float]:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    species

    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l
    return [l, h]


@njit(fastmath=True)
def get_li_mass(m: float, d: float, r: float) -> float:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    species
    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    return l


@njit(fastmath=True)
def get_frac(m: float, l: float, a: float) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha on
    the ratio between the light and heavy isotope.

    Note that alpha needs to be given as fractional value, i.e., 1.07 rather
    than 70 (i.e., (alpha-1) * 1000

    """

    # if a.any() > 1.1 or a.any()  < 0.9:
    #     raise ValueError("alpha needs to be given as fractional value not in permil")

    li: float = -l * m / (a * l - a * m - l)
    hi: float = m - li  # get the new heavy isotope value
    return li, hi


@njit(fastmath=True)
def get_new_ratio_from_alpha(
    ref_mass: float,  # reference mass
    ref_l: float,  # reference light istope
    a: float,  # fractionation factor
) -> [float, float]:
    """Calculate the effect of the istope fractionation factor alpha on
    the ratio between the mass of the light isotope devided by the total mass

    Note that alpha needs to be given as fractional value, i.e., 1.07 rather
    than 70 (i.e., (alpha-1) * 1000
    """

    if ref_mass > 0.0:
        new_ratio = -ref_l / (a * ref_l - a * ref_mass - ref_l)
    else:
        new_ratio = 0.0

    return new_ratio


@njit(fastmath=True)
def get_flux_data(m: float, d: float, r: float) -> NDArrayFloat:
    """
    Calculate the isotope masses from bulk mass and delta value.
    Arguments are m = mass, d= delta value, r = abundance ratio
    species. Unlike get_mass, this function returns the full array

    """

    l: float = (1000.0 * m) / ((d + 1000.0) * r + 1000.0)
    h: float = m - l

    return np.array([m, l, h, d])


@njit(fastmath=True)
def get_delta(l: NDArrayFloat, h: NDArrayFloat, r: float) -> NDArrayFloat:
    """Calculate the delta from the mass of light and heavy isotope

    :param l: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    return 1000 * (h / l - r) / r


def get_delta_from_concentration(c, l, r):
    """Calculate the delta from the mass of light and heavy isotope

    :param c: total mass/concentration
    :param l: light isotope mass/concentration
    :param r: reference ratio

    """
    h = c - l
    d = 1000 * (h / l - r) / r
    return d


def get_delta_i(l: float, h: float, r: float) -> float:
    """Calculate the delta from the mass of light and heavy isotope
    Arguments are l and h which are the masses of the light and
    heavy isotopes respectively, r = abundance ratio of the
    respective element. Note that this equation can result in a
    siginificant loss of precision (on the order of 1E-13). I
    therefore round the results to numbers largers 1E12 (otherwise a
    delta of zero may no longer be zero)

    """
    return 1e3 * (h / l - r) / r if l > 0 else 0


def get_flux_delta(f) -> float:
    """Calculate the delta of flux f"""

    m = f.fa[0]
    l = f.fa[1]
    h = m - l
    r = f.species.r
    return 1e3 * (h / l - r) / r


def get_delta_h(R) -> float:
    """Calculate the delta of a flux or reservoir

    :param R: Reservoir or Flux handle

    returns d as vector of delta values
    R.c = total concentration
    R.l = concentration of the light isotope
    """

    from esbmtk import Reservoir, GasReservoir, Flux

    r = R.species.r  # reference ratio
    if isinstance(R, (Reservoir,GasReservoir)):
        d = np.where(R.l > 0, 1e3 * ((R.c - R.l) / R.l - r) / r, 0)
    elif isinstance(R, Flux):
        d = np.where(R.l > 0, 1e3 * ((R.m - R.l) / R.l - r) / r, 0)
    else:
        raise ValueError(
            f"{R.full_name} must be of type Flux or Reservoir, not {type(R)}"
        )

    return d


def execute(
    time: NDArrayFloat,
    lop: list,
    lor: list,
    lpc_f: list,
    lpc_r: list,
) -> None:
    """This is the original object oriented solver"""

    start: float = process_time()
    i = 1  # some processes refer to the previous time step
    for _ in time[1:-1]:
        # we first need to calculate all fluxes
        # for r in lor:  # loop over all reservoirs
        #     for p in r.lop:  # loop over reservoir processes
        #         p(i)  # update fluxes

        for p in lop:  # loop over reservoir processes
            p(i)  # update fluxes

        for p in lpc_f:  # update all process based fluxes.
            p(i)

        for r in lor:  # loop over all reservoirs
            flux_list = r.lof

            m = l = 0
            for f in flux_list:  # do sum of fluxes in this reservoir
                direction = r.lio[f]
                m += f.fa[0] * direction  # current flux and direction
                l += f.fa[1] * direction  # current flux and direction

            r.m[i] = r.m[i - 1] + m * r.mo.dt
            r.l[i] = r.l[i - 1] + l * r.mo.dt
            r.c[i] = r.m[i] / r.v[i]

        # update reservoirs which do not depend on fluxes but on
        # functions
        for p in lpc_r:
            p(i)

        i = i + 1

    duration: float = process_time() - start
    print(f"\n Execution time {duration:.2e} cpu seconds\n")
