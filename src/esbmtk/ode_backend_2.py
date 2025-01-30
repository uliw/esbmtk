"""esbmtk: A general purpose Earth Science box model toolkit.

Copyright (C), 2020 Ulrich G.  Wortmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]

if tp.TYPE_CHECKING:
    from esbmtk import Model


def build_eqs_matrix(M: Model) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Build the coefficient matrix C and Flux vector f.

    So that we can solve dC_dt = C dot flux_values
    """
    i = 0
    for f in M.lof:  # get fluxe index positions
        print(f"f = {f.full_name}")
        f.idx = i  # set starting index
        if f.isotopes:
            i = i + 1  # isotopes count as additional flux
        i = i + 1

    flux_values = np.zeros(i)  # initialize flux value vector
    # Initialize Coefficient matrix, assume that all reservoirs have isotopes
    C = np.zeros((len(M.lor) * 2, i))  # Initialize Coefficient matrix:
    ri = 0
    for r in M.lor:  # loop over M.lor to build the coefficient matrix
        r.idx = ri  # record index position
        for f in r.lof:  # loop over reservoir fluxes
            # check similar code!
            sign = -1 if f.parent.source == r else 1
            C[ri, f.idx] = sign  # 1
            if f.isotopes:  # add equation for isotopes
                ri = ri + 1
                C[ri, f.idx + 1] = sign  # 2
        ri = ri + 1
    return C[:ri, :], flux_values


def write_equations_3(
    M: Model,
    R: list[float],
    icl: dict,
    cpl: list,
    ipl: list,
    eqs_fn: str,
) -> tuple:
    """Write file that contains the ode-equations for the Model.

    Returns the list R that contains the initial condition for each
    reservoir

    :param Model: Model handle
    :param R: tp.List of floats with the initial conditions for each
              reservoir
    :param icl: dict of reservoirs that have actual fluxes
    :param cpl: tp.List of reservoirs that have no fluxes but are
        computed based on other reservoirs
    :param ipl: tp.List of reservoir that do not change in concentration
    :param fn: str, filename
    """
    # construct header and static code:
    # add optional import statements
    h1 = """from __future__ import annotations
from numpy import array as npa
# from numba import njit
"""

    h2 = """# @njit(fastmath=True)
def eqs(t, R, M, gpt, toc, area_table, area_dz_table, Csat_table) -> list:
        '''Auto generated esbmtk equations do not edit
        '''
"""
    ind2 = 8 * " "  # indention
    ind3 = 12 * " "  # indention
    hi = ""

    # ensure there are no duplicates
    M.lpc_i = set(M.lpc_i)
    M.lpc_f = set(M.lpc_f)

    if len(M.lpc_f) > 0:
        hi += "from esbmtk import "
        for f in M.lpc_f:
            hi += f"{f} ,"
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if len(M.lpc_i) > 0:  # test if functions imports are required
        hi += f"from esbmtk.bio_pump_functions{M.bio_pump_functions} import "
        for f in M.lpc_i:
            hi += f"{f} ,"
        hi = f"{hi[:-2]}\n"  # strip comma and space

    if M.luf:  # test if user defined function imports are required
        for function, args in M.luf.items():
            source = args[0]
            hi += f"from {source} import {function}\n"

    header = f"{h1}{hi}\n{h2}"

    rel = ""  # list of return values
    # """
    # write file
    with open(eqs_fn, "w", encoding="utf-8") as eqs:
        eqs.write(header)
        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do not depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        """  Computed reservoirs and computed fluxes are both stored in  M.lpc_r
        At this point, we only print the reservoirs.
        TODO: This should really be cleaned up
        """
        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "std":
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)

        # write all fluxes in M.lof
        flist = []
        sep = "# ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")
        for flux in M.lof:  # loop over fluxes
            """This loop will only write regular flux equations, withe the sole
            exception of fluxes belonging to ExternalCode objects that need to be
            in a given sequence" All other fluxes must be on the M.lpr_r list
            below.
            """
            if flux.ftype == "computed":
                continue

            if flux not in flist:
                # functions can return more than one flux, but we only need to
                # call the function once
                if flux.computed_by != "None":
                    flist = flist + flux.computed_by.lof
                else:
                    flist.append(flux)  # add to list of fluxes already computed

                # include computed fluxes that need to be in sequence
                if flux.ftype == "in_sequence":
                    rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)
                    continue
                else:
                    ex, exl = get_flux(flux, M, R, icl)  # get flux expressions
                    fn = flux.full_name.replace(".", "_")
                    # all others types that have separate expressions/isotope
                    eqs.write(f"{ind2}{fn} = {ex}\n")
                    if flux.parent.isotopes:  # add line for isotopes
                        eqs.write(f"{ind2}{fn}_l =  {exl}\n")

        sep = (
            "# ---------------- write computed reservoir equations -------- #\n"
            + "# that do depend on fluxes"
        )
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list
            if r.ftype == "computed":  #
                rel = write_ef(eqs, r, icl, rel, ind2, ind3, M.gpt)

        sep = "# ---------------- write input only reservoir equations -------- #"
        eqs.write(f"\n{sep}\n")
        for r in ipl:
            rname = r.full_name.replace(".", "_")
            eqs.write(f"{ind2}{rname} = 0.0")

        sep = "# ---------------- write regular reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- write isotope reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        rel = write_reservoir_equations_with_isotopes(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- bits and pieces --------------------------- #"
        eqs.write(f"\n{sep}\n{ind2}return [\n")
        # Write all initial conditions that are recorded in icl
        for k, v in icl.items():
            eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')},  # {v[0]}\n")
            if k.isotopes:
                eqs.write(f"{ind3}dCdt_{k.full_name.replace('.', '_')}_l,  # {v[1]}\n")

        eqs.write(f"{ind2}]\n")

    return eqs_fn.stem
