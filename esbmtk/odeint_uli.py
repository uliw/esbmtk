"""
     esbmtk/odeint_uli: A general purpose Earth Science box model toolkit
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
from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Reservoir, Model, Connection, Connect


def check_signal(ex: str, c: union(Connection, Connect)) -> str:
    """Test if connection requires a signal"""

    sign = ""
    if c.signal != "None":
        # get signal type
        print(f"signame =  {c.signal.full_name}")
        if c.signal.stype == "addition":
            sign = "+"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        ex = ex + f"\n\t{sign} {c.signal.full_name}(t)[0]  # Signal"

    return ex


def connection_types(f: Flux, r: Reservoir, R: list, M: Model) -> tuple(str, str):
    """Create formula expression that describes the flux f
    returns ex as string
    """
    ex = ""
    c = f.parent  # shorthand for the connection object
    cfn = f.parent.full_name

    if c.source == r:
        sign = "-"
    elif c.sink == r:
        sign = "+"
    else:
        raise ValueError("This should not happen")

    if c.ctype == "regular":
        # ex = f"{c.rate:.16e}  # {c.id} rate"
        ex = f"{cfn}.rate  # {c.id} rate"
        ex = check_signal(ex, c)

    elif c.ctype == "scale_with_concentration":
        ici = M.lic.index(c.source)  # index into initial conditions
        ex = f"{cfn}.scale * R[{ici}]  # {c.id} scale with conc in {c.source.full_name}"
        ex = check_signal(ex, c)

    elif c.ctype == "scale_with_flux":
        p = f.parent.ref_flux.parent
        ici = M.lic.index(c.source)  # index into initial conditions
        #ex = f"{c.scale:.16e} * R[{ici}] * {f.parent.scale:.16e}  # {f.parent.id} scale with c in {c.source.full_name}"
        # ex = check_signal(ex, c)
        ex = f"{cfn}.scale * R[{ici}] * {p.full_name}.scale # {f.parent.id} scale with f in {c.source.full_name}"
        ex = check_signal(ex, c)

    else:
        raise ValueError(f"{c.ctype} is not implmented")

    return ex, sign


def write_equations(M: Model) -> list:
    """Write file that contains the ode-equations for M
    Returns the list R that contains the initial condition
    for each reserevoir

    """
    import pathlib as pl

    # get list of initial conditions
    R = []
    for e in M.lor:
        R.append(e.c[0])

    # get pathlib object
    fn: str = "equations.py"  # file name
    cwd: pl.Path = pl.Path.cwd()  # get the current working directory
    fqfn: pl.Path = pl.Path(f"{cwd}/{fn}")  # fully qualified file name

    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:
        eqs.write("from __future__ import annotations\n\n\n")
        eqs.write("def eqs(t, R: list, M: Model) -> list:\n\n")
        eqs.write('\t"""Auto generated esbmtk equations do not edit"""\n\n')

        rel = ""  # list of return values
        for r in M.lor:  # loop over reservoirs

            fex = ""
            for f in r.lof:
                ex, sign = connection_types(f, r, R, M)
                fex = fex + f"\t{sign} {ex}\n"

            eqs.write(f"\t{r.name} = (\n{fex}\t)/{r.volume:.16e}\n\n")
            rel = rel + f"{r.name}, "

        eqs.write(f"\treturn [{rel}]\n")

    return R
