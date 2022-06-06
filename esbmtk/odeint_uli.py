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


def check_signal(ex: str, c: tp.union(Connection, Connect)) -> str:
    """Test if connection requires a signal"""

    sign = ""
    ind3 = 12 * " "  # indentation
    if c.signal != "None":
        # get signal type
        print(f"signame =  {c.signal.full_name}")
        if c.signal.stype == "addition":
            sign = "+"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        ex = ex + f"\n{ind3}{sign} {c.signal.full_name}(t)[0]  # Signal"

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

    elif c.ctype == "scale_with_mass":
        ici = M.lic.index(c.source)  # index into initial conditions
        ex = (
            f"{cfn}.scale * {c.source.full_name}.volume * R[{ici}]"
            f"# {c.id} scale with conc in {c.source.full_name}"
        )
        ex = check_signal(ex, c)

    elif c.ctype == "scale_with_flux":
        p = f.parent.ref_flux.parent
        ici = M.lic.index(c.source)  # index into initial conditions

        if f.parent.ref_flux.parent.ctype == "scale_with_concentration":
            ex = (
                f"{cfn}.scale * R[{ici}] * {p.full_name}.scale"
                f"# {f.parent.id} scale with f in {c.source.full_name}"
            )
        elif f.parent.ref_flux.parent.ctype == "scale_with_mass":
            ex = (
                f"{cfn}.scale * R[{ici}] * {p.full_name}.scale * {p.full_name}.source.volume"
                f"# {f.parent.id} scale with f in {c.source.full_name}"
            )
        else:
            raise ValueError(f"{f.parent.ref_flux.parent.ctype} is not implmented")
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

    # construct header and static code:
    header = """from __future__ import annotations\n\n
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Model


class setup_ode():
    '''Class stub to enable state in the equation system passed to ODEINT
    '''

    def __init__(self):
        ''' Use this method to initialize all variables that require the state
            t-1
        '''
    
        self.i: int = 0
        self.last_t: float = 0.0

    def eqs(self, t, R: list, M: Model) -> list:
        '''Auto generated esbmtk equations do not edit
        '''

"""

    # """
    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:

        rel = ""  # list of return values
        ind1 = 4 * " "
        ind2 = 8 * " "  # indention
        ind3 = 12 * " "  # indention

        eqs.write(header)

        for r in M.lor:  # loop over reservoirs

            # create unique variable name. Reservoirs are typically called
            # M.rg.r so we replace all dots with underscore
            name = r.full_name.replace(".", "_")
            fex = ""
            for f in r.lof:
                ex, sign = connection_types(f, r, R, M)
                fex = fex + f"{ind3}{sign} {ex}\n"

            eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
            rel = rel + f"{name}, "

        eqs.write(f"{ind2}return [{rel}]\n")

    return R
