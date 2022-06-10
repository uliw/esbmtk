from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Reservoir, Model, Connection, Connect


def write_equations_2(M: Model) -> list:
    """Write file that contains the ode-equations for M
    Returns the list R that contains the initial condition
    for each reservoir

    """
    from esbmtk import Model, ReservoirGroup
    import pathlib as pl

    # get list of initial conditions
    R = []
    for e in M.lic:
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

    from esbmtk import Model, ReservoirGroup

    def __init__(self, M: Model)->None:
        ''' Use this method to initialize all variables that require the state
            t-1
        '''
        import numpy as np

        self.i: int = np.zeros(len(M.time))
        self.last_t: float = np.zeros(len(M.time))

    def eqs(self, t, R: list, M: Model) -> list:
        '''Auto generated esbmtk equations do not edit
        '''

        from esbmtk import carbonate_system_1_ode, carbonate_system_2_ode
    
        # flux equations
"""

    # """
    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:

        rel = ""  # list of return values
        ind1 = 4 * " "
        ind2 = 8 * " "  # indention
        ind3 = 12 * " "  # indention

        eqs.write(header)

        eqs.write(f"{ind2}{M.name} = M\n")

        flist = list()
        for flux in M.lof:  # loop over fluxes

            # fluces blong to at least 2 reservoirs, so we need to avoid duplication
            # we cannot use a set, since we need to preserv order
            if flux not in flist:
                fex = ""
                fex = get_flux(flux, M)
                fn = flux.full_name.replace(".", "_")
                eqs.write(f"{ind2}{fn} = {fex}\n")
                eqs.write(f"{ind2}print({fn})")
                flist.append(flux)

        message = (
            f"\n{ind2}'''\n"
            f"{ind2}Carbonate System 1, has no return values, it just calculates\n"
            f"{ind2}values that are used elsewhere.\n\n"
            f"{ind2}Carbonate System 2, returns the carbonate burial/dissolution\n"
            f"{ind2}flux for DIC\n"
            f"{ind2}'''\n\n"
        )

        eqs.write(message)
        # Setup carbonate chemistry calculations
        for r in M.lpc_r:
            if r.ftype == "cs1":
                # carbonate_system_1_ode(rg: ReservoirGroup
                eqs.write(
                    f"{ind2}carbonate_system_1_ode({r.parent.full_name}, "
                    f"R[{M.lic.index(r.parent.DIC)}], "
                    f"R[{M.lic.index(r.parent.TA)}])  # cs1 \n"
                )
            elif r.ftype == "cs2":
                fn_dic = f"{r.register.DIC.full_name}.burial".replace(".", "_")
                fn_ta = f"{r.register.TA.full_name}.burial".replace(".", "_")
                influx = r.parent.cs.ref_flux[0].full_name.replace(".", "_")
                eqs.write(
                    f"{ind2}{fn_dic} = carbonate_system_2_ode(t, {r.parent.full_name}, {influx}, "
                    f"R[{M.lic.index(r.parent.DIC)}], "
                    f"R[{M.lic.index(r.parent.TA)}])  # cs2 \n"
                )
                eqs.write(f"{ind2}{fn_ta} = {fn_dic} * 2  # cs2\n")
            else:
                raise ValueError(f"{r.ftype} is undefined")

        eqs.write(f"\n{ind2}# Reservoir Equations\n")

        for r in M.lor:  # loop over reservoirs

            # create unique variable name. Reservoirs are typically called
            # M.rg.r so we replace all dots with underscore
            name = r.full_name.replace(".", "_")
            fex = ""
            for flux in r.lof:
                if flux.parent.source == r:
                    sign = "-"
                elif flux.parent.sink == r:
                    sign = "+"
                fex = fex + f"{ind3}{sign} {flux.full_name.replace('.', '_')}\n"

            # check if reservoir requires carbonate burial fluxes
            if isinstance(r.parent, ReservoirGroup):
                if r.parent.has_cs2:
                    if r.species.name == "DIC" or r.species.name == "TA":
                        fn = f"{r.full_name}.burial".replace(".", "_")
                        fex = f"{fex}{ind3}- {fn}\n"

            # avoid reservoirs without active fluxes
            if len(r.lof) > 0:
                eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
            else:
                eqs.write(f"{ind2}{name} = R[{M.lic.index(r)}]\n"),

            # but we need to return the full set
            rel = rel + f"{name}, "

        eqs.write(f"{ind2}self.i += 1\n")
        eqs.write(f"{ind2}self.last_t = t\n")
        eqs.write(f"{ind2}return [{rel}]\n")

        if len(R) != len(rel.split(",")) - 1:
            raise ValueError(
                f"number of initial conditions ({len(R)})"
                f"does not match number of return values ({len(rel.split(','))-1}')"
            )

    return R


def get_flux(flux: Flux, M: Model) -> str:
    """Create formula expression that describes the flux f
    returns ex as string
    """

    ex = ""
    c = flux.parent  # shorthand for the connection object
    cfn = flux.parent.full_name

    if c.ctype.casefold() == "regular":
        ex = f"{flux.full_name}.rate"
        ex = check_signal_2(ex, c)
        ex = ex + "  # fixed rate"

    elif c.ctype == "scale_with_concentration":
        ici = M.lic.index(c.source)  # index into initial conditions
        ex = (
            f"{cfn}.scale * R[{ici}]"  # {c.id} scale with conc in {c.source.full_name}"
        )
        ex = check_signal_2(ex, c)
        ex = ex + "  # scale with concentration"

    elif c.ctype == "scale_with_mass":
        ici = M.lic.index(c.source)  # index into initial conditions
        ex = f"{cfn}.scale * {c.source.full_name}.volume * R[{ici}]"
        ex = check_signal_2(ex, c)
        ex = ex + "  # scale with mass"

    elif c.ctype == "scale_with_flux":
        p = flux.parent.ref_flux.parent
        ex = f"{cfn}.scale * {p.full_name.replace('.', '_')}__F"
        ex = check_signal_2(ex, c)
        ex = ex + "  # scale with flux"

    elif c.ctype == "weathering":
        # c.reservoir_ref.full_name needs to be replaced with stateful reference or initial conditions?
        # how do we fine the correct R[] ?
        ici = M.lic.index(c.reservoir_ref)
        ex = f"{cfn}.rate * {cfn}.scale * (R[{ici}]/{cfn}.pco2_0) **  {cfn}.ex"
        ex = check_signal_2(ex, c)
        ex = ex + "  # weathering"

    elif c.ctype == "gas_exchange":  # Gasexchange
        ex = f"{flux.full_name}._PGex.ode()"
        ex = check_signal_2(ex, c)
        ex = ex + "  # gas_exchange"

    else:
        pass
        raise ValueError(f"{c.ctype} is not implmented")

    return ex


def check_signal_2(ex: str, c: tp.union(Connection, Connect)) -> str:
    """Test if connection requires a signal"""

    sign = ""
    ind3 = 12 * " "  # indentation
    if c.signal != "None":
        # get signal type
        if c.signal.stype == "addition":
            sign = "+"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        ex = ex + f" {sign} {c.signal.full_name}(t)[0]  # Signal"

    return ex
