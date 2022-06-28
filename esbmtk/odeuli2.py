from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import Flux, Reservoir, Model, Connection, Connect


def get_initial_conditions(M: Model) -> tuple[list, list, list, list]:
    """get list of initial conditions. THis list needs to match
    the number of equations. We need to consider 3 types reservoirs:

    1) Reservoirs that change as a result of physical fluxes
       i.e. r.lof > 0. These require a flux statements and a
       reservoir equation.

    2) Reservoirs that do not have active fluxes but are computed
       as a tracer, i.e.. HCO3. These only require a reservoir
       equation

    3) Reservoirs that do not change but are used as input. Those
       should not happen in a well formed model, but we cannot
       exclude the possibility. In this case, there is no flux
       equation, and we state that dR/dt = 0

    see also get_ic()

    """

    R = []  # list of initial conditions
    icl: list = []  # list of reservoirs that depend on fluxes
    cpl: list = []  # list of reservoirs that are computed
    ipl: list = []  # list of static reservoirs that serve as input

    for r in M.lic:
        # print(f"R={r.full_name} lof = {len(r.lof)}")
        if len(r.lof) > 0:  # list of reservoirs that depend on fluxes
            R.append(r.c[0])
            icl.append(r)

        if r.rtype == "computed":
            R.append(r.c[0])
            icl.append(r)

    return R, icl, cpl, ipl


def write_reservoir_equations(
    eqs,
    M: Model,
    rel: str,  # string with reservoir names used in return function
    ind2: str,  # indent 2 times
    ind3: str,  # indent 3 times
) -> str:  # updated list of reservoirs
    """Loop over reservoirs and their fluxes to build the reservoir equation"""

    from esbmtk import ReservoirGroup

    for r in M.lor:  # loop over reservoirs
        # Write equations for each reservoir
        # create unique variable names. Reservoirs are typiclally called
        # M.rg.r so we replace all dots with underscore

        name = f'{r.full_name.replace(".", "_")}'
        fex = ""

        # add all fluxes
        for flux in r.lof:  # check if in or outflux
            if flux.parent.source == r:
                sign = "-"
            elif flux.parent.sink == r:
                sign = "+"

            fname = f'{flux.full_name.replace(".", "_")}'
            fex = fex + f"{ind3}{sign} {fname}\n"

        # check if reservoir requires carbonate burial fluxes
        if isinstance(r.parent, ReservoirGroup):
            if r.parent.has_cs2:
                if r.species.name == "DIC" or r.species.name == "TA":
                    fn = f"{r.full_name}.burial".replace(".", "_")
                    fex = f"{fex}{ind3}- {fn}\n"

        # avoid reservoirs without active fluxes
        if len(r.lof) > 0:
            eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
            rel = rel + f"{ind3}{name},\n"

    return rel


def write_reservoir_equations_with_istopes(
    eqs,
    M: Model,
    rel: str,  # string with reservoir names used in return function
    ind2: str,  # indent 2 times
    ind3: str,  # indent 3 times
) -> str:  # updated list of reservoirs
    """Loop over reservoirs and their fluxes to build the reservoir equation"""

    from esbmtk import ReservoirGroup

    for r in M.lor:  # loop over reservoirs
        # Write equations for each reservoir
        # create unique variable names. Reservoirs are typiclally called
        # M.rg.r so we replace all dots with underscore

        # need to query the connection and decide if isotopes are
        # needed and then only write the isotope equation?  Ort do
        # this on a per reservoir base

        name = f'{r.full_name.replace(".", "_")}_l'
        fex = ""

        # add all fluxes
        for flux in r.lof:  # check if in or outflux
            if flux.parent.source == r:
                sign = "-"
            elif flux.parent.sink == r:
                sign = "+"

            fname = f'{flux.full_name.replace(".", "_")}_l'
            fex = fex + f"{ind3}{sign} {fname}\n"

        # check if reservoir requires carbonate burial fluxes
        if isinstance(r.parent, ReservoirGroup):
            if r.parent.has_cs2:
                if r.species.name == "DIC" or r.species.name == "TA":
                    fn = f"{r.full_name}.burial".replace(".", "_")
                    fex = f"{fex}{ind3}- {fn}\n"

        # avoid reservoirs without active fluxes
        if len(r.lof) > 0:
            eqs.write(f"{ind2}{name} = (\n{fex}{ind2})/{r.full_name}.volume\n\n")
            rel = rel + f"{ind3}{name},\n"

    return rel


def write_equations_2(
    M: Model, R: list[float], icl: list, cpl: list, ipl: list
) -> tuple:
    """Write file that contains the ode-equations for M
    Returns the list R that contains the initial condition
    for each reservoir

    icl: list of reservoirs that have actual fluxes
    cpl: list of reservoirs that hjave no fluxes but are computed based on other R's
    ipl: list of reservoir that do not change in concentration

    """
    from esbmtk import Model, ReservoirGroup
    import pathlib as pl

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

        self.i = 0
        self.t = []
        self.last_t = 0

    def eqs(self, t, R: list, M: Model) -> list:
        '''Auto generated esbmtk equations do not edit
        '''

        from esbmtk import carbonate_system_1_ode, carbonate_system_2_ode
        from esbmtk import gas_exchange_ode, get_hplus

        max_i = len(M.time)-1

        # flux equations
"""

    # """
    # write file
    with open(fqfn, "w", encoding="utf-8") as eqs:

        rel = ""  # list of return values
        # ind1 = 4 * " "
        ind2 = 8 * " "  # indention
        ind3 = 12 * " "  # indention

        eqs.write(header)

        eqs.write(f"{ind2}{M.name} = M\n")

        sep = "# ---------------- write computed reservoir equations -------- #\n"
        sep = sep + "# that do not depend on fluxes"

        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list

            if r.ftype == "cs1":
                rel = write_cs_1(eqs, r, icl, rel, ind2, ind3)
            elif r.ftype == "cs2":
                pass  # see below
            else:
                raise ValueError(f"{r.ftype} is undefined")

        flist = list()

        sep = "# ---------------- write all flux equations ------------------- #"
        eqs.write(f"\n{sep}\n")

        for flux in M.lof:  # loop over fluxes
            # fluxes belong to at least 2 reservoirs, so we need to avoid duplication
            # we cannot use a set, since we need to preserve order
            if flux not in flist:
                ex, exl = get_flux(flux, M, R, icl)
                fn = flux.full_name.replace(".", "_")
                eqs.write(f"{ind2}{fn} = {ex}\n")
                flist.append(flux)

                # add equations for light isotope
                if flux.parent.isotopes:
                    fn = flux.full_name.replace(".", "_")
                    eqs.write(f"{ind2}{fn}_l = {exl}\n")
                    flist.append(flux)

        sep = "# ---------------- write computed reservoir equations -------- #\n"
        sep = sep + "# that do depend on fluxes"
        eqs.write(f"\n{sep}\n")

        for r in M.lpc_r:  # All virtual reservoirs need to be in this list

            if r.ftype == "cs1":
                pass  # see above
            elif r.ftype == "cs2":  #
                rel = write_cs_2(eqs, r, icl, rel, ind2, ind3)
            else:
                raise ValueError(f"{r.ftype} is undefined")

        sep = "# ---------------- write input only reservoir equations -------- #"
        eqs.write(f"\n{sep}\n")

        for r in ipl:
            rname = r.full_name.replace(".", "_")
            eqs.write(f"{ind2}{rname} = 0.0")

        sep = "# ---------------- write regular reservoir equations ------------ #"
        eqs.write(f"\n{sep}\n")

        rel = write_reservoir_equations(eqs, M, rel, ind2, ind3)

        sep = "# ---------------- bits and pieces --------------------------- #"
        eqs.write(
            f"\n{sep}\n"
            f"{ind2}self.i += 1\n"
            f"{ind2}self.t.extend([t])\n"
            f"{ind2}self.last_t = t\n"
            f"{ind2}return [\n"
        )
        for i, r in enumerate(icl):
            eqs.write(f"{ind3}{r.full_name.replace('.', '_')},  # {i}\n")

        eqs.write(f"{ind2}]\n")

        if len(R) != len(rel.split(",")) - 1:
            raise ValueError(
                f"number of initial conditions ({len(R)})"
                f"does not match number of return values ({len(rel.split(','))-1}')\n\n"
                f"R = {R}\n"
                f"rv = {rel}\n"
            )

    return fqfn


def get_flux(flux: Flux, M: Model, R: list[float], icl: list) -> tuple(str, str):
    """Create formula expressions that describes the flux f
    return expression ex as string
    """

    ind2 = 8 * " "  # indentation
    ind3 = 12 * " "  # indentation

    ex = ""  # expression string
    exl = ""  # expression string for light isotope
    c = flux.parent  # shorthand for the connection object
    cfn = flux.parent.full_name  # shorthand for the connection object name

    if c.ctype.casefold() == "regular":
        ex, exl = get_regular_flux(flux, c)

    elif c.ctype == "scale_with_concentration":
        ex, exl = get_scale_with_concentration(flux, c, cfn, icl)

    elif c.ctype == "scale_with_mass":
        ex, exl = get_scale_with_concentration(flux, c, cfn, icl)

    elif c.ctype == "scale_with_flux":
        ex, exl = get_scale_with_flux(flux, c, cfn, icl)

    elif c.ctype == "weathering":
        ex, exl = get_weathering(flux, c, cfn, icl, ind2, ind3)

    elif c.ctype == "gas_exchange":  # Gasexchange
        ex, exl = get_gas_exchange(flux, c, cfn, icl, ind2, ind3)

    else:
        raise ValueError(
            f"Connection type {c.ctype} for {c.full_name} is not implmented"
        )

    return ex, exl


def check_signal_2(ex: str, c: tp.union(Connection, Connect)) -> str:
    """Test if connection requires a signal"""

    sign = ""
    if c.signal != "None":
        # get signal type
        if c.signal.stype == "addition":
            sign = "+"
        elif c.signal.stype == "scale":
            sign = "*"
        else:
            raise ValueError(f"stype={c.signal.stype} not implemented")

        ex = ex + f" {sign} {c.signal.full_name}(t)[0]  # Signal"

    return ex


def get_ic(r: Reservoir, icl: list) -> str:
    """Get initial condition. If r in icl, return index
    expression into R. If r is not in index, return
    Reservoir concentration a t=0

    In both cases return these a string
    """

    s = ""

    if r in icl:
        s = f"R[{icl.index(r)}]"
    else:
        s = f"{r.full_name}.c[0]"

    return s


# ------------------------ define processes ------------------------- #


def write_cs_1(eqs, r: Reservoir, icl: list, rel: str, ind2: str, ind3: str) -> list:
    """Write the python code that defines carbonate system 1"""

    fname = f"{r.parent.full_name}.Hplus".replace(".", "_")
    cname = f"{r.parent.full_name}.CO2aq".replace(".", "_")

    eqs.write(
        f"{ind2}{fname}, {cname} = carbonate_system_1_ode(\n"
        f"{ind3}{r.parent.full_name},\n"
        f"{ind3}{get_ic(r.parent.DIC, icl)},\n"
        f"{ind3}{get_ic(r.parent.TA, icl)},\n"
        f"{ind3}{get_ic(r.parent.Hplus, icl)},\n"
        f"{ind3}self.i,\n"
        f"{ind3}max_i)  # cs1\n"
    )
    rel = rel + f"{ind3}{fname},\n"

    return rel


def write_cs_2(eqs, r: Reservoir, icl: list, rel: str, ind2: str, ind3: str) -> list:
    """Write the python code that defines carbonate system 2"""

    fn_dic = f"{r.register.DIC.full_name}.burial".replace(".", "_")
    fn_ta = f"{r.register.TA.full_name}.burial".replace(".", "_")
    influx = r.parent.cs.ref_flux[0].full_name.replace(".", "_")
    fname = f"{r.parent.full_name}.Hplus".replace(".", "_")
    zname = f"{r.parent.full_name}.zsnow".replace(".", "_")
    zmax = r.parent.cs.function_params[15]
    eqs.write(
        f"{ind2}{fn_dic}, {fname}, {zname} = carbonate_system_2_ode(\n"
        f"{ind3}t,\n{ind3}{r.parent.full_name},\n{ind3}{influx},\n"
        f"{ind3}{get_ic(r.parent.DIC, icl)},\n"
        f"{ind3}{get_ic(r.parent.TA, icl)},\n"
        f"{ind3}{get_ic(r.parent.Hplus, icl)},\n"
        f"{ind3}{get_ic(r.parent.zsnow, icl)},\n"
        f"{ind3}self.i,\n"
        f"{ind3}max_i,\n"
        f"{ind3}self.last_t,\n"
        f"{ind2})  # cs2\n"
        f"{ind2}{fn_ta} = {fn_dic} * 2  # cs2\n"
        f"{ind2}# Limit zsnow >= zmax\n"
        f"{ind2}if {get_ic(r.parent.zsnow, icl)} > {zmax}:"
        f" {get_ic(r.parent.zsnow, icl)} = {zmax}\n"
    )
    # add Hplus to the list of return values
    rel = rel + f"{ind3}{fname},\n"
    rel = rel + f"{ind3}{zname},\n"

    return rel


def get_regular_flux(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
) -> tuple:
    """Equation defining a fixed rate flux
    Example:

    M1_volcanic_flux__F = M1.C_Fw_to_CO2_At_DIC_volcanic_flux._F.rate

    """

    ex = f"{flux.full_name}.rate"
    ex = check_signal_2(ex, c)
    ex = ex + "  # fixed rate"

    if c.isotopes:
        exl = ""
    else:
        exl = ""

    return ex, exl


def get_scale_with_concentration(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: list,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration in the upstream
    reservoir

    Example:

    M1_CG_D_b_to_L_b_TA_thc__F = M1.CG_D_b_to_L_b.TA_thc.scale * R[5]
    """

    ici = icl.index(c.source)  # index into initial conditions
    # this should probably handled by get_ic() see below

    ex = f"{cfn}.scale * R[{ici}]"  # {c.id} scale with conc in {c.source.full_name}"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with concentration"

    if c.isotopes:
        exl = ""
    else:
        exl = ""

    return ex, exl


def get_scale_with_mass(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: list,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration in the upstream
    reservoir

    Example:

    M1_CG_D_b_to_L_b_TA_thc__F = M1.CG_D_b_to_L_b.TA_thc.scale * R[5]
    """

    ici = icl.index(c.source)  # index into initial conditions
    ex = f"{cfn}.scale * {c.source.full_name}.volume * R[{ici}]"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with mass"
    exl = ""
    if c.isotopes:
        exl = ""
    else:
        exl = ""

    return ex, exl


def get_scale_with_flux(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: list,  # list of initial conditions
) -> tuple(str, str):
    """Equation defining a flux that scales with strength of another flux
    reservoir

    Example:

    M1_C_Fw_to_L_b_TA_wca_ta__F = M1.C_Fw_to_L_b_TA_wca_ta.scale * M1_C_Fw_to_L_b_DIC_Ca_W__F
    """

    p = flux.parent.ref_flux.parent

    ex = f"{cfn}.scale * {p.full_name.replace('.', '_')}__F"
    ex = check_signal_2(ex, c)
    ex = ex + "  # scale with flux"

    if c.isotopes:
        exl = ""
    else:
        exl = ""

    return ex, exl


def get_weathering(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: list,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of pcO2

    F = F_0 (pcO2/pCO2_0) * c, see Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    Example:

     M1_C_Fw_to_L_b_DIC_Ca_W__F = (
            M1.C_Fw_to_L_b_DIC_Ca_W.rate
            * M1.C_Fw_to_L_b_DIC_Ca_W.scale
            * (R[10]/M1.C_Fw_to_L_b_DIC_Ca_W.pco2_0)
            **  M1.C_Fw_to_L_b_DIC_Ca_W.ex
        )

    """

    ici = icl.index(c.reservoir_ref)
    ex = (
        f"(\n{ind3}{cfn}.rate\n"
        f"{ind3}* {cfn}.scale\n"
        f"{ind3}* (R[{ici}]/{cfn}.pco2_0)\n"
        f"{ind3}**  {cfn}.ex\n"
        f"{ind2})"
    )
    ex = check_signal_2(ex, c)
    ex = ex + "  # weathering\n"

    if c.isotopes:
        exl = ""
    else:
        exl = ""

    return ex, exl


def get_gas_exchange(
    flux: Flux,  # flux instance
    c: Connect,  # connection instance
    cfn: str,  # full name of the connection instance
    icl: list,  # list of initial conditions
    ind2: str,
    ind3: str,
) -> tuple(str, str):
    """Equation defining a flux that scales with the concentration of pcO2
    see Zeebe, 2012, doi:10.5194/gmd-5-149-2012

    M1_C_H_b_to_CO2_At_gex_hb_F = gas_exchange_ode(
            M1.C_H_b_to_CO2_At.scale,
            R[10],  # pco2 in atmosphere
            M1.C_H_b_to_CO2_At.water_vapor_pressure,
            M1.C_H_b_to_CO2_At.solubility,
            M1_H_b_CO2aq, # [co2]aq
            )  # gas_exchange

    """

    # get co2_aq reference
    co2aq = f"{c.liquid_reservoir.parent.full_name}.CO2aq".replace(".", "_")
    # get atmosphere pcO2 reference
    pco2 = get_ic(c.gas_reservoir, icl)
    ex = (
        f"gas_exchange_ode(\n"
        f"{ind3}{cfn}.scale,\n"
        f"{ind3}{pco2},\n"
        f"{ind3}{cfn}.water_vapor_pressure,\n"
        f"{ind3}{cfn}.solubility,\n"
        f"{ind3}{co2aq},\n"
        f"{ind3})"
    )
    ex = check_signal_2(ex, c)
    ex = ex + "  # gas_exchange\n"

    if c.isotopes:
        exl = ""
    else:
        exl = ""

    exl = ""

    return ex, exl
