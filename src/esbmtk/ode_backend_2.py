from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt

if tp.TYPE_CHECKING:
    from esbmtk import Model

# declare numpy types
NDArrayFloat = npt.NDArray[np.float64]


def set_tolerance(v: float, rtol: float) -> float:
    """Set tolerance values for solve_ivp. See the solve_ivp doumentation

    :param v: value if inititial condition
    :param rtol: relative tolerance
    :return atol: absolute tolerance

    """
    if v > 0.0:
        atol = rtol * abs(v) / 10.0
    else:
        atol = 1e-6
    return atol


def sort_fluxes_by_type(M) -> dict:
    """Sort all model fluxes in to two lists, one containig all constant fluxes,
    and one containing all computed fluxes (i.e. those that require a function).
    All others, will be collected in the 'other' list.

    :param M: Model handle

    :returns: a dictionary of lists containhg the respective flux/connection types
    """
    f_dict = {
        "variable": [],
        "ref": [],
        "fixed": [],
        "other": [],
    }

    for f in M.lof:
        if f.register.ctype == "weathering".lower():
            f_dict["variable"].append(f)
        elif f.register.ctype == "scale_with_concentration".lower():
            f_dict["other"].append(f)
        elif f.register.ctype == "scale_with_flux".lower():
            f_dict["ref"].append(f)
        elif f.register.ctype == "regular".lower():
            f_dict["fixed"].append(f)
        else:
            raise ValueError(
                f"{f.register.ctype} is not implmented in sort_fluxes_by_type"
            )

    return f_dict


def get_types(M: Model, rtol: float) -> tuple(NDArrayFloat, NDArrayFloat, dict, dict):
    """Create the data that is needed to assemble the equation matrix

    :param M: Model hande
    :param rtol: relative tolerance,

    :return atol: NDArrayFloat of absolute tolerances
    :return IC: NDArrayFloat of initial conditions and constant/computed Fluxes
    :return IC_i: dict of reservoir/flux index positions
    :return f_dict: dictionary of flux types
    """
    from esbmtk import Source, Sink

    atol = list()  # list of tolerances
    IC = list()  # IC = r1.l, r1.c, r2.l, r3.l etc)
    IC_i = dict()  # IC_i[r] = index position of r in IC
    M.num_ic = dict()

    # get initial conditions tolerances and indexes for each reservoir
    i = 0
    for r in M.lic:
        IC.append(r.c[0])
        atol.append(set_tolerance(r.c[0], rtol))
        IC_i[r.full_name] = i
        i = i + 1
        if r.isotopes:
            IC.append(r.l[0])
            atol.append(set_tolerance(r.l[0], rtol))
            IC_i[f"{r.full_name}_i"] = i
            i = i + 1

    M.num_ic["reservoirs"] = i

    # get fluxes sorted by connection type and enter them in list of
    # initial conditions
    f_dict = sort_fluxes_by_type(M)

    types_to_append = ["fixed"]
    for tta in types_to_append:
        if tta in f_dict:
            for f in f_dict[tta]:
                # print(f"setting {f.full_name} with {f.rate}")
                IC.append(f.rate)  # add constant to inittial conditions
                atol.append(set_tolerance(r.m[-1], rtol))
                if isinstance(f.register.source, Source):
                    IC_i[f.register.source.full_name] = i
                elif isinstance(f.register.sink, Sink):
                    IC_i[f.register.sink.full_name] = i
                else:
                    IC_i[f.full_name] = i

                i = i + 1
                M.num_ic[tta] = i

    return np.asarray(atol, dtype=float), np.asarray(IC, dtype=float), IC_i, f_dict


def get_matrix_coordinates(
    source,
    sink,
    IC_i,
    scale: float,
    c,
    bypass=False,
) -> None:
    """Set matrix coefficients for source and sinks.

    :param C: Coefficient Matrix
    :param row: row index
    :param col: cold index
    :param scale: value
    :param bypass: whether to bypass the sink term
    """
    from esbmtk import Source, Sink

    coords: list[list] = list()

    if isinstance(c.source, Source):
        row = IC_i[sink]  # this will set the row
        col = IC_i[source]
        coords.append([row, col, scale])
    elif isinstance(c.sink, Sink):
        row = IC_i[source]  # this will set the row
        col = IC_i[sink]
        coords.append([row, col, -scale])
    else:
        row = IC_i[source]  # this will set the row
        col = IC_i[source]
        coords.append([row, col, -scale])

        if not bypass:
            row = IC_i[sink]  # this will set the row
            col = IC_i[source]
            coords.append([row, col, +scale])

    return coords


def set_matrix_coefficients(C, coords):
    """Set matrix coefficients. If values equals zero, set with scale.
    If value is unequal zero, add to existing value

    :param C: Matrix
    :param coords: Matrix coordinates and values [[0,1,1e4],[1,0,1e3]] etc,

    """
    for c in coords:
        # print(f"row = {c[0]}, col = {c[1]}, val = {c[2]:.2e}")
        if C[c[0], c[1]] == 0:
            C[c[0], c[1]] = c[2]
        else:
            # print(f"C before = {C}")
            # print(
            #     f"current = {C[c[0], c[1]]:.2e}, new = {c[2]:.2e}, future = {C[c[0], c[1]] - c[2]:.2e}"
            # )
            C[c[0], c[1]] += c[2]
            # print(f"C after = {C}\n")
        # print(C)


def build_matrix(M: Model, IC: NDArrayFloat, IC_i: dict, f_dict: dict):
    """Create the initial co-efficient Matrix C[rows, cols]), where
    rows = number of reservoirs
    cols = number of initial condition + fixed an computed fluxes

    :param M: Model handle
    :param IC: array of initial conditions and fixed parameters
    :param IC_i: dict of initial condtions index positions
    :param f_dict: dict of flux_types

    :return C: coefficient matrix
    """
    # create initial coefficient matrix
    rows = len(M.lor)
    cols = len(IC)
    C = np.zeros((rows, cols), dtype=float)

    # loop over reservoirs
    for f in M.lof:
        c = f.register  # get connection

        if c.ctype == "scale_with_flux":
            source = c.ref_flux.register.source.full_name
            sink = c.sink.full_name
            scale = c.ref_flux.register.scale * c.scale
            print(f"source = {source}")
            print(f"sink = {sink}")
        else:
            source = c.source.full_name
            sink = c.sink.full_name
            scale = c.scale

        coords = get_matrix_coordinates(
            source,
            sink,
            IC_i,
            scale,
            c,
        )
        set_matrix_coefficients(C, coords)
    return C
