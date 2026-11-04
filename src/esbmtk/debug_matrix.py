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

Debug helper that reports all fluxes of all reservoirs by name.

ESBMTK translates the model objects into a coefficient matrix ``CM`` and a
flux vector ``F``, where reservoirs and fluxes are only referred to by
index.  ``debug_matrix()`` evaluates the generated equations once and writes
the result in a form that can be inspected by eye (and in a spreadsheet).
"""

from __future__ import annotations

import csv
import logging
import re
import typing as tp
from itertools import zip_longest

import numpy as np
import pandas as pd

if tp.TYPE_CHECKING:
    from .model import Model

CATEGORIES = ("thermohaline", "mixing", "process")
TITLES = {
    "thermohaline": "Thermohaline fluxes",
    "mixing": "Mixing fluxes",
    "process": "Process fluxes (e.g., primary productivity, weathering, burial)",
}
COLUMNS = [
    "flux_in",
    "flux_in_value",
    "flux_in_delta",
    "matched_flux_out",
    "flux_out_value",
    "flux_out_delta",
    "flux_in_minus_flux_out",
    "flags",
]


class _Rec(tp.NamedTuple):
    """One flux as seen from one reservoir."""

    reservoir: tp.Any  # the reservoir object
    flux: tp.Any  # total flux object
    conn: tp.Any  # connection (or None)
    direction: str  # "in" or "out"
    value: float
    delta: float
    category: str
    partner: tp.Any  # reservoir/source/sink at the other end
    flags: list


def _is_light(f, lof: list) -> bool:
    """Return True if f is the light-isotope companion of the preceding flux."""
    i = getattr(f, "idx", None)
    if i is None or i < 1 or i >= len(lof):
        return False
    prev = lof[i - 1]
    return prev.parent is f.parent and f.id == f"{prev.id}_l"


def _mass(r) -> float:
    """Return the reservoir mass (or 1) by which CM scales fluxes."""
    from esbmtk import GasReservoir, Species

    if isinstance(r, Species):
        density = r.parent.swc.density / 1000 if hasattr(r.parent, "swc") else 1.0
        return r.volume.to(r.model.v_unit).magnitude * density
    if isinstance(r, GasReservoir):
        return r.v[0]
    return 1.0


def _delta(total: float, light: float, ratio: float) -> float:
    """Return delta (permil) or NaN if it cannot be computed."""
    if not (np.isfinite(total) and np.isfinite(light)) or light <= 0:
        return np.nan
    return float(1000 * ((total - light) / light - ratio) / ratio)


def _category(conn, thermohaline_tags, mixing_tags) -> str:
    """Classify a connection by its id."""
    cid = str(getattr(conn, "id", "")).casefold()
    if any(t.casefold() in cid for t in thermohaline_tags):
        return "thermohaline"
    if any(t.casefold() in cid for t in mixing_tags):
        return "mixing"
    return "process"


def _names(obj) -> set:
    """Return the names that may be used to refer to an object in a name."""
    n = {str(obj.name)}
    p = getattr(obj, "parent", None)
    if p is not None and hasattr(p, "name"):
        n.add(str(p.name))
    return n


def _check_flux_name(M, f) -> str | None:
    """Return a description if the flux name looks unexpected, else None."""
    from esbmtk import Species2Species

    c = f.parent
    if not isinstance(c, Species2Species):
        return f"flux is not owned by a connection (parent = {type(c).__name__})"
    if str(c.id) in ("None", ""):
        return "connection has no id"
    expected = f"{c.full_name}._F{c.id}"
    if f.full_name != expected:
        return f"expected name {expected}"
    sp = re.escape(str(c.sp.name))
    cid = re.escape(str(c.id))
    for so in _names(c.source):
        for si in _names(c.sink):
            pat = (
                rf"^{re.escape(M.name)}\.Conn_{re.escape(so)}_to_{re.escape(si)}"
                rf"_{sp}(_{cid})?\._F{cid}$"
            )
            if re.match(pat, f.full_name):
                return None
    return (
        "name does not follow Conn_<source>_to_<sink>_<species>_<id> for "
        f"source={c.source.full_name} sink={c.sink.full_name}"
    )


def _evaluate(M, state: str):
    """Evaluate the equations file once, return (t, R, F)."""
    R0, _atol, eqs = M._prepare_equations(force_temp=True)
    if state == "initial":
        t, R = float(M.time[0]), R0
    else:
        if getattr(M, "results", None) is None:
            raise ValueError(
                "state='final' requires a solved model. Call M.run() first, "
                "or use state='initial'."
            )
        t, R = float(M.results.t[-1]), M.results.y[:, -1]

    F = np.full(len(M.F), np.nan)  # NaN marks fluxes that were never written
    eqs(t, R, M, M.gpt, M.toc, M.area_table, M.area_dz_table, M.Csat_table, M.CM, F)
    return t, R, F


def _collect(M, F, thermohaline_tags, mixing_tags):
    """Collect one record per (reservoir, flux)."""
    lof = M.lof
    CM = M.CM
    recs: list[_Rec] = []
    keys: dict = {}
    pkeys: dict = {}
    for f in lof:
        if f.parent is None or _is_light(f, lof):
            continue
        c = f.parent
        if hasattr(c, "source") and hasattr(c, "sink"):
            keys.setdefault((id(c.source), id(c.sink), str(c.id)), []).append(f)
            cat = _category(c, thermohaline_tags, mixing_tags)
            if cat != "process":
                pkeys.setdefault((id(c.source), id(c.sink), cat), []).append(f)

    for r in M.lor:
        if r.rtype == "computed":
            continue
        mass = _mass(r)
        step = 2 if r.isotopes else 1
        for fi in range(0, len(r.lof), step):
            f = r.lof[fi]
            c = f.parent
            flags: list[str] = []
            has_conn = hasattr(c, "source") and hasattr(c, "sink")
            if has_conn:
                outflow = c.source == r
            else:  # e.g. fluxes of the carbonate system: sign from the matrix
                outflow = float(CM[r.idx, f.idx]) < 0
            val = float(F[f.idx])
            d = np.nan
            if r.isotopes:
                d = _delta(val, float(F[r.lof[fi + 1].idx]), r.species.r)
            if f in r.lif:
                flags.append("IGNORED")
            if not np.isfinite(val):
                flags.append("NOT_EVALUATED")
            # does the matrix agree with the connection topology?
            if has_conn:
                want = 0.0 if f in r.lif else (-1.0 if outflow else 1.0)
                have = float(CM[r.idx, f.idx]) * mass
                if not np.isclose(want, have, rtol=1e-6, atol=1e-12):
                    flags.append(f"MATRIX_COEFF={have:.3g} expected {want:.3g}")
                if len(keys.get((id(c.source), id(c.sink), str(c.id)), [])) > 1:
                    flags.append("DUPLICATE")
                cat = _category(c, thermohaline_tags, mixing_tags)
                if (
                    cat != "process"
                    and len(pkeys.get((id(c.source), id(c.sink), cat), [])) > 1
                ):
                    flags.append("PARALLEL")
                partner = c.sink if outflow else c.source
            else:
                cat, partner = "process", None
            recs.append(
                _Rec(r, f, c, "out" if outflow else "in", val, d, cat, partner, flags)
            )
    return recs


def _pair(ins: list, outs: list, by_partner: bool) -> list:
    """Pair in- and outflows of one reservoir."""
    ins = sorted(ins, key=lambda x: x.flux.full_name)
    outs = sorted(outs, key=lambda x: x.flux.full_name)
    if not by_partner:
        return list(zip_longest(ins, outs))
    rows, free = [], list(outs)
    for i in ins:
        m = next(
            (o for o in free if i.partner is not None and o.partner is i.partner), None
        )
        if m is not None:
            free.remove(m)
        rows.append((i, m))
    rows.extend((None, o) for o in free)
    return rows


def _fmt(x) -> str:
    """Format numbers for the csv file."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.6e}" if isinstance(x, float | np.floating) else str(x)


def _row(i: _Rec | None, o: _Rec | None):
    """Return the csv cells for one pair, and the flag string."""
    vi = i.value if i else 0.0
    vo = o.value if o else 0.0
    flags = (i.flags if i else []) + (o.flags if o else [])
    diff = np.nansum([vi, -vo])
    return [
        i.flux.full_name if i else "",
        _fmt(i.value) if i else "",
        _fmt(i.delta) if i else "",
        o.flux.full_name if o else "",
        _fmt(o.value) if o else "",
        _fmt(o.delta) if o else "",
        _fmt(float(diff)),
        "; ".join(flags),
    ]


def debug_matrix(
    M: Model,
    filename: str = "debug_matrix.csv",
    state: str = "initial",
    thermohaline_tags: tuple = ("thermohaline",),
    mixing_tags: tuple = ("mix",),
) -> pd.DataFrame:
    """Write all fluxes of all reservoirs to a csv file for debugging.

    The generated equations are evaluated once, using the actual
    ``eqs()`` function, so the values are those seen by the solver.
    The file contains three sections (thermohaline, mixing, and process
    fluxes), each sorted by reservoir and full flux name, followed by a
    balance check, the sources and sinks, and a list of fluxes whose name
    does not look as expected.

    Parameters
    ----------
    M : Model
        The model object. It may or may not have been run.
    filename : str
        Name of the csv file.
    state : str
        ``"initial"`` evaluates the fluxes with the initial conditions
        (works before and after ``M.run()``), ``"final"`` uses the
        state at the end of the integration (requires ``M.run()``).
    thermohaline_tags : tuple of str
        Connections whose id contains one of these are thermohaline fluxes.
    mixing_tags : tuple of str
        Connections whose id contains one of these are mixing fluxes.

    Returns
    -------
    pandas.DataFrame
        One row per (reservoir, flux) with the same information.

    Notes
    -----
    Thermohaline fluxes are unidirectional, so in- and outflows of a
    reservoir are listed side by side. For mixing and process fluxes the
    outflow that goes back to the source of the inflow is shown as the
    matching flux.  Values are in model units (mass/time), deltas in permil.
    Flags: IGNORED (flux is in the ignore list of the reservoir),
    NOT_EVALUATED (equations never set the flux), MATRIX_COEFF (the
    coefficient matrix disagrees with the connection), DUPLICATE (same
    source, sink and id twice) and PARALLEL (several thermohaline or mixing
    fluxes between the same reservoirs).
    """
    from esbmtk import Sink, Source

    state = str(state).casefold()
    if state not in ("initial", "final"):
        raise ValueError("state must be 'initial' or 'final'")

    t, _R, F = _evaluate(M, state)
    recs = _collect(M, F, thermohaline_tags, mixing_tags)
    dcdt = np.nan_to_num(M.CM.dot(np.nan_to_num(F)))

    out: list[list] = []
    flat: list[dict] = []
    head = ["reservoir"] + COLUMNS

    def line(text=""):
        out.append([text])

    line(f"# ESBMTK debug_matrix: model {M.name}, state = {state}, t = {t:g} {M.t_unit}")
    line(f"# fluxes in {M.f_unit}, delta in permil, only 'flags' needs attention")
    line()

    reservoirs = sorted({r.reservoir.full_name for r in recs})
    byres: dict = {}
    for r in recs:
        byres.setdefault((r.category, r.reservoir.full_name), []).append(r)

    for cat in CATEGORIES:
        line(f"# {TITLES[cat]} (sorted by reservoir and full flux name)")
        for rn in reservoirs:
            rr = byres.get((cat, rn), [])
            if not rr:
                continue
            line(f"## {rn}")
            out.append(["flux_in"] + COLUMNS[1:])
            ins = [x for x in rr if x.direction == "in"]
            outs = [x for x in rr if x.direction == "out"]
            for i, o in _pair(ins, outs, by_partner=cat != "thermohaline"):
                cells = _row(i, o)
                out.append(cells)
                flat.append(
                    dict(zip(head, [rn, *cells], strict=True), category=cat),
                )
            tin = float(np.nansum([x.value for x in ins if "IGNORED" not in x.flags]))
            tout = float(np.nansum([x.value for x in outs if "IGNORED" not in x.flags]))
            out.append(
                ["SUM", _fmt(tin), "", "", _fmt(tout), "", _fmt(tin - tout), ""]
            )
            line()
        line()

    # ---- balance check: topology versus coefficient matrix ---- #
    line("# Balance check: sum(in) - sum(out) versus CM.dot(F) * mass")
    out.append(["reservoir", "sum_in_minus_out", "CM_dot_F_times_mass", "difference", "flags"])
    problems: list[str] = []
    for r in M.lor:
        if r.rtype == "computed":
            continue
        mine = [x for x in recs if x.reservoir is r and "IGNORED" not in x.flags]
        net = float(
            np.nansum([x.value if x.direction == "in" else -x.value for x in mine])
        )
        mat = float(dcdt[r.idx]) * _mass(r)
        flag = "" if np.isclose(net, mat, rtol=1e-6, atol=1e-9) else "MISMATCH"
        if flag:
            problems.append(f"{r.full_name}: balance {flag}")
        out.append([r.full_name, _fmt(net), _fmt(mat), _fmt(net - mat), flag])
    line()

    # ---- sources and sinks ---- #
    line("# Sources and Sinks")
    out.append(["source_or_sink", "flux", "direction", "value", "delta", "flags"])
    seen = set()
    lof = M.lof
    for f in sorted(lof, key=lambda x: x.full_name):
        c = f.parent
        if not (hasattr(c, "source") and hasattr(c, "sink")) or _is_light(f, lof):
            continue
        for o, dirn in ((c.source, "out"), (c.sink, "in")):
            if isinstance(o, Source | Sink) and (o, f) not in seen:
                seen.add((o, f))
                d = np.nan
                if f.isotopes and f.idx + 1 < len(lof):
                    d = _delta(float(F[f.idx]), float(F[f.idx + 1]), f.species.r)
                kind = "to" if dirn == "out" else "from"
                other = c.sink if dirn == "out" else c.source
                out.append(
                    [
                        o.full_name,
                        f.full_name,
                        f"{kind} {other.full_name}",
                        _fmt(float(F[f.idx])),
                        _fmt(d),
                        "" if np.isfinite(F[f.idx]) else "NOT_EVALUATED",
                    ]
                )
    line()

    # ---- odd names ---- #
    line("# Fluxes with unexpected names")
    out.append(["flux", "reason"])
    names: dict = {}
    for f in lof:
        names.setdefault(f.full_name, []).append(f)
    n_odd = 0
    for f in sorted(lof, key=lambda x: x.full_name):
        if _is_light(f, lof):
            continue
        why = _check_flux_name(M, f)
        if len(names[f.full_name]) > 1:
            why = f"{why + '; ' if why else ''}name is used by {len(names[f.full_name])} fluxes"
        if why:
            n_odd += 1
            out.append([f.full_name, why])
    if n_odd == 0:
        line("none")
    line()

    # ---- flagged rows summary ---- #
    for r in recs:
        for fl in r.flags:
            if fl != "IGNORED":
                problems.append(f"{r.reservoir.full_name} {r.flux.full_name}: {fl}")
    line(f"# Summary: {len(problems)} flagged items, {n_odd} unexpected names")
    for p in sorted(set(problems)):
        line(p)

    with open(filename, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(out)
    logging.info(f"debug_matrix written to {filename}")
    print(f"debug_matrix: wrote {filename} ({len(problems)} flagged, {n_odd} odd names)")
    return pd.DataFrame(flat)
