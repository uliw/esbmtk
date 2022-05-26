from __future__ import annotations
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import EQ_Terms


class Constructor:
    """constructs the equation from the terms
    in res_flux dictionary"""

    def __init__(self, terms: EQ_Terms):
        self.terms = terms
        # tuple of resevoir ids (also res_flux keys)
        self.var_ids = tuple(terms.res_flux.keys())
        # holds the equations
        self.eqs = []
        self.ivp = []
        vars = []
        for var_id in self.var_ids:
            # reservoir id
            res_id = terms.res_flux[var_id]
            # list of variables
            vars.append(self.terms.var_dict[res_id[2]])
            # volume of the given reservoir
            vol = res_id[3].volume
            # creates the IVP
            self.ivp.append(res_id[3].concentration)

            # this is the equation
            eq = (sum(res_id[0]) - sum(res_id[1])) / vol
            self.eqs.append(eq)
        # turns the above list of variables into a tuple
        self.vars = tuple(vars)


class run_solver:
    """runs the odeint solver on the equations
    and if want_a_plot==True, it also plots the graph"""

    def __init__(self, M, want_a_plot: bool = False):
        """activates the solver and the plotting method
        M is a Model class object n esbmtk"""
        from esbmtk import EQ_Terms

        eq_terms = EQ_Terms(M)
        construct = Constructor(eq_terms)
        self._solve(construct)
        if want_a_plot:
            self.plot(construct)

    def _solve(self, K: Constructor):
        """solves the ODE system"""

        from esbmtk import Q_

        eqs = K.eqs
        vars = K.vars

        def kin(z, t):
            """the ode system function"""

            dzdt = []
            dics = {}
            for v in range(len(vars)):
                dics[vars[v]] = z[v]
            for e in range(len(eqs)):
                dzdt.append(eqs[e].subs(dics))
            return dzdt

        start = K.terms.M.start
        stop = K.terms.M.stop
        timestep = Q_(K.terms.M.timestep).magnitude  # timestep is a str
        x0 = K.ivp
        self.t = np.linspace(start, stop, int((stop - start) / timestep))
        self.sol = odeint(kin, x0, self.t)

        for var in K.vars:
            var_index = K.vars.index(var)
            res_id = int(str(var)[2:])
            K.terms.res_flux[res_id][3].c = self.sol[:, var_index]

    def plot(self, K: Constructor):
        """plots the ODE system"""

        from esbmtk import Q_

        for var in K.vars:
            var_index = K.vars.index(var)
            res_id = int(str(var)[2:])
            res_name = K.terms.res_flux[res_id][3].name

            plt.plot(self.t, self.sol[:, var_index], label=res_name)

        plt.legend(loc="best")
        plt.xlabel(f"time in {str(Q_(K.terms.M.timestep).units)}s")
        plt.ylabel("concentration in " + K.terms.M.concentration_unit)
        plt.grid()
        plt.show()
