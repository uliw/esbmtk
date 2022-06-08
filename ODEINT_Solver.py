"""
 esbmtk: A general purpose Earth Science box model toolkit
 Copyright (C), 2022 Ruben Navasardyan

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
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import typing as tp
if tp.TYPE_CHECKING:
    from esbmtk import Model

class run_solver:
    def __init__(self, M: Model, want_a_plot: bool = False):
        from esbmtk import EQ_Terms, Construct

        eq_terms = EQ_Terms(M)
        construct = Construct(eq_terms)

        self._solve(construct)
        if want_a_plot:
            self.plot(construct)

    def _solve(self, K: Constructor):
        """solves the ODE system and plots it"""
        vars = K.vars

        def kin(z, t):
            """evaluates the system at
            the given z and t values"""
            dics = {}
            dics["t"] = t
            for v in range(len(vars)):
                dics[vars[v]] = z[v]
            return K.sum_vol(dics)

        x0 = K.ivp
        self.t = K.terms.M.time
        self.sol = odeint(kin, x0, self.t)
        for var in K.vars:
            var_index = K.vars.index(var)
            res_id = int(var[2:])
            K.terms.res_flux[res_id][3].c = self.sol[:, var_index]

    def plot(self, K: Constructor):
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
