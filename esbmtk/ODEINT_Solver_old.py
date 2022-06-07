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

from __future__ import annotations
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import typing as tp

if tp.TYPE_CHECKING:
    from esbmtk import EQ_Terms


class Constructor:
    """This class is for constructing the equations
    when the EQ_Terms object is inputted together
    with the res_flux dictionary (which is an attribute).
    
    It constructs the equations by looping over the tuples
    in the res_flux dictionary and for every tuple,
    it summs up the elements of the first list and sums the
    elementrs of the second list and subtracts the second
    sum from the first and divides the result by the volume
    of the reservoir for whovh the equation is constructed.
    Thus getting a sympy evaluable expression of the form
    
    concentration = 
    (inflowing-terms - outflowing-terms)/volume-of-reservoir
    
    and appends a list to include these expressions in order
    """

    def __init__(self, terms: EQ_Terms):
        """The entire aforementioned process
        is carried during the initialization of
        the Constructor object"""
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
    """In order to generate the equations, solve the
    system, and plot it the wanted Model M is inputted
    for initialization.
    
     It utelizes the EQ_Terms and Constructor classes
     to first generate the terms of the equations and store
     them in res_flux and later inputs this EQ_Terms object into
     Constructor to generate the system itself.
     
     The class uses scipy.integrate.odeint package
     to solve the system of ODEs.
     
     If want_a_plot = True during initialization,
     then the plot will be provided but the default
     if False."""

    def __init__(self, M, want_a_plot: bool = False):
        """Activates the solver and the plotting methods
        M is a Model class object in esbmtk"""
        from esbmtk import EQ_Terms

        eq_terms = EQ_Terms(M)
        construct = Constructor(eq_terms)
        self._solve(construct)
        if want_a_plot:
            self.plot(construct)

    def _solve(self, K: Constructor):
        """Solves the ODE system.
        The K is the Constructor object
        from __init__
        """

        from esbmtk import Q_

        eqs = K.eqs
        vars = K.vars

        def kin(z, t):
            """The ode system function.
            Computes the derivative of z at t."""

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
        """Plots the output of the _solve method
        if want_a_plot = True. It plots all of the
        plots on one graph."""

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
