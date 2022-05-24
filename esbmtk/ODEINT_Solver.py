from .esbmtk import Q_, Model, Reservoir, Connect, Source, Sink
from .Constructor import Constructor
from .Equation_Terms import EQ_Terms
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

class run_solver:
      def __init__(self, M: Model, want_a_plot: bool=False):

            eq_terms = EQ_Terms(M)
            construct = Constructor(eq_terms)
            self._solve(construct)
            if want_a_plot:
                  self.plot(construct)

      def _solve(self, K: Constructor):
            '''solves the ODE system and plots it'''
            eqs = K.eqs
            vars = K.vars

            def kin(z, t):
                  dzdt = []
                  dics = {}
                  for v in range(len(vars)):
                        dics[vars[v]] = z[v]
                  for e in range(len(eqs)):
                        dzdt.append(eqs[e].subs(dics))
                  return dzdt

            start = K.terms.M.start
            stop = K.terms.M.stop
            timestep = Q_(K.terms.M.timestep).magnitude # timestep is a str
            x0 = K.ivp
            self.t = np.linspace(start, stop, int((stop-start)/timestep))
            self.sol = odeint(kin, x0, self.t)
            for var in K.vars:
                  var_index = K.vars.index(var)
                  res_id = int(str(var)[2:])
                  K.terms.res_flux[res_id][3].c = self.sol[:, var_index]


      def plot(self, K: Constructor):
            for var in K.vars:
                  var_index = K.vars.index(var)
                  res_id = int(str(var)[2:])
                  res_name = K.terms.res_flux[res_id][3].name

                  plt.plot(self.t, self.sol[:, var_index],  label=res_name)

            plt.legend(loc='best')
            plt.xlabel(f'time in {str(Q_(K.terms.M.timestep).units)}s')
            plt.ylabel('concentration in ' + K.terms.M.concentration_unit)
            plt.grid()
            plt.show()
