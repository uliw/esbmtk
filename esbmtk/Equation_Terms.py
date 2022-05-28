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
from sympy import Symbol
import typing as tp

if tp.TYPE_CHECKING:
    from typing import Union


class EQ_Terms:

    """Given a Model class object M, this class generates the terms of
    the system of ODE equations where each equation corresponds to the rate
    of change of concentration of a specific substance in a specific reservoir
    
    The class further sorts these terms of equations into two different lists
    stored within a tuple of the form (list, list, str, Reservoir). These tuples
    are values in a dictionary called res_flux where the keys are corresponding
    reservoir memory ids.
    
    In any given tuple, the first list holds the terms of an equation that represent
    an influx into the reservoir (hence, substance is being added to the reservoir).
    The second list holds terms that represent an outflux out of a reservoir, so 
    substance is being removed out of reseroir
    
    The str in the suple is of the form x_id(Reservoir) and will be the x in the dx/dt
    terms in the equation that will represent the concentration of a substance in the
    reservoir. We use the reservoir id to guarantee sufficiency and uniqueness of x terms
    in every equation within the system.
    
    Lastly, The last item in the tuple is the reservoir itself for which the equation is built.
    """

    def __init__(self, M):
        """During initialization, the empty res_flux dictionary is created and later
        filled. It there are different types of equation terms and these types are dependednt
        on the connection types.
        
        - regular: the flow is constant and the rate is given
            then scale_w_concentration_or_regular will be used
        - scale_with_concentration: rate is given so scale_w_concentration_or_regular
            will be used
        - scale_with_concentration: rate is NOT given so variable_rate is used
            in this case, we assume that since the rate is not provided, then it is
                 a variable and may change over time
        - scale_with_mass: scale_with_mass method will be used to generate the eq term
        - scale_with_flux: use scale_with_flux method and provide the reference flux
            as this case assumes that the eq terms will be a scaled offo of an already
            existing flux (hence an equation term)"""

        self.res_flux = (
            {}
        )  # dict that holds variables, reservoirse and in/out-flux (keys are int [res ids])
        self.var_dict = {}  # dict that holds variables {'x_123':Symbol('x_123')}
        self.M = M

        for res in self.M.lor:
            # influx, outflux lists, concentration variable for a reservoir
            var_str = f"x_{id(res)}"
            self.res_flux[id(res)] = ([], [], var_str, res)
            self.var_dict[var_str] = Symbol(var_str)

        for connection in M.loc:
            # I need to further split these if blocks into a separate function
            if (
                connection.ctype == "scale_with_concentration"
                or connection.ctype == "regular"
            ):
                if connection.rate != "None":
                    self.scale_w_concentration_or_regular(connection)
                else:
                    # if the rate is not given, it assumes a variable rate
                    self.variable_rate(connection)

            elif connection.ctype == "scale_with_mass":
                self.scale_with_mass(connection)

            elif connection.ctype == "scale_with_flux":
                if isinstance(M.burial.ref_flux.parent, type(connection)):
                    self.scale_with_flux(connection)
                else:
                    raise TypeError("Please input proper ref_flux for scale_with_flux")
            else:
                raise TypeError("Please check all connection types inputted")

    # the methods have the "factor" parameter that is there in order
    # to be able to add terms to the equations that are some already existing
    # terms in that equation from other connections but scaled by a factor
    # the default is 1

    # if the equation term is scaled off of another term, then the term
    # generating methods the factor_scaling_connection is set to be equal the
    # the method that is generating this scaled term otherwise, factor_scaling_connection
    # parameter is set at None by default

    def scale_w_concentration_or_regular(
        self, connection, factor=1, factor_scaling_connection=None
    ):
        """This method is used when we work with regular or
        scale_with_concentration connections where the rate
        provided. The term it genereates in a constant and is
        the provided rate itself"""
        if factor_scaling_connection is None:
            self.res_flux_filler(connection.rate * factor, connection)
        else:
            self.res_flux_filler(connection.rate * factor, factor_scaling_connection)

    def scale_with_flux(self, connection):
        """Given a reference flux, we calculate the
        new flux as a percentage of the given flux.
        We don't need a factor parameter (which is the scale)
        here as this method itself is using the factor parameter
        granted by the connection that gets inputted 
        """
        original_flux = connection.ref_flux.parent
        factor = connection.scale

        if (
            original_flux.ctype == "scale_with_concentration"
            or original_flux.ctype == "regular"
        ):
            if original_flux.rate != "None":
                self.scale_w_concentration_or_regular(original_flux, factor, connection)
            else:
                # if the rate is not given, it assumes a variable rate
                self.variable_rate(original_flux, factor, connection)

        elif original_flux.ctype == "scale_with_mass":
            self.scale_with_mass(original_flux, factor, connection)

        else:
            raise TypeError("Please input proper ref_flux connection type")

    def scale_with_mass(self, connection, factor=1, factor_scaling_connection=None):
        """The variable for this equation term is the
        same as for the source variable and the coeff
        is the product of the scale (1/tau) multipled
        by the volume of the source reservoir"""

        coeff = connection.source.volume * connection.scale
        var_str = self.res_flux[id(connection.source)][2]

        if factor_scaling_connection is None:
            self.res_flux_filler(coeff * self.var_dict[var_str] * factor, connection)
        else:
            self.res_flux_filler(
                coeff * self.var_dict[var_str] * factor, factor_scaling_connection
            )

    def variable_rate(
        self, connection, factor: Union[float, int] = 1, factor_scaling_connection=None
    ):

        """If the connection is of scale_with_concentration type but without a provided
        rate, then it will be calculated as scale*x-variable where x-variable is the source
        reservoir corresponding x variable as x_id(Reservoir). Since these equations will
        further need to be evaluated, the x-variable is not a string but in a sympy.Symbol
        class object with the same name"""

        scale_mag = connection.scale
        var_str = self.res_flux[id(connection.source)][2]

        if factor_scaling_connection is None:
            self.res_flux_filler(
                scale_mag * self.var_dict[var_str] * factor, connection
            )
        else:
            self.res_flux_filler(
                scale_mag * self.var_dict[var_str] * factor, factor_scaling_connection
            )

    def res_flux_filler(self, input, connection):
        """Fills the influx and outflux lists in the res_flux dictionary
        filling the res_flux dict depending on the source and sink of the flux

        - if both source and sink are reservoirs the input (which is the equation term in
          form of a Sympy object) will be filled into both the influx and outflux lists of
          corresponding reservoirs

        - if only the source is a reservoir, then the input will be filled into the source
          list of the corresponding reservoir

        - if only the sink is a reservoir, then the input will be filled into the sink
          list of the corresponding reservoir"""

        type_reservoir = type(self.M.lor[0])

        if isinstance(connection.source, type_reservoir) and isinstance(
            connection.sink, type_reservoir
        ):
            self.res_flux[id(connection.source)][1].append(
                input
            )  # outflowing is the first list
            self.res_flux[id(connection.sink)][0].append(
                input
            )  # inflowing is the second list

        elif isinstance(
            connection.source, type_reservoir
        ):  # and not isinstance(connection.sink, Reservoir)
            # outflowing list
            self.res_flux[id(connection.source)][1].append(input)

        else:
            # inflowing list
            self.res_flux[id(connection.sink)][0].append(input)

        return self.res_flux
