from .esbmtk import Q_, Model, Reservoir, Connect, Source, Sink
from sympy import Symbol
import numpy as np
from typing import Union

class EQ_Terms:
    '''terms of ode equations'''
    def __init__(self, M):
        self.res_flux = {} # dict that holds variables, reservoirse and in/out-flux (keys are int [res ids])
        self.var_dict = {} # dict that holds variables {'x_123':Symbol('x_123')}
        self.M = M

        for res in self.M.lor:
            # influx, outflux lists, concentration variable for a reservoir
            var_str = f'x_{id(res)}'
            self.res_flux[id(res)] = ([], [], var_str, res)
            self.var_dict[var_str] = Symbol(var_str)

        for connection in M.loc:
            # I need to further split these if blocks into a separate function
            if connection.ctype == "scale_with_concentration" or connection.ctype == "regular":
                if connection.rate != "None":
                    self.scale_w_concentration_or_regular(connection)
                else:
                    # if the rate is not given, it assumes a variable rate
                    self.variable_rate(connection)

            elif connection.ctype == "scale_with_mass":
                self.scale_with_mass(connection)

            elif connection.ctype == "scale_with_flux":
                if isinstance(M.burial.ref_flux.parent, Connect):
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

    def scale_w_concentration_or_regular(self, connection: Connect, factor=1, factor_scaling_connection=None):
        '''this method is used when we work with regular or
        scale_with_concentration connections'''
        if factor_scaling_connection is None:
            self.res_flux_filler(connection.rate*factor, connection)
        else:
            self.res_flux_filler(connection.rate*factor, factor_scaling_connection)

    def scale_with_flux(self, connection: Connect):
        '''given a reference flux, we calculate the
        new flux as a percentage of that other flux'''
        # we don't need a factor parameter here as this method
        # itself is using the factor parameters from other methods

        original_flux = connection.ref_flux.parent
        factor = connection.scale

        if original_flux.ctype == "scale_with_concentration" or original_flux.ctype == "regular":
            if original_flux.rate != "None":
                self.scale_w_concentration_or_regular(original_flux, factor, connection)
            else:
                # if the rate is not given, it assumes a variable rate
                self.variable_rate(original_flux, factor, connection)

        elif original_flux.ctype == "scale_with_mass":
            self.scale_with_mass(original_flux, factor, connection)

        else:
            raise TypeError("Please input proper ref_flux connection type")


    def scale_with_mass(self, connection: Connect, factor=1, factor_scaling_connection=None):
        '''the variable for this equation term is the
        same as for the source variable and the coeff
        is the product of the scale (1/tau) multipled
        by the volume of the source reservoir '''
        coeff = connection.source.volume*connection.scale
        var_str = self.res_flux[id(connection.source)][2]

        if factor_scaling_connection is None:
            self.res_flux_filler(
                coeff*self.var_dict[var_str]*factor,
                connection
                )
        else:
            self.res_flux_filler(
                coeff*self.var_dict[var_str]*factor,
                factor_scaling_connection
                )


    def variable_rate(self, connection: Connect, factor: Union[float, int] =1, 
                        factor_scaling_connection: Union[None, Connect] =None):

        '''if the connection has a specified rate then
        use this one to get the units matching'''

        scale_mag = connection.scale
        var_str = self.res_flux[id(connection.source)][2]

        if factor_scaling_connection is None:
            self.res_flux_filler(
                scale_mag*self.var_dict[var_str]*factor,
                connection
                )
        else:
            self.res_flux_filler(
                scale_mag*self.var_dict[var_str]*factor,
                factor_scaling_connection
                )          

    def res_flux_filler(self, input, connection: Connect):
        ''' Fills the influx and outflux lists in the res_flux dictionary
            filling the res_flux dict depending on the source and sink of the flux
            
            - if both source and sink are reservoirs the input (which is the equation term in 
              form of a Sympy object) will be filled into both the influx and outflux lists of
              corresponding reservoirs
            
            - if only the source is a reservoir, then the input will be filled into the source
              list of the corresponding reservoir
            
            - if only the sink is a reservoir, then the input will be filled into the sink
              list of the corresponding reservoir'''

        if isinstance(connection.source, Reservoir) and isinstance(connection.sink, Reservoir):
            self.res_flux[id(connection.source)][1].append(input) # outflowing is the first list
            self.res_flux[id(connection.sink)][0].append(input) # inflowing is the second list

        elif isinstance(connection.source, Reservoir): # and not isinstance(connection.sink, Reservoir)
            # outflowing list
            self.res_flux[id(connection.source)][1].append(input)

        else:
            # inflowing list
            self.res_flux[id(connection.sink)][0].append(input)

        return self.res_flux
