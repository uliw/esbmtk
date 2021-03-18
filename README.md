
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#orgdf00cee)
2.  [News](#org9ae8582)
3.  [Contributing](#org20dc123)
4.  [Installation](#org8bd7ec3)
5.  [Documentation](#org2b8eeb9)
6.  [Todo](#org75dfa96)
7.  [License](#org794af26)


<a id="orgdf00cee"></a>

# ESBMTK - An Earth-sciences box modeling toolkit

ESBMTK is python library which aims to simplify typical box modeling
projects the in Earth-Sciences. The general focus is to make box
modeling more approachable for classroom teaching. So performance and
scalability currently no priority. Specifically, the solver is just a
simple forward Euler scheme, so stiff problems are not handled
gracefully.

At present, it will calculate masses/concentrations in reservoirs and
fluxes including isotope ratios. It provides a variety of classes
which allow the creation and manipulation of input signals, and the
generation of graphical results.


<a id="org9ae8582"></a>

# News

-   March 18<sup>th</sup> esbmtk 0.4.0.0 now has a carbonate chemistry module
    which currently includes methods to calculate PCO<sub>2</sub>, CA, and H<sup>+</sup>
    concentrations from TA and DIC. The seawater class has been renamed
    `SeawaterConstants` and provides access to a limited set of
    seawater species concentrations and their K and Pk constants at
    given set of temperature, salinity and pressure conditions. This
    version also includes some refactoring in the `Connnection` and
    `ConnectionmGroup` classes. It is likely that this broke some
    connection types.

-   March 13<sup>th</sup>, cleaned up the use of the `k_value` keyword which is
    now restricted to the `flux_balance` connection type. In all other
    instances use the `scale` keyword instead. The old keyword is still
    working, but will print a warning message. The `describe()` method
    is now called `info()`.

-   March 11<sup>th</sup>, added a seawater class which provides access to
    K-values, and concentrations.

-   March 10<sup>th</sup>, the code documentation is now available at <https://uliw.github.io/esbmtk/esbmtk/index.html>

-   March 6<sup>th</sup>, the plot reservoir function now takes and additional
    filename argument e.g., (fn="foo.pdf"). Signals now accept an
    optional reservoir argument. This simplifies signal creation as the
    source and reservoir connection can be created implicitly.

-   Feb. 28<sup>th</sup>, added a VirtualReservoir class. This class allows the
    definition of reservoirs which depend on the execution of a
    user-defined function. See the class documentation for details.
    
    Display precision can now be set independently for each Reservoir,
    Flux, Signal, Datafield and VirtualReservoir

-   Jan. 30<sup>th</sup>, added oxygen and nitrogen species definitions

-   Jan. 18<sup>th</sup>, Reading a previous model state is now more robust. It no
    longer requires the models model have the same numbers of
    fluxes. It will attempt to match by name, and print a warning for
    those fluxes it could not match.

-   Jan. 12<sup>th</sup>, The model object now accepts a `plot_style` keyword

-   Jan. 5<sup>th</sup>, Connector objects and fluxes use now a more consistent
    naming scheme: `Source_2_Sink_Connector`, and the associated flux
    is named `Source_2_Sink_Flux`. Processes acting on flux are named
    `Source_2_Sink_Pname`
    
    The model type (`m_type`) now defaults to `mass_only`, and will
    ignore isotope calculations. Use `m_type = "both"` to get the old
    behavior.

-   Dec. 30<sup>th</sup>, the connection object has now a generalized update
    method which allows to update all or a subset of all parameters

-   Dec. 23<sup>rd</sup>, the connection object has now the basic machinery to
    allow updates to the connection properties after the connection has
    been established. If need be, updates will trigger a change to the
    connection type and re-initialize the associated processes. At
    present this works for changes to the rate, the fractionation
    factor, possibly delta.

-   Dec. 20<sup>th</sup>, added a new connection type (`flux_balance`) which
    allows equilibration fluxes between two reservoirs without the need
    to specify forward and backwards fluxes explicitly. See the
    equilibration example in the example directory.

-   Dec. 9<sup>th</sup>, added a basic logging infrastructure. Added `describe()`
    method to `Model`, `Reservoir` and `Connnection` classes. This will
    list details about the fluxes and processes etc. Lot's of code
    cleanup and refactoring.

-   Dec. 7<sup>th</sup>, When calling an instance without arguments, it now
    returns the values it was initialized with. In other words, it will
    print the code which was used to initialize the instance.

-   Dec. 5<sup>th</sup>, added a DataField Class. This allows for the integration of data
    which is computed after the model finishes into the model summary
    plots.

-   Nov. 26<sup>th</sup>  Species definitions now accept an optional display string. This
    allows pretty printed output for chemical formulas.

-   Nov. 24<sup>th</sup> New functions to list all connections of a reservoir, and
    to list all processes associated with a connection. This allows the
    use of the help system on process names. New interface to specify
    connections with more complex characteristics (e.g., scale a flux
    in response to reservoir concentration). This will breaks existing
    scripts which use these kind of connections. See the Quickstart
    guide how to change the connection definition.

-   Nov. 23<sup>rd</sup> A model can now save it's state, which can then be used
    to initialize a subsequent model run. This is particularly useful
    for models which require a spin up phase to reach equilibrium

-   Nov. 18<sup>th</sup>, started to add unit tests for selected modules. Added
    unit conversions to external data sets. External data can now be
    directly associated with a reservoir.

-   Nov. 5<sup>th</sup>, released version 0.2. This version is now unit aware. So
    rather than having a separate keyword for `unit`, quantities are
    now specified together wit their unit, e.g., `rate = "15
       mol/s"`. This breaks the API, and requires that existing scripts
    are modified. I thus also removed much of the existing
    documentation until I have time to update it.

-   Oct. 27<sup>th</sup>, added documentation on how to integrate user written
    process classes, added a class which allows for concentration
    dependent flux. Updated the documentation, added examples

-   Oct. 25<sup>th</sup>, Initial release on github.


<a id="org20dc123"></a>

# Contributing

Don't be shy. Contributing is as easy as finding bugs by using the
code, or maybe you want to add a new process code? If you have plenty
of time to spare, ESMBTK could use a solver for stiff problems, or a
graphical interface ;-) See the todo section for ideas.


<a id="org8bd7ec3"></a>

# Installation

ESBMTK relies on the following python versions and libraries

-   python > 3.6
-   matplotlib
-   numpy
-   pandas
-   typing
-   nptyping
-   pint

If you work with conda, it is recommended to install the above via
conda. If you work with pip, the installer should install these
libraries automatically. ESBMTK itself can be installed with pip

-   pip install esbmtk


<a id="org2b8eeb9"></a>

# Documentation

The documentation is available in org format or in pdf format. 
See the documentation folder, [specifically the quickstart guide](https://github.com/uliw/esbmtk/blob/master/Documentation/ESBMTK-Quick-Start_Guide.org).

At present, I also provide the following example cases (as py-files
and in jupyter notebook format)

-   A trivial carbon cycle model which shows how to set up the model,
    and read an external csv file to force the model.
-   


<a id="org75dfa96"></a>

# Todo

-   expand the documentation
-   provide more examples
-   do more testing


<a id="org794af26"></a>

# License

ESBMTK: A general purpose Earth Science box model toolkit
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

