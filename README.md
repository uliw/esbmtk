
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#org35fb3ae)
2.  [News](#orgdd001fe)
3.  [Contributing](#orgce391b2)
4.  [Installation](#org57ec755)
5.  [Documentation](#org9af042b)
6.  [Todo](#org3fe5993)
7.  [License](#orgfa74a0a)


<a id="org35fb3ae"></a>

# ESBMTK - An Earth-sciences box modeling toolkit

ESBMTK is python library which aims to simplify typical box modeling
projects the in Earth-Sciences. I started this project to teach myself
python and explore object oriented programming. The code became
however functional enough that the focus of this project has shifted
to make box modeling more approachable for classroom teaching. So
performance and scalability are not great. Specifically, the solver is
just a simple forward Euler scheme, so stiff problems are not handled
gracefully.

At present, it will calculate masses/concentrations in reservoirs and
fluxes including isotope ratios. It provides a variety of classes
which allow the creation and manipulation of input signals, and the
generation of graphical results. There is however no support for
chemical reactions (including equilibrium reactions).


<a id="orgdd001fe"></a>

# News

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


<a id="orgce391b2"></a>

# Contributing

Don't be shy. Contributing is as easy as finding bugs by using the
code, or maybe you want to add a new process code? If you have plenty
of time to spare, ESMBTK could use a solver for stiff problems, or a
graphical interface ;-) See the todo section for ideas.


<a id="org57ec755"></a>

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


<a id="org9af042b"></a>

# Documentation

The documentation is available in org format or in pdf format. 
See the documentation folder, [specifically the quickstart guide](https://github.com/uliw/esbmtk/blob/master/Documentation/ESBMTK-Quick-Start_Guide.org).

At present, I also provide the following example cases (as py-files
and in jupyter notebook format)

-   A trivial carbon cycle model which shows how to set up the model,
    and read an external csv file to force the model.
-   


<a id="org3fe5993"></a>

# Todo

-   expand the documentation
-   provide more examples
-   do more testing


<a id="orgfa74a0a"></a>

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

