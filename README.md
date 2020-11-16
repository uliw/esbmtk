
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#org8459e67)
2.  [News](#org3d03696)
3.  [Contributing](#orgb476a5d)
4.  [Installation](#orgfafabf4)
5.  [Documentation](#org4642bd0)
6.  [Todo](#orgc917b2e)
7.  [License](#org27820a0)


<a id="org8459e67"></a>

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


<a id="org3d03696"></a>

# News

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


<a id="orgb476a5d"></a>

# Contributing

Don't be shy. Contributing is as easy as finding bugs by using the
code, or maybe you want to add a new process code? If you have plenty
of time to spare, ESMBTK could use a solver for stiff problems, or a
graphical interface ;-) See the todo section for ideas.


<a id="orgfafabf4"></a>

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


<a id="org4642bd0"></a>

# Documentation

The documentation is available in org format or in pdf format. 
See the documentation folder, [specifically the quickstart guide](https://github.com/uliw/esbmtk/blob/master/Documentation/ESBMTK-Quick-Start_Guide.org).

At present, I also provide the following example cases (as py-files
and in jupyter notebook format)

-   A trivial carbon cycle model which shows how to set up the model,
    and read an external csv file to force the model.
-   


<a id="orgc917b2e"></a>

# Todo

-   expand the documentation
-   provide more examples
-   do more testing


<a id="org27820a0"></a>

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

