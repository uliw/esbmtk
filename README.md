
# Table of Contents

1.  [ESBMTK - An Earth-sciences box modeling toolkit](#orgd114727)
2.  [News](#orge9abfe5)
3.  [Contributing](#org10786eb)
4.  [Installation](#org9904f8c)
5.  [Documentation](#orgff85c0b)
6.  [License](#org8820bbd)


<a id="orgd114727"></a>

# ESBMTK - An Earth-sciences box modeling toolkit

ESBMTK is python library which aims to simplify typical box modeling
projects the in Earth-Sciences. I started this project to teach myself
python and explore object oriented programming.  The code became
however functional enough that the focus of this project has shifted
to make box modeling more approachable for classroom teaching. So
performance and scalability are not great. Specifically, the solved is
a simple forward Euler scheme, so stiff problems are not handled
gracefully.

At present, it will calculate masses/concentrations in reservoirs and
fluxes including isotope ratios. It provides a variety of classes
which allow the creation and manipulation of input signals, and the
generation of graphical results. There is however no support for
chemical reactions (including equilibrium reactions).


<a id="orge9abfe5"></a>

# News

-   Oct. 27<sup>th</sup>, added documentation on how to integrate user written
    process classes, added a class which allows for concentration
    dependent flux. Updated the documentation, added examples
-   Oct. 25<sup>th</sup>, Initial release on github.


<a id="org10786eb"></a>

# Contributing

Don't be shy. Contributing is as easy as finding bugs by using the
code, or maybe you want to add a new process code? If you have plenty
of time to spare, ESMBTK could use a solver for stiff problems, or a
graphical interface ;-)


<a id="org9904f8c"></a>

# Installation

ESBMTK relies on the following python versions and libraries

-   python > 3.6
-   matplotlib
-   numpy
-   pandas
-   logging
-   nptyping
-   numbers

please install the above with pip or conda etc. You can install esbmtk with pip or in a local working directory from github

-   pip install esbmtk
-   <https://github.com/uliw/esbmtk/archive/main.zip>
-   <https://github.com/uliw/esbmtk.git>
-   git@github.com:uliw/esbmtk.git


<a id="orgff85c0b"></a>

# Documentation

See the documentation folder, [specifically the quickstart guide](https://github.com/uliw/esbmtk/blob/main/Documentation/ESBMTK-Quick-Start_Guide.org).

At present, I also provide the following example cases (as py-files
and in jupyter notebook format)

-   A trivial carbon cycle model which shows how to set up the model,
    and read an external [csv file to force the model](https://github.com/uliw/esbmtk/blob/main/Examples/A%20simple%20carbon%20cycle%20example/C_cycle.org)
-   The same model as be before but now to demonstrate how to add
    pyramid shaped signal, and how to use the rate constant process to
    adjust concentration dependent flux rates[concentration dependent flux rates](https://github.com/uliw/esbmtk/blob/main/Examples/Using%20a%20rate%20constant/rate_example.org).

Last but not least, I added a short [guide how to add your own process
classes to the ESBMTK](https://github.com/uliw/esbmtk/blob/main/Documentation/Adding_your_own_Processes.org) 


<a id="org8820bbd"></a>

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

